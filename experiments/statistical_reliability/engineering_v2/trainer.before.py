"""
================================================================================
MARS Trainer - 噪声消融实验版本
================================================================================
"""
import os
os.environ["XFORMERS_DISABLED"] = "1"
import time
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tool import calculate_metric, calculate_average_metric, plot_sample_cv2
from .anomaly_mamba import AnomalyMamba


class _PadToMultiple:
    """把张量 (C,H,W) 的 H,W 补边到 multiple 的倍数 (右/下方补边, 不缩放)."""

    def __init__(self, multiple=14, fill=0.0):
        self.multiple = multiple
        self.fill = fill

    def __call__(self, t):
        h, w = t.shape[-2], t.shape[-1]
        ph = (self.multiple - h % self.multiple) % self.multiple
        pw = (self.multiple - w % self.multiple) % self.multiple
        if ph or pw:
            t = torch.nn.functional.pad(t, (0, pw, 0, ph), value=self.fill)
        return t


class _CropLR:
    """裁掉 PIL 图像左右两侧各 n 个像素 (陶瓷边缘的黄边非异常, 不裁上下)."""

    def __init__(self, n=0):
        self.n = n

    def __call__(self, img):  # PIL.Image
        w, h = img.size
        if self.n > 0 and w > 2 * self.n:
            img = img.crop((self.n, 0, w - self.n, h))
        return img


class AnomalyMambaTrainer(nn.Module):
    def __init__(
            self,
            backbone='wide_resnet50_2',
            norm=False,
            hierarchies=[1, 2, 3],
            depths=[2, 2],
            valid_path=0,
            bos_length=4,
            pred_next_n=0,
            lambda_spatial=1.,
            learnable_bos=False,
            use_pe=False,
            is_proj=True,
            use_path_weights=False,
            image_size=256,
            device='cuda:0',
            learning_rate=1e-3,
            pretrained=None,
            adapter=['linear'],
            use_sequence_prediction=True,
            seq_loss_weight=0.5,
            # ⭐ 噪声消融参数
            noise_type='divnoise',
            noise_pattern='all',
            noise_op='all',
            noise_ratio_min=0.02,
            noise_ratio_max=0.22,
            clean_ratio=0.5,
            decoder_type='vssm',
            native_res=False,
            edge_crop=0,
            loss_type='mse_l1',
    ):
        super(AnomalyMambaTrainer, self).__init__()
        self.device = device
        self.native_res = native_res

        # 原生分辨率模式下解码器与尺寸无关，仅用一个小方形 dummy 构建解码器即可
        cache_size = 14 * 37 if native_res else image_size
        cache_data = torch.zeros((1, 3, cache_size, cache_size)).to(device)
        self.model = AnomalyMamba(
            pred_next_n=pred_next_n,
            backbone=backbone,
            hierarchies=hierarchies,
            norm=norm,
            depths=depths,
            bos_length=bos_length,
            valid_path=valid_path,
            lambda_spatial=lambda_spatial,
            image_size=image_size,
            learnable_bos=learnable_bos,
            use_pe=use_pe,
            is_proj=is_proj,
            cache_data=cache_data,
            use_path_weights=use_path_weights,
            adapter=adapter,
            use_sequence_prediction=use_sequence_prediction,
            seq_loss_weight=seq_loss_weight,
            # ⭐ 传递噪声参数
            noise_type=noise_type,
            noise_pattern=noise_pattern,
            noise_op=noise_op,
            noise_ratio_min=noise_ratio_min,
            noise_ratio_max=noise_ratio_max,
            clean_ratio=clean_ratio,
            decoder_type=decoder_type,
            loss_type=loss_type,
        )
        self.model.to(device)
        
        mean_train = [0.485, 0.456, 0.406]
        std_train = [0.229, 0.224, 0.225]
        self.image_size = image_size

        # 先裁掉左右黄边 (若启用), 再做后续变换; 图像与 mask 使用同一裁剪保证对齐
        crop_prefix = [_CropLR(edge_crop)] if edge_crop > 0 else []
        if native_res:
            # 原生分辨率: 不 resize, 仅把长宽补边到 14 的倍数 (右/下方补边)
            pad14 = _PadToMultiple(multiple=14, fill=0.0)
            self.data_transform = transforms.Compose(crop_prefix + [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_train, std=std_train),
                pad14,
            ])
            self.target_transform = transforms.Compose(crop_prefix + [
                transforms.ToTensor(),
                pad14,
            ])
        else:
            self.data_transform = transforms.Compose(crop_prefix + [
                transforms.Resize((image_size, image_size),
                                  interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_train, std=std_train)
            ])

            self.target_transform = transforms.Compose(crop_prefix + [
                transforms.Resize((image_size, image_size),
                                  interpolation=transforms.InterpolationMode.NEAREST),
                transforms.CenterCrop(image_size),
                transforms.ToTensor()
            ])

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate,
            betas=(0.5, 0.999)
        )

        if pretrained:
            self.load(pretrained)

    @staticmethod
    def setup_paths(args):
        save_root = os.path.join(args.save_path, 'VarAD')
        model_root = os.path.join(save_root, 'models')
        log_root = os.path.join(save_root, 'logs')
        csv_root = os.path.join(save_root, 'csvs')
        image_root = os.path.join(save_root, 'images')
        tensorboard_root = os.path.join(save_root, 'tensorboard')

        os.makedirs(model_root, exist_ok=True)
        os.makedirs(log_root, exist_ok=True)
        os.makedirs(csv_root, exist_ok=True)
        os.makedirs(image_root, exist_ok=True)
        os.makedirs(tensorboard_root, exist_ok=True)

        # ⭐ 模型名包含噪声类型
        noise_type = getattr(args, 'noise_type', 'all')
        decoder_type = getattr(args, 'decoder_type', 'vssm')
        model_name = '{:}-{:}-{:}-{:}-{:}-{:}-{:}-noise_{:}'.format(
            args.dataset, args.model,
            args.image_size, args.pred_next_n,
            args.adapter, args.hierarchies, args.valid_path,
            noise_type
        )
        # ortho 模式下把空间模板/腐蚀算子并入模型名，避免不同消融组合互相覆盖
        if noise_type == 'ortho':
            noise_pattern = getattr(args, 'noise_pattern', 'all')
            noise_op = getattr(args, 'noise_op', 'all')
            model_name = f'{model_name}-pat_{noise_pattern}-op_{noise_op}'
        if decoder_type != 'vssm':
            model_name = f'{model_name}-decoder_{decoder_type}'

        lambda_spatial = getattr(args, 'lambda_spatial', 1.0)
        seq_loss_weight = getattr(args, 'seq_loss_weight', 0.5)
        model_name = f'{model_name}-lam{lambda_spatial}-seq{seq_loss_weight}'

        # loss_type 仅在非默认 'mse_l1' 时并入模型名, 避免改变既有 checkpoint 命名,
        # 同时防止 mse/l1/mse_l1 三者在同一目录互相覆盖
        loss_type = getattr(args, 'loss_type', 'mse_l1')
        if loss_type != 'mse_l1':
            model_name = f'{model_name}-loss{loss_type}'

        ckp_path = os.path.join(model_root, f'{model_name}-{args.category}')

        tensorboard_dir = os.path.join(tensorboard_root, f'{model_name}-{args.category}')
        if os.path.exists(tensorboard_dir):
            import shutil
            shutil.rmtree(tensorboard_dir, ignore_errors=True)
        tensorboard_logger = SummaryWriter(log_dir=tensorboard_dir)

        csv_path = os.path.join(csv_root, f'{model_name}.csv')

        image_dir = os.path.join(image_root, f'{model_name}')
        os.makedirs(image_dir, exist_ok=True)

        log_path = os.path.join(log_root, f'{model_name}.txt')

        return model_name, image_dir, csv_path, log_path, ckp_path, tensorboard_logger

    def save(self, path):
        self.save_dict = {}
        for param, value in self.state_dict().items():
            self.save_dict[param] = value
        torch.save(self.save_dict, path)

    def load(self, path):
        # Legacy checkpoints without BOS cannot guarantee identical predictions.
        self.load_state_dict(torch.load(path, map_location=self.device, weights_only=True), strict=True)

    def train_epoch(self, loader):
        self.model.train()
        loss_list = []
        for idx, items in enumerate(loader):
            data = items['img']
            data = data.to(self.device)
            outputs = self.model(data)
            loss = self.model.cal_loss(outputs)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_list.append(loss.item())
        return np.mean(loss_list)

    @torch.no_grad()
    @torch.cuda.amp.autocast()
    def compute_fps(self, dataloader):
        global_image = None
        for indx, items in enumerate(dataloader):
            image = items['img'].to(self.device)
            outputs = self.model(image)
            anomaly_map = self.model.cal_am(outputs)
            global_image = image
        global_image = global_image[0:1, :, :, :]
        time_begin = time.time()
        image_number = 0
        for _ in range(100):
            outputs = self.model(global_image)
            anomaly_map = self.model.cal_am(outputs)
            image_number += global_image.shape[0]
        time_end = time.time()
        try:
            from thop import profile
            flops, params = profile(self.model, (global_image,))
            print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
        except:
            pass
        fps = image_number / (time_end - time_begin)
        return fps

    @torch.no_grad()
    def evaluation(self, dataloader, obj_list, save_fig, save_fig_dir=None, cal_pro=False, by_specie=False):
        self.model.eval()
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        from tqdm import tqdm
        results = {}
        results['cls_names'] = []
        results['imgs_gts'] = []
        results['anomaly_scores'] = []
        results['imgs_masks'] = []
        results['anomaly_maps'] = []
        results['imgs'] = []
        results['names'] = []
        results['specie_name'] = []
        image_indx = 0
        for indx, items in enumerate(tqdm(dataloader, desc="Evaluating")):
            path = items['img_path']
            specie_name = items['specie_name']
            cls_name = items['cls_name']
            for _cls_name, _specie_name in zip(cls_name, specie_name):
                image_indx += 1
                results['names'].append('{:}-{:}-{:03d}'.format(_cls_name, _specie_name, image_indx))
                results['specie_name'].append(_specie_name)
            gt_mask = items['img_mask']
            gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
            for _gt_mask in gt_mask:
                mask_np = _gt_mask.squeeze(0).numpy()
                if mask_np.shape[0] > 512:
                    mask_np = cv2.resize(mask_np, (512, 512), interpolation=cv2.INTER_NEAREST)
                results['imgs_masks'].append(mask_np)
            is_anomaly = np.array(items['anomaly'])
            for _is_anomaly in is_anomaly:
                results['imgs_gts'].append(_is_anomaly)
            if save_fig:
                for _path in path:
                    vis_image = cv2.resize(cv2.imread(_path), (512, 512))
                    results['imgs'].append(vis_image)
            image = items['img'].to(self.device)
            cls_name = items['cls_name']
            results['cls_names'].extend(cls_name)
            outputs = self.model(image)
            anomaly_map = self.model.cal_am(outputs)
            anomaly_score = [np.max(s) for s in anomaly_map]
            for _anomaly_map, _anomaly_score in zip(anomaly_map, anomaly_score):
                if _anomaly_map.shape[0] > 512:
                    _anomaly_map = cv2.resize(_anomaly_map, (512, 512))
                results['anomaly_maps'].append(_anomaly_map)
                results['anomaly_scores'].append(_anomaly_score)
            del image, outputs, anomaly_map
            torch.cuda.empty_cache()
            import sys
            sys.stdout.flush()
        if save_fig:
            print('saving fig.....')
            plot_sample_cv2(
                results['names'],
                results['imgs'],
                {'mamba': results['anomaly_maps']},
                results['imgs_masks'],
                save_fig_dir,
                norm_by_sample=True
            )
        if by_specie:
            specie_set = sorted(list(set(results['specie_name'])))
        metric_dict = dict()
        for obj in obj_list:
            if by_specie:
                for specie in specie_set:
                    if specie == 'good':
                        continue
                    else:
                        metric_dict[f'{obj}-{specie}'] = dict()
            else:
                metric_dict[obj] = dict()
        for obj in obj_list:
            if by_specie:
                for specie in specie_set:
                    if specie == 'good':
                        continue
                    else:
                        metric = calculate_metric(results, obj, cal_pro, by_specie, specie)
                        obj_full_name = f'{obj}-{specie}'
                        metric_dict[obj_full_name] = metric
            else:
                metric = calculate_metric(results, obj, cal_pro)
                obj_full_name = f'{obj}'
                metric_dict[obj_full_name] = metric
        metric_dict['Average'] = calculate_average_metric(metric_dict)
        return metric_dict