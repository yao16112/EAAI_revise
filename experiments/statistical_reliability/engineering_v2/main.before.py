"""
================================================================================
VarAD Main - 支持批量训练所有 MVTec AD 类别
================================================================================
用法:
    python main.py --image_size 512 --model dinov2_vits14  # 训练所有类别
    python main.py --category bottle  # 只训练单个类别
    python main.py --category bottle cable capsule  # 训练多个指定类别
================================================================================
"""

import os
import json
from pathlib import Path
from config import PATHS
import warnings

# 忽略 FutureWarning 警告
warnings.filterwarnings("ignore", category=FutureWarning)

# 禁用 xformers 以兼容新 GPU (必须在 import torch 之前)
os.environ["XFORMERS_DISABLED"] = "1"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import torch

# Importing from local modules
from dataset import get_dataset
from model.VarAD.trainer import AnomalyMambaTrainer
from tool import save_metric, setup_seed, Logger, log_metrics



# MVTec AD 数据集的 15 个类别
MVTEC_CATEGORIES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]

VISA_CATEGORIES = [
    'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
    'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3',
    'pcb4', 'pipe_fryum'
]

TAOCI_CATEGORIES = ['taoci']
BTAD_CATEGORIES=[
    '01','02','03'
]

DTD_CATEGORIES = [
    'Blotchy_099', 'Fibrous_183', 'Marbled_078', 'Matted_069',
    'Mesh_114', 'Perforated_037', 'Stratified_154', 'Woven_001',
    'Woven_068', 'Woven_104', 'Woven_125', 'Woven_127'
]

def build_trainer(args, device):
    return AnomalyMambaTrainer(
        backbone=args.model,
        norm=True,
        depths=[1],
        hierarchies=args.hierarchies,
        pred_next_n=args.pred_next_n,
        valid_path=args.valid_path,
        image_size=args.image_size,
        learning_rate=args.learning_rate,
        device=device,
        pretrained=args.ckt_path,
        is_proj=args.is_proj,
        adapter=args.adapter,
        use_sequence_prediction=args.use_seq_pred,
        lambda_spatial=args.lambda_spatial,
        seq_loss_weight=args.seq_loss_weight,
        noise_type=args.noise_type,
        noise_pattern=args.noise_pattern,
        noise_op=args.noise_op,
        noise_ratio_min=args.noise_ratio_min,
        noise_ratio_max=args.noise_ratio_max,
        clean_ratio=args.clean_ratio,
        decoder_type=args.decoder_type,
        native_res=getattr(args, 'native_res', False),
        edge_crop=getattr(args, 'edge_crop', 0),
        loss_type=getattr(args, 'loss_type', 'mse_l1'),
    )


def train_single_category(args, category):
    """训练单个类别"""
    setup_seed(args.seed)
    # Configurations
    epochs = args.epoch
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    image_size = args.image_size
    
    # 使用指定的GPU
    if torch.cuda.is_available():
        device = f'cuda:{args.gpu}'
        print("1111111111111111111")
    else:
        device = 'cpu'
        print("2222222222222222222")

    # 临时修改 args.category 为当前类别
    original_category = args.category
    args.category = category

    # Set up paths
    model_name, image_dir, csv_path, log_path, ckp_path, tensorboard_logger = AnomalyMambaTrainer.setup_paths(args)

    # Extract configurations
    save_fig = args.save_fig
    do_train = args.do_train
    cal_pro = args.cal_pro

    # Logger
    logger = Logger(log_path)

    # Print basic information
    for key, value in sorted(vars(args).items()):
        logger.info(f'{key} = {value}')

    logger.info('Model name: {:}'.format(model_name))

    model = build_trainer(args, device)

    train_data_cls_names, train_data, train_data_root = get_dataset(
        dataset_types=args.dataset,
        transform=model.data_transform,
        target_transform=model.target_transform,
        training=True, mode='FS', k_shot=1e6, synthesis_anomalies=False, class_name_list=[[category]])

    test_data_cls_names, test_data, test_data_root = get_dataset(
        dataset_types=args.dataset,
        transform=model.data_transform,
        target_transform=model.target_transform,
        training=False, mode='FS', k_shot=0, synthesis_anomalies=False, class_name_list=[[category]])

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)  # 测试时用batch_size=1

    if args.compute_fps:
        torch.backends.cudnn.benchmark = True
        fps_test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
        fps = model.compute_fps(fps_test_dataloader)
        print(f'fps: {fps:.2f}')
        args.category = original_category
        return None

    by_specie = (args.dataset == 'mvtec_loco')

    final_metrics = None

    config_path = Path(ckp_path + '_config.json')
    config_path.write_text(json.dumps({"arguments": vars(args), "paths": PATHS,
                                      "checkpoint_selection": "fixed_final_epoch",
                                      "f1_threshold": "test_oracle"}, indent=2))
    if do_train:
        for epoch in tqdm(range(epochs), desc=f"Training {category}"):
            loss = model.train_epoch(train_dataloader)

            # Logs
            if (epoch + 1) % args.print_freq == 0:
                logger.info('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, loss))
                tensorboard_logger.add_scalar('loss', loss, epoch)

            # Validation
            if epoch == epochs - 1:
                model.save(ckp_path + '_final.pth')
                if epoch == epochs - 1:
                    save_fig_flag = save_fig
                else:
                    save_fig_flag = False

                logger.info('=============================Testing ====================================')
                metric_dict_novel = model.evaluation(
                    test_dataloader,
                    test_data_cls_names,
                    save_fig_flag,
                    image_dir,
                    cal_pro,
                    by_specie
                )

                log_metrics(
                    metric_dict_novel,
                    logger,
                    tensorboard_logger,
                    epoch
                )

                for k in metric_dict_novel:
                    if k != 'Average':
                        save_metric(metric_dict_novel[k], test_data_cls_names, k, csv_path)
                final_metrics = metric_dict_novel

    else:
        if args.ckt_path:
            ckt_path = args.ckt_path
            logger.info(f'use manually specified ckt path: {ckt_path}')
        else:
            ckt_path = ckp_path + '_final.pth'
            logger.info(f'automatically generate ckt path: {ckt_path}')

        logger.info(f'trying to load {ckt_path}')
        assert os.path.isfile(ckt_path), 'Please train first'
        model.load(ckt_path)

        save_fig_flag = save_fig
        logger.info('=============================Testing ====================================')

        metric_dict = model.evaluation(
            test_dataloader,
            test_data_cls_names,
            save_fig_flag,
            image_dir,
            cal_pro,
            by_specie
        )

        log_metrics(
            metric_dict,
            logger,
            tensorboard_logger,
            epochs
        )

        for k in metric_dict.keys():
            save_metric(metric_dict[k], test_data_cls_names, k, csv_path)
        
        final_metrics = metric_dict

    Path(ckp_path + '_metrics.json').write_text(json.dumps(final_metrics, indent=2))
    tensorboard_logger.close()
    # 恢复原始 category
    args.category = original_category
    
    # 清理显存
    del model
    torch.cuda.empty_cache()
    
    return final_metrics


def train(args):
    """主训练函数 - 支持训练多个类别"""
        # ============ 新增：支持训练所有数据集 ============
    if args.dataset == 'all':
        all_datasets = ['mvtec', 'visa', 'btad', 'dtd']
        print("="*60)
        print("训练所有数据集")
        print("="*60)
        
        for i, dataset in enumerate(all_datasets):
            print(f"\n{'#'*60}")
            print(f"# [{i+1}/{len(all_datasets)}] 数据集: {dataset}")
            print(f"{'#'*60}\n")
            
            args.dataset = dataset
            args.category = 'all'
            train(args)  # 递归调用
        
        print("\n" + "="*60)
        print("所有数据集训练完成！")
        print("="*60)
        return
    # ============ 新增结束 ============

    # 根据数据集选择类别列表
    if args.dataset == 'visa':
        ALL_CATEGORIES = VISA_CATEGORIES
    elif args.dataset == 'btad':
        ALL_CATEGORIES = BTAD_CATEGORIES
    elif args.dataset == 'taoci':
        ALL_CATEGORIES = TAOCI_CATEGORIES
    elif args.dataset == 'dtd':
        ALL_CATEGORIES = DTD_CATEGORIES
    else:
        ALL_CATEGORIES = MVTEC_CATEGORIES
    
    # 确定要训练的类别列表
    if args.category == 'all':
        categories = ALL_CATEGORIES.copy()
    elif isinstance(args.category, list):
        categories = args.category
    else:
        categories = [args.category]
    
    # 验证类别名称
    for cat in categories:
        if cat not in ALL_CATEGORIES and cat != 'custom':
            print(f"警告: '{cat}' 不是有效的 {args.dataset} 类别，将被跳过")
    
    categories = [c for c in categories if c in ALL_CATEGORIES or c == 'custom']
    
    if not categories:
        print("错误: 没有有效的类别可以训练")
        return
    
    print("="*60)
    print("VarAD 训练")
    print("="*60)
    print(f"数据集: {args.dataset}")
    print(f"模型: {args.model}")
    print(f"图像尺寸: {args.image_size}")
    print(f"GPU: {args.gpu}")
    print(f"要训练的类别 ({len(categories)}个): {', '.join(categories)}")
    print("="*60)
    
    # 存储所有类别的结果
    all_results = {}
    
    # 训练每个类别
    for i, category in enumerate(categories):
        print(f"\n{'#'*60}")
        print(f"# [{i+1}/{len(categories)}] 开始训练类别: {category}")
        print(f"{'#'*60}\n")
        
        try:
            metrics = train_single_category(args, category)
            if metrics:
                all_results[category] = metrics
                print(f"\n[✓] {category} 训练完成")
                if 'Average' in metrics:
                    avg = metrics['Average']
                    auroc_im = avg.get('auroc_im', 0)
                    auroc_px = avg.get('auroc_px', 0)
                    print(f"    I-AUROC: {auroc_im:.2f}, P-AUROC: {auroc_px:.2f}")
        except Exception as e:
            print(f"\n[✗] {category} 训练失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    # 打印总结
    if len(categories) > 1:
        print("\n" + "="*60)
        print("训练总结")
        print("="*60)
        
        if not all_results:
            print("  (没有收集到训练结果)")
            print("="*60)
            return
        
        print(f"  共完成 {len(all_results)} 个类别")
        print("-"*60)
        
        auroc_im_list = []
        auroc_px_list = []
        
        for cat, metrics in all_results.items():
            if 'Average' in metrics:
                avg = metrics['Average']
                auroc_im = avg.get('auroc_im', 0)
                auroc_px = avg.get('auroc_px', 0)
                
                auroc_im_list.append(float(auroc_im))
                auroc_px_list.append(float(auroc_px))
                print(f"  {cat}: I-AUROC={auroc_im:.2f}, P-AUROC={auroc_px:.2f}")
            else:
                print(f"  {cat}: (没有 Average 数据)")
        
        if auroc_im_list:
            print("-"*60)
            print(f"  平均: I-AUROC={sum(auroc_im_list)/len(auroc_im_list):.2f}, "
                  f"P-AUROC={sum(auroc_px_list)/len(auroc_px_list):.2f}")
        else:
            print("  (没有找到指标数据)")
        print("="*60)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("AnomalyMamba", add_help=True)
    
    # Paths and configurations
    parser.add_argument("--dataset", type=str, default="mvtec", help="Dataset for training")
    parser.add_argument("--category", type=str, nargs='+', default='all',
                        help="Category for training. Use 'all' for all categories, or specify one or more categories")
    parser.add_argument("--save_path", type=str, required=True, help="New output directory; existing paths are rejected")

    parser.add_argument("--save_fig", type=str2bool, default=False)
    parser.add_argument("--compute_fps", type=str2bool, default=False)
    parser.add_argument("--do_train", type=str2bool, default=True)
    parser.add_argument("--cal_pro", type=str2bool, default=False)
    parser.add_argument("--ckt_path", type=str, default='')

    parser.add_argument("--is_proj", type=str2bool, default=True)
    parser.add_argument("--valid_path", type=int, default=-1)

    parser.add_argument("--hierarchies", type=int, nargs='+', default=[1, 2, 3], help="Hierarchies")
    parser.add_argument("--adapter", type=str, nargs='+', default=['linear'], help="Adapter type")

    parser.add_argument("--model", type=str, default="dinov2_vits14")

    # Hyper-parameters
    parser.add_argument("--gpu", type=int, default=0, help="GPU device id")
    parser.add_argument("--exp_indx", type=int, default=0, help="Experiment index")
    parser.add_argument("--epoch", type=int, default=10, help="Epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")

    parser.add_argument("--image_size", type=int, default=512, help="Image size")
    parser.add_argument("--native_res", type=str2bool, default=False,
                        help="使用原生分辨率, 不resize, 仅pad到14倍数; 每图尺寸不同, 强制batch_size=1")
    parser.add_argument("--edge_crop", type=int, default=0,
                        help="裁掉左右两侧各 N 像素 (去除陶瓷黄边, 非异常); 0=不裁")
    parser.add_argument("--print_freq", type=int, default=1, help="Print frequency")
    parser.add_argument("--valid_freq", type=int, default=5, help="Validation frequency")

    parser.add_argument('--noise_type', type=str, default='divnoise',
                    choices=['none', 'perlin', 'gaussian', 'stripe', 'mixed', 'all', 'ortho', 'simple', 'divnoise', 'divsigma'])
    parser.add_argument('--noise_pattern', type=str, default='all',
                    choices=['blob', 'stripe', 'scatter', 'mixed', 'all'],
                    help="空间模板(WHERE)，仅 --noise_type ortho 生效")
    parser.add_argument('--noise_op', type=str, default='all',
                    choices=['gauss', 'shift', 'scale', 'swap', 'all'],
                    help="腐蚀算子(WHAT)，仅 --noise_type ortho 生效")
    parser.add_argument('--noise_ratio_min', type=float, default=0.02)
    parser.add_argument('--noise_ratio_max', type=float, default=0.22)
    parser.add_argument('--clean_ratio', type=float, default=0.5)
    parser.add_argument('--pred_next_n', type=int, default=0)
    parser.add_argument('--use_seq_pred', type=str2bool, default=True)
    parser.add_argument('--lambda_spatial', type=float, default=1.0,
                        help="L1在重建损失中的权重: loss = MSE + lambda_spatial * L1; 0=不使用L1")
    parser.add_argument('--seq_loss_weight', type=float, default=0.5,
                        help="序列预测损失权重: total = recon + seq_loss_weight * seq; 0=不使用序列预测损失")
    parser.add_argument('--decoder_type', type=str, default='vssm', choices=['vssm', 'cnn', 'transformer'])
    parser.add_argument('--loss_type', type=str, default='mse_l1', choices=['mse_l1', 'mse', 'l1', 'cosine'],
                        help="重建损失类型: mse_l1(默认,原行为)/mse/l1/cosine; 用于 loss-function 消融")

    parser.add_argument("--seed", type=int, default=111)
    with (Path(__file__).resolve().parent / 'configs/observed.json').open() as f:
        parser.set_defaults(**json.load(f))
    args = parser.parse_args()
    if args.dataset not in ('mvtec', 'visa', 'btad', 'dtd', 'all'):
        parser.error("Only the four public MARS benchmarks are included")
    if args.model != 'dinov2_vitb14' or args.decoder_type != 'vssm':
        parser.error("This extraction supports DINOv2 ViT-B/14 with VSSM")
    if args.epoch < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs, batch size and learning rate must be positive")
    if not args.do_train and not Path(args.ckt_path).is_file():
        parser.error("Evaluation requires an existing --ckt_path")
    Path(args.save_path).mkdir(parents=True, exist_ok=False)

    # 处理 category 参数
    if isinstance(args.category, list):
        if len(args.category) == 1:
            args.category = args.category[0]
        # 如果是多个类别，保持为列表

    if args.native_res:
        # 原生分辨率: 不 resize, 每张图尺寸不同 -> 必须 batch_size=1
        args.batch_size = 1
        print("[native_res] 使用原生分辨率, 不 resize, 仅 pad 到 14 倍数, batch_size 强制为 1")
    elif args.model.count('DINO') > 0 or args.model.count('dinov2') > 0:
        args.image_size = 14 * (args.image_size // 14)

    train(args)