"""
================================================================================
MARS - 噪声消融实验版本
================================================================================

================================================================================
"""

import math
import numpy as np
from .vssm import VSSM
from .dino_vision_tokenizer import VisionTokenizer as DINOVisionTokenizer
from torch import nn
from torch.nn import functional as F
from scipy.ndimage import gaussian_filter
import torch


class AnomalyMamba(nn.Module):
    def __init__(self,
                 backbone,
                 norm=False,
                 hierarchies=[1, 2, 3],
                 depths=[2, 2],
                 bos_length=4,
                 valid_path=0,
                 pred_next_n=0,
                 image_size=256,
                 lambda_spatial=1.,
                 learnable_bos=True,
                 use_pe=True,
                 is_proj=True,
                 use_path_weights=True,
                 cache_data=None,
                 adapter=['linear'],
                 use_sequence_prediction=True,
                 clean_ratio=0.5,
                 seq_loss_weight=0.5,
                 decoder_type='vssm',
                 loss_type='mse_l1',  # 重建损失类型: 'mse_l1'(默认,原行为), 'mse', 'l1'
                 # ⭐ 噪声消融参数
                 noise_type='divnoise',  # default: diverse zero-mean unit-variance noise
                 noise_ratio_min=0.02,
                 noise_ratio_max=0.22,
                 noise_pattern='all',  # blob, stripe, scatter, mixed, all (仅 noise_type='ortho' 生效)
                 noise_op='all',       # gauss, shift, scale, swap, all (仅 noise_type='ortho' 生效)
                 ):
        super(AnomalyMamba, self).__init__()

        self.image_size = image_size
        self.norm = norm
        self.use_sequence_prediction = use_sequence_prediction
        self.lambda_spatial = lambda_spatial
        self.clean_ratio = clean_ratio
        self.seq_loss_weight = seq_loss_weight
        self.decoder_type = decoder_type.lower()
        self.loss_type = loss_type.lower()
        assert self.loss_type in ('mse_l1', 'mse', 'l1', 'cosine'), \
            f"Unsupported loss_type={loss_type!r} (expected 'mse_l1', 'mse', 'l1' or 'cosine')"
        
        # ⭐ 噪声消融参数
        self.noise_type = noise_type
        self.noise_ratio_min = noise_ratio_min
        self.noise_ratio_max = noise_ratio_max
        self.noise_pattern = noise_pattern
        self.noise_op = noise_op
        print(f"[AnomalyMamba] noise_type: {self.noise_type}, "
              f"ratio: [{self.noise_ratio_min}, {self.noise_ratio_max}], "
              f"clean_ratio: {self.clean_ratio}"
              + (f", pattern: {self.noise_pattern}, op: {self.noise_op}"
                 if self.noise_type == 'ortho' else ""))

        # Tokenizer
        if backbone.count('DINO') > 0 or backbone.count('dinov2') > 0:
            self.tokenizer = DINOVisionTokenizer(backbone=backbone, hierarchies=hierarchies, 
                                                  norm=norm, adapter=adapter)
        else:
            raise ValueError("Only the DINOv2 MARS backbone is included")

        if cache_data is not None:
            device = cache_data.device
            self.tokenizer = self.tokenizer.to(device)

        dims = self.tokenizer.target_dimension
        x = self.tokenizer(cache_data)

        # Decoder模块
        self.vssm = nn.ModuleList()
        n_blocks = 1
        for indx, d in enumerate(dims):
            if self.decoder_type == 'vssm':
                decoder_temp = nn.Sequential(*[
                    VSSM(depths=depths,
                         dims=d,
                         pred_next_n=pred_next_n // (2 ** indx),
                         valid_path=valid_path,
                         learnable_bos=learnable_bos,
                         use_pe=use_pe,
                         cache_data=x[indx],
                         use_path_weights=use_path_weights,
                         ) for _ in range(n_blocks)
                ])
            else:
                raise ValueError(f"Unsupported decoder_type={decoder_type!r}")
            self.vssm.append(decoder_temp)

        # 序列预测头
        if self.use_sequence_prediction:
            self.seq_pred_heads = nn.ModuleList()
            for d in dims:
                self.seq_pred_heads.append(
                    nn.Sequential(
                        nn.Linear(d, d * 2),
                        nn.GELU(),
                        nn.Linear(d * 2, d)
                    )
                )

    def train(self, mode: bool = True):
        super().train(mode)
        self.tokenizer.frozen_tokenizer.eval()
        self.tokenizer.proj.train(mode)
        self.vssm.train(mode)
        if self.use_sequence_prediction:
            self.seq_pred_heads.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def inject_token_noise(self, x):
        """两正交轴的特征空间腐蚀: 空间模板(WHERE) × 腐蚀算子(WHAT).

        空间模板 (noise_pattern) 只决定噪声出现的位置 (二值掩码 M):
          - 'blob'   : Perlin 平滑场阈值 -> 连续块状 (污渍/斑块)
          - 'stripe' : 随机整行/整列      -> 直线条带 (划痕/裂纹)
          - 'scatter': 随机离散 token     -> 散点     (坏点/噪点)
          - 'mixed'  : 上述三种掩码的并集
        腐蚀算子 (noise_op) 决定在 M=1 处如何改写特征:
          - 'gauss'  : 加性零均值高斯       E + M⊙(σ_E·ε)
          - 'shift'  : 非零均值偏移         E + M⊙(β·σ_E)
          - 'scale'  : 乘性幅值畸变         E·(1+(γ-1)·M)
          - 'swap'   : 特征替换(空间打乱)   (1-M)⊙E + M⊙shuffle(E)
        σ_E 为逐通道空间标准差, 掩码沿通道共享, 每通道独立采样噪声.

        noise_type 作为高层预设:
          - 'none'  : 不加噪 (纯重建 baseline)
          - 'perlin'/'gaussian'/'stripe'/'mixed' : 旧式单策略 (对应模板 + op='gauss')
          - 'all'   : 旧默认 (随机模板 + op='gauss'), 复现历史消融
          - 'ortho' : 启用 noise_pattern × noise_op 两轴独立采样 (新设计)
          - 'simple': 随机散点位置(比例随机) + 标准高斯 N(0,1) 加噪
                      E + M⊙η, η~N(0,1), 不按 σ 缩放
          - 'divnoise': 形状轴不变(4 种模板随机) + 分布轴多样化
                      E + M⊙η, η 从 {Gaussian,Laplace,Uniform,Rayleigh,
                      Gamma,Poisson} 随机取一种, 每种都归一化到均值0方差1.
                      纯加性零均值噪声, 作为打破恒等映射的去噪正则
                      (对齐导师 AAAI25 "deep noise evaluation" 的高多样性加噪).
        """
        if not self.training or self.noise_type == 'none':
            return x

        B, C, H, W = x.shape
        device = x.device
        x_noisy = x.clone()

        # 每个样本以 clean_ratio 概率保持干净
        clean_mask = torch.rand(B, device=device) < self.clean_ratio
        noise_range = self.noise_ratio_max - self.noise_ratio_min
        noise_ratios = torch.rand(B, device=device) * noise_range + self.noise_ratio_min
        noise_ratios[clean_mask] = 0.0

        for b in range(B):
            ratio = noise_ratios[b].item()
            if ratio <= 0:
                continue
            pattern, op = self._resolve_strategy()
            mask = self._make_mask(pattern, H, W, ratio, device)
            x_noisy[b:b + 1] = self._apply_corruption(op, x[b:b + 1], mask, device)

        return x_noisy

    # ---- 策略解析 ---------------------------------------------------------
    def _resolve_strategy(self):
        """返回 (pattern, op). 兼容旧 noise_type, 并支持 'ortho' 两轴采样."""
        nt = self.noise_type
        legacy = {
            'perlin': ('blob', 'gauss'),
            'gaussian': ('scatter', 'gauss'),
            'scatter': ('scatter', 'gauss'),
            'stripe': ('stripe', 'gauss'),
            'mixed': ('mixed', 'gauss'),
        }
        if nt in legacy:
            return legacy[nt]
        if nt == 'all':
            pattern = ['blob', 'stripe', 'scatter', 'mixed'][torch.randint(0, 4, (1,)).item()]
            return pattern, 'gauss'
        if nt == 'simple':
            # 随机散点位置 (比例由外层随机采样) + 标准高斯 N(0,1)
            return 'scatter', 'std_gauss'
        if nt == 'divnoise':
            # 形状轴: 4 种模板随机 (同旧 'all'); 分布轴: 多样化零均值单位方差噪声
            pattern = ['blob', 'stripe', 'scatter', 'mixed'][torch.randint(0, 4, (1,)).item()]
            return pattern, 'divnoise'
        if nt == 'divsigma':
            # 同 divnoise 的形状轴(4 模板随机)与分布轴(多样化零均值单位方差 η),
            # 但第⑥步把噪声按逐通道 σ_z 缩放: E + M⊙(σ_z·η) (而非严格单位方差).
            pattern = ['blob', 'stripe', 'scatter', 'mixed'][torch.randint(0, 4, (1,)).item()]
            return pattern, 'divsigma'
        if nt == 'ortho':
            pattern = self._pick(self.noise_pattern, ['blob', 'stripe', 'scatter', 'mixed'])
            op = self._pick(self.noise_op, ['gauss', 'shift', 'scale', 'swap'])
            return pattern, op
        # 未知预设回退到散点高斯
        return 'scatter', 'gauss'

    @staticmethod
    def _pick(value, choices):
        """value=='all' 时在 choices 中均匀随机, 否则取 value 本身."""
        if value == 'all':
            return choices[torch.randint(0, len(choices), (1,)).item()]
        return value

    # ---- 空间模板 (WHERE): 生成二值掩码 M, 形状 (1,1,H,W) -----------------
    def _make_mask(self, pattern, H, W, ratio, device):
        if pattern == 'blob':
            return self._mask_blob(H, W, ratio, device)
        if pattern == 'stripe':
            return self._mask_stripe(H, W, ratio, device)
        if pattern == 'scatter':
            return self._mask_scatter(H, W, ratio, device)
        if pattern == 'mixed':
            splits = torch.rand(3, device=device)
            splits = splits / splits.sum()
            m = self._mask_blob(H, W, ratio * splits[0].item(), device)
            m = torch.maximum(m, self._mask_stripe(H, W, ratio * splits[1].item(), device))
            m = torch.maximum(m, self._mask_scatter(H, W, ratio * splits[2].item(), device))
            return m
        raise ValueError(f"Unknown noise pattern: {pattern!r}")

    @staticmethod
    def _mask_blob(H, W, ratio, device):
        """Perlin 平滑场取阈值 -> 连续块状掩码."""
        L = H * W
        num = int(L * ratio)
        if num <= 0:
            return torch.zeros(1, 1, H, W, device=device)
        freq_h = torch.randint(2, 8, (1,)).item()
        freq_w = torch.randint(2, 8, (1,)).item()
        field = F.interpolate(torch.rand(1, 1, freq_h, freq_w, device=device),
                              size=(H, W), mode='bicubic', align_corners=False)
        flat = field.view(-1)
        val = torch.kthvalue(flat, max(1, L - num)).values
        return (field > val).float()

    @staticmethod
    def _mask_scatter(H, W, ratio, device):
        """随机离散 token -> 散点掩码."""
        L = H * W
        num = int(L * ratio)
        if num <= 0:
            return torch.zeros(1, 1, H, W, device=device)
        scores = torch.rand(1, L, device=device)
        _, idx = torch.topk(scores, num, dim=1)
        mask = torch.zeros(1, L, device=device)
        mask.scatter_(1, idx, 1.0)
        return mask.view(1, 1, H, W)

    @staticmethod
    def _mask_stripe(H, W, ratio, device):
        """随机整行/整列 -> 直线条带掩码."""
        mask = torch.zeros(1, 1, H, W, device=device)
        if torch.rand(1).item() > 0.5:
            n = max(1, int(H * ratio * 2))
            idx = torch.randperm(H, device=device)[:n]
            mask[:, :, idx, :] = 1.0
        else:
            n = max(1, int(W * ratio * 2))
            idx = torch.randperm(W, device=device)[:n]
            mask[:, :, :, idx] = 1.0
        return mask

    # ---- 腐蚀算子 (WHAT): 在 M=1 处改写特征 ------------------------------
    def _apply_corruption(self, op, x_clean, mask, device):
        if op == 'gauss':
            return self._op_gauss(x_clean, mask, device)
        if op == 'shift':
            return self._op_shift(x_clean, mask, device)
        if op == 'scale':
            return self._op_scale(x_clean, mask, device)
        if op == 'swap':
            return self._op_swap(x_clean, mask, device)
        if op == 'std_gauss':
            return self._op_std_gauss(x_clean, mask, device)
        if op == 'divnoise':
            return self._op_divnoise(x_clean, mask, device)
        if op == 'divsigma':
            return self._op_divsigma(x_clean, mask, device)
        raise ValueError(f"Unknown noise op: {op!r}")

    @staticmethod
    def _op_std_gauss(x_clean, mask, device):
        """标准高斯加噪: E + M⊙η, η~N(0,1) (均值0方差1, 不按 σ 缩放)."""
        noise = torch.randn_like(x_clean)
        return x_clean + noise * mask

    # 多样化噪声分布 (每种都归一化到均值0方差1), 对齐导师 AAAI25 noise-evaluation
    _NOISE_DISTS = ['gaussian', 'laplace', 'uniform', 'rayleigh', 'gamma', 'poisson']

    @staticmethod
    def _sample_unit_noise(shape, device, dist=None):
        """采样均值0、方差1的噪声; dist 为空时在 6 种分布中均匀随机 (高多样性)."""
        if dist is None:
            dist = AnomalyMamba._NOISE_DISTS[torch.randint(0, len(AnomalyMamba._NOISE_DISTS), (1,)).item()]
        if dist == 'gaussian':
            return torch.randn(shape, device=device)
        if dist == 'laplace':
            # Laplace(0,b) 方差=2b^2; b=1/√2 -> 方差1
            # clamp 远离 0/1, 避免 |u|=0.5 时 log1p(-1)=-inf -> NaN
            u = torch.rand(shape, device=device).clamp(1e-6, 1.0 - 1e-6) - 0.5
            return -torch.sign(u) * torch.log1p(-2.0 * u.abs()) / math.sqrt(2.0)
        if dist == 'uniform':
            # U[-√3,√3] -> 均值0方差1
            return (torch.rand(shape, device=device) * 2.0 - 1.0) * math.sqrt(3.0)
        if dist == 'rayleigh':
            # Rayleigh(s=1): 均值√(π/2), 方差(4-π)/2 -> 平移+缩放到均值0方差1
            u = torch.rand(shape, device=device).clamp_min(1e-12)
            r = torch.sqrt(-2.0 * torch.log(u))
            mean = math.sqrt(math.pi / 2.0)
            std = math.sqrt((4.0 - math.pi) / 2.0)
            return (r - mean) / std
        if dist == 'gamma':
            # Exp(1)(=Gamma(k=1)): 均值1方差1 -> 平移到均值0 (右偏)
            u = torch.rand(shape, device=device).clamp_min(1e-12)
            return (-torch.log(u)) - 1.0
        if dist == 'poisson':
            # Poisson(λ=1): 均值1方差1 -> 平移到均值0 (离散右偏)
            return torch.poisson(torch.ones(shape, device=device)) - 1.0
        return torch.randn(shape, device=device)

    def _op_divnoise(self, x_clean, mask, device):
        """多样化零均值单位方差噪声: E + M⊙η, η 随机取一种分布 (均值0方差1)."""
        noise = self._sample_unit_noise(x_clean.shape, device)
        return x_clean + noise * mask

    def _op_divsigma(self, x_clean, mask, device):
        """divnoise + 逐通道 σ_z 缩放: E + M⊙(σ_z·η), η 多样化零均值单位方差,
        噪声幅值随每通道特征标准差 σ_z 自适应 (对齐 ortho 的 σ 缩放)."""
        noise = self._sample_unit_noise(x_clean.shape, device)
        sigma = self._channel_std(x_clean)
        return x_clean + (sigma * noise) * mask

    @staticmethod
    def _channel_std(x_clean):
        """逐通道空间标准差 σ_E, 形状 (1,C,1,1)."""
        return x_clean.std(dim=[2, 3], keepdim=True).clamp(min=1e-6)

    def _op_gauss(self, x_clean, mask, device):
        """加性零均值高斯: E + M⊙(σ_E·ε)."""
        std = self._channel_std(x_clean)
        scale = std * (torch.rand(1, 1, 1, 1, device=device) * 1.2 + 0.3)
        noise = torch.randn_like(x_clean) * scale
        return x_clean + noise * mask

    def _op_shift(self, x_clean, mask, device):
        """非零均值偏移: E + M⊙(β·σ_E), 模拟系统性特征漂移."""
        std = self._channel_std(x_clean)
        sign = 1.0 if torch.rand(1).item() > 0.5 else -1.0
        beta = (torch.rand(1, 1, 1, 1, device=device) * 1.0 + 0.5) * sign  # |β|∈[0.5,1.5]
        offset = std * beta
        return x_clean + offset * mask

    @staticmethod
    def _op_scale(x_clean, mask, device):
        """乘性幅值畸变: E·(1+(γ-1)·M), 模拟过/欠激活."""
        gamma = torch.rand(1, 1, 1, 1, device=device) * 1.5 + 0.5  # γ∈[0.5,2.0]
        return x_clean * (1.0 + (gamma - 1.0) * mask)

    @staticmethod
    def _op_swap(x_clean, mask, device):
        """特征替换: (1-M)⊙E + M⊙shuffle(E), 用空间打乱特征顶替, 模拟结构异常."""
        _, C, H, W = x_clean.shape
        flat = x_clean.reshape(1, C, H * W)
        perm = torch.randperm(H * W, device=device)
        shuffled = flat[:, :, perm].reshape(1, C, H, W)
        return x_clean * (1.0 - mask) + shuffled * mask

    def forward(self, x):
        patch_embed = self.tokenizer(x)
        predictive_embed = []
        seq_pred_losses = []

        for indx, patch in enumerate(patch_embed):
            if self.training:
                input_patch = self.inject_token_noise(patch)
            else:
                input_patch = patch

            pred_embed = self.vssm[indx](input_patch)

            if self.norm:
                pred_embed = F.normalize(pred_embed, dim=1)

            predictive_embed.append(pred_embed)

            if self.training and self.use_sequence_prediction:
                seq_loss = self._compute_sequence_prediction_loss(
                    patch,
                    pred_embed,
                    indx
                )
                seq_pred_losses.append(seq_loss)

        if self.training and self.use_sequence_prediction:
            return patch_embed, predictive_embed, seq_pred_losses
        else:
            return patch_embed, predictive_embed

    def _compute_sequence_prediction_loss(self, clean_feat, output_feat, layer_idx):
        B, C, H, W = clean_feat.shape
        
        clean_seq = clean_feat.view(B, C, -1).permute(0, 2, 1)
        output_seq = output_feat.view(B, C, -1).permute(0, 2, 1)
        
        pred_next = self.seq_pred_heads[layer_idx](output_seq[:, :-1, :])
        target_next = clean_seq[:, 1:, :].detach()
        
        mse_loss = F.mse_loss(pred_next, target_next)
        l1_loss = F.l1_loss(pred_next, target_next)
        loss = mse_loss + l1_loss
        
        return loss
        
    def _cal_loss(self, a, b):
        """重建损失 - 按 loss_type 选择 MSE / L1 / MSE+L1 / Cosine"""
        if self.loss_type == 'mse':
            loss = F.mse_loss(a, b)
        elif self.loss_type == 'l1':
            loss = F.l1_loss(a, b)
        elif self.loss_type == 'cosine':
            loss = (1 - F.cosine_similarity(
                a.view(a.shape[0], a.shape[1], -1),
                b.view(b.shape[0], b.shape[1], -1),
                dim=1,
            )).mean()
        else:  # 'mse_l1' (默认, 原行为)
            mse_loss = F.mse_loss(a, b)
            l1_loss = F.l1_loss(a, b)
            loss = mse_loss + self.lambda_spatial * l1_loss
        loss = loss * a.shape[0]
        return loss

    def _cal_am(self, a, b):
        """异常图计算 - 评分函数与 loss_type 保持一致"""
        mse_dist = (a - b).pow(2).mean(dim=1, keepdim=True)
        l1_dist = (a - b).abs().mean(dim=1, keepdim=True)
        if self.loss_type == 'mse':
            dist = mse_dist
        elif self.loss_type == 'l1':
            dist = l1_dist
        elif self.loss_type == 'cosine':
            dist = 1 - F.cosine_similarity(a, b, dim=1, eps=1e-8).unsqueeze(1)
        else:  # 'mse_l1' (默认, 原行为)
            dist = mse_dist + self.lambda_spatial * l1_dist
        anomaly_map = F.interpolate(dist, size=(self.image_size, self.image_size),
                                    mode='bilinear', align_corners=True)
        return anomaly_map

    def cal_loss(self, outputs):
        if self.training and self.use_sequence_prediction:
            gt, pred, seq_losses = outputs
        else:
            gt, pred = outputs
            seq_losses = []

        recon_loss = 0
        for a, b in zip(gt, pred):
            recon_loss += self._cal_loss(a, b)

        if seq_losses:
            seq_loss = sum(seq_losses) / len(seq_losses)
            total_loss = recon_loss + self.seq_loss_weight * seq_loss
        else:
            total_loss = recon_loss

        return total_loss

    @torch.no_grad()
    def cal_am(self, outputs):
        if isinstance(outputs, tuple) and len(outputs) == 3:
            gt, pred, _ = outputs
        else:
            gt, pred = outputs

        anomaly_map_list = []
        for a, b in zip(gt, pred):
            anomaly_map_list.append(self._cal_am(a, b))

        anomaly_map = torch.mean(torch.cat(anomaly_map_list, dim=1), dim=1)
        am_np = anomaly_map.squeeze(1).cpu().numpy()
        am_np_list = []

        for i in range(am_np.shape[0]):
            am_np[i] = gaussian_filter(am_np[i], sigma=4)
            if np.sum(np.isnan(am_np[i])) > 0:
                print(f'error: {np.sum(np.isnan(am_np[i]))} values are NaN')
            am_np_list.append(am_np[i])

        return am_np_list