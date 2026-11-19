# EAAI 多随机种子实验启动核查

**最终状态：PARTIAL**

日期：2026-11-13。固定种子：`[111, 222, 333, 444, 555]`。
本报告针对本项目当前 observed 配置。BOS checkpoint 缺陷已经修复并通过真实模型验证；部分公开数据及训练入口可用。论文方法/协议一致性、逐图结果持久化和完整续训仍有限制，因此不能据此直接宣称已具备“四数据集原论文统计复现”的全部条件。

没有执行完整 epoch、60 epoch、多 seed 正式训练或 benchmark。只执行 L0、真实模型初始化/推理、一次合成 batch 的优化和两图 smoke 评估。Smoke 的指标不得进入论文结果表。

## 1. 证据及范围

检查了 `main.py` → `AnomalyMambaTrainer` → `AnomalyMamba` → `VSSM`/DINO 的调用、配置、数据、保存/加载及评价链；检查 `dataset/base.py`、四个数据集 loader、`tool/utils.py`、`tool/metrics.py`、`configs/*.json`、迁移报告及本项目 provenance/smoke 记录。

论文依据为本项目 `Titlelabel1.pdf`（35 页），重点是 §3.3–3.6、§5.1 和附录 C。提取文字保存在 `paper_text.txt`，PDF 页号为物理页号。临时 PDF 读取工具仅下载至 `/tmp/varad-pdf-tool/`，没有安装或升级 vmamba_new 依赖。

历史依据仅为本项目 `MIGRATION_REPORT.md` 已记录的历史日志核查结论和 provenance；本次没有重新访问其他研究项目的源代码、历史日志、checkpoint 或结果。历史日志自身没有复制到本项目，故“与当时正式投稿运行完全对应”仍未验证。本项目没有 `.git`，源仓库 HEAD 不能代替当前项目状态；本次保存当前文件清单及改动 patch 作为辅助 provenance。

## 2. Checkpoint / BOS

### 根因与修复

- `AnomalyMambaTrainer` 默认 `learnable_bos=False`，入口没有覆盖它。
- 原 `VSSM.__init__` 非可学习分支先 `torch.rand(K,J,C)`，再 `nn.init.normal_(std=0.02)`；结果是普通 Tensor 属性，既不是参数，也不是 buffer，缺席于 `state_dict()`。训练中不更新，但推理依赖它。
- Trainer 原 save 保存全部 `state_dict`，因此“已注册状态保存完整”不代表“实际模型状态保存完整”；原 `strict=False` 又掩盖缺失。
- 最小修复：将上述同一次 `torch.rand` 注册为默认持久化的 `BOS` buffer，随后保留原 normal 初始化。保留相同 RNG 消耗顺序、形状、分布、不可学习属性和前向公式。learnable 分支仍为 `nn.Parameter`。
- Trainer load 改为 `weights_only=True, strict=True`。新 checkpoint 必须完整匹配；缺失或多余 key 不再悄悄忽略。
- 实际三个 BOS 为 `[4,12,768]`、`[4,6,768]`、`[4,4,768]`；对应 decoder offset 为 8、4、2。均在 state_dict 中，均无梯度。
- 其余非持久化张量 `inv_freq` 和位置编码 cache 来自确定性公式；当前入口 `use_pe=False`。未发现当前 eval 路径另有未保存的随机初始化 Tensor。运行时训练噪声不是 checkpoint 模型参数状态。

### 验证结果（真实 FP32 / 1022×1022）

`tests/check_statistical_checkpoint.py` 逐个启动五个独立 Python 进程，均用完整当前模型；使用同一张 MVTec bottle 异常图，仅 seed111 保存 checkpoint。每个进程先用自己的 seed 初始化，再严格加载同一 checkpoint。

| 检查 | 结果 |
|---|---|
| 保存、同进程清零 BOS 后重载 | 六个特征输出和异常图逐元素相等 |
| seed222、333、444、555 跨进程重载 | 六个特征输出和异常图逐元素相等，全部最大差值 **0.0** |
| 五个 seed 初始可训练参数的整体 SHA256 | 五个均不同；每个模型 69 个可训练 Tensor |
| 模拟旧状态：删除三个 BOS 后 permissive load | 与参考图最大差分别为 0.000297118 / 0.000414202 / 0.000407017 / 0.000347712 |
| 缺 BOS 的状态 strict load | 明确失败 |
| 修改后的原 smoke | 一次优化、保存/异 seed 重载、真实图推理、两图评估均通过；重载异常图最大差 0.0 |

证据：`checkpoint_check_20261113/summary.json`、各 `seed_*.json`、`checkpoint_check.log`、`smoke_after_bos_20261113/smoke_result.json`。生成的两个 checkpoint 仅是 L1 验证产物，不能作正式 baseline。独立进程相等只证明本机、当前环境、该输入和 FP32 路径，不证明所有硬件、混合精度或所有输入上的位级一致性。

### 兼容性限制

旧 checkpoint 不含 BOS，**新 Trainer 默认拒绝加载**。不能从旧 checkpoint 字节恢复原始 BOS。只有掌握原 seed、精确配置、环境和完整初始化顺序时才可能重建，仍需逐元素验证；本任务没有自动转换或猜测。旧 smoke checkpoint 和历史结果全部保留。新 checkpoint 的 key 不变，仅新增三个 BOS buffer key；不承诺旧代码能无损加载新 checkpoint。

## 3. 实际训练配置与论文对照

入口先以 `configs/observed.json` 覆盖 argparse 默认，再应用 CLI；非 native DINO 输入向下对齐 14 的倍数。下表是有效默认值，不能只读取 argparse 声明。

| Parameter | Actual Implementation | Paper Description | Status |
|---|---|---|---|
| Input Resolution | 1022×1022，73×73 token；native_res=false，edge_crop=0 | §5.1：1024×1024 | 差异，未改动 |
| Batch Size | 2 | 8 | 差异，未改动 |
| Learning Rate | 5e-4 | 1e-3 | 差异，未改动 |
| Epochs | 60 | 60 | 一致；未执行正式训练 |
| Optimizer | AdamW，betas=(0.5,0.999)，无 scheduler；未显式指定 weight_decay，环境默认 0.01 | AdamW、相同 betas；weight decay 未明确 | 部分一致 |
| Backbone / Feature Layers | 冻结 DINOv2 ViT-B/14，本地官方 cache/预训练权重；blocks 4/8/12，3×768D；linear 1×1 adapter | 效率表标 ViT-B/14；正文未充分明确三层抽取配置 | backbone 可对照；层选择投稿对应关系 UNKNOWN |
| Perturbation Settings | divnoise；blob/stripe/scatter/mixed，Gaussian/Laplace/Uniform/Rayleigh/Gamma/Poisson，零均值单位方差；ratio∈[.02,.22]，clean=.5 | §3.3、算法1、§5.1 给出相同模板/分布及比例 | 名义设置一致；未独立验证每种噪声的统计性质 |
| Reconstruction Loss | 各层 mean MSE + 1×mean L1，乘 batch size 后跨层求和 | 式(6)(7)：空间平均的通道 L2平方和与 L1和 | 存在归一化/聚合差异；不能假设相对 sequence 权重等效 |
| Sequence Loss | 每层一次行优先展平；一个 MLP head；预测下一个位置；MSE+L1；跨层平均，权重 .5 | §3.5 式(8)：每个方向独立展开和方向专属 head，再跨方向平均；α=.5 | **实质方法差异** |
| VSSM scanning | CrossScan 四路，decoder 每层 depth=1，offset=8/4/2，CrossMerge | §3.4 VSSM | 四路扫描不能充当四方向辅助 loss 的实现证据 |
| Augmentation / preprocessing | RGB；bicubic Resize→CenterCrop→Tensor→ImageNet Normalize；mask nearest Resize；没有随机翻转/颜色抖动；像素级异常合成关闭 | §5.1：随机水平/垂直翻转、轻微 color jitter | 差异，未改动 |
| Anomaly Scoring | 通道 mean MSE + mean L1，双线性上采样、跨3层平均、Gaussian sigma=4；image score 为平滑后 1022图的 max | §3.6 式(10)：通道平方 L2残差，上采样、image max | **实质 scoring 差异**；论文未在该段明确上述 L1和平滑 |
| Pixel Evaluation Geometry | masks/maps 高度>512 时统一 cv2 resize 至512×512；mask nearest；map 默认 linear | §5.1：异常图恢复到原始图像分辨率 | **像素指标协议差异** |
| Checkpoint Selection | 固定第60 epoch，保存后只测试一次；不据测试结果选 epoch/seed | 公共数据部分未明确 checkpoint 选择；附录 D 的验证选择规则仅针对陶瓷 | 当前规则独立于 test；与原历史 test-best 规则不同 |

历史日志结论来自迁移报告：42 个 header 支持 1022、batch2、5e-4、60 epoch、ViT-B/14、offset8、sequence enabled、divnoise。其他项有些仅是现代码默认，不能推断所有历史运行都采用了它们。

Sequence 的确切路径：`forward` 对每层取合并后的预测图 → `_compute_sequence_prediction_loss` 的 `view(B,C,-1).permute(0,2,1)` → `head(output_seq[:,:-1,:])` 对 `clean_seq[:,1:,:].detach()` → mean MSE+mean L1。这里没有按四个扫描顺序分别展开，没有四个方向 head，也没有方向维平均。跨行边界同样被当作相邻序列位置。不能将其描述为论文四方向正则的完整实现。本次没有改变任何 loss、perturbation、扫描、预处理或 scoring。

## 4. 数据集现状

| Dataset | Root | Categories | Train N | Test N / A | Mask / split audit | Status |
|---|---|---:|---:|---:|---|---|
| MVTec AD | `/opt/data/private/MVTecAD` | 15 | 3629 | 467 / 1258 | 1258 个有效异常mask；无路径/字节跨split重复 | 当前文件/划分检查通过；原论文协议仍有差异 |
| VisA | `/opt/data/private/Data/VisA` | 12 | 8659 | 962 / 1200 | 1200 个有效异常mask；无路径/字节跨split重复 | 当前文件/划分检查通过；与本地1cls.csv一致 |
| DTD-Synthetic | `/opt/data/private/data/dtd-synthetic` | 12 | 1200 | 357 / 947 | 947 个有效异常mask；无路径/字节跨split重复 | 文件检查通过；与论文split不一致 |
| BTAD | `/opt/data/private/Btad` | 3 | 0 | 0 / 0 | 空目录，不能把 errors=[]解释为可用 | 缺失，不可启动 |

三组现有数据共 **18,679** 张图像、**3,405** 个异常mask；全部12/15类均有正常训练、正常测试、异常测试。BTAD表中3类只是预期类别列表，不表示发现了3类数据。类别逐项计数、路径及SHA256见 `dataset_check_20261113/dataset_audit.json` 和 `dataset_inventory.jsonl`，最终命令 exit=0。

论文附录C：MVTec 3629/467/1258、VisA 8659/962/1200、BTAD 2540/290/290、DTD 459/518/947。当前前两者计数一致，DTD为1200/357/947；没有调整任何划分。

清单对应实际 loader 的 FS 全正常训练设置：`k_shot=1e6` 取全部正常训练图，训练时仅重排；test 不抽样；不移动或重分数据。VisA 优先读本地 `meta.json`，其余三个 BaseSolver 动态遍历 `train/good`（fallback `ok`）和 `test/*`。异常 mask 按 `_mask.png` / 同名 png/bmp / 原扩展名寻找；缺失 mask 会被原 loader 静默作全零，故正式运行必须先保证审计无异常。

全量图像做文件存在、PIL verify、原尺寸检查，并保存图像 SHA256 用于检查**跨 split 字节相同**；全部异常 mask 解码为灰度，检查原尺寸相等、非空及路径复用。没有做感知相似/近重复、生产来源分组泄漏或所有正常 RGB 的逐像素解码认证。字节不同不能证明语义独立。

BTAD：`/opt/data/private/Btad` 为空，`BTech_Dataset_transformed` 指向它。补充检查 `/opt/data` 下公开数据容器和 `/mnt` 的有限目录搜索，没有发现可用 BTAD。搜索非全盘穷举；没有下载、修改数据集或搜索其他研究项目内部。找不到数据不能计为零分实验。

VisA CSV 还有独立限制：本地 `1cls.csv` 使用 `object` 表头，原 loader fallback 却读取 `category`。当前存在 meta.json，实际不走 fallback，因此本次不改 loader；若 meta.json 缺失则会失败。审计脚本首次对照 CSV 也误用了 category，发生 KeyError；该失败日志/部分清单保留，修正审计脚本为真实 object 表头后使用全新输出目录重跑。该失败不算数据集通过证据。

DTD 根目录还存在嵌套数据目录/生成记录；实际 loader 只使用配置根目录下的12类，不递归并入其他目录。论文 train459/test normal518 与当前计数不同，不能靠增删/挪动图像补齐论文统计。嵌套版本的来源与投稿 split 对应关系未认证。

## 5. 五个 seed 实验可执行性

| 核查 | 已验证行为 / 限制 |
|---|---|
| 独立初始化 | 每个类别进入 train_single_category 后 setup_seed，再构建模型；5个独立进程的整体可训练参数指纹不同。常数初始化的 bias/LayerNorm 不要求随 seed 改变；这不等于5次完整训练的统计独立已实证 |
| 随机状态控制 | Python random、NumPy 全局 RNG、Torch CPU/CUDA/all CUDA 均设 seed；cuDNN deterministic=true、benchmark=false。五个 seed 的直接采样和 num_workers=8、shuffle DataLoader 的重复回放测试均通过，见 seed_controls.json |
| 确定性边界 | 未启用 torch.use_deterministic_algorithms；没有额外保证 CUDA/cuBLAS/自定义 selective scan 的训练确定性；没有保存 optimizer/RNG/worker数据位置。Seeded run 不等于统计稳定、确定性训练或跨设备位级复现 |
| 统一 checkpoint 选择 | 入口 mock 三epoch事件严格为 train/train/train/save/test；零分也写出最终结果。统一60 epoch；不据 test 选 epoch或seed |
| 指标 | sklearn ROC AUC / average_precision_score / PR 阈值最大F1，均×100。P 指标在每类所有图像像素上 pooled；I 指标在每类图像上 pooled；最终类别等权宏平均，不能把类别图像数作权重 |
| 指标验证 | 已用可手算4样本确认 AUROC75、AP83.3333、maxF180，以及等权类别均值；见 metric_sanity.json。论文同样明确 oracle maxF1。AUPRO 默认关闭且非论文六个主指标；其实现未在本次认证 |
| 与论文可比性 | 六个指标的名称/基本定义一致，但 scoring、pixel geometry、loss、超参数和 checkpoint 来源仍不完全对应。不得声称协议等同原论文 |
| 每图分数 / 像素热图持久化 | **未具备正式完整导出**。evaluation 内存有 anomaly_scores/maps，但只返回聚合指标；默认 save_fig=false；true 时输出归一化可视化，不能代替带原路径的原始数值分数/热图。L1测试保存一张原始 npy 仅证明样例，不代表正式入口全量导出 |
| 失败与覆盖 | CLI save_path.mkdir(exist_ok=False)；实际重复目录运行已在训练前以 FileExistsError 非零退出。每 seed 必须新目录，失败重试用新 attempt；历史结果保留 |
| 完整断点恢复 | **未支持**。只保存模型 state_dict；没有 optimizer、epoch、RNG、scheduler/scaler（当前也无 scheduler/scaler）。训练中断在最终epoch之前没有正式模型快照；--ckt_path 只是载模型权重后从epoch0用新optimizer训练，不是续训 |
| 失败后的可用策略 | 从失败类别/seed 的epoch0重新跑到固定60epoch，使用新的目录，保留失败尝试并记录替代关系；不要将失败/缺失seed删掉后只报最优seed。可以对已有完整final checkpoint在新目录重新评估；不可把多次尝试择优合并 |

## 6. 尚未解决的阻断项

1. 对“原论文四数据集多 seed 统计复现”的声明：四方向 sequence loss、scoring、像素几何、batch/LR/input/augmentation 等尚未对应，投稿实际 run/config/checkpoint 仍 UNKNOWN。需先决定评估目标是当前 observed 实现还是论文描述；本任务不替用户修改科学协议。
2. BTAD 缺失；DTD 版本/划分与论文不一致。禁止为匹配统计而调整 split。
3. 正式入口缺少逐图原始 score/heatmap 导出，无法直接满足逐图存档及后续 bootstrap 审计。
4. 完整中断续训不具备；只能保留失败记录并以新 attempt 从头重跑。若要求恢复 optimizer/RNG/epoch，则这是启动前工程阻断项。
5. 本次只认证当前环境/FP32 的有限 L1 样例，未测全60epoch收敛或所有类别 GPU 资源；论文指标未复现。

这些限制不否定已通过的 checkpoint 修复与数据文件审计，但不能由 smoke 成功越级推断为论文复现或五 seed 可靠性结论。

## 7. 正式训练命令和输出计划（仅文档，未执行）

保留 observed 配置，不修改论文参数。以下命令是明确完成上述目标/限制决策后的启动模板；当前不自动执行。先只考虑已审计的 MVTec/VisA。DTD 不纳入论文可比表，BTAD 不启动。

```bash
cd /opt/data/private/VarAD_EAAI
source /opt/conda/etc/profile.d/conda.sh
conda activate vmamba_new
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
attempt=$(date +%Y%m%d_%H%M%S)
for dataset in mvtec visa; do
  for seed in 111 222 333 444 555; do
    run="experiments/statistical_reliability/formal_${dataset}_seed${seed}_${attempt}"
    test ! -e "$run" || exit 1
    mkdir "$run" || exit 1
    python -u main.py --dataset "$dataset" --seed "$seed" --gpu 0 \
      --image_size 1022 --batch_size 2 --learning_rate 0.0005 --epoch 60 \
      --model dinov2_vitb14 --hierarchies 1 2 3 --pred_next_n 8 \
      --noise_type divnoise --use_seq_pred true --seq_loss_weight 0.5 \
      --native_res false --save_fig false --cal_pro false \
      --save_path "$run/artifacts" > "$run/train.log" 2>&1
    result=$?
    printf '%s\n' "$result" > "$run/exit_code"
    test "$result" -eq 0 || exit "$result"
  done
done
```

每个 seed 一个新根目录；未指定 category 时入口默认 all，MVTec15类/VisA12类。正式60epoch共 `5×(15+12)×60 = 8100` 类别epoch；没有测完整epoch吞吐，故时间/总存储 UNKNOWN。正式启动应放入持久任务机制，记录 PID、完整命令、cwd、日志路径，验证成功启动后交回控制；本任务没有创建此正式作业。

输出规则：`$run/artifacts/VarAD/models/*-<category>_final.pth`、同名前缀 `_config.json` 和 `_metrics.json`；`logs/`、`csvs/`、`tensorboard/` 位于 `$run/artifacts/VarAD/`。现有 config 记录实际 args、外部路径、fixed_final_epoch 和 test_oracle；环境/当前代码清单需与各正式 run 一并冻结，checkpoint 和退出码必须在完成后核验。

五个 seed 全部保留并报告逐seed值、均值和样本标准差（ddof=1），不按test结果筛seed；任何 CI 方法需另行固定。这里没有生成正式统计表或虚构指标。

## 8. 复核与产物

- `tests/verify_migration.py`：通过。只在内存逆转明确批准的 BOS 注册和 strict-load 两处变化后核对原迁移字节/方法指纹，原 manifest 保持不变；不是把任意新代码豁免。
- `tests/test_entrypoint.py`：通过，无真实训练；日志 entrypoint_check.log。
- `tests/check_statistical_checkpoint.py`：通过，五个独立进程；没有训练。
- `tests/check_seed_controls.py`：通过；没有训练。
- `tests/smoke_test.py`：通过，仅一次合成优化；原测试描述已更新为异seed重载。
- 指标小样例和旧输出目录拒绝：通过；后一检查期望 exit=1，见 existing_path_rejection.log。
- 数据审计以最终 dataset_check_20261113 中的清单和汇总为准；首轮错误保留以便追踪。
- README 更新 checkpoint 兼容性说明；没有调整原 MARS 数学、实验默认配置、数据、预训练权重或历史结果。


## 2026-11-13 用户确认后的正式配置更正

未来正式实验的 Batch Size=8、LR=5e-4、patch size=14 已由用户确认。本文早前batch2记录仍准确描述迁移来源与当前Pilot，不再作为正式实验的batch依据。实际投稿尺寸是否为1036、是否由Resize1024后Padding得到，仍未证实；当前普通入口的实际输入为1022、73×73。详见 [TRAINING_CONFIGURATION_AUDIT.md](TRAINING_CONFIGURATION_AUDIT.md) 与新 `official_protocol_v2_confirmed_hparams/` 候选配置。旧Pilot与旧结果保持不变，正式五seed未启动。
