# MARS 训练配置与输入几何核查

日期：2026-11-13。环境：`/opt/conda/envs/vmamba_new/bin/python`。

**结论：当前项目实际普通路径输入为1022×1022、73×73 patch grid；投稿训练输入是否为1036×1036仍未证实。未来正式训练的 Batch Size=8、LR=5e-4、patch size=14 已由用户明确确认，优先于旧迁移配置。PAPER REPRODUCTION UNVERIFIED。**

本次不修改运行中的 Pilot，不修改 `main.py`、模型、loss、scoring、预处理、旧 `configs/observed.json` 或任何旧实验输出；没有启动正式五 Seed。新增内容仅为独立核查脚本、新配置及文档。Pilot 的 PID83105 / runner PID82325 仍为原进程，参数仍为1022/batch2/LR5e-4。所有 Pilot 冻结的代码/配置 SHA256 均与原 official_protocol_v1 完全一致。

## 1. Resize / Crop / Padding 顺序

普通公共数据路径 `native_res=false, edge_crop=0`：

1. `main.py:514` **先**执行 `args.image_size = 14 * (args.image_size // 14)`。
2. Dataset 打开原图并转 RGB；本路径 `synthesis_anomalies=False`。
3. Trainer 的可选 `_CropLR(edge_crop)`（本配置禁用）在所有其他变换之前。
4. `Resize((image_size,image_size), BICUBIC)` → `CenterCrop(image_size)` → `ToTensor()` → ImageNet `Normalize`。
5. **没有 Padding**。Resize 已输出同尺寸正方形，后面的 CenterCrop 在此不再裁掉像素。
6. Mask 对应流程为可选 CropLR → nearest Resize → CenterCrop → ToTensor。

所以 `--image_size 1024` 的真实过程是先将参数改为1022，再直接 Resize1022；不是先产生1024图再裁掉2像素，也不是 Resize1024 后 Padding1036。当前默认 observed.json 已直接设置1022。

native 路径 `native_res=true`：

- 可选 CropLR → ToTensor → Normalize → **右/下补零至14的倍数**；无 Resize，也无 CenterCrop。
- Mask：对应可选 CropLR → ToTensor → 同样右/下补零。
- `ph=(14-H%14)%14`、`pw=(14-W%14)%14`，补值是在标准化后的张量空间中为0。
- 如果原图确为1024×1024，会右/下各补12，得到1036×1036。若原图为900×900，则得到910×910；`image_size=1024` 参数不会在此强制统一原图尺寸。
- **当前 main.py:509–512 在 native 模式强制 batch_size=1。** 因此不能简单启用 native 来声称实现了正式 Batch8 的“1024 Resize +1036 Padding”。新正式候选配置明确 `native_res=false`，避开该分支；本次不修改该代码以保护 Pilot。

迁移前代码证据：`provenance/source_dirty.patch` 的 trainer 原有删除行同样是 Resize→CenterCrop→Tensor→Normalize；新增的 Padding 是 native 条件分支。它支持“该迁移来源版本普通路径没有 Padding”，不等于证明投稿历史版本从未采用其他预处理。

另一个1024来源：`dataset/base.py:363–364` 的 `transform_image()` 为像素级合成异常辅助函数，内部 Resize1024。其调用受 `training and synthesis_anomalies` 控制；当前入口 train/test 均显式传 False (`main.py:136,142`)，所以不能拿这处1024解释当前正式数据路径。

## 2. 实际 DINO 输入与 Tokens：动态证据

新增 `tests/verify_input_geometry.py`，使用真实 `AnomalyMambaTrainer` 的变换构建逻辑；仅将其构造阶段不相关的 AnomalyMamba 用一个 CPU 占位模块代替以避免训练模型初始化。DINO 则使用真实本地官方cache、实际 ViT-B/14 和严格加载的预训练权重。在真实 `dino.patch_embed` 前后挂 hook，执行实际 PatchEmbed 和位置编码，在第一个 Transformer block 前停止。

没有修改任何项目运行函数，没有注入 Pilot。全程 CPU、2线程，`torch.cuda.is_initialized()==False`，没有完整 Transformer/VSSM forward/backward。此证据证明形状和 patch 几何，不证明 Batch8 GPU 训练容量。

| 测试路径 | 原图 | 请求 image_size | PatchEmbed 实际输入 | 实际 patch 输出 | Spatial grid |
|---|---|---:|---|---|---|
| 普通 Resize1022 | 真实 bottle900×900 | 1022 | `[8,3,1022,1022]` | `[8,5329,768]` | 73×73 |
| 普通 CLI1024 | 同上 | 1024 | `[8,3,1022,1022]` | `[8,5329,768]` | 73×73 |
| 普通显式 Resize1036 | 同上 | 1036 | `[8,3,1036,1036]` | `[8,5476,768]` | 74×74 |
| native1024 | 合成1024×1024，仅验证补边 | 1024 | `[1,3,1036,1036]` | `[1,5476,768]` | 74×74 |
| native真实 bottle | 真实900×900 | 1024 | `[1,3,910,910]` | `[1,4225,768]` | 65×65 |

CLS token 另计：73×73空间tokens为5329，首个 Transformer 输入为5330 tokens；74×74空间tokens为5476，首个 Transformer 输入为5477 tokens。Patch size 实际对象确认为14，PatchEmbed kernel/stride为(14,14)。

直接把未对齐的 `[1,3,1024,1024]` 送入真实 DINO 会报：

```text
Input image height 1024 is not a multiple of patch height 14
```

因此 DINO 自身不自动将1024 Padding到1036。实际依赖的官方 `dinov2/layers/patch_embed.py` 有高宽整除断言。项目 DINO wrapper 的 prepare_tokens_with_masks 在 patch_embed 前也没有 Resize、Crop 或 Padding。

证据：`config_audit_20261113/input_geometry.json`、同名 `.log`，exit0。先前完整 GPU smoke 的 `[2,768,73,73]` 三层特征同样支持当前1022路径；没有将 CPU 部分执行说成完整模型训练。

## 3. 1022 /1024 /1036 的来源分类

| 尺寸 | 本项目中可验证来源 | 证据层级 / 限制 |
|---|---|---|
| 1022 | configs/observed.json；当前 Pilot 实际CLI、日志、每类配置、完整训练checkpoint配置；早前 GPU smoke 的输入及73×73特征；迁移清单中的旧序列消融文件名 | 当前执行多源一致；不是用户确认的投稿输入 |
| 1024 | Titlelabel1.pdf §5.1及多处效率/分辨率描述；CLI请求1024可以被向下对齐；禁用的dataset像素合成函数；`dinov2_vitl14:[1024,...]` 是通道维而非图像尺寸 | 不能把全部“1024”搜索结果视为实际输入 |
| 1036 | provenance/copy_manifest.json:410–414 列出旧 `mvtec_fig/...1036-8-...noise_all` 的 CSV、日志、TensorBoard路径和hash；动态CPU显式Resize1036 / native1024补边验证 | **旧文件名确实存在于清单**，但旧文件正文未复制进本项目；无法从清单证明具体 Resize/Padding、Batch、训练与投稿对应关系 |

特别是模型命名 `'{dataset}-{model}-{image_size}-{pred_next_n}-...'` (`trainer.py:182–186`) 中 `1036-8` 的8表示 **预测offset `pred_next_n`**，不能读成Batch8。该组旧文件标为 noise_all，也不能直接当成当前 divnoise 运行的完整配置。

### 1024是否名义Resize、1036是否Padding后尺寸？

- 数学及当前 native 单元路径：1024右/下补12 →1036成立。
- 当前普通公共数据路径：**不成立**，1024参数被floor至1022后再Resize。
- “投稿确实 Resize1024 → Padding1036”：**UNKNOWN，缺原始配置/transform代码/运行输入记录**。
- “投稿就是直接 Resize1036”：同样UNKNOWN。显式Resize1036与Resize1024再Padding1036的图像内容不同，不能因最终tokens相同就互换。

## 4. Checkpoint 与历史记录

只读解析本项目已经完成且不再写入的 Pilot epoch1 快照，使用 CPU mmap、weights_only=True：

- image_size=1022；batch_size=2；learning_rate=0.0005。
- native_res=false、edge_crop=0、ViT-B/14、pred_next_n=8。
- optimizer param-group lr=0.0005；completed_epochs=1，global_step=105，对应209训练图、batch2的105步。
- `pos_embed` shape为 `[1,1370,768]`，来自预训练37×37 patch基准位置表（加CLS）；运行时可插值到73×73或74×74。因此 **不能从这个权重形状推断训练输入为518、1022或1036**。

证据：`config_audit_20261113/checkpoint_metadata.json`。本项目没有迁移历史正式checkpoint；当前检查的是本项目 Pilot，而非EAAI投稿checkpoint。不能靠 checkpoint文件名或其他项目的邻近产物补足未知来源。

旧迁移报告记录过42个旧日志header，结论为batch2/LR5e-4/input1022。那是本项目保存的二手核查结论，旧header正文未迁入，本次没有越界重新访问其他研究项目。source_dirty.patch 也可见来源 main.py 的batch默认值为2。

### Batch2 是否显存受限替代配置？

目前只能验证“来源默认/已记录旧日志与当前Pilot确实采用2”，**不能验证其原因**。本项目没有把8降到2的OOM日志、明确降级说明、硬件对应关系或投稿checkpoint配置。没有证据支持将“显存受限降级”写成事实，也不能据这些batch2记录否定用户刚确认的正式Batch8。先前文档中“observed batch2”仅指该来源配置，不再作为未来正式实验的Batch依据。

## 5. 论文LaTeX / PDF / 用户确认 / 代码对照

本项目没有 `.tex` 或 `.bib` 源文件。provenance 的 source_status_before 只记录过 `manuscript_table2_experiments.tex` 文件名，没有该文件正文；故无法完成LaTeX源码逐行核对。可直接核对的论文为用户指定的 `Titlelabel1.pdf`，已有 `paper_text.txt` 提取记录。没有伪称查看或修改LaTeX。

| 项目 | 用户确认的正式设置 | 本地论文PDF | 当前普通路径 / Pilot | 后续行动 |
|---|---|---|---|---|
| Batch Size | **8** | 8 | 2 | 新配置明确8；不改在跑Pilot；不能以旧batch2替代 |
| LR | **5e-4** | §5.1写1e-3 | 5e-4；checkpoint optimizer一致 | 论文LR应修正为5e-4 |
| Patch size | **14**，高宽必须为14倍数 | ViT-B/14 | 真实PatchEmbed14 | 论文应明确patch约束及整除处理 |
| 实际输入 | 可能1036，未确认 | 名义1024×1024 | 实际1022×1022、73×73 | 投稿实际输入UNKNOWN；不擅自定为1036 |
| Sequence Loss | 未授权改变 | 四方向独立展开/head | 每层单次展平、一个head | 仍有实质差异，不修改公式 |
| Anomaly scoring | 未授权改变 | 平方L2残差，上采样、image max | 通道mean MSE+mean L1、3层平均、Gaussian sigma4、image max；pixel评价512 | 仍有实质差异，不修改公式 |

论文修正建议：

1. LR 的 `1×10^{-3}` 改为用户确认的 `5×10^{-4}`；Batch8保持。
2. 分开列出“名义尺寸”“真实网络输入高宽”“预处理顺序”“patch grid”。不能继续把不可整除的1024写成未经处理的ViT-B/14实际输入。
3. 若未来证实投稿使用Resize1024→Padding1036，则明确Resize1024、右/下各补12、实际1036、74×74；同时给出原始运行证据。**本次证据还不足以写这句话。**
4. 若以当前实现进行新实验，则应据实写普通Resize1022、无Padding、73×73，不能声称它与1036 padding方案等价。新实验结果也不能自动补作旧投稿运行的出处。
5. 在上述方法、评分及提交版本对应关系解决前继续标记 **PAPER REPRODUCTION UNVERIFIED**。

## 6. Batch8 运行能力与 Pilot 保护

本次CPU形状验证实际用了batch8，但不是完整GPU训练。GPU0同时运行原Pilot，容量门检查时 used=16267MiB、free=80995MiB、total=97887MiB。为不干扰Pilot，本次没有并发分配新CUDA上下文或启动新的forward/backward。

`tests/probe_batch8_capacity.py` 明确固定batch8、LR5e-4、FP32，要求显式14倍数输入，**没有减batch或改精度的fallback**。GPU占用>256MiB会在Torch导入/上下文初始化前退出4。本次实际结果：

- status=`NOT_RUN_GPU_BUSY`；training_steps=0；capacity_verified=false；oom_observed=false。
- 没有发生或观察到Batch8 OOM；**不能断言Batch8无法运行，也不能把空闲显存数字当作训练通过**。
- Batch8训练峰值分配/保留显存需求：UNKNOWN。不能用Batch2峰值简单乘4冒充实测。

证据：`config_audit_20261113/batch8_capacity_gate/capacity.json`、gate.log、exit_code=4。仅在Pilot完成且GPU空闲后，可人工执行以下一次batch8优化测试；本任务没有排队或自动执行它：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate vmamba_new
PYTHONDONTWRITEBYTECODE=1 python tests/probe_batch8_capacity.py \
  --image-size 1022 --gpu 0 --output experiments/statistical_reliability/NEW_batch8_capacity
```

若确认提交实际尺寸为1036，则容量测试应显式传1036。成功仅证明一次batch8优化可执行；失败会保留实际异常类型、完整错误字符串和本进程CUDA peak allocated/reserved，不会退回Batch2。

## 7. 新正式实验候选配置（未启动）

`official_protocol_v2_confirmed_hparams/` 包含：

- `confirmed_parameters.json`：用户确认Batch8/LR5e-4/patch14及固定五seeds。
- `protocol.json`：Batch8、LR5e-4、native_res=false、edge_crop=0；其余原MARS数学设置保持；限MVTec/VisA。
- **当前可执行尺寸候选显式为1022、73×73**，依据当前代码动态验证；不是宣称已确定投稿尺寸。`submission_actual_input_hw=null`、`submission_resolution_verified=false`，并单列1036候选及74×74。
- `paper_protocol_verified=false`、`execution_authorized=false`、`batch8_gpu_capacity_verified=false`。尺寸、原论文方法一致性和GPU容量未全部核清前不把它称为最终可启动冻结协议。
- `run_five_seeds.sh`：两数据集各5个独立运行命令，新路径 `formal_v2_<dataset>_seed<seed>_attempt01`。现有launcher会拒绝启动；实际测试期望exit2，未创建任何正式输出目录。
- 数据清单/摘要为指向原完整冻结清单的相对symlink，hash与v1一致；没有复制数据，没有改数据或旧清单。

保留旧 `configs/observed.json` / official_protocol_v1 的原因是它们属于现有Pilot及历史来源证据，不应因用户确认新的正式超参数而改写历史或破坏resume identity。未来正式命令必须使用新候选配置，经解决剩余阻断后重新冻结，不能直接沿用旧默认batch2。

## 8. 检查结果与未完成项

通过：实际变换与DINO PatchEmbed CPU hook、三种请求尺寸、native反例、裸1024断言、checkpoint元数据/LR、代码/配置未变核对、JSON解析、shell语法、正式启动门控拒绝。

未完成：投稿LaTeX正文、旧1036日志正文和提交checkpoint均不在本项目；Batch2降级原因无证据；Batch8完整GPU训练容量未测；投稿实际尺寸尚未认证。未终止/改写Pilot，未启动五seed，未调整算法、Sequence Loss或评分。
