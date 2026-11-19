# MARS for the EAAI revision

Standalone extraction from `/opt/data/private/VarAD`, using the existing DINOv2 ViT-B/14 + Adapter + VSSM MARS implementation. The source repository is not on the runtime import path. See `MIGRATION_REPORT.md` before interpreting any result as a paper reproduction.

Use the existing environment:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate vmamba_new
cd /opt/data/private/VarAD_EAAI
export PYTHONDONTWRITEBYTECODE=1
```

`configs/paths.json` configures the external datasets, local official DINOv2 hub cache, and pretrained weight file. Set `MARS_PATHS_CONFIG=/absolute/path/to/paths.json` to use another path configuration. No data, pretrained weights, or historical checkpoints are bundled or automatically downloaded. The cached official DINOv2 implementation is an external dependency, not old VarAD Python source.

`configs/observed.json` contains the observed public-run settings: 60 epochs, batch 2, LR 5e-4, 1022 input, DINOv2 blocks 4/8/12, predictive offset 8, divnoise, and the existing sequence loss. CLI flags override these settings. This is distinct from the paper's batch 8 / LR 1e-3 / 1024 description.

Training command (documentation only; not launched during migration):

```bash
python main.py --dataset mvtec --category bottle --seed 111   --save_path experiments/mvtec_bottle_seed111
```

Evaluation uses the same model configuration; new checkpoints restore BOS independently of the initialization seed:

```bash
python main.py --dataset mvtec --category bottle --seed 111   --do_train false --ckt_path /absolute/path/to/model_final.pth   --save_path experiments/mvtec_bottle_seed111_eval
```

Output directories must be new. Training checkpoints end in `_final.pth` and are saved before the single final test evaluation. Existing trainer metric definitions, including 512-square pixel evaluation for larger masks/maps, remain unchanged. Image/pixel maxF1 are **Oracle maxF1**, selected by sweeping the test thresholds. Dataset aggregation is the original equal category mean.

Fixed random BOS tensors are now persistent buffers. New checkpoints restore them with strict state loading; legacy checkpoints missing BOS are rejected because their exact BOS values cannot be recovered from checkpoint bytes. Epoch-boundary full training resume and lossless prediction export are now implemented; strict bitwise training replay is not guaranteed by the current GPU backward backend. Current readiness evidence and limits are in [READINESS_REPORT.md](experiments/statistical_reliability/READINESS_REPORT.md).

Verification only:

```bash
python tests/verify_migration.py
python -u tests/smoke_test.py --output experiments/smoke_check
```

The smoke test uses the observed 1022 resolution and batch 2 in FP32, with one synthetic training iteration, checkpoint reload, real-image inference, and a two-image evaluation. It is not a formal run or performance estimate. BTAD availability and DTD split differences remain documented limitations.


Epoch-boundary recovery (new output directory, same scientific configuration and total epoch budget):

```bash
python main.py --dataset mvtec --category bottle --seed 111 --epoch 60 \
  --resume /absolute/path/to/complete_epoch.training.pth --save_path experiments/NEW_resume_attempt
```

Every complete epoch saves full model/BOS, optimizer, RNG, counters, configuration and data/code identity. Recovery replays only an unfinished epoch. The evaluator writes lossless original/evaluation maps and paired masks plus raw image scores under `predictions/<category>/`; no per-image normalization is used for metrics. `scripts/recompute_predictions.py` independently verifies the export. Existing outputs are rejected.

See [IMPLEMENTATION_REPORT.md](experiments/statistical_reliability/IMPLEMENTATION_REPORT.md) for the measured backward nondeterminism and paper discrepancies, and [PILOT_RESULTS.md](experiments/statistical_reliability/PILOT_RESULTS.md) for the separate60epoch Pilot. Draft five-seed commands and manifests are under `official_protocol_v1/`; their formal launcher is disabled while paper/protocol verification remains unresolved.

2026-11-16 current user-confirmed checkpoint policy: use `scripts/train_reference_best.py --plan PATH_TO_PLAN.json`. It selects the best mean image/pixel AUROC on the test set every5 epochs and at the final epoch, and atomically replaces one model-only `_best.pth` per category. No per-epoch full snapshot or extra final model is written. This intentionally changes the earlier fixed-final protocol. New best files include BOS but omit optimizer/RNG state; full training resume is unavailable from them. Images/heatmap export is disabled. Legacy `main.py` remains frozen for old checkpoint identity verification. See `experiments/statistical_reliability/reference_best_migration_path.txt` and that directory's POLICY_CHANGE.md for active VisA/DTD plans, logs, migration evidence and limitations.
