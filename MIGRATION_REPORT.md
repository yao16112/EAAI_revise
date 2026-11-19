# MARS EAAI migration report

Status: **CODE READY / PAPER REPRODUCTION UNVERIFIED**.

The independent project was created at `/opt/data/private/VarAD_EAAI`. All 26 copied source files came exclusively from `/opt/data/private/VarAD`; no other research checkout was consulted or used. Newly authored material is limited to path/profile configuration, documentation, provenance, and bounded tests. Formal training was **NOT STARTED**. A single synthetic optimizer iteration was performed only as the explicitly authorized L1 smoke test.

## Source identity and preservation

- Source HEAD: `390051c7a52a71962974ce5ada68ebd6073aa741`. The source already had uncommitted changes; HEAD alone does not describe the extraction. `provenance/source_dirty.patch` and `source_status_before.txt` preserve that state.
- Explicit whitelist: 26 copied files, 163,836 original bytes. Original and extracted SHA-256 for each file are in [copy_manifest.json](provenance/copy_manifest.json). Exact extraction changes are in [extraction.patch](provenance/extraction.patch).
- Before/after verification: 230 tracked source files and every copied source file matched their original hashes; Git status was unchanged. See [source_preservation_check.json](provenance/source_preservation_check.json). No old data, checkpoints, or experiment outputs were written. Their complete contents were not redundantly hashed.
- Target did not exist before creation. Creation used an explicit list and exclusive file creation, not recursive repository copying. No pre-existing destination files were overwritten.

## File retention

| Target (same source-relative path) | Class | Treatment |
|---|---|---|
| LICENSE | KEEP | byte-identical |
| main.py | KEEP | imports/paths/entry control only |
| model/__init__.py | KEEP | byte-identical |
| model/VarAD/__init__.py | KEEP | byte-identical |
| model/VarAD/anomaly_mamba.py | KEEP | imports/paths/entry control only |
| model/VarAD/trainer.py | KEEP | byte-identical |
| model/VarAD/vssm.py | KEEP | byte-identical |
| model/VarAD/dino_vision_tokenizer.py | KEEP | imports/paths/entry control only |
| model/VarAD/lora.py | KEEP | byte-identical |
| model/VarAD/tokenizer_backbones/__init__.py | KEEP | imports/paths/entry control only |
| model/VarAD/tokenizer_backbones/dino/__init__.py | KEEP | byte-identical |
| model/VarAD/tokenizer_backbones/dino/model.py | KEEP | imports/paths/entry control only |
| dataset/__init__.py | KEEP | imports/paths/entry control only |
| dataset/base.py | KEEP | byte-identical |
| dataset/mvtec.py | KEEP | imports/paths/entry control only |
| dataset/visa.py | KEEP | imports/paths/entry control only |
| dataset/btad.py | KEEP | imports/paths/entry control only |
| dataset/dtd.py | KEEP | imports/paths/entry control only |
| dataset/synthesis_utils.py | KEEP | byte-identical |
| config/__init__.py | CONFIG | imports/paths/entry control only |
| config/global_config.py | CONFIG | imports/paths/entry control only |
| tool/__init__.py | KEEP | byte-identical |
| tool/logger.py | KEEP | byte-identical |
| tool/metrics.py | KEEP | byte-identical |
| tool/utils.py | KEEP | byte-identical |
| tool/visualization.py | KEEP | byte-identical |

KEEP dependency closure: `main.py` -> `dataset` and `model.VarAD.trainer` -> `anomaly_mamba`, `vssm`, `dino_vision_tokenizer` -> local DINO wrapper and `lora`; trainer -> original `tool` metrics/logger/visualization. `BaseDataset` imports `synthesis_utils` and imgaug even when pixel synthesis is disabled, so their import closure was retained without changing data-loader behavior. LoRA remains an imported helper; the observed profile activates only the original linear Adapter. The license preserves source attribution.

Reused VarAD components include Adapter/tokenization plumbing, VSSM directional scanning, dataset loaders, trainer and metric utilities. The MARS path additionally uses the existing structured `divnoise` perturbation and auxiliary next-token prediction. Whole source files in the closure are preserved rather than rewriting their mathematics to remove inactive functions.

CONFIG: `config/` exports paths loaded from the new `configs/paths.json`; `configs/observed.json` records the selected observed settings. Neither file reads old repository Python.

EXTERNAL: MVTec/VisA/BTAD/DTD data and masks; local cached official DINOv2 source and ViT-B/14 pretrained weights; existing Conda environment and CUDA extension. Historical checkpoints remain outside this project and are not required by the smoke test. No external data, pretrained weights or historical checkpoint was copied. The new `experiments/smoke_20261113/smoke_only.pth` was produced by this task's one-iteration test and is explicitly not a benchmark model.

SKIP: other pretrained backbones and their configuration/assets; non-DINO `vision_tokenizer.py`; CNN/Transformer `decoder_variants.py`; ceramic loaders/scripts; backup/date-suffixed models; historical experiment/analysis/debug/plot scripts; old logs, checkpoints, outputs, figures, caches, `.git`, virtual environments. They are outside the selected DINOv2/VSSM MARS run path. Removing eager unrelated imports prevents their transitive import trees being dragged into this project.

UNRESOLVED: exact EAAI submission source/run mapping, paper four-direction auxiliary-loss description versus observed single-direction implementation, BTAD image location, DTD split/count differences, and legacy non-parameter BOS checkpoint serialization.

## Preserved computation and dependency handling

The VSSM implementation, trainer, metrics, RNG helper, dataset parsing and preprocessing methods are copied unchanged. AnomalyMamba changes remove only inactive alternate-backbone/decoder imports and branches; all its computational methods except constructor are AST-identical. The retained DINOv2 and VSSM constructor branches preserve the source expressions. DINO projection/normalization methods are AST-identical. The DINO wrapper's constructor now reads configured local source/weight paths, constructs the same architecture with `pretrained=False`, and loads the explicitly configured pretrained state strictly; no download is attempted. All feature extraction/position interpolation methods remain AST-identical. `tests/verify_migration.py` checks saved source method fingerprints without reading the old repository.

The external cached official DINOv2 implementation is a declared backbone dependency, analogous to installed libraries. This is independent of old VarAD Python source; it is not a vendor-complete offline bundle. Its hub path and the pretrained weight path/hash are recorded in [external_paths_audit.json](provenance/external_paths_audit.json).

## Observed configuration versus paper

| Setting | Observed code / historical public logs | Paper description |
|---|---|---|
| Backbone | DINOv2 ViT-B/14 | DINOv2 ViT-B/14 |
| Feature blocks | 4/8/12; zero-based 3/7/11; three 768-channel maps | Must retain recorded submission provenance |
| Input | 1022 x 1022 after the source DINO multiple-of-14 alignment | 1024 x 1024 |
| Epochs | 60 | 60 |
| Batch | 2 | 8 |
| LR | 5e-4 | 1e-3 |
| AdamW betas | (0.5, 0.999), source trainer | (0.5, 0.999) |
| Corruption | source divnoise; four spatial templates and six standardized distributions | Structured feature-space perturbation |
| Corruption range / clean gate | [0.02, 0.22] / 0.5, source defaults | [0.02, 0.22] / 0.5 |
| Auxiliary loss | one flattened next-token sequence, one head per feature level; weight 0.5 | Four directional predictions; weight 0.5 |
| VSSM scanning | Four directions; offset 8 at first level | Distinct from the auxiliary loss definition |
| Augmentation | source resize/center crop/tensor/ImageNet normalization | Random flips and slight color jitter described |
| Image score | maximum of the source smoothed anomaly map | Submission scoring version not fully traced |
| Pixel evaluation | original source evaluator reduces larger masks/maps to 512-square | Original-image-resolution evaluation described |

Historical evidence used: `scripts/experiments/run_noise_all_*.py` and all 42 headers under source `outputs/divnoise_{mvtec,visa,btad,dtd}/logs/`. Those artifacts were inspected but not copied. They agree on 60 epochs, batch 2, LR 5e-4, input 1022, DINOv2 ViT-B/14, predictive offset 8, sequence prediction enabled and divnoise. Some remaining values come from current code defaults, not explicit historical log fields. No claim is made that the current source, defaults, and historical checkpoints fully reconstruct the submitted tables.

No four-direction auxiliary loss, new perturbation, new decoder, new precision mode or input change was introduced to match the paper. The default profile encodes observed settings rather than silently adopting paper settings.

## Entry-point policy changes (operational, not model mathematics)

The copied historical `main.py` used test `(I-AUROC + P-AUROC)/2` to choose the best epoch and seeded 111 at import. The independent entry instead takes an explicit `--seed` (default 111), seeds before each category's model construction, saves a fixed final-epoch checkpoint **before** test evaluation, and tests only once at the end. This satisfies the requested AGENTS rule and is an explicit protocol departure from historical epoch selection; results must not be described as identical to that old selection procedure.

`--save_path` is required and must be new, including for evaluation. Defaults are loaded from `configs/observed.json`, with CLI overrides recorded in a per-category config. Training errors propagate as nonzero exits rather than being swallowed. Evaluation requires an explicit existing checkpoint. CLI supports only the selected ViT-B/14 + VSSM path and the four public datasets. Full interrupted-run resume is not implemented.

Original anomaly-map formulas, image score, pixel geometry, six metrics, threshold sweep and equal category averaging are retained. Both maxF1 metrics are **Oracle maxF1**. The source's permissive checkpoint loading and missing-mask fallbacks remain legacy limitations; strict registered-state loading is additionally tested by the smoke test, not silently assumed for arbitrary external checkpoints.

## External data/path audit

No data were downloaded or modified. The configured paths and current manifest/file-existence observations are recorded in [external_paths_audit.json](provenance/external_paths_audit.json).

| Dataset | External root | Current manifest train N / test N / test A | Status |
|---|---|---|---|
| MVTec AD | `/opt/data/private/MVTecAD` | 3629 / 467 / 1258 | Present; real images exercised by smoke |
| VisA | `/opt/data/private/Data/VisA` | 8659 / 962 / 1200 | Manifest and representative file present; no full evaluation here |
| BTAD | `/opt/data/private/Btad` | UNKNOWN | Directory empty; alias points to it; not ready |
| DTD-Synthetic | `/opt/data/private/data/dtd-synthetic` | 1200 / 357 / 947 | Present; differs from paper 459 / 518 / 947 |

These migration-stage observations do not replace a frozen full dataset audit. Content duplicates, split/version provenance, full RGB decoding and all-category mask semantics are not certified by the one-category smoke. BTAD remains a missing external resource, not synthetic data or a zero-valued performance result.

## Environment and smoke evidence

Environment: `/opt/conda/envs/vmamba_new/bin/python`; PyTorch `2.11.0.dev20260119+cu128`, CUDA `12.8`, cuDNN `91501`; GPU `NVIDIA RTX PRO 6000 Blackwell Server Edition`. `selective_scan_cuda_oflex` is the installed extension used through the unchanged source VSSM adapter; missing alternate module/package names did not prevent this smoke from completing. No dependencies were installed or upgraded.

The normal Codex shell's known bwrap failure was not repaired or hidden. All filesystem actions and GPU verification used the existing per-command approval mechanism for this explicitly authorized migration and bounded smoke. No Docker, host, Codex sandbox/approval configuration or privileged setting was changed. Successful approved execution does not imply ordinary sandbox recovery.

Exact smoke command, cwd `/opt/data/private/VarAD_EAAI`:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate vmamba_new
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
python -u tests/smoke_test.py --output experiments/smoke_20261113
```

Actual invocation redirected stdout/stderr to [smoke_20261113.log](experiments/smoke_20261113.log) and recorded its exit code separately in [smoke_20261113.exit_code](experiments/smoke_20261113.exit_code): **0**. To repeat, use a fresh output/log name. JSON stage evidence: [smoke_result.json](experiments/smoke_20261113/smoke_result.json).

| Exercised stage | Result |
|---|---|
| imports | PASS |
| model_initialization | PASS |
| dummy_forward | PASS |
| loss | PASS |
| backward | PASS |
| single_training_iteration | PASS |
| checkpoint_save_load | PASS |
| real_image_inference | PASS |
| evaluation_entry_path | PASS |
| runtime_independence | PASS |

Actual smoke settings: 1022-square input, batch 2, FP32, three `[2,768,73,73]` feature tensors; finite loss; 69 gradient-bearing tensors with finite gradients; frozen DINO parameters had no gradients; one AdamW step changed the Adapter parameter. Runtime 14.76 seconds; peak allocated GPU memory 13.70 GiB. Real anomaly map is `[1022,1022]`, finite, and saved to `experiments/smoke_20261113/real_anomaly_map.npy`; image identity and score are in `real_image.json`. The two-image evaluation JSON is explicitly smoke-only and must never enter benchmark or multi-seed tables.

The smoke process installs an audit hook denying reads under `/opt/data/private/VarAD`, removes that root from its import path, and verifies module origins. No old repository module was loaded. Local imports resolve to the new project; external installed libraries and the configured official DINOv2 cache remain declared dependencies.

## Checkpoint reproducibility limit

The unchanged legacy VSSM creates random nonlearnable `BOS` tensors as plain attributes. They are absent from `state_dict`. The migrated model therefore **does not claim seed-independent or portable checkpoint restoration**. Same seed/configuration/model-construction order rebuilt these tensors exactly in the tested environment, and a freshly constructed model with strictly loaded registered state produced exactly the same anomaly map (maximum difference 0.0). The original algorithms and serialization were not patched to hide this inherited limitation. An arbitrary historical checkpoint without its initialization provenance cannot be certified by this test. Optimizer/RNG/epoch restoration for training resume is also unsupported.

## Remaining limits and next use

This delivery establishes code execution for the tested MARS path, not paper reproduction, statistical reliability, all-dataset training readiness, or deterministic behavior across hardware/library revisions. The source's single-direction sequence loss, observed hyperparameters, evaluator geometry, checkpoint selection difference, and BOS limitation must be resolved or explicitly disclosed before a formal EAAI experiment. BTAD data and the DTD version/count discrepancy remain unresolved. No baseline, bootstrap, multi-seed training, full epoch or full benchmark evaluation was run.

Training/evaluation examples are in [README.md](README.md). Project rules are in [AGENTS.md](AGENTS.md). Final status remains **CODE READY / PAPER REPRODUCTION UNVERIFIED**, with BTAD unavailable and full checkpoint portability unverified.

## Final entry-point and preservation checks

`python tests/verify_migration.py`: PASS (copied hashes, computational method ASTs, Python syntax and JSON). `python tests/test_entrypoint.py`: PASS, using a mock model for three fake epochs; the observed event order was train/train/train/save/test, including zero test metrics. This verifies final-epoch selection is independent of test values; it is not additional GPU training. Both checks completed in the successful verification invocation. Final copied source/target hash readback also passed. See [verification_summary.json](provenance/verification_summary.json).
