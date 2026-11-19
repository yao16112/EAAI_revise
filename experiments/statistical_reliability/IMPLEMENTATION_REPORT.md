# MARS Multi-Seed Engineering Implementation

Date: 2026-11-13. Workspace: `/opt/data/private/VarAD_EAAI`. Environment: existing `vmamba_new`; no dependency upgrades. Read README.md, MIGRATION_REPORT.md and READINESS_REPORT.md first. Other research projects were not accessed.

| Decision | Current evidence |
|---|---|
| ENGINEERING READY | Export and epoch-boundary state recovery implemented and validated at L0/L1. Full 60-epoch single-category Pilot launched; completion is separately tracked. **Stepwise bitwise training reproducibility is NOT achieved.** |
| PAPER PROTOCOL VERIFIED | **NO — PAPER REPRODUCTION UNVERIFIED.** Current mathematics and protocol differ from the supplied paper. |
| MULTI-SEED READY | Draft MVTec/VisA configuration, manifests, independent seed commands, recovery and Mean ± SD tooling prepared. **Formal launch disabled** until paper/protocol decisions and authorization are recorded. DTD/BTAD excluded. |

This report supersedes only the export/resume limitations in the earlier READINESS_REPORT. Existing migration/readiness reports and previous experimental artifacts are preserved. No multi-seed formal result, other baseline or bootstrap significance test was produced.

## Scope and preservation

Modified `main.py` and `model/VarAD/trainer.py` to connect lifecycle operations; added `tool/experiment.py` for state, provenance and export. Updated affected tests and added bounded verification. New scripts freeze/launch/aggregate experiments and recompute metrics. README documents the new interfaces.

Architecture, perturbation, reconstruction/sequence losses, preprocessing, scoring formulas and `tool/metrics.py` are unchanged. The previous BOS buffer repair remains. `tests/verify_migration.py` still checks immutable core files/methods, allowing that BOS registration; intentional entry/trainer lifecycle changes are covered by behavioral tests instead of pretending their old byte hashes match. Prior manifests remain unchanged. Before-edit main/trainer copies and an implementation patch are in `engineering_v2/` for review.

## Prediction export

Normal CLI training/evaluation now always writes `<new save_path>/predictions/<category>/`:

- `predictions.jsonl`: one record per actual evaluated image, with dataset, category, stable image ID/relative path, GT label, raw image anomaly score, seed, checkpoint SHA256, mask relative path (or explicit synthetic zero-mask policy for normals), defect type and an image cluster ID.
- `000000.npz`, etc.: lossless `anomaly_map` at the actual model output geometry (1022-square here), `evaluation_map` at the geometry actually passed to pixel metrics (512-square here), and the corresponding binary `evaluation_mask`. Float precision is preserved; no JPEG or per-image normalization.
- `manifest.json`: **written last**, marks complete only when all actual test IDs were seen exactly once, records checkpoint absolute path/SHA256, dataset root, seed, count, metric settings and all evaluator metrics.

The evaluator still takes image scores as the max of the **raw smoothed 1022 map, before 512 resize**, and computes metrics from the same original score/map arrays as before. Saving display figures with their existing normalization never changes these inputs. Duplicate IDs, label/category mismatches, nonfinite predictions, incomplete exports and existing export directories fail explicitly. Full training checkpoints retain original seed; evaluation rejects known checkpoint/CLI seed mismatches, also checking the sidecar of the standard final checkpoint. An arbitrary imported raw checkpoint without a sidecar cannot establish training-seed provenance and should not enter formal collections.

The stable image ID permits pairing predictions across seeds and sampling at image level later. No physical part/batch IDs are available: image-level grouping is provided, not proof of workpiece independence. No bootstrap was run.

Verification: full bottle test set, 83 images (20 normal / 63 anomalous), after an explicitly bounded 4-training-image test. Export disabled vs enabled returned **identical values for every evaluator metric**. Independently reading the lossless NPZ/JSONL and recomputing all enabled metrics also gave **maximum difference 0.0**. The verification checks max(raw map)==stored image score, raw→evaluation resize equality, binary mask geometry, IDs and checkpoint/seed pairing. Default `cal_pro=false`; AUPRO/maxIoU were disabled, not independently validated as nonzero metrics.

Evidence: `engineering_v2/resume_export/export_metric_equality.json`, `recomputed_metrics.json`, `predictions/`. These metrics are L1 engineering evidence, not 60-epoch Pilot or paper results.

```bash
python scripts/recompute_predictions.py PATH/TO/predictions/bottle --output NEW_metrics_check.json
```

## Complete epoch checkpoints and recovery

Every completed epoch publishes a unique `*_epoch_001.training.pth` snapshot, through a temporary file, fsync and atomic no-replace link. A partial temporary file is never a valid checkpoint; a kill may leave such a hidden temporary file, which is ignored. Previous snapshots are never overwritten. Directory reuse and duplicate categories fail; the former TensorBoard recursive deletion was replaced by rejection.

Each version-1 snapshot contains:

- full model state_dict, including all three BOS buffers;
- AdamW state_dict, including moments and optimizer step counts;
- scheduler state or explicit None (current training has no scheduler or AMP scaler);
- completed epoch count and global optimizer-step count;
- Python, NumPy, Torch CPU and all CUDA RNG states;
- actual training arguments and seed;
- scientific configuration, execution code hashes, environment and the actual **ordered data records plus image/mask hashes** used to validate a resume;
- explicit DataLoader recovery policy.

Only a complete training snapshot is accepted by `--resume`; a plain inference/model-only checkpoint cannot be mistaken for full resume. Model state is loaded strictly. The entry first reconstructs datasets/loaders/model/optimizer, checks identity, loads trained parameters and optimizer, then restores RNG **after reconstruction and before the next loader iterator**. The loop starts at `completed_epochs`, so completed epochs are not repeated. The current DataLoader has shuffle=true, 8 workers, persistent_workers=false and no separate generator. Saved global Torch RNG regenerates both RandomSampler permutations and worker base seeds. Workers are recreated at each epoch and their default Python/NumPy/Torch seeds are reconstructed. Current transforms are deterministic and synthetic image augmentation is disabled. No change of sampling semantics was introduced.

Mid-epoch iterator/prefetch state is not serialized. An interruption resumes from the preceding completed epoch; work in the unfinished epoch is discarded. This is the promised recovery boundary. Config (including total epochs, seed and batch), ordered data bytes, code or recorded environment mismatch fails explicitly rather than silently resuming a different experiment. Runtime output/resume paths, pause control, print frequency and display-figure setting are excluded from scientific identity. CUDA device count must match. Custom CUDA kernel determinism and OS/driver/library identity beyond the recorded fields are not guaranteed.

Resume into a **new** directory with identical total --epoch and configuration:

```bash
python main.py --dataset mvtec --category bottle --seed 111 --epoch 60 \
  --resume /absolute/path/to/epoch_020.training.pth \
  --save_path experiments/statistical_reliability/NEW_resume_attempt
```

`--ckt_path` is weights-only initialization/evaluation, not training recovery. `--resume` and `--ckt_path` are mutually exclusive. The original `*_final.pth` model-only artifact is still saved before the only final test evaluation. Trainer evaluation accepts both new full training snapshots and registered-state-complete model-only checkpoints. Old pre-BOS checkpoints remain rejected.

`--stop_after_epoch 1` pauses a single explicitly named category at the epoch boundary without changing total `--epoch 60`. It writes PAUSED.json and exits without test evaluation. Resuming an already-completed epoch60 snapshot performs final export/evaluation without another training step, in another fresh directory. Test labels never select a checkpoint, epoch or seed.

## Resume tests: what did and did not match

1. Cheap state test verifies exact model/BOS, AdamW and RNG roundtrip, atomic overwrite refusal and identity-mismatch rejection.
2. Real MARS/1022/FP32 test uses 4 normal bottle training images, batch2, shuffle and 8 workers. Three fresh processes run continuous epochs1–2, interrupted epoch1, and recovery epoch2. Recovered start_epoch=1, final completed_epochs=2, global_step=4. Final Python/NumPy/CPU/CUDA RNG states match exactly.
3. **Trained weights/moments were not bitwise identical.** Max model absolute difference `0.0001293513923883438`; max optimizer moment difference `1.0043004294857383e-07`. Epoch1 mean losses matched `0.1151377335190773`; epoch2 continuous/resumed losses were `0.09592554718255997` / `0.09592555835843086` (difference ≈1.12e-8). No scientific-equivalence tolerance was invented.
4. Additional diagnosis performs no optimizer update: same input, same weights, restored RNG; forward tensors and loss match exactly twice, but 18 gradient tensors differ, max `2.546585164964199e-10`. Thus the installed GPU backward path itself is nondeterministic, independently of serialization. The existing selective-scan extension / Torch backend is involved; precise kernel attribution has not been proved. Adaptive optimizer updates can propagate small gradient differences, but this does not quantify long-run divergence.
5. No architecture/kernel/precision/dependency change was made to conceal the discrepancy. `torch.use_deterministic_algorithms` remains disabled as in the observed setup; enabling it alone would not establish custom extension determinism. **A claim of strict stepwise or cross-hardware bitwise replay is unsupported.**

Evidence: `engineering_v2/resume_export/resume_comparison.json` preserves the original strict-test failure and per-tensor differences; `resume_verification.json` reports the qualified outcome. `backward_replay.json` records independent diagnosis. The strict test invocation exited 1 and was not relabeled a pass. Subsequent state/RNG verification and export recomputation completed separately. Small numerical differences are disclosed, not treated as proof of scientific interchangeability.

Five fresh-process seeds `[111,222,333,444,555]` have five different aggregate initial trainable-parameter hashes (69 trainable tensors each), and loading the same complete registered model state gives identical predictions. This is initialization/reload evidence only, not five independent trained outcomes. See `engineering_v2/five_seed_initialization/summary.json`.

## Paper consistency (source: local Titlelabel1.pdf)

| Item | Current implementation | Paper | Decision |
|---|---|---|---|
| Sequence loss | Per feature level: merged prediction map → row-major view(B,C,-1) → one MLP head → next token; mean MSE+L1; average over 3 feature levels; weight0.5 | §3.5, eq8 (p12): directional sequences and direction-specific heads, averaged over scanning directions | Not equivalent. Four-direction CrossScan inside VSSM does not implement four-direction auxiliary loss. Unchanged. |
| Scoring | `_cal_am`: channel mean squared residual + mean absolute residual; bilinear upsample to1022 (align_corners=true); mean3levels; gaussian_filter sigma4; image max | §3.6, eq10 (pp12–13): channel squared L2 sum residual, upsample, image max | L1 is an extra term; averaging/pooling and smoothing details are not fully specified in that paper section. Unchanged. |
| Pixel evaluation | Masks/maps >512 high resized to512×512; masks nearest, maps cv2 linear | §5.1 (p17): resize anomaly maps to original image resolution | Pixel protocol differs. Unchanged. |
| Input / batch / LR | 1022 / 2 / 5e-4 | §5.1: 1024 / 8 / 1e-3 | Different. Unchanged. |
| Epoch / selection | 60, fixed final before testing | 60; public-data selection not fully specified | Same nominal budget; exact selection correspondence unproved. |
| Augmentation | Resize/CenterCrop/ImageNet normalization; no random flips/jitter | §5.1: horizontal/vertical flips + slight jitter | Different. Unchanged. |

Actual current paths: `AnomalyMamba._compute_sequence_prediction_loss`, `cal_loss`, `_cal_am`, `cal_am`, `VSSM.forward` and `AnomalyMambaTrainer.evaluation`. Core method fingerprints remain unchanged. Current `configs/observed.json`, local earlier smoke/checkpoint configs and this task's actual Pilot arguments corroborate the current settings. The only available historical-log conclusion is the existing migration report's inspection of 42 headers. Those historical headers are not in this project and were not re-accessed; they are not primary evidence that this is the submitted implementation. **PAPER REPRODUCTION UNVERIFIED** remains mandatory.

## MVTec / VisA frozen preparation

`official_protocol_v1/dataset_files.jsonl` freshly records every referenced image and anomalous mask with split, label, class and SHA256. Summary and complete training/code freeze are adjacent:

| Dataset | Categories | Train normal | Test normal | Test abnormal |
|---|---:|---:|---:|---:|
| MVTec AD | 15 | 3629 | 467 | 1258 |
| VisA | 12 | 8659 | 962 | 1200 |

No training anomalies, no cross-split byte-identical images; VisA actual meta.json agrees exactly with local official split_csv/1cls.csv (object column). Previous full mask/size validation is preserved in `dataset_check_20261113/`. Current freeze does not rerun a perceptual/near-duplicate or physical-source-group audit. No split/data edits. Counts matching the paper do not prove complete historical protocol identity. DTD-Synthetic and BTAD are absent from formal configuration and scripts.

`official_protocol_v1/protocol.json` is a frozen **DRAFT_OBSERVED_IMPLEMENTATION**, with both `paper_protocol_verified=false` and `execution_authorized=false`. It specifies observed 60epoch/1022/batch2/LR5e-4, all other retained settings and seed list. `run_five_seeds.sh` lists ten independent commands (five seeds per dataset); it is syntax-checked but **not executed**. The launcher was tested to reject the current gate (expected exit2), without creating an output directory.

The launcher freezes logs/argv/PID/exit status and verifies code plus image/mask bytes before any authorized launch. Each seed gets a fresh attempt directory. Failure recovery launches only the failed category with --resume and a new output; completed categories remain untouched. Explicit collection indexes select each category's complete final export from its retained attempt, avoiding accidental merging of duplicate partial attempts. No test-metric-based attempt selection is implemented.

```bash
# Future use only after protocol and execution authorization are resolved:
python scripts/run_statistical_seed.py --protocol experiments/statistical_reliability/official_protocol_v1/protocol.json \
  --dataset mvtec --seed 111 --category bottle --resume /path/to/complete.training.pth --output NEW_attempt
# Edit a copy of the collection template to name each completed export path:
python scripts/summarize_seeds.py --index NEW_collection.json --output NEW_mean_sd.json
```

`summarize_seeds.py` requires all five fixed seeds, all15/12 categories, complete exports, equal test IDs/labels and independent checkpoint IDs. It computes equal category macro means per seed and then mean/sample SD(ddof=1) over the five seed values. It rejects missing seeds/categories rather than silently shrinking the sample. A synthetic test checks mean3 and SDsqrt(2.5). No bootstrap, significance test, confidence interval, seed selection or formal statistic was run on these pilot results.

## Single-category original-budget Pilot

MVTec bottle seed111, 209 normal training images, 83 test images; original observed60epoch/1022/batch2/LR5e-4. Only epoch60 is evaluated, after saving it. It deliberately pauses after complete epoch1 and resumes epochs2–60 in a new process/directory. Output: `pilot_bottle_seed111_20261113_v2/`. Detached launcher and commands are recorded under that directory and `engineering_v2/pilot_launch.log` / `pilot_launcher.pid`.

It is **PILOT, not formal five-seed evidence**. See PILOT_RESULTS.md for live/completed status. The runner records stage PIDs, exact argv, exit codes and recomputes exported metrics on completion; it writes PILOT_COMPLETE.json only after all checks finish. Failure writes PILOT_FAILED.json. Successful launch is not completion. Time/ETA are reported only from observed progress; no60epoch metric has been invented.

## Verification summary

- L0: syntax/configs; immutable core file/method preservation; shell syntax.
- Cheap behavior: state/BOS/optimizer/RNG restore, identity mismatch, no overwrite, fixed-final-before-test, export identity guards, Mean±SD arithmetic and missing-seed refusal.
- L1: five seed initialization/model reload; real-model continuous/interrupted training comparison with **numerical caveat**; full83-image export on/off metric equality; independent lossless-export metric recomputation difference0.
- L2: original60epoch one-seed bottle Pilot running separately; not yet a completed result at report creation.
- Formal: not launched. Bitwise training equivalence, all-class full training, other hardware, AUPRO enabled, physical cluster independence and submitted-version correspondence remain unverified.
