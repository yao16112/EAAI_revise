# Legacy best-only checkpoint entry point

Use `python scripts/train_best.py --dataset mvtec --save_path experiments/NEW_best_only` for a new run. Use an idle GPU through `CUDA_VISIBLE_DEVICES`; the output directory must be new. Defaults remain seed111, Batch8, LR5e-4, input1022 and60 total epochs, using the existing MARS architecture, perturbations, losses, preprocessing and scoring. External dataset paths retain the existing `MARS_PATHS_CONFIG` support.

This entry point implements the user's explicitly requested old VarAD selection rule:

- Evaluate the test set every5 epochs for MVTec/VisA, every1 epoch for BTAD/DTD, and always at the final epoch. `--valid_freq` can override the frequency.
- Select the highest `(test I-AUROC + test P-AUROC) / 2`; update only on strict improvement. A tie keeps the earlier checkpoint. The first finite evaluation is saved even when its score is zero.
- Each category retains one `*_best.pth`, atomically replaced on improvement. No per-epoch checkpoints or extra final checkpoint are produced. Old experiment files are never deleted or overwritten.
- The checkpoint includes model state (including BOS), seed, actual configuration and selection metadata. Its version2 container is supported by the current inference loader. It intentionally omits optimizer/RNG state and cannot provide full training resume.
- Final `*_metrics.json` describes the selected best epoch, not automatically the last epoch. `*_selection.json` records the selected epoch, score and the use of test labels. Scalar logs and provenance are retained; no per-image prediction, heatmap or visualization files are saved.

The selection uses test labels and can bias reported test performance. It is explicitly labeled `uses_test_labels=true` and must not be represented as independent test evaluation or as the previous fixed-final-epoch protocol. PAPER REPRODUCTION UNVERIFIED.

This is an isolated entry point. Existing `main.py`, model, utilities, frozen configurations and already-running processes are unchanged. They will not switch policies automatically. No training is started by installing or testing this entry point.
