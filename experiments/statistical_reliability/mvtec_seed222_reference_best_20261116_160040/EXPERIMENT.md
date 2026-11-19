# Fresh MARS MVTec AD Seed222

User requested the remaining GPU for a different MVTec seed; use physicalGPU2, seed222, all15 categories,1022×1022, Batch8, LR5e-4,60 epochs/category, FP32. Independent trainable initialization; no Seed111 trained model/optimizer is reused. DINO pretrained backbone remains the same frozen external model. Seed is reset before each category exactly as in the existing entry.

Uses scripts/train_reference_best.py: test every5 epochs and final, strictly improve mean I-AUROC/P-AUROC, retain one model-only _best.pth per category including BOS. No per-epoch/full/final-extra checkpoint; no image or heatmap export. Model-only best lacks optimizer/RNG state, so cannot provide exact interrupted-training resume.

All actual data/core-file identities are checked against the existing MVTec frozen inventory before training. Exact launch command, output directory, GPU UUID, PID and exit status are recorded. Existing GPU0/1 jobs and all old results are untouched. BTAD is on hold per user request.

Results are test-selected; old Seed111 MVTec results used fixed final epoch. Do not directly combine them into a supposedly same-protocol Mean±SD. PAPER REPRODUCTION UNVERIFIED. Approximate retained best-model space7.5GB; complete runtime remains unmeasured.
