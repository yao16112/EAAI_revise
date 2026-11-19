# Authorized current-implementation MVTec AD experiment

All15 categories, seed111 only, 60 epochs/category, input1022, Batch8, LR5e-4, DINOv2 patch14/grid73x73, FP32. Fixed final epoch selected before test. Original architecture, losses and scoring retained. PAPER REPRODUCTION UNVERIFIED.

The current user request authorizes this run. The old global draft gate and existing Pilot are unchanged. This run caps its allocator at70% of GPU memory to protect the Pilot. OOM under that cap is a shared-resource failure, not proof Batch8 cannot run alone; no batch/precision fallback. A post-step observer records the first real AdamW step without modifying inputs, parameters or gradients. Approximately0.8TB for per-epoch checkpoints; complete-dataset runtime is not yet measured.
