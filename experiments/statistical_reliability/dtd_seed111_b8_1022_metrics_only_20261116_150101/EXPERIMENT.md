# DTD-Synthetic MARS experiment

User-authorized current-implementation single-seed run. GPU0, seed111, all12 categories, 1022×1022, Batch8, LR5e-4, FP32, 60 epochs/category. Fixed final epoch saved before testing. No changes to model, losses, scoring or dataset. Other GPU jobs left unchanged.

Actual local split:1200 normal training /357 normal test /947 anomalous test. Paper split:459/518/947. No split adjustment. PAPER REPRODUCTION UNVERIFIED. Formal five-seed launch not performed.

Save final aggregate metrics/logs and per-epoch recovery checkpoints. Per-image predictions, images and anomaly-map disk export disabled by output-only runtime adapter; evaluator calculations unchanged. Existing datasets/results preserved. Approximate additional checkpoint budget620GB; initial free space3.3TB. No automatic batch or precision fallback.

Read nohup.log for progress, first_optimizer_step.json for startup evidence, exit_status.json for eventual completion/failure. Per-category metrics are under artifacts/VarAD/models/*_metrics.json. Full-dataset runtime is not yet measured.
