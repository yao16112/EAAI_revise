# User-confirmed reference checkpoint policy

2026-11-16: user explicitly approved changing the earlier fixed-final/non-test-selection rule to the save policy in /opt/data/private/VarAD/main.py: default every5 epochs and final epoch, strictly improve (I-AUROC+P-AUROC)/2, replace one _best.pth per category. The first eligible candidate is always saved (including a zero score); ties keep the earlier candidate.

New entry: scripts/train_reference_best.py. Main/model/metric/core files remain frozen so old full checkpoint identities remain verifiable. Source-reference project is read-only. Mathematical model, BOS buffer, losses, scoring, data, batch8, LR5e-4,1022, seed111 and total60 epochs remain unchanged. Evaluation frequency and checkpoint selection have changed and results are test-selected, not comparable as the same protocol to old final-epoch results. PAPER REPRODUCTION UNVERIFIED.

Only model state_dict including BOS is saved, with atomic overwrite of the current run's single owned best file. No per-epoch full checkpoint, optimizer, RNG or extra final model is saved in the new run. New best files therefore do not provide complete interrupted-training resume. Old full snapshots remain untouched and can still be used for migration.

VisA and DTD are migrated from validated complete old snapshots, in new directories. Incomplete epoch work is discarded when stopping the old verified job sessions. Historical eligible checkpoints are evaluated to recover the past best; then the latest complete optimizer/RNG/model state is restored before remaining training. Completed categories undergo candidate evaluation without retraining. Old fixed-final weights/metrics are not modified. Historical candidates originated under the former training/evaluation schedule; do not claim a wholly new reference-policy run from epoch1.

No images or anomaly maps exported. Per-category evaluation_history.jsonl records candidates; *_best.json records selection score/epoch and metrics. Only the best .pth is retained in each new category directory. Final RESULTS.json is an equal-category macro average of this test-selected policy, not unbiased held-out performance or five-seed Mean±SD.

Tests: tests/test_reference_best.py passed AUROC criterion, ties, single-file overwrite, BOS/model-only content, cadence and final epoch; frozen-core verification passed. Migration preflight validated real DTD/VisA model+optimizer/RNG/config/data identity and BOS without training on the spare GPU. See migration_preflight.json and stop_record.json.
