# Pilot Results

**Pilot: COMPLETE** — MVTec AD bottle, seed111, fixed final epoch60.

This is an engineering pilot, excluded from the official five-seed statistics. PAPER REPRODUCTION UNVERIFIED.

- Training: 209 normal images; batch2; input1022; LR5e-4; 60 total epochs.
- Paused after epoch1 / step105, resumed in a new process and output directory; completed epochs2–60.
- Full test: 20 normal + 63 abnormal images; 83 lossless exports.
- Recomputed all evaluator metrics from exports: maximum difference 0.0.
- Elapsed: 4077.8 seconds. Both training stage exit codes: 0.
- Metrics (%): `{"auroc_px": 98.70225942590758, "auroc_im": 99.92063492063492, "f1_px": 78.99055011545317, "f1_im": 99.21259842019717, "ap_px": 81.7689920187435, "ap_im": 99.97519841269839, "aupro": 0, "max_iou": 0, "fnr": 0.0, "fpr": 4.999999999975}`.
- Exact commands, PIDs, checkpoints and logs: `experiments/statistical_reliability/pilot_bottle_seed111_20261113_v2`.
- Completed evidence: `experiments/statistical_reliability/pilot_bottle_seed111_20261113_v2/PILOT_COMPLETE.json`.

Resume caveat: separate L1 tests restored all RNG/counters/optimizer state, but CUDA backward produced small numerical differences; stepwise bitwise equality is not promised. See IMPLEMENTATION_REPORT.md.

ENGINEERING READY: epoch recovery and export tested. PAPER PROTOCOL VERIFIED: NO. MULTI-SEED READY: formal launch remains disabled pending protocol decisions.
