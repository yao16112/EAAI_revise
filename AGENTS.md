# MARS EAAI

- Use the existing `vmamba_new` Conda environment.
- Preserve the copied MARS architecture, perturbations, losses, preprocessing and scoring.
- Keep datasets and pretrained weights external; do not copy historical checkpoints or outputs.
- Use a new output directory for every experiment. Never overwrite existing results.
- Do not use test labels to select an epoch or seed. The entry point saves the fixed final epoch before testing.
- Record actual configuration, seed, environment, checkpoint and exit status. Report only real results.
- No formal training is authorized by the migration task; only its bounded smoke test is authorized.

## Current user-authorized exception (2026-11-16)

The user explicitly confirmed the reference `/opt/data/private/VarAD` save policy: test every5 epochs and at the final epoch; select strictly better mean I-AUROC/P-AUROC and replace one model-only `_best.pth` per category. This overrides the earlier fixed-final/no-test-selection requirement for new reference-best runs only. Label them test-selected. Keep existing results/full checkpoints unchanged; retain BOS. Use `scripts/train_reference_best.py` for these runs, with no per-epoch snapshots or image/heatmap export. Model-only best checkpoints cannot provide full optimizer/RNG resume. Do not claim protocol equivalence with earlier results or paper reproduction.
