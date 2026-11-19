source /opt/conda/etc/profile.d/conda.sh
conda activate vmamba_new
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=GPU-ea7e1b69-d452-bab6-4924-67e9e8f6f09a
nohup setsid python -u /opt/data/private/VarAD_EAAI/experiments/statistical_reliability/visa_seed111_b8_1022_metrics_only_20261116_144832_retry1/runner.py > /opt/data/private/VarAD_EAAI/experiments/statistical_reliability/visa_seed111_b8_1022_metrics_only_20261116_144832_retry1/nohup.log 2>&1 < /dev/null &
