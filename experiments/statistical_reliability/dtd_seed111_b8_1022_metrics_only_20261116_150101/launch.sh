source /opt/conda/etc/profile.d/conda.sh
conda activate vmamba_new
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=GPU-87c9ac48-ee96-e2f0-5553-cf2bd2756984
nohup setsid python -u /opt/data/private/VarAD_EAAI/experiments/statistical_reliability/dtd_seed111_b8_1022_metrics_only_20261116_150101/runner.py > /opt/data/private/VarAD_EAAI/experiments/statistical_reliability/dtd_seed111_b8_1022_metrics_only_20261116_150101/nohup.log 2>&1 < /dev/null &
