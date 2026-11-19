source /opt/conda/etc/profile.d/conda.sh
conda activate vmamba_new
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
nohup setsid python -u /opt/data/private/VarAD_EAAI/experiments/statistical_reliability/mvtec_seed111_recovery_3gpu_20261116_132337/dispatch.py > /opt/data/private/VarAD_EAAI/experiments/statistical_reliability/mvtec_seed111_recovery_3gpu_20261116_132337/nohup.log 2>&1 < /dev/null &
