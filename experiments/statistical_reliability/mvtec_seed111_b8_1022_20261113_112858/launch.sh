source /opt/conda/etc/profile.d/conda.sh
conda activate vmamba_new
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
nohup setsid python -u /opt/data/private/VarAD_EAAI/experiments/statistical_reliability/mvtec_seed111_b8_1022_20261113_112858/runner.py > /opt/data/private/VarAD_EAAI/experiments/statistical_reliability/mvtec_seed111_b8_1022_20261113_112858/nohup.log 2>&1 < /dev/null &
