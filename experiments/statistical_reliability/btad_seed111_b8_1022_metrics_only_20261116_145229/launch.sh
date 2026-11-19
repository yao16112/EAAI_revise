source /opt/conda/etc/profile.d/conda.sh
conda activate vmamba_new
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=GPU-54d94bb2-5b56-1e25-3550-a3217be361fa
nohup setsid python -u /opt/data/private/VarAD_EAAI/experiments/statistical_reliability/btad_seed111_b8_1022_metrics_only_20261116_145229/prepare_and_train.py > /opt/data/private/VarAD_EAAI/experiments/statistical_reliability/btad_seed111_b8_1022_metrics_only_20261116_145229/nohup.log 2>&1 < /dev/null &
