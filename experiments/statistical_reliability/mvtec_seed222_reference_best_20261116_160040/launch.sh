source /opt/conda/etc/profile.d/conda.sh
conda activate vmamba_new
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=GPU-54d94bb2-5b56-1e25-3550-a3217be361fa
nohup setsid python -u scripts/train_mvtec_fresh_best.py --plan /opt/data/private/VarAD_EAAI/experiments/statistical_reliability/mvtec_seed222_reference_best_20261116_160040/launch_plan.json > /opt/data/private/VarAD_EAAI/experiments/statistical_reliability/mvtec_seed222_reference_best_20261116_160040/nohup.log 2>&1 < /dev/null &
