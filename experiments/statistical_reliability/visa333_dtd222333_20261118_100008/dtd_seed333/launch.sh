#!/usr/bin/env bash
set -e
cd /opt/data/private/VarAD_EAAI
source /opt/conda/etc/profile.d/conda.sh
conda activate vmamba_new
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=GPU-54d94bb2-5b56-1e25-3550-a3217be361fa
nohup setsid python -u /opt/data/private/VarAD_EAAI/experiments/statistical_reliability/visa333_dtd222333_20261118_100008/runner.py --plan /opt/data/private/VarAD_EAAI/experiments/statistical_reliability/visa333_dtd222333_20261118_100008/dtd_seed333/launch_plan.json > /opt/data/private/VarAD_EAAI/experiments/statistical_reliability/visa333_dtd222333_20261118_100008/dtd_seed333/nohup.log 2>&1 < /dev/null &
training_pid=$!
printf "%s\n" "$training_pid" > /opt/data/private/VarAD_EAAI/experiments/statistical_reliability/visa333_dtd222333_20261118_100008/dtd_seed333/launcher.pid
printf "DTD333_PID=%s\n" "$training_pid"
