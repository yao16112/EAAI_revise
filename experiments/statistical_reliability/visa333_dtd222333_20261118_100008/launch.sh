#!/usr/bin/env bash
set -e
cd /opt/data/private/VarAD_EAAI
source /opt/conda/etc/profile.d/conda.sh
conda activate vmamba_new
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES=GPU-87c9ac48-ee96-e2f0-5553-cf2bd2756984 nohup setsid python -u /opt/data/private/VarAD_EAAI/experiments/statistical_reliability/visa333_dtd222333_20261118_100008/runner.py --plan /opt/data/private/VarAD_EAAI/experiments/statistical_reliability/visa333_dtd222333_20261118_100008/visa_seed333/launch_plan.json > /opt/data/private/VarAD_EAAI/experiments/statistical_reliability/visa333_dtd222333_20261118_100008/visa_seed333/nohup.log 2>&1 < /dev/null &
visa_pid=$!
printf "%s\n" "$visa_pid" > /opt/data/private/VarAD_EAAI/experiments/statistical_reliability/visa333_dtd222333_20261118_100008/visa_seed333/launcher.pid
CUDA_VISIBLE_DEVICES=GPU-ea7e1b69-d452-bab6-4924-67e9e8f6f09a nohup setsid bash /opt/data/private/VarAD_EAAI/experiments/statistical_reliability/visa333_dtd222333_20261118_100008/dtd_queue.sh > /opt/data/private/VarAD_EAAI/experiments/statistical_reliability/visa333_dtd222333_20261118_100008/dtd_queue.log 2>&1 < /dev/null &
queue_pid=$!
printf "%s\n" "$queue_pid" > /opt/data/private/VarAD_EAAI/experiments/statistical_reliability/visa333_dtd222333_20261118_100008/dtd_queue.pid
printf "VISA_PID=%s DTD_QUEUE_PID=%s\n" "$visa_pid" "$queue_pid"
