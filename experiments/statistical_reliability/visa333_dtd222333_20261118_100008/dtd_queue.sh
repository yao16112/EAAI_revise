#!/usr/bin/env bash
set -e
cd /opt/data/private/VarAD_EAAI
trap 'queue_code=$?; printf '"'"'{"exit_code":%s}\n'"'"' "$queue_code" > /opt/data/private/VarAD_EAAI/experiments/statistical_reliability/visa333_dtd222333_20261118_100008/dtd_queue_exit.json' EXIT
printf "START DTD Seed222\n"
python -u /opt/data/private/VarAD_EAAI/experiments/statistical_reliability/visa333_dtd222333_20261118_100008/runner.py --plan /opt/data/private/VarAD_EAAI/experiments/statistical_reliability/visa333_dtd222333_20261118_100008/dtd_seed222/launch_plan.json > /opt/data/private/VarAD_EAAI/experiments/statistical_reliability/visa333_dtd222333_20261118_100008/dtd_seed222/nohup.log 2>&1 < /dev/null
printf "COMPLETE DTD Seed222\n"
printf "START DTD Seed333\n"
python -u /opt/data/private/VarAD_EAAI/experiments/statistical_reliability/visa333_dtd222333_20261118_100008/runner.py --plan /opt/data/private/VarAD_EAAI/experiments/statistical_reliability/visa333_dtd222333_20261118_100008/dtd_seed333/launch_plan.json > /opt/data/private/VarAD_EAAI/experiments/statistical_reliability/visa333_dtd222333_20261118_100008/dtd_seed333/nohup.log 2>&1 < /dev/null
printf "COMPLETE DTD Seed333\n"
