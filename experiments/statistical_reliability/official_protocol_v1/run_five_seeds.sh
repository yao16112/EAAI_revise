#!/usr/bin/env bash
set -euo pipefail
cd /opt/data/private/VarAD_EAAI
source /opt/conda/etc/profile.d/conda.sh
conda activate vmamba_new
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
# Draft commands only. The launcher refuses the unresolved paper protocol.
python scripts/run_statistical_seed.py --protocol experiments/statistical_reliability/official_protocol_v1/protocol.json --dataset mvtec --seed 111 --output experiments/statistical_reliability/formal_mvtec_seed111_attempt01
python scripts/run_statistical_seed.py --protocol experiments/statistical_reliability/official_protocol_v1/protocol.json --dataset mvtec --seed 222 --output experiments/statistical_reliability/formal_mvtec_seed222_attempt01
python scripts/run_statistical_seed.py --protocol experiments/statistical_reliability/official_protocol_v1/protocol.json --dataset mvtec --seed 333 --output experiments/statistical_reliability/formal_mvtec_seed333_attempt01
python scripts/run_statistical_seed.py --protocol experiments/statistical_reliability/official_protocol_v1/protocol.json --dataset mvtec --seed 444 --output experiments/statistical_reliability/formal_mvtec_seed444_attempt01
python scripts/run_statistical_seed.py --protocol experiments/statistical_reliability/official_protocol_v1/protocol.json --dataset mvtec --seed 555 --output experiments/statistical_reliability/formal_mvtec_seed555_attempt01
python scripts/run_statistical_seed.py --protocol experiments/statistical_reliability/official_protocol_v1/protocol.json --dataset visa --seed 111 --output experiments/statistical_reliability/formal_visa_seed111_attempt01
python scripts/run_statistical_seed.py --protocol experiments/statistical_reliability/official_protocol_v1/protocol.json --dataset visa --seed 222 --output experiments/statistical_reliability/formal_visa_seed222_attempt01
python scripts/run_statistical_seed.py --protocol experiments/statistical_reliability/official_protocol_v1/protocol.json --dataset visa --seed 333 --output experiments/statistical_reliability/formal_visa_seed333_attempt01
python scripts/run_statistical_seed.py --protocol experiments/statistical_reliability/official_protocol_v1/protocol.json --dataset visa --seed 444 --output experiments/statistical_reliability/formal_visa_seed444_attempt01
python scripts/run_statistical_seed.py --protocol experiments/statistical_reliability/official_protocol_v1/protocol.json --dataset visa --seed 555 --output experiments/statistical_reliability/formal_visa_seed555_attempt01
