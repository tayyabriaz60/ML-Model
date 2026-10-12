#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git python3-pip rsync curl software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update -qq
apt-get install -y -qq python3.10-venv python3.10-dev
PY=python3.10
cd /workspace
if [[ ! -d ML-Model ]]; then
  git clone https://github.com/tayyabriaz60/ML-Model.git
fi
cd ML-Model
git pull origin main
mkdir -p code/data/processed code/outputs/models code/outputs/tables code/outputs/figures deliverables
cd code
rm -rf .venv
${PY:-python3.10} -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel -q
pip install -r requirements.txt -q
python -c "import xgboost, lightgbm; print('xgb', xgboost.__version__, 'lgb', lightgbm.__version__)"
echo "BOOTSTRAP_OK python=$(python --version 2>&1)"
