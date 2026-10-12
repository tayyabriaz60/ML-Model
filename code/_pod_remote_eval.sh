#!/usr/bin/env bash
set -euo pipefail
cd /workspace/ML-Model/code
source .venv/bin/activate
export REGEN_TRAIN_CAP=0
bash run_pod_eval.sh
ls -lh ../deliverables/T5_milestone_compute.zip
