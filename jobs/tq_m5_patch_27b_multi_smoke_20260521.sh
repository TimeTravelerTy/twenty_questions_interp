#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=01:00:00
#$ -N tq_m5_patch_27b_multismoke
#$ -o ./logs/$JOB_NAME.$JOB_ID.out
#$ -e ./logs/$JOB_NAME.$JOB_ID.err
#$ -V

# Smoke test for the multi-anchor extension of patch_anchor.py.
# 1 src x 1 tgt per class, 5 anchors, narrow layer band L12-14.
# Validates: multi-anchor arg parsing, per-anchor source-residual
# loading, multi-position hook registration, position indexing.
# ~5 min compute after 27B model load.

set -eu

cd /gs/fs/tga-sip_arase/tyrone/twenty_questions_interp
mkdir -p logs jobs runs

module purge
module load cuda
source /apps/t4/rhel9/free/miniconda/24.1.2/bin/activate ~/.conda/envs/py311

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export HF_HOME=/gs/bs/tga-sip_arase/tyrone/hf_cache

RUN_DIR="runs/diag/selfchosen_ready_20bank_27b_default_20260427"
SRC_DIR="runs/positional_residuals/27b_default_n80_v2"
OUT_JSON="runs/m5_patch_27b_default_multi_smoke.json"
ANCHORS="end_ready,end_model_q1,end_model_q2,end_model_q3,end_model_q4"
LAYERS="12,13,14"

echo "git=$(git rev-parse --short HEAD)"
echo "anchors=$ANCHORS"
echo "layers=$LAYERS"

python - <<'PY'
import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0))
PY

python scripts/patch_anchor.py \
    --run-dir "$RUN_DIR" \
    --src-residuals-dir "$SRC_DIR" \
    --anchor "$ANCHORS" \
    --model google/gemma-3-27b-it \
    --device auto \
    --dtype bfloat16 \
    --layers "$LAYERS" \
    --n-source-per-class 1 \
    --n-target-per-class 1 \
    --out-json "$OUT_JSON"

echo "DONE $OUT_JSON"
