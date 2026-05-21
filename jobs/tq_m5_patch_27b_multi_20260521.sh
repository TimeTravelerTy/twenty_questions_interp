#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=08:00:00
#$ -N tq_m5_patch_27b_multi
#$ -o ./logs/$JOB_NAME.$JOB_ID.out
#$ -e ./logs/$JOB_NAME.$JOB_ID.err
#$ -V

# M5 multi-anchor patch at 27B. Patches end_ready AND every end_model_qN
# (q1..q4) simultaneously in the same forward pass, same source run,
# same layer band L12-L48 at each anchor.
#
# Rationale: the single-anchor band patches (end_ready/L12-20,
# pa4/L27-62) are both null after excluding the boundary-degenerate
# target attempt_593. But a single blocked site leaves the model free
# to re-derive class info downstream. This experiment blocks every
# commitment/re-derivation site at once: end_ready is the initial
# commitment locus; end_model_qN is the post-answer state right after
# each of the 4 dialogue turns. L12-L48 spans the end_ready commitment
# band (L12-20, LR peak 0.51 @ L16) and the decodable mid-late class
# carry band (the STATUS L20-45 probe summary zone).
#
# Predicted null. If it fires, the load-bearing locus is the AGGREGATE
# of re-derivation sites, not any single anchor.
#
# 5 src x 5 tgt per class x 7x7 cells = 1225 patched trials + 35
# baselines. gpu_1 (H100 80GB) for 27B bf16. ~15-20 min compute after
# model load.

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
OUT_JSON="runs/m5_patch_27b_default_multi_endready_endmodelq1-4_L12-48.json"
ANCHORS="end_ready,end_model_q1,end_model_q2,end_model_q3,end_model_q4"
LAYERS="12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48"

echo "git=$(git rev-parse --short HEAD)"
echo "run_dir=$RUN_DIR"
echo "src_dir=$SRC_DIR"
echo "anchors=$ANCHORS"
echo "layers=$LAYERS"
echo "out=$OUT_JSON"

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
    --n-source-per-class 5 \
    --n-target-per-class 5 \
    --out-json "$OUT_JSON"

echo "DONE $OUT_JSON"
