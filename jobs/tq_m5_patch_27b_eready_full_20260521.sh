#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=08:00:00
#$ -N tq_m5_patch_27b_erfull
#$ -o ./logs/$JOB_NAME.$JOB_ID.out
#$ -e ./logs/$JOB_NAME.$JOB_ID.err
#$ -V

# M5 full-residual patch at 27B, anchor end_ready, every layer L1-L62.
# Strongest-possible single-anchor intervention at the 27B-only
# commitment locus (end_ready: LR LOO 0.508 @ L16, 3.55x chance; 12B
# is at chance here). The L12-20 band patch was null; this replaces
# the entire residual stream at end_ready to test whether the new
# early commitment legibility is load-bearing under the maximal
# intervention. Predicted null.
#
# 1225 patched trials + 35 baselines. gpu_1 (H100 80GB) for 27B bf16.

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
OUT_JSON="runs/m5_patch_27b_default_endready_L1-62full.json"
LAYERS=$(python3 -c "print(','.join(str(i) for i in range(1,63)))")

echo "git=$(git rev-parse --short HEAD)"
echo "run_dir=$RUN_DIR"
echo "src_dir=$SRC_DIR"
echo "anchor=end_ready"
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
    --anchor end_ready \
    --model google/gemma-3-27b-it \
    --device auto \
    --dtype bfloat16 \
    --layers "$LAYERS" \
    --n-source-per-class 5 \
    --n-target-per-class 5 \
    --out-json "$OUT_JSON"

echo "DONE $OUT_JSON"
