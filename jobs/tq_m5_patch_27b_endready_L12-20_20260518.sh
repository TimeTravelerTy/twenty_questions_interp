#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=08:00:00
#$ -N tq_m5_patch_27b_eready
#$ -o ./logs/$JOB_NAME.$JOB_ID.out
#$ -e ./logs/$JOB_NAME.$JOB_ID.err
#$ -V

# M5 Exp-B: 27B causal patch at end_ready, layer band L12-L20. The new
# 27B-only commitment locus: LR LOO peaks at L16 (0.508, 3.55x chance),
# with the band L12-L20 all > 0.42 (3x chance). 12B has no signal here
# at end_ready (LR ~chance), so this experiment is intrinsically a
# scale-only finding — there is no 12B baseline to repeat.
#
# Tests: is the new early-network commitment load-bearing across 4
# intervening dialogue turns? End_ready is far upstream of the reveal
# in the prefill (~330 tokens between end_ready and reveal generation
# in 27B's tokenization). The patched residual propagates via the KV
# cache; the reveal-token logits at first generation step are read out
# to measure flip rate + logit-diff delta vs unpatched baseline.
#
# Uses the new patch_anchor.py + v2 capture as source residual store.
# 1225 trials + 35 baselines; gpu_1, 8h walltime.

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
OUT_JSON="runs/m5_patch_27b_default_endready_L12-20.json"
LAYERS="12,13,14,15,16,17,18,19,20"

echo "git=$(git rev-parse --short HEAD)"
echo "run_dir=$RUN_DIR"
echo "src_dir=$SRC_DIR"
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
