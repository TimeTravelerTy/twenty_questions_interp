#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=08:00:00
#$ -N tq_m5_patch_27b_pa4
#$ -o ./logs/$JOB_NAME.$JOB_ID.out
#$ -e ./logs/$JOB_NAME.$JOB_ID.err
#$ -V

# M5 Exp-A: 27B causal patch at turn-4 pre-answer, layer band L27-L62.
# Apples-to-apples scale comparator to 12B M4 phase-2a (L27-L48 band,
# 0/2280 reveal flips). Layer band picked from 27B LR > 0.50 cells at
# pre_answer_q4: LR(L27-L62) is uniformly in [0.50, 0.67], the broad
# carry band where the late-network class direction lives.
#
# 5 src x 5 tgt per class x 7x7 cells = 1225 patched trials + 35
# baselines. Same patch_turn4.py script as 12B; only --model and
# --run-dir change. gpu_1 (H100 80GB) needed for 27B in bf16 (~54GB
# weights). Walltime 8h: 27B forward ~2-3x slower than 12B; per-trial
# ~3-5s; 1225 trials ~= 1-2h actual + load time + 35 baselines.

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
OUT_JSON="runs/m5_patch_27b_default_pa4_L27-62.json"
LAYERS="27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62"

echo "git=$(git rev-parse --short HEAD)"
echo "run_dir=$RUN_DIR"
echo "layers=$LAYERS"
echo "out=$OUT_JSON"

python - <<'PY'
import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0))
PY

python scripts/patch_turn4.py \
    --run-dir "$RUN_DIR" \
    --model google/gemma-3-27b-it \
    --device auto \
    --dtype bfloat16 \
    --layers "$LAYERS" \
    --turn 4 \
    --n-source-per-class 5 \
    --n-target-per-class 5 \
    --out-json "$OUT_JSON"

echo "DONE $OUT_JSON"
