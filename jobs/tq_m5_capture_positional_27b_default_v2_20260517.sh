#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=01:00:00
#$ -N tq_m5_pos_27b_v2
#$ -o ./logs/$JOB_NAME.$JOB_ID.out
#$ -e ./logs/$JOB_NAME.$JOB_ID.err
#$ -V

# M5 scale comparison: re-capture 27B default positional residuals with the
# upgraded 16-anchor v2 schema (adds pre_answer_qN for N in 1..4). Re-uses
# the existing 600-run self-chosen collection at
# runs/diag/selfchosen_ready_20bank_27b_default_20260427.
#
# Storage: 16 anchors x 63 layers x 5120-d float32 ~= 10 MB/run x 600 ~= 6 GB.
# gpu_1 (full H100, 80GB) — 27B in bf16 is ~54GB so doesn't fit in gpu_h.

set -eu

cd /gs/fs/tga-sip_arase/tyrone/twenty_questions_interp
mkdir -p logs jobs runs runs/positional_residuals

module purge
module load cuda
source /apps/t4/rhel9/free/miniconda/24.1.2/bin/activate ~/.conda/envs/py311

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export HF_HOME=/gs/bs/tga-sip_arase/tyrone/hf_cache

RUN_DIR="runs/diag/selfchosen_ready_20bank_27b_default_20260427"
OUT_DIR="runs/positional_residuals/27b_default_n80_v2"

echo "git=$(git rev-parse --short HEAD)"
echo "run_dir=$RUN_DIR"
echo "out_dir=$OUT_DIR"

python scripts/capture_positional_residuals.py \
    --run-dir "$RUN_DIR" \
    --out-dir "$OUT_DIR" \
    --model google/gemma-3-27b-it \
    --device auto \
    --dtype bfloat16 \
    --prompt-variant default

echo "DONE $OUT_DIR"
