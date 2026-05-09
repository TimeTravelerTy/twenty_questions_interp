#!/bin/sh
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=01:00:00
#$ -N tq_m5_sae_L41_pa1234
#$ -o ./logs/$JOB_NAME.$JOB_ID.out
#$ -e ./logs/$JOB_NAME.$JOB_ID.err
#$ -V

# M5 Phase A second cut: L41 resid_post (Gemma Scope 2's deepest
# fixed-depth layer at 85% depth, in the M4-identified L40-48 monotonic-
# growth band) across all four pre_answer_qN anchors. Decision gate:
# if balanced LR LOO at any anchor reaches >= 0.7 of M3 residual probe,
# misalignment is L31-specific; else SAE-retrieval is dead at default
# Gemma Scope 2 12b-it layers.

set -eu
cd /gs/fs/tga-sip_arase/tyrone/twenty_questions_interp
mkdir -p logs runs
module purge
module load cuda
source /apps/t4/rhel9/free/miniconda/24.1.2/bin/activate ~/.conda/envs/py311
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
mkdir -p /gs/bs/tga-sip_arase/tyrone/.cache/huggingface
export HF_HOME=/gs/bs/tga-sip_arase/tyrone/.cache/huggingface

echo "git=$(git rev-parse --short HEAD)"
python scripts/sae_feature_firings.py \
    --residuals-dir runs/positional_residuals/12b_default_n80_v2 \
    --hf-repo google/gemma-scope-2-12b-it \
    --hf-subfolder resid_post/layer_41_width_65k_l0_medium \
    --capture-index 42 \
    --block-id 41 \
    --anchors pre_answer_q1 pre_answer_q2 pre_answer_q3 pre_answer_q4 \
    --top-k 64 \
    --out runs/m5_sae_firings_12b_default_resid_post_L41_pre_answer_q1q2q3q4.pt \
    --device auto \
    --dtype bfloat16

echo DONE
