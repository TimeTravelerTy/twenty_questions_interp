#!/bin/sh
#$ -cwd
#$ -l cpu_4=1
#$ -l h_rt=00:30:00
#$ -N tq_sae_27b_L40_qanchors
#$ -o ./logs/$JOB_NAME.$JOB_ID.out
#$ -e ./logs/$JOB_NAME.$JOB_ID.err
#$ -V

# SAE features at the QUESTION anchors. The Q1+ probe/steering signal peaks
# around L38; nearest Gemma Scope 2 flagship is layer_40 (resid_post[40] =
# capture-index 41). Encode end_ready + pre_answer_q1..q4 all at L40 so the
# anchor effect is isolated from the layer effect (same-layer comparison).
#
# Hypothesis: at "Ready" only formatting features fire; once a question lands,
# class/attribute features should start firing -> the question is what
# activates the animal space. Width/L0 = 65k/medium to match the L16 run.

set -eu
cd /gs/fs/tga-sip_arase/tyrone/twenty_questions_interp
mkdir -p logs runs
module purge
source /apps/t4/rhel9/free/miniconda/24.1.2/bin/activate ~/.conda/envs/py311
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export HF_HOME=/gs/bs/tga-sip_arase/tyrone/hf_cache

echo "git=$(git rev-parse --short HEAD)"
python scripts/sae_feature_firings.py \
    --residuals-dir runs/positional_residuals/27b_default_n80_v2 \
    --hf-repo google/gemma-scope-2-27b-it \
    --hf-subfolder resid_post/layer_40_width_65k_l0_medium \
    --capture-index 41 \
    --block-id 40 \
    --anchors end_ready pre_answer_q1 pre_answer_q2 pre_answer_q3 pre_answer_q4 \
    --top-k 64 \
    --out runs/m5_sae_firings_27b_default_resid_post_L40_qanchors.pt \
    --device cpu \
    --dtype float32

echo DONE
