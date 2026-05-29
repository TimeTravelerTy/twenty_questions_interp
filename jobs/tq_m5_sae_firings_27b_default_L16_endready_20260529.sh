#!/bin/sh
#$ -cwd
#$ -l cpu_4=1
#$ -l h_rt=00:30:00
#$ -N tq_m5_sae_27b_L16_endready
#$ -o ./logs/$JOB_NAME.$JOB_ID.out
#$ -e ./logs/$JOB_NAME.$JOB_ID.err
#$ -V

# Exp 2 (M5 legibility) — encode 27B end_ready (probe peak) residuals through
# the Gemma Scope 2 27B SAE at resid_post layer_16 (25% depth flagship), the
# layer whose linear class probe newly lights up at 27B (end_ready LR 0.508,
# 3.55x chance; 12B was at chance). Tests whether the newly-legible class
# direction sparsifies into SAE features.
#
# Layer indexing: probe peak "L16" = capture-index 16 = resid_post[15]. The
# flagship SAE layer_16 is trained on resid_post[16] = capture-index 17, one
# block past the peak but inside the L12-L18 decodable plateau (~0.44-0.51).
# Width/L0 = 65k/medium to match the 12B SAE (gemma-scope-2-12b-it layer_31
# width_65k_l0_medium) for an apples-to-apples sparsification comparison.
#
# Also encodes the four pre_answer_qN anchors in the same pass for a
# turn-progressive sidebar (their own probe peaks are at other layers, so
# treat as context, not the headline). CPU job: the SAE encode is light.

set -eu
cd /gs/fs/tga-sip_arase/tyrone/twenty_questions_interp
mkdir -p logs runs
module purge
source /apps/t4/rhel9/free/miniconda/24.1.2/bin/activate ~/.conda/envs/py311
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export HF_HOME=/gs/bs/tga-sip_arase/tyrone/hf_cache

echo "git=$(git rev-parse --short HEAD)"
python scripts/sae_feature_firings.py \
    --residuals-dir runs/positional_residuals/27b_default_n80_v2 \
    --hf-repo google/gemma-scope-2-27b-it \
    --hf-subfolder resid_post/layer_16_width_65k_l0_medium \
    --capture-index 17 \
    --block-id 16 \
    --anchors end_ready pre_answer_q1 pre_answer_q2 pre_answer_q3 pre_answer_q4 \
    --top-k 64 \
    --out runs/m5_sae_firings_27b_default_resid_post_L16_endready_pa1234.pt \
    --device cpu \
    --dtype float32

echo DONE
