#!/bin/sh
#$ -cwd
#$ -l cpu_4=1
#$ -l h_rt=00:30:00
#$ -N tq_m5_anlz_L41
#$ -o ./logs/$JOB_NAME.$JOB_ID.out
#$ -e ./logs/$JOB_NAME.$JOB_ID.err
#$ -V

set -eu
cd /gs/fs/tga-sip_arase/tyrone/twenty_questions_interp
source /apps/t4/rhel9/free/miniconda/24.1.2/bin/activate ~/.conda/envs/py311
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

FIRINGS=runs/m5_sae_firings_12b_default_resid_post_L41_pre_answer_q1q2q3q4.pt

for ANCHOR in pre_answer_q1 pre_answer_q2 pre_answer_q3 pre_answer_q4; do
  python scripts/analyze_sae_features.py \
      --firings $FIRINGS \
      --anchor $ANCHOR \
      --balance 20 --seed 0 \
      --top-n 30 --lr-c 1.0 \
      --out runs/m5_sae_analysis_12b_default_resid_post_L41_${ANCHOR}_balanced.json
done

echo DONE
