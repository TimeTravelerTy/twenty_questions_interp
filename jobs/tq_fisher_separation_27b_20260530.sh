#!/bin/sh
#$ -cwd
#$ -l cpu_4=1
#$ -l h_rt=00:20:00
#$ -N tq_fisher_sep
#$ -o ./logs/$JOB_NAME.$JOB_ID.out
#$ -e ./logs/$JOB_NAME.$JOB_ID.err
#$ -V

# Appendix A number: normalized class separation (d-prime) at end_ready/L16 vs
# pre_answer_q1/L38, to show the raw mean-diff norm differs ~100x while the
# scale-free SNR (what the probe reads) is comparable. 6 classes (drop shark).

set -eu
cd /gs/fs/tga-sip_arase/tyrone/twenty_questions_interp
module purge
source /apps/t4/rhel9/free/miniconda/24.1.2/bin/activate ~/.conda/envs/py311
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4

echo "git=$(git rev-parse --short HEAD)"
python scripts/fisher_separation.py \
    --in-dir runs/positional_residuals/27b_default_n80_v2 \
    --cells end_ready:16 pre_answer_q1:38 pre_answer_q4:38 \
    --drop-class shark \
    --n-per-class 20
echo DONE
