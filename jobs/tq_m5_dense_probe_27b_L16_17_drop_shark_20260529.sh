#!/bin/sh
#$ -cwd
#$ -l cpu_4=1
#$ -l h_rt=00:30:00
#$ -N tq_m5_dense_L16_17_ds
#$ -o ./logs/$JOB_NAME.$JOB_ID.out
#$ -e ./logs/$JOB_NAME.$JOB_ID.err
#$ -V

# Exp 2 prong-A baseline: dense residual LR LOO at end_ready for capture-index
# 16 (probe peak "L16") and 17 (the SAE layer_16 location, resid_post[16]),
# balanced 20/class over the 6 real classes (drop shark, n=2). This is the
# apples-to-apples dense counterpart to the balanced sparse SAE LR LOO (0.300,
# chance 0.167) so we can state dense-vs-sparse at matched class set/chance.

set -eu
cd /gs/fs/tga-sip_arase/tyrone/twenty_questions_interp
mkdir -p logs runs
module purge
source /apps/t4/rhel9/free/miniconda/24.1.2/bin/activate ~/.conda/envs/py311
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

echo "git=$(git rev-parse --short HEAD)"
python scripts/probe_positional_anchors.py \
    --in-dir runs/positional_residuals/27b_default_n80_v2 \
    --out-prefix runs/m5_dense_probe_27b_end_ready_L16_17_drop_shark \
    --layers 16,17 \
    --n-per-class 20 \
    --drop-class shark

echo DONE
