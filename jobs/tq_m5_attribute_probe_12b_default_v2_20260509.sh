#!/bin/sh
#$ -cwd
#$ -l cpu_40=1
#$ -l h_rt=04:00:00
#$ -N tq_m5_attrprobe
#$ -o ./logs/$JOB_NAME.$JOB_ID.out
#$ -e ./logs/$JOB_NAME.$JOB_ID.err
#$ -V

# M5 attribute-bundle probe: direct-fit per-attribute binary LR LOO at
# every (anchor x layer) cell on the v2 self-chosen capture. Tests
# whether the residual encodes answer-relevant binary attributes more
# readily (and at earlier anchors) than 4-way class identity.

set -eu
cd /gs/fs/tga-sip_arase/tyrone/twenty_questions_interp
source /apps/t4/rhel9/free/miniconda/24.1.2/bin/activate ~/.conda/envs/py311
# Use all cores; sklearn OMP picks it up.
export OPENBLAS_NUM_THREADS=40
export OMP_NUM_THREADS=40
export MKL_NUM_THREADS=40

python scripts/probe_attribute_anchors.py \
    --in-dir runs/positional_residuals/12b_default_n80_v2 \
    --classes cow,dog,elephant,horse \
    --n-per-class 20 \
    --include-class \
    --out runs/m5_attribute_probe_12b_default_v2_n80.json

echo DONE
