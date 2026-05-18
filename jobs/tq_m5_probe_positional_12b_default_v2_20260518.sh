#!/bin/sh
#$ -cwd
#$ -l cpu_40=1
#$ -l h_rt=02:00:00
#$ -N tq_m5_probe_12b_v2
#$ -o ./logs/$JOB_NAME.$JOB_ID.out
#$ -e ./logs/$JOB_NAME.$JOB_ID.err
#$ -V

# M5 scale comparator: re-probe the 12B v2 (16-anchor) positional capture
# to get an apples-to-apples 16-anchor x 49-layer LR LOO heatmap at 12B,
# matched to the 27B v2 probe shipped today. Output is the clean v2
# 12B-vs-27B comparator the M5-positional-probe-27b writeup flagged.
#
# 12B has 49 layers (vs 27B's 63) and hidden=3840 (vs 5376), so per-fit
# compute is ~3x lighter than 27B's 5h cpu_40 probe. 2h walltime headroom.

set -eu

cd /gs/fs/tga-sip_arase/tyrone/twenty_questions_interp
mkdir -p logs runs

module purge
source /apps/t4/rhel9/free/miniconda/24.1.2/bin/activate ~/.conda/envs/py311

export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

IN_DIR="runs/positional_residuals/12b_default_n80_v2"
OUT_PREFIX="runs/m5_positional_probe_12b_default_v2_n80"

echo "git=$(git rev-parse --short HEAD)"
echo "in_dir=$IN_DIR"
echo "out_prefix=$OUT_PREFIX"

python - <<'PY'
import sys
print('python', sys.version.split()[0])
import sklearn, numpy, torch
print('sklearn', sklearn.__version__, 'numpy', numpy.__version__, 'torch', torch.__version__)
PY

python scripts/probe_positional_anchors.py \
    --in-dir "$IN_DIR" \
    --out-prefix "$OUT_PREFIX" \
    --n-per-class 20

echo "DONE $OUT_PREFIX.json"
