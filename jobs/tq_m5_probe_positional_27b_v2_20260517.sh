#!/bin/sh
#$ -cwd
#$ -l cpu_40=1
#$ -l h_rt=06:00:00
#$ -N tq_m5_probe_27b_v2
#$ -o ./logs/$JOB_NAME.$JOB_ID.out
#$ -e ./logs/$JOB_NAME.$JOB_ID.err
#$ -V

# M5 scale comparison: probe-anchors LR LOO + centroids on the 27B v2
# (16-anchor) positional capture. Headline observable per STATUS:
# end_ready LR LOO at 27B — if >0.40, scale grants explicit
# pre-commitment (turn-4 + end_ready), and the M4 patch sweep gets
# repeated at 27B; if at chance (<0.30), improvisation is scale-robust.
#
# Compute: 16 anchors × 63 layers = 1008 (anchor, layer) cells; each does
# 7-class LR LOO over ~122 runs at hidden=5376. cpu_40 with 6h walltime.
# Previous 12-anchor probe job at cpu_8/3.5h appears to have run out of
# time silently (no JSON written) — this run upgrades both axes.

set -eu

cd /gs/fs/tga-sip_arase/tyrone/twenty_questions_interp
mkdir -p logs runs

module purge
source /apps/t4/rhel9/free/miniconda/24.1.2/bin/activate ~/.conda/envs/py311

export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

IN_DIR="runs/positional_residuals/27b_default_n80_v2"
OUT_PREFIX="runs/m5_positional_probe_27b_default_v2_n120"

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
