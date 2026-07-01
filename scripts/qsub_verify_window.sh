#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=1:00:00
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/calib/verify_window.$JOB_NAME.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/calib/verify_window.$JOB_NAME.err.log
source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt
export OMP_NUM_THREADS=4 PYTHONPATH=/gs/bs/tga-RLA/yoshida/blt
python scripts/verify_window_truncation.py --config "$1" --n-batches "${2:-100}"
echo "=== rc=$? ==="
