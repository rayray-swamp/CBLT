#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=4:00:00
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/calib/calib_packed.$JOB_NAME.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/calib/calib_packed.$JOB_NAME.err.log
source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH=/gs/bs/tga-RLA/yoshida/blt
echo "=== packed θ較正 $1 start $(date '+%F %H:%M:%S') ==="
python scripts/calibrate_theta_packed.py --config "$1" --thetas "$2" --targets "${3:-4,5,6,7,8}" --n-batches "${4:-150}"
echo "=== end $(date '+%F %H:%M:%S') rc=$? ==="
