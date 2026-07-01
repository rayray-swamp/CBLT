#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=12:00:00
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/calib/calib_dw.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/calib/calib_dw.err.log
source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTHONPATH=/gs/bs/tga-RLA/yoshida/blt
echo "=== data-weighted θ 再較正 start $(date '+%F %H:%M:%S') stride=${1:-20} ==="
python scripts/calibrate_theta_dataweighted.py \
  --out-csv /gs/bs/tga-RLA/yoshida/blt/flores_dumps/theta_calibration_dataweighted.csv \
  --shards all --stride ${1:-20}
echo "=== end $(date '+%F %H:%M:%S') rc=$? ==="
