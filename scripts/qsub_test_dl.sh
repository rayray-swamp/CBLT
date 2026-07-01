#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=1:00:00
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/calib/t3.$JOB_NAME.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/calib/t3.$JOB_NAME.err.log
source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt
export OMP_NUM_THREADS=4 PYTHONPATH=/gs/bs/tga-RLA/yoshida/blt
python scripts/test_dataloader_byte_budget.py --config "$1" --n-batches 100
echo "=== rc=$? ==="
