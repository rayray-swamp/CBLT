#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=0:50:00
#$ -N calibtheta
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/calib.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/calib.err.log
source /gs/bs/tga-RLA/yoshida/activate_blt.sh
export OPENBLAS_NUM_THREADS=4
cd /gs/bs/tga-RLA/yoshida/blt
CK=/gs/bs/tga-RLA/yoshida/blt_runs/entropy_stage_run/frozen_entropy_285000/consolidated
python scripts/calibrate_theta.py --ckpt-dir "$CK" --out-dir /gs/bs/tga-RLA/yoshida/blt/flores_dumps --n-per-lang 150 --max-chars 4000
echo DONE
