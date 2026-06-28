#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=0:50:00
#$ -N intrinsic
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/intrinsic.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/intrinsic.err.log
source /gs/bs/tga-RLA/yoshida/activate_blt.sh
export OPENBLAS_NUM_THREADS=4
cd /gs/bs/tga-RLA/yoshida/blt
CK=/gs/bs/tga-RLA/yoshida/blt_runs/entropy_stage_run/frozen_entropy_285000/consolidated
python scripts/intrinsic_compare.py --ckpt-dir "$CK" --out-dir /gs/bs/tga-RLA/yoshida/blt/flores_dumps
echo DONE
