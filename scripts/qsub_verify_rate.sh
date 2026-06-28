#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=0:30:00
#$ -N verifyrate
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/body/verify_rate.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/body/verify_rate.err.log
source /gs/bs/tga-RLA/yoshida/activate_blt.sh
export OPENBLAS_NUM_THREADS=4
cd /gs/bs/tga-RLA/yoshida/blt
CK=/gs/bs/tga-RLA/yoshida/blt_runs/entropy_stage_run/frozen_entropy_285000/consolidated
python scripts/verify_cblt_rate.py --ckpt-dir "$CK" --n-per-lang 80
echo DONE
