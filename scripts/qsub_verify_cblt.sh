#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=0:10:00
#$ -N verify_cblt
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/verify_cblt.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/verify_cblt.err.log
source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt
python scripts/verify_cblt_align.py
echo DONE
