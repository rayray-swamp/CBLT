#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=0:10:00
#$ -N vcblt
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/verify_conv.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/verify_conv.err.log
source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt
python scripts/verify_cblt_convention.py
echo DONE
