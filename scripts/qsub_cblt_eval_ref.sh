#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=0:10:00
#$ -N cblteval
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/cblt_eval_ref.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/cblt_eval_ref.err.log
source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt
python scripts/cblt_eval_reference.py
echo DONE
