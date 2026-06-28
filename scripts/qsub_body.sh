#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=0:35:00
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/body/$JOB_NAME.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/body/$JOB_NAME.err.log
source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt
echo "=== body run: $1 start $(date '+%H:%M:%S') ==="
python -m bytelatent.train config=bytelatent/configs/$1.yaml
echo "=== end $(date '+%H:%M:%S') ==="
