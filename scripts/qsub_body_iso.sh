#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=0:15:00
#$ -N bodyiso
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/body/iso.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/body/iso.err.log
source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt
echo "=== iso start $(date '+%H:%M:%S') ==="
python -m bytelatent.train config=bytelatent/configs/blt_body_iso.yaml
echo "=== end $(date '+%H:%M:%S') rc=$? ==="
