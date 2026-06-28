#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=0:20:00
#$ -N bodydbg
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/body/debug.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/body/debug.err.log
source /gs/bs/tga-RLA/yoshida/activate_blt.sh
export CUDA_LAUNCH_BLOCKING=1
cd /gs/bs/tga-RLA/yoshida/blt
echo "=== debug start $(date '+%H:%M:%S') ==="
python -m bytelatent.train config=bytelatent/configs/blt_body_debug.yaml
echo "=== end $(date '+%H:%M:%S') rc=$? ==="
