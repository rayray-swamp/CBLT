#!/bin/bash
#$ -cwd
#$ -l node_h=1
#$ -l h_rt=24:00:00
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/body/$JOB_NAME.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/body/$JOB_NAME.err.log
source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt
echo "=== node_h(2GPU) run: $1 start $(date '+%H:%M:%S') NVIDIA_VISIBLE=$NVIDIA_VISIBLE_DEVICES ==="
torchrun --nproc_per_node=2 -m bytelatent.train config=bytelatent/configs/$1.yaml
echo "=== end $(date '+%H:%M:%S') rc=$? ==="
