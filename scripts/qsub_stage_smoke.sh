#!/bin/bash
# node_f マルチGPU smoke（4GPU・300step）: FSDP/NCCL疎通 + マルチGPU学習ループ確認
# qsub -g tga-RLA scripts/qsub_stage_smoke.sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=0:30:00
#$ -N stage_smoke
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/stage_smoke.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/stage_smoke.err.log

source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt

echo "=== GPU確認 ==="; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "=== 4GPU torchrun smoke ==="
torchrun --nproc_per_node=4 -m bytelatent.train config=bytelatent/configs/entropy_stage_smoke.yaml

echo "DONE"
