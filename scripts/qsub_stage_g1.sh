#!/bin/bash
# gpu_1(1GPU) batch4 並行hedge run。node_h(stage_run)とは完全分離(別dump_dir/config/ログ)。
# 共有プール(gg_mig)=AR非対象=ほぼ即起動。resumable(ckpt 20000毎/keep-1)。
# qsub -g tga-RLA scripts/qsub_stage_g1.sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=24:00:00
#$ -N stage_g1
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/stage_g1.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/stage_g1.err.log

source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt

echo "=== 開始(gpu_1 batch4) ==="; date; nvidia-smi --query-gpu=index,name --format=csv,noheader
torchrun --nproc_per_node=1 -m bytelatent.train config=bytelatent/configs/entropy_stage_g1.yaml
echo "=== 終了 ==="; date; echo "DONE"
