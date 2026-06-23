#!/bin/bash
# エントロピー継続学習 本走(gpu_1版): 共有プール gpu_1(1GPU) で即起動・連続run
# AR squeeze 中の暫定本走。24h上限のため切れたら同スクリプト再投入→最新ckptから自動再開(resumable)。
# dump_dir は node_h版(entropy_stage_run)と分離してあるので二重書き込み事故なし。
# qsub -g tga-RLA scripts/qsub_stage_run_gpu1.sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=24:00:00
#$ -N stage_gpu1
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/stage_gpu1.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/stage_gpu1.err.log

source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt

echo "=== 開始 ==="; date; nvidia-smi --query-gpu=index,name --format=csv,noheader; df -h /gs/bs/tga-RLA | tail -1
torchrun --nproc_per_node=1 -m bytelatent.train config=bytelatent/configs/entropy_stage_run_gpu1.yaml
echo "=== 終了 ==="; date
echo "DONE"
