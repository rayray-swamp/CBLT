#!/bin/bash
# エントロピー継続学習 本走: node_h(2GPU) 24h run（corpus 36.7GB, ckpt 1000毎）
# 24h上限のため頭打ち前に切れたら同スクリプトを再投入→最新ckptから自動再開(resumable)。
# AUC推移を見て best を後から凍結。
# qsub -g tga-RLA scripts/qsub_stage_run.sh
#$ -cwd
#$ -l node_h=1
#$ -l h_rt=24:00:00
#$ -N stage_run
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/stage_run.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/stage_run.err.log

source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt

echo "=== 開始 ==="; date; nvidia-smi --query-gpu=index,name --format=csv,noheader; df -h /gs/bs/tga-RLA | tail -1
torchrun --nproc_per_node=2 -m bytelatent.train config=bytelatent/configs/entropy_stage_run.yaml
echo "=== 終了 ==="; date
echo "DONE"
