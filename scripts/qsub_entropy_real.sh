#!/bin/bash
# 本番エントロピーモデル学習 — qsub バッチ投入用
# 使い方: qsub scripts/qsub_entropy_real.sh
# 投入したら SSH を切ってよい（計算ノードを占有し続けない）
#
# h_rt はプローブ実測 ~2.54h + マージンで 4h。
# checkpoint で resumable。万一 walltime で切れても再開可能。

#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=4:00:00
#$ -N entropy_real
# 注意: グループ -g tga-RLA は qsub コマンドラインで渡す（#$ ディレクティブ不可）
#   qsub -g tga-RLA scripts/qsub_entropy_real.sh
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/entropy_real/qsub.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/entropy_real/qsub.err.log

source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt

mkdir -p /gs/bs/tga-RLA/yoshida/blt_runs/entropy_real

echo "=== 本番エントロピーモデル学習 開始 ==="
date
df -h /gs/bs/tga-RLA
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

torchrun --nproc_per_node=1 -m bytelatent.train \
  config=bytelatent/configs/entropy_real.yaml

echo ""
echo "=== 学習完了 ==="
date
