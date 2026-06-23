#!/bin/bash
# ゲート② Step2: CBLT 日本語 smoke 学習 (50 steps)
# 計算ノード (qrsh -g tga-RLA -l gpu_1=1) 上で実行する
# run_preprocess_ja.sh の完了後に実行

set -e
cd /gs/bs/tga-RLA/yoshida/blt

echo "=== CBLT ゲート② 日本語 smoke 学習 ==="
df -h /gs/bs/tga-RLA  # storage 確認
echo ""

python -m bytelatent.train \
  config=bytelatent/configs/blt_cblt_gate2_ja.yaml
