#!/bin/bash
# D1: entropy_d1 の全 checkpoint を held-out / train-sample で eval し held-out U字を観測
# 計算ノード（qrsh gpu_1）で entropy_d1 学習完走後に実行する。
# 各 checkpoint を consolidate → eval_entropy_bpb.py で bpb 測定。

set -e
cd /gs/bs/tga-RLA/yoshida/blt
RUN=/gs/bs/tga-RLA/yoshida/blt_runs/entropy_d1
HELD=/gs/bs/tga-RLA/yoshida/blt_data/eval/heldout.jsonl
TRAIN=/gs/bs/tga-RLA/yoshida/blt_data/eval/train_sample.jsonl

echo "step,held_out_bpb,train_bpb"
for ckpt in $(ls -d $RUN/checkpoints/*/ | sort); do
  step=$(basename "$ckpt")
  # consolidate（未実施なら）
  if [ ! -f "$ckpt/consolidated/consolidated.pth" ]; then
    python -m bytelatent.checkpoint consolidate "$ckpt" > /dev/null 2>&1
  fi
  cdir="$ckpt/consolidated"
  ho=$(python scripts/eval_entropy_bpb.py --ckpt-dir "$cdir" --jsonl "$HELD" 2>/dev/null | grep -oP "bpb = \K[0-9.]+")
  tr=$(python scripts/eval_entropy_bpb.py --ckpt-dir "$cdir" --jsonl "$TRAIN" 2>/dev/null | grep -oP "bpb = \K[0-9.]+")
  echo "$step,$ho,$tr"
done
