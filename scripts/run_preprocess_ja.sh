#!/bin/bash
# ゲート② Step1: 日本語 smoke → entropy Arrow 生成
# 計算ノード (qrsh -g tga-RLA -l gpu_1=1) 上で実行する
# source /gs/bs/tga-RLA/yoshida/activate_blt.sh が必要

set -e
cd /gs/bs/tga-RLA/yoshida/blt

INPUT=/gs/bs/tga-RLA/yoshida/blt_data/smoke_ja/wikipedia_ja_smoke/wikipedia_ja_smoke.chunk.00.jsonl
OUTPUT_DIR=/gs/bs/tga-RLA/yoshida/blt_data/smoke_ja/preprocess/wikipedia_ja_smoke/transformer_100m
OUTPUT=${OUTPUT_DIR}/wikipedia_ja_smoke.chunk.00.jsonl.shard_0.arrow
ENTROPY_CKPT=/gs/bs/tga-RLA/yoshida/blt_runs/entropy_smoke/checkpoints/0000000300/consolidated
TOKENIZER=/gs/bs/tga-RLA/yoshida/blt_data/tokenizer/tokenizer.model

mkdir -p "$OUTPUT_DIR"

echo "=== 日本語 entropy Arrow 生成 ==="
echo "入力: $INPUT"
echo "出力: $OUTPUT"
df -h /gs/bs/tga-RLA  # storage 確認
echo ""

python bytelatent/preprocess/preprocess_entropies.py \
  "$INPUT" "$OUTPUT" \
  --entropy-model-checkpoint-dir "$ENTROPY_CKPT" \
  --entropy-model-state-dict-path "${ENTROPY_CKPT}/consolidated.pth" \
  --bpe-tokenizer-path "$TOKENIZER" \
  --patching-device cuda

echo ""
echo "Arrow サイズ:"
ls -lh "$OUTPUT" "$OUTPUT.complete"
