#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=0:30:00
#$ -N precompslice
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/body/precompute_slice.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/body/precompute_slice.err.log
source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt
SD=/gs/bs/tga-RLA/yoshida/blt_data/shakedown
FROZEN=/gs/bs/tga-RLA/yoshida/blt_runs/entropy_stage_run/frozen_entropy_285000/consolidated
OUT=$SD/preprocess/corpus/transformer_100m/corpus.chunk.00.jsonl.shard_0.arrow
echo "=== precompute slice start $(date '+%H:%M:%S') ==="
python bytelatent/preprocess/preprocess_entropies.py \
  $SD/corpus/corpus.chunk.00.jsonl $OUT \
  --entropy-model-checkpoint-dir $FROZEN \
  --entropy-model-state-dict-path $FROZEN/consolidated.pth \
  --bpe-tokenizer-path /gs/bs/tga-RLA/yoshida/blt_data/tokenizer/tokenizer.model
touch ${OUT}.complete
echo "=== end $(date '+%H:%M:%S') rc=$? ==="
