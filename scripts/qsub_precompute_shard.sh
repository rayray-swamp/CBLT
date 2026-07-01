#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=12:00:00
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/body/pre_shard$1.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/body/pre_shard$1.err.log
source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt
SD=/gs/bs/tga-RLA/yoshida/blt_data/largecorpus_clean
FROZEN=/gs/bs/tga-RLA/yoshida/blt_runs/entropy_stage_run/frozen_entropy_285000/consolidated
OUT=$SD/preprocess/corpus/transformer_100m/corpus.chunk.00.jsonl.shard_$1.arrow
echo "=== precompute shard $1 start $(date '+%H:%M:%S') ==="
python bytelatent/preprocess/preprocess_entropies.py \
  $SD/shards/part_0$1.jsonl $OUT \
  --entropy-model-checkpoint-dir $FROZEN \
  --entropy-model-state-dict-path $FROZEN/consolidated.pth \
  --bpe-tokenizer-path /gs/bs/tga-RLA/yoshida/blt_data/tokenizer/tokenizer.model
touch ${OUT}.complete
echo "=== shard $1 end $(date '+%H:%M:%S') rc=$? ==="
