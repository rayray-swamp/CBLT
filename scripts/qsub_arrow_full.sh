#!/bin/bash
# fullcorpus 3チャンク（stage1/2/3）を placeholder Arrow 化（CPU）
# chunk.00(stage1) を最初に作る → smoke が早く着手できる
# qsub -g tga-RLA scripts/qsub_arrow_full.sh
#$ -cwd
#$ -l cpu_8=1
#$ -l h_rt=2:00:00
#$ -N arrow_full
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/arrow_full.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/arrow_full.err.log

source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt
export OMP_NUM_THREADS=1
LC=/gs/bs/tga-RLA/yoshida/blt_data/largecorpus
PRE=$LC/preprocess

for c in 00 01 02; do
  echo "=== arrow chunk.$c ==="
  python scripts/make_placeholder_arrow.py "$LC/fullcorpus/fullcorpus.chunk.$c.jsonl" "$PRE"
  ls -lh "$PRE/fullcorpus/transformer_100m/" 2>/dev/null | tail -3
done
echo "=== df ==="; df -h /gs/bs/tga-RLA | tail -1
echo "DONE"
