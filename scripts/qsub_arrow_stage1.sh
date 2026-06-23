#!/bin/bash
#$ -cwd
#$ -l cpu_8=1
#$ -l h_rt=1:00:00
#$ -N arrow_stage1
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/arrow_stage1.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/arrow_stage1.err.log
source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt
export OMP_NUM_THREADS=1
python scripts/make_placeholder_arrow.py \
  /gs/bs/tga-RLA/yoshida/blt_data/largecorpus/stage1/stage1.chunk.00.jsonl \
  /gs/bs/tga-RLA/yoshida/blt_data/largecorpus/preprocess
echo "=== 出力 ==="
ls -lh /gs/bs/tga-RLA/yoshida/blt_data/largecorpus/preprocess/stage1/transformer_100m/
echo "DONE"
