#!/bin/bash
# 段階2-B: Wiki+CulturaX 文書シャッフル → 8GB3分割（CPU・GPU不使用）
# qsub -g tga-RLA scripts/qsub_make_stages.sh
#$ -cwd
#$ -l cpu_8=1
#$ -l h_rt=3:00:00
#$ -N make_stages
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/make_stages.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/make_stages.err.log

source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt
export OMP_NUM_THREADS=1

echo "=== df 開始前 ==="; df -h /gs/bs/tga-RLA | tail -1
python scripts/make_stages.py
echo "=== df 終了後 ==="; df -h /gs/bs/tga-RLA | tail -1
echo "=== stage 出力 ==="; ls -lh /gs/bs/tga-RLA/yoshida/blt_data/largecorpus/stage*/ 2>/dev/null
echo "DONE"
