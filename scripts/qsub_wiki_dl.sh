#!/bin/bash
# 段階2-A: Wikipedia 8言語 実DL（CPU・GPU不使用）
# qsub -g tga-RLA scripts/qsub_wiki_dl.sh
#$ -cwd
#$ -l cpu_8=1
#$ -l h_rt=6:00:00
#$ -N wiki_dl
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/wiki_dl.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/wiki_dl.err.log

source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1

echo "=== df 開始前 ==="; df -h /gs/bs/tga-RLA | tail -1
python scripts/make_wiki_data.py
echo "=== df 終了後 ==="; df -h /gs/bs/tga-RLA | tail -1
echo "=== 出力 ==="; ls -lh /gs/bs/tga-RLA/yoshida/blt_data/largecorpus/wiki/
echo "DONE"
