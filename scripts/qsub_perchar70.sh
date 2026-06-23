#!/bin/bash
# 70文書 per-char CBLT エントロピー → CSV + PNG70枚（GPU・計算ノード）
# qsub -g tga-RLA scripts/qsub_perchar70.sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=0:40:00
#$ -N perchar70
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/perchar70.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/perchar70.err.log

source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt

python scripts/perchar_entropy_70.py

echo "=== tar.gz 作成 ==="
cd /gs/bs/tga-RLA/yoshida/tmp
tar czf perchar70.tar.gz perchar70/
ls -lh /gs/bs/tga-RLA/yoshida/tmp/perchar70.tar.gz
echo "PNG枚数:"; ls /gs/bs/tga-RLA/yoshida/tmp/perchar70/png/ | wc -l
echo "CSV行数:"; wc -l /gs/bs/tga-RLA/yoshida/tmp/perchar70/perchar_entropy_70.csv
echo "DONE"
