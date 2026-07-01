#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=4:00:00
#$ -N cleancorpus
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/body/clean.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/body/clean.err.log
source /gs/bs/tga-RLA/yoshida/activate_blt.sh
export OPENBLAS_NUM_THREADS=4
cd /gs/bs/tga-RLA/yoshida/blt
echo "=== clean start $(date '+%H:%M:%S') ==="
python scripts/clean_corpus.py
echo "=== end $(date '+%H:%M:%S') rc=$? ==="
