#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=0:10:00
#$ -N test_cblt
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/test_cblt.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/test_cblt.err.log
source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt
python scripts/test_cblt_align.py && echo "EXIT=PASS" || echo "EXIT=FAIL"
echo DONE
