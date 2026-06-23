#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=0:30:00
#$ -N ofat_N
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/ofat/N.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/ofat/N.err.log
source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt
torchrun --standalone --nproc_per_node=1 -m bytelatent.train config=bytelatent/configs/ofat/N.yaml
python scripts/ofat/summarize.py /gs/bs/tga-RLA/yoshida/blt_runs/ofat/N N
echo DONE
