#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=0:30:00
#$ -N ofat_O6b
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/ofat/O6b.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/ofat/O6b.err.log
source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt
torchrun --standalone --nproc_per_node=1 -m bytelatent.train config=bytelatent/configs/ofat/O6b.yaml
python scripts/ofat/summarize.py /gs/bs/tga-RLA/yoshida/blt_runs/ofat/O6b O6b
echo DONE
