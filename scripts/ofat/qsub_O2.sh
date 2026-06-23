#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=0:30:00
#$ -N ofat_O2
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/ofat/O2.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/ofat/O2.err.log
source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt
torchrun --standalone --nproc_per_node=1 -m bytelatent.train config=bytelatent/configs/ofat/O2.yaml
python scripts/ofat/summarize.py /gs/bs/tga-RLA/yoshida/blt_runs/ofat/O2 O2
echo DONE
