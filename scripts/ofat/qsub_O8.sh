#!/bin/bash
#$ -cwd
#$ -l node_h=1
#$ -l h_rt=0:30:00
#$ -N ofat_O8
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/ofat/O8.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/ofat/O8.err.log
source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt
torchrun --standalone --nproc_per_node=2 -m bytelatent.train config=bytelatent/configs/ofat/O8.yaml
python scripts/ofat/summarize.py /gs/bs/tga-RLA/yoshida/blt_runs/ofat/O8 O8
echo DONE
