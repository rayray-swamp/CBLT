#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=0:30:00
#$ -N ofat_O1b
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/ofat/O1b.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/ofat/O1b.err.log
source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt
export TORCHDYNAMO_SUPPRESS_ERRORS=1
torchrun --standalone --nproc_per_node=1 -m bytelatent.train config=bytelatent/configs/ofat/O1b.yaml
python scripts/ofat/summarize.py /gs/bs/tga-RLA/yoshida/blt_runs/ofat/O1b O1b
echo DONE
