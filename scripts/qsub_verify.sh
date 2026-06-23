#!/bin/bash
# 凍結前最終検証: healthy(step5000) vs degenerate(102k) を同一処理で対比
# qsub -g tga-RLA scripts/qsub_verify.sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=0:20:00
#$ -N verify_ent
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/verify_ent.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/verify_ent.err.log

source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt

python scripts/verify_entropy_model.py \
  --ckpt-dir /gs/bs/tga-RLA/yoshida/blt_runs/entropy_d1/checkpoints/0000005000/consolidated \
  --label "healthy_step5000"

python scripts/verify_entropy_model.py \
  --ckpt-dir /gs/bs/tga-RLA/yoshida/blt_runs/entropy_real/checkpoints/0000102000/consolidated \
  --label "degenerate_102k"

echo "DONE"
