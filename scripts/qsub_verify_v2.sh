#!/bin/bash
# 追加検証v2: フルコンテキスト+warm-up除外+英35日35。healthy vs degenerate
# qsub -g tga-RLA scripts/qsub_verify_v2.sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=0:30:00
#$ -N verify_v2
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/verify_v2.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/verify_v2.err.log

source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt
J=/gs/bs/tga-RLA/yoshida/blt_data/eval/heldout_v2.jsonl

python scripts/verify_entropy_v2.py --jsonl "$J" --label healthy_step5000 \
  --ckpt-dir /gs/bs/tga-RLA/yoshida/blt_runs/entropy_d1/checkpoints/0000005000/consolidated

python scripts/verify_entropy_v2.py --jsonl "$J" --label degenerate_102k \
  --ckpt-dir /gs/bs/tga-RLA/yoshida/blt_runs/entropy_real/checkpoints/0000102000/consolidated

echo "DONE"
