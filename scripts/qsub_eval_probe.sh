#!/bin/bash
# メモリ化切り分けプローブ — 4セットの bpb を測る（GPU・短時間バッチ）
# qsub -g tga-RLA scripts/qsub_eval_probe.sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=0:25:00
#$ -N eval_probe
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/eval_probe.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/eval_probe.err.log

source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt
CK=/gs/bs/tga-RLA/yoshida/blt_runs/entropy_real/checkpoints/0000102000/consolidated
E=/gs/bs/tga-RLA/yoshida/blt_data/eval

for name in train_sample train_sample_perturbed heldout heldout_jawiki; do
  echo "######## $name ########"
  python scripts/eval_entropy_bpb.py --ckpt-dir "$CK" --jsonl "$E/$name.jsonl"
  echo ""
done
echo "DONE"
