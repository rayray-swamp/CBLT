#!/bin/bash
# LR検証プローブ: D1 と同一・LR のみ 1e-4。学習→全checkpoint eval で U字比較
# qsub -g tga-RLA scripts/qsub_d1_lowlr.sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=0:50:00
#$ -N d1_lowlr
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/d1_lowlr.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/d1_lowlr.err.log

source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt

echo "==== 学習 (LR=1e-4, 8000 step) ===="
torchrun --nproc_per_node=1 -m bytelatent.train config=bytelatent/configs/entropy_d1_lowlr.yaml

echo ""
echo "==== 全 checkpoint を held-out/train で eval (U字) ===="
RUN=/gs/bs/tga-RLA/yoshida/blt_runs/entropy_d1_lowlr
HELD=/gs/bs/tga-RLA/yoshida/blt_data/eval/heldout.jsonl
TRAIN=/gs/bs/tga-RLA/yoshida/blt_data/eval/train_sample.jsonl
echo "step,held_out_bpb,train_bpb"
for ckpt in $(ls -d $RUN/checkpoints/*/ | sort); do
  step=$(basename "$ckpt")
  [ -f "$ckpt/consolidated/consolidated.pth" ] || python -m bytelatent.checkpoint consolidate "$ckpt" >/dev/null 2>&1
  ho=$(python scripts/eval_entropy_bpb.py --ckpt-dir "$ckpt/consolidated" --jsonl "$HELD" 2>/dev/null | grep -oP "bpb = \K[0-9.]+")
  tr=$(python scripts/eval_entropy_bpb.py --ckpt-dir "$ckpt/consolidated" --jsonl "$TRAIN" 2>/dev/null | grep -oP "bpb = \K[0-9.]+")
  echo "$step,$ho,$tr"
done
echo "DONE"
