#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=0:20:00
#$ -N jasamp
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/ja_samples.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/ja_samples.err.log
source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt
SRC=/gs/bs/tga-RLA/yoshida/blt_runs/entropy_stage_run/checkpoints
DST=/gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/samp_ckpt
LATEST=""; for d in $(ls -d $SRC/[0-9]*/ 2>/dev/null|sort -r); do [ -f "$d/train_state_00000.json" ]&&{ LATEST="$d";break;};done
echo "ckpt=$(basename $LATEST)"
rm -rf "$DST"; cp -r "$LATEST" "$DST"; python -m bytelatent.checkpoint consolidate "$DST" >/dev/null 2>&1
python scripts/ja_raw_csv_samples.py --ckpt-dir "$DST/consolidated" \
  --out /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/ja_raw_samples.csv --n 25 --seed 42
rm -rf "$DST"; echo DONE
