#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=0:40:00
#$ -N flores
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/flores.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/flores.err.log
source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt
SRC=/gs/bs/tga-RLA/yoshida/blt_runs/entropy_stage_run/checkpoints
DST=/gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/flores_ckpt
LATEST=""; for d in $(ls -d $SRC/[0-9]*/ 2>/dev/null|sort -r); do [ -f "$d/train_state_00000.json" ]&&{ LATEST="$d";break;};done
echo "ckpt=$(basename $LATEST)"
rm -rf "$DST"; cp -r "$LATEST" "$DST"; python -m bytelatent.checkpoint consolidate "$DST" >/dev/null 2>&1
python scripts/flores_eval.py --ckpt-dir "$DST/consolidated" \
  --out-file /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/flores_results.txt
rm -rf "$DST"; echo DONE
