#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=0:20:00
#$ -N ja_csv
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/ja_csv.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/ja_csv.err.log
source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt
SRC=/gs/bs/tga-RLA/yoshida/blt_runs/entropy_stage_run/checkpoints
DST=/gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/ja_csv_ckpt
LATEST=$(ls -d $SRC/[0-9]*/ | sort | tail -1); echo "ckpt=$(basename $LATEST)"
rm -rf "$DST"; cp -r "$LATEST" "$DST"
python -m bytelatent.checkpoint consolidate "$DST" >/dev/null 2>&1
python scripts/ja_raw_csv.py --ckpt-dir "$DST/consolidated" --out /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/ja_raw_entropy.csv
rm -rf "$DST"; echo DONE
