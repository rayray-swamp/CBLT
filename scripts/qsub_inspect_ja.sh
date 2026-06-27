#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=0:30:00
#$ -N inspect_ja
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/inspect_ja.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/inspect_ja.err.log
source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt
SRC=/gs/bs/tga-RLA/yoshida/blt_runs/entropy_stage_run/checkpoints
DST=/gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/inspect_ckpt
HV=/gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/heldout_v3.jsonl
# 本走の最新 checkpoint をコピー（本走フォルダは読み取りのみ）
LATEST=$(ls -d $SRC/[0-9]*/ | sort | tail -1)
step=$(basename "$LATEST")
echo "=== 現モデル checkpoint=$step をコピーして検査 ==="; date
rm -rf "$DST"; cp -r "$LATEST" "$DST"
python -m bytelatent.checkpoint consolidate "$DST" >/dev/null 2>&1
python scripts/inspect_ja_boundary.py --ckpt-dir "$DST/consolidated" --jsonl "$HV" --sample 2
rm -rf "$DST"
echo DONE
