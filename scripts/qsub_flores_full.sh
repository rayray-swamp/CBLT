#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=0:45:00
#$ -N floresfull
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/flores_full.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/flores_full.err.log
source /gs/bs/tga-RLA/yoshida/activate_blt.sh
export OPENBLAS_NUM_THREADS=4
cd /gs/bs/tga-RLA/yoshida/blt
SRC=/gs/bs/tga-RLA/yoshida/blt_runs/entropy_stage_run/checkpoints
DST=/gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/full_ckpt
LATEST=""; for d in $(ls -d $SRC/[0-9]*/ 2>/dev/null|sort -r); do [ -f "$d/train_state_00000.json" ]&&{ LATEST="$d";break;};done
echo "ckpt=$(basename $LATEST)"
rm -rf "$DST"; cp -r "$LATEST" "$DST"; python -m bytelatent.checkpoint consolidate "$DST" >/dev/null 2>&1
python scripts/flores_full.py --ckpt-dir "$DST/consolidated" --out-dir /gs/bs/tga-RLA/yoshida/blt/flores_dumps
rm -rf "$DST"; echo DONE
