#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=0:20:00
#$ -N cblt_en
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/cblt_en.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/cblt_en.err.log
source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt
SRC=/gs/bs/tga-RLA/yoshida/blt_runs/entropy_stage_run/checkpoints
DST=/gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/viz_en_ckpt
LATEST=$(ls -d $SRC/[0-9]*/ | sort | tail -1)
rm -rf "$DST"; cp -r "$LATEST" "$DST"
python -m bytelatent.checkpoint consolidate "$DST" >/dev/null 2>&1
python scripts/cblt_viz.py --ckpt-dir "$DST/consolidated" --text "The quick brown fox jumps over the lazy dog." --out /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/cblt_viz_en.html
rm -rf "$DST"; echo DONE
