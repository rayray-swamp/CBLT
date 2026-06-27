#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=0:30:00
#$ -N cbltws
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/cblt_wikisents.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/cblt_wikisents.err.log
source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt
SRC=/gs/bs/tga-RLA/yoshida/blt_runs/entropy_stage_run/checkpoints
DST=/gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/ws_ckpt
LATEST=""; for d in $(ls -d $SRC/[0-9]*/ 2>/dev/null|sort -r); do [ -f "$d/train_state_00000.json" ]&&{ LATEST="$d";break;};done
echo "ckpt=$(basename $LATEST)"
rm -rf "$DST"; cp -r "$LATEST" "$DST"; python -m bytelatent.checkpoint consolidate "$DST" >/dev/null 2>&1
echo "########## Wikipedia 文単位(短文・5文と同条件) ##########"
python scripts/cblt_eval_corpus.py --ckpt-dir "$DST/consolidated" \
  --jsonl /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/jawiki_sents.jsonl --lang-prefix ja --min-toks 2
rm -rf "$DST"; echo DONE
