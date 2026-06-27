#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=0:40:00
#$ -N recheckja
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/recheck_ja.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/recheck_ja.err.log
source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt
SRC=/gs/bs/tga-RLA/yoshida/blt_runs/entropy_stage_run/checkpoints
DST=/gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/recheck_ckpt
LATEST=""; for d in $(ls -d $SRC/[0-9]*/ 2>/dev/null|sort -r); do [ -f "$d/train_state_00000.json" ]&&{ LATEST="$d";break;};done
echo "ckpt=$(basename $LATEST)"
rm -rf "$DST"; cp -r "$LATEST" "$DST"; python -m bytelatent.checkpoint consolidate "$DST" >/dev/null 2>&1
echo "########## heldout_jawiki (clean wiki, 旧0.51) gold修正版 ##########"
python scripts/cblt_eval_corpus.py --ckpt-dir "$DST/consolidated" \
  --jsonl /gs/bs/tga-RLA/yoshida/blt_data/eval/heldout_jawiki.jsonl --lang-prefix ja
echo "########## heldout_v3 (spam corpus, 旧0.51) gold修正版 ##########"
python scripts/cblt_eval_corpus.py --ckpt-dir "$DST/consolidated" \
  --jsonl /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/heldout_v3.jsonl --lang-prefix ja
rm -rf "$DST"; echo DONE
