#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=0:30:00
#$ -N cbltwiki
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/cblt_jawiki.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/cblt_jawiki.err.log
source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt
SRC=/gs/bs/tga-RLA/yoshida/blt_runs/entropy_stage_run/checkpoints
DST=/gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/wiki_ckpt
LATEST=""
for d in $(ls -d $SRC/[0-9]*/ 2>/dev/null | sort -r); do
  [ -f "$d/train_state_00000.json" ] && [ -f "$d/params.json" ] && { LATEST="$d"; break; }
done
echo "ckpt=$(basename $LATEST)"
rm -rf "$DST"; cp -r "$LATEST" "$DST"
python -m bytelatent.checkpoint consolidate "$DST" >/dev/null 2>&1
echo "########## 綺麗な日本語Wikipedia (heldout_jawiki, lang=ja_wiki) ##########"
python scripts/cblt_eval_corpus.py --ckpt-dir "$DST/consolidated" \
  --jsonl /gs/bs/tga-RLA/yoshida/blt_data/eval/heldout_jawiki.jsonl --lang-prefix ja
rm -rf "$DST"; echo DONE
