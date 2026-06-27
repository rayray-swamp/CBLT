#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=0:30:00
#$ -N cbltcorp
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/cblt_corpus.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/cblt_corpus.err.log
source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt
SRC=/gs/bs/tga-RLA/yoshida/blt_runs/entropy_stage_run/checkpoints
DST=/gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/corpus_ckpt
# 書きかけ回避: train_state_00000.json を持つ=完成 の最新ckptを選ぶ
LATEST=""
for d in $(ls -d $SRC/[0-9]*/ 2>/dev/null | sort -r); do
  [ -f "$d/train_state_00000.json" ] && [ -f "$d/params.json" ] && { LATEST="$d"; break; }
done
echo "ckpt=$(basename $LATEST) (完成チェックポイント)"
rm -rf "$DST"; cp -r "$LATEST" "$DST"
python -m bytelatent.checkpoint consolidate "$DST" >/dev/null 2>&1
python scripts/cblt_eval_corpus.py --ckpt-dir "$DST/consolidated" \
  --jsonl /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/heldout_v3.jsonl
rm -rf "$DST"; echo DONE
