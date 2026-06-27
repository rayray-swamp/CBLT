#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=0:30:00
#$ -N cbltboth
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/cblt_both.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/cblt_both.err.log
source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt
SRC=/gs/bs/tga-RLA/yoshida/blt_runs/entropy_stage_run/checkpoints
DST=/gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/both_ckpt
LATEST=$(ls -d $SRC/[0-9]*/ | sort | tail -1); echo "ckpt=$(basename $LATEST)"
rm -rf "$DST"; cp -r "$LATEST" "$DST"
python -m bytelatent.checkpoint consolidate "$DST" >/dev/null 2>&1
echo "########## (A) 5文 (同一コード, min_toks=2) ##########"
python scripts/cblt_eval_corpus.py --ckpt-dir "$DST/consolidated" \
  --jsonl /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/five_sents.jsonl --min-toks 2
echo "########## (B) コーパス heldout_v3 (同一コード, min_toks=64) ##########"
python scripts/cblt_eval_corpus.py --ckpt-dir "$DST/consolidated" \
  --jsonl /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/heldout_v3.jsonl --min-toks 64
rm -rf "$DST"; echo DONE
