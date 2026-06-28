#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=2:30:00
#$ -N auccurve
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/auc_curve.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/auc_curve.err.log
source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt
SRC=/gs/bs/tga-RLA/yoshida/blt_runs/entropy_stage_run/checkpoints
DST=/gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/curve_ckpt
RES=/gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/auc_curve.tsv
JW=/gs/bs/tga-RLA/yoshida/blt_data/eval/heldout_jawiki.jsonl
echo -e "step\tja_AUC\t境界H\t非境界H" > $RES
for step in 5000 25000 50000 75000 100000 125000 150000 175000 200000 225000 250000 265000 275000 285000; do
  CK=$SRC/$(printf "%010d" $step)
  [ -f "$CK/train_state_00000.json" ] || { echo "skip $step"; continue; }
  rm -rf "$DST"; cp -r "$CK" "$DST"
  python -m bytelatent.checkpoint consolidate "$DST" >/dev/null 2>&1
  OUT=$(python scripts/cblt_eval_corpus.py --ckpt-dir "$DST/consolidated" --jsonl "$JW" --lang-prefix ja 2>/dev/null)
  AUC=$(echo "$OUT" | grep "pooled AUC" | grep -oE "0\.[0-9]+" | head -1)
  HB=$(echo "$OUT" | grep -oE "境界平均H [0-9.]+" | grep -oE "[0-9.]+" | head -1)
  HN=$(echo "$OUT" | grep -oE "非境界 [0-9.]+" | grep -oE "[0-9.]+" | head -1)
  echo -e "$step\t$AUC\t$HB\t$HN" >> $RES
  echo "step $step: AUC=$AUC (境界H=$HB 非境界=$HN)"
  rm -rf "$DST"
done
echo "=== curve ==="; cat $RES
echo DONE
