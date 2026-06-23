#!/bin/bash
# 本走 node_h(entropy_stage_run) の checkpoint を「コピーして」AUC評価。本走フォルダは読み取りのみ。
# コピー先 blt_runs/auc_eval/ で consolidate→eval→コピー削除（auc_curve.csv だけ残す）。
# baseline entropy_d1/0000005000 は別フォルダ・consolidate済なので in-place 評価。
# gpu_1。修正済み eval_boundary_auc.py（en基準0.99検証済）。
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=1:30:00
#$ -N auc_curve3
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/auc_curve_v3.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/auc_curve_v3.err.log

source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt

SRC=/gs/bs/tga-RLA/yoshida/blt_runs/entropy_stage_run/checkpoints   # 本走（読み取りのみ）
DST=/gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/ckpt                   # コピー先（隔離）
HV=/gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/heldout_v3.jsonl
OUT=/gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/auc_curve_v3.csv
D1=/gs/bs/tga-RLA/yoshida/blt_runs/entropy_d1/checkpoints/0000005000/consolidated
mkdir -p "$DST"
echo "step,lang,auc,boundary_rate" > "$OUT"

echo "=== baseline d1_5000（in-place, 本走外）==="; date
python scripts/eval_boundary_auc.py --ckpt-dir "$D1" --jsonl "$HV" --label d1_5000 2>/dev/null \
  | awk -F, 'NR>2 && NF==4 {print "BASE_d1_5000,"$1","$4","$3}' >> "$OUT"

for ck in $(ls -d $SRC/[0-9]*/ 2>/dev/null | sort); do
  step=$(basename "$ck")
  echo "=== $step: copy→consolidate→eval ==="; date
  rm -rf "$DST/$step"; cp -r "$ck" "$DST/$step"          # 本走から読み取りコピー
  python -m bytelatent.checkpoint consolidate "$DST/$step" >/dev/null 2>&1
  python scripts/eval_boundary_auc.py --ckpt-dir "$DST/$step/consolidated" --jsonl "$HV" --label "$step" 2>/dev/null \
    | awk -F, -v s="$step" 'NR>2 && NF==4 {print s","$1","$4","$3}' >> "$OUT"
  rm -rf "$DST/$step"                                     # コピー削除（容量節約）
  echo "  done $step"
done

echo "=== AUC推移（ja=主役・BASE_d1_5000=0.49基準と比較）==="
column -t -s, "$OUT"
date; echo "DONE"
