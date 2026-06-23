#!/bin/bash
# 境界整合AUC サニティ: node_h連続run の全checkpoint を consolidate→AUC推移CSV
# + healthy基準 entropy_d1/0000005000 を同CSVに baseline 行として併記。
# gpu_1（共有プール, AR非対象, 即起動）で実行。
# qsub -g tga-RLA scripts/qsub_eval_ckpts.sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=3:00:00
#$ -N eval_auc
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/eval_auc.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/eval_auc.err.log

source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt

RUN=/gs/bs/tga-RLA/yoshida/blt_runs/entropy_stage_run
HV=/gs/bs/tga-RLA/yoshida/blt_data/eval/heldout_v2.jsonl
OUT=$RUN/auc_curve.csv

echo "=== node_h 連続run の checkpoint AUC 推移 ==="; date
bash scripts/eval_checkpoints_auc.sh "$RUN"

echo ""; echo "=== healthy基準 entropy_d1/0000005000 を baseline として追記 ==="
python scripts/eval_boundary_auc.py \
  --ckpt-dir /gs/bs/tga-RLA/yoshida/blt_runs/entropy_d1/checkpoints/0000005000/consolidated \
  --jsonl "$HV" --label "d1_5000_baseline" 2>/dev/null \
  | awk -F, 'NR>2 && NF==4 {print "BASELINE_d1_5000,"$1","$4","$3}' >> "$OUT"

echo ""; echo "=== 最終 AUC 一覧（lang=ja が主役、BASELINE_d1_5000 と比較）==="
column -t -s, "$OUT"
date; echo "DONE"
