#!/bin/bash
# 連続run の全 checkpoint を consolidate → 境界整合AUC 評価 → AUC推移CSV
# 進捗指標（§7）: 日本語 MeCab 境界AUC が主役。en/ru はサニティ。
# gpu_1 で実行（学習と並行 or 走行後）。引数: RUN dir（省略時 entropy_stage_run）
# qsub -g tga-RLA scripts/qsub_eval_ckpts.sh  などから呼ぶ
set -e
RUN=${1:-/gs/bs/tga-RLA/yoshida/blt_runs/entropy_stage_run}
HV=/gs/bs/tga-RLA/yoshida/blt_data/eval/heldout_v2.jsonl
OUT=$RUN/auc_curve.csv
cd /gs/bs/tga-RLA/yoshida/blt

echo "step,lang,auc,boundary_rate" > "$OUT"
for ck in $(ls -d $RUN/checkpoints/*/ 2>/dev/null | sort); do
  step=$(basename "$ck")
  # consolidate（未実施なら）
  [ -f "$ck/consolidated/consolidated.pth" ] || python -m bytelatent.checkpoint consolidate "$ck" >/dev/null 2>&1
  # 境界AUC（lang別に出力 → step付きで集約）
  python scripts/eval_boundary_auc.py --ckpt-dir "$ck/consolidated" --jsonl "$HV" --label "$step" 2>/dev/null \
    | awk -F, -v s="$step" 'NR>2 && NF==4 {print s","$1","$4","$3}' >> "$OUT"
  echo "  done $step"
done
echo "=== AUC推移 (lang=ja が主役) ==="
column -t -s, "$OUT"
