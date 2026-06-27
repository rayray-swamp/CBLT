#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=0:20:00
#$ -N cblt_viz
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/cblt_viz.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/cblt_viz.err.log
source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt
SRC=/gs/bs/tga-RLA/yoshida/blt_runs/entropy_stage_run/checkpoints
DST=/gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/viz_ckpt
LATEST=$(ls -d $SRC/[0-9]*/ | sort | tail -1); step=$(basename "$LATEST")
echo "=== 現モデル ckpt=$step をコピーして可視化 ==="; date
rm -rf "$DST"; cp -r "$LATEST" "$DST"
python -m bytelatent.checkpoint consolidate "$DST" >/dev/null 2>&1
# 検証用の複数テキストで HTML 生成
python scripts/cblt_viz.py --ckpt-dir "$DST/consolidated" --text "I have a りんご!" --out /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/cblt_viz_sample.html
echo "--- 日本語サンプル ---"
python scripts/cblt_viz.py --ckpt-dir "$DST/consolidated" --text "参議院議員選挙の投票日は来週の日曜日です。" --out /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/cblt_viz_ja.html
rm -rf "$DST"
echo DONE
