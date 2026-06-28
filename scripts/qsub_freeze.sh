#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=0:20:00
#$ -N freeze
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/freeze.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/freeze.err.log
source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt
SRC=/gs/bs/tga-RLA/yoshida/blt_runs/entropy_stage_run/checkpoints/0000285000
FROZEN=/gs/bs/tga-RLA/yoshida/blt_runs/entropy_stage_run/frozen_entropy_285000
echo "=== 凍結: 285000 → frozen_entropy_285000 ==="
rm -rf "$FROZEN"; cp -r "$SRC" "$FROZEN"
python -m bytelatent.checkpoint consolidate "$FROZEN" 2>&1 | tail -3
echo "--- consolidated 中身 ---"; ls -la "$FROZEN/consolidated/" 2>/dev/null
echo ""
echo "ENTROPY_MODEL_CHECKPOINT_DIR=$FROZEN"
echo "ENTROPY_MODEL_STATE_DICT_PATH=$FROZEN/consolidated/consolidated.pth"
echo "params.json: $([ -f $FROZEN/params.json ] && echo "$FROZEN/params.json" || echo "$FROZEN/consolidated/params.json")"
echo DONE
