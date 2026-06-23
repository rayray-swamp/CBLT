#!/bin/bash
# off-by-one 修正の検証: healthy基準 entropy_d1/0000005000 を再評価。
# en AUC が 0.53(修正前) → 0.8前後 に跳ねれば修正成功（構造を正しく捉えている）。
# gpu_1（共有プール, 即起動）。
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=0:30:00
#$ -N verify_auc
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/verify_auc.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/verify_auc.err.log

source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt

echo "=== off-by-one 修正検証: entropy_d1/5000 (healthy基準) ==="; date
python scripts/eval_boundary_auc.py \
  --ckpt-dir /gs/bs/tga-RLA/yoshida/blt_runs/entropy_d1/checkpoints/0000005000/consolidated \
  --jsonl /gs/bs/tga-RLA/yoshida/blt_data/eval/heldout_v2.jsonl --label "d1_5000_FIXED"
echo "=== 期待: en AUC が 0.53→0.8前後 / ja も chance(0.49)から上昇すれば修正成功 ==="
date; echo "DONE"
