#!/bin/bash
# エントロピー学習の再開(tail): 270000→287000。残り~1.6h相当なので h_rt=3h(短くしてbackfillで早く動く)。
# 最新ckptから自動再開(resumable)。元のフル走スクリプトは qsub_stage_run.sh(24h)を温存。
# qsub -g tga-RLA scripts/qsub_stage_resume.sh
#$ -cwd
#$ -l node_h=1
#$ -l h_rt=3:00:00
#$ -N stage_run
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/stage_resume.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/stage_resume.err.log
source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt
echo "=== 再開 ==="; date; nvidia-smi --query-gpu=index,name --format=csv,noheader
torchrun --nproc_per_node=2 -m bytelatent.train config=bytelatent/configs/entropy_stage_run.yaml
echo "=== 終了 ==="; date; echo "DONE"
