#!/bin/bash
# C-3 ローダ重複監査 — CPU バッチ（GPU不使用）
# qsub -g tga-RLA scripts/qsub_audit_loader.sh
#$ -cwd
#$ -l cpu_4=1
#$ -l h_rt=0:40:00
#$ -N audit_loader
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/audit_loader.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/audit_loader.err.log

source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt
mkdir -p /gs/bs/tga-RLA/yoshida/blt_runs

echo "======== MODE: continuous (MPなし・コア配線) ========"
python scripts/audit_loader_dup.py --mode continuous --n-batches 3000

echo ""
echo "======== MODE: checkpoint (学習と同じ refresh サイクル) ========"
python scripts/audit_loader_dup.py --mode checkpoint --n-batches 3000 --ckpt-every 500

echo ""
echo "DONE"
