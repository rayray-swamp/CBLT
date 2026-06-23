#!/bin/bash
# 最適化A/B実験: gpu_1 batch4 + torch.compile。700step短時間。
# baseline=現 stage_g1(batch4 compile無, MFU~24%/wps~2e5)と steady MFU を比較。
# 別dump_dir(entropy_opt_test)・別gpu_1スライス。node_h/現gpu_1 に無干渉。
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=0:30:00
#$ -N opt_test
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/opt_test.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/opt_test.err.log

source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt

echo "=== 開始(gpu_1 batch4 + torch.compile) ==="; date
torchrun --nproc_per_node=1 -m bytelatent.train config=bytelatent/configs/entropy_opt_test.yaml
echo "=== 終了 ==="; date; echo "DONE"
