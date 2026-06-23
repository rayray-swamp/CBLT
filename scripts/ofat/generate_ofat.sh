#!/bin/bash
# OFAT 全 variant の config + qsub を生成。既存 config をコピーし1項目だけ変更。
# 出力は全て blt_runs/ofat/ 配下に隔離。既存フォルダには一切書かない。
set -e
cd /gs/bs/tga-RLA/yoshida/blt
mkdir -p bytelatent/configs/ofat scripts/ofat /gs/bs/tga-RLA/yoshida/blt_runs/ofat
G1=bytelatent/configs/entropy_stage_g1.yaml    # batch4 baseline(gpu_1)
NH=bytelatent/configs/entropy_stage_run.yaml   # batch8 (node_h, O8用)
DD=/gs/bs/tga-RLA/yoshida/blt_runs/ofat

# gpu_1 variant config: $1=name $2=steps
mkg1(){ cp $G1 bytelatent/configs/ofat/$1.yaml
  sed -i -e "s#^dump_dir:.*#dump_dir: $DD/$1#" -e "s#^name:.*#name: \"ofat_$1\"#" -e "s#^steps:.*#steps: $2#" bytelatent/configs/ofat/$1.yaml; }
# gpu_1 qsub: $1=name
mkqg1(){ cat > scripts/ofat/qsub_$1.sh <<EOF
#!/bin/bash
#\$ -cwd
#\$ -l gpu_1=1
#\$ -l h_rt=0:30:00
#\$ -N ofat_$1
#\$ -o $DD/$1.out.log
#\$ -e $DD/$1.err.log
source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt
torchrun --nproc_per_node=1 -m bytelatent.train config=bytelatent/configs/ofat/$1.yaml
python scripts/ofat/summarize.py $DD/$1 $1
echo DONE
EOF
}

mkg1 N 2000;  mkqg1 N
mkg1 O1 2000; sed -i '/^distributed:/a\  compile: true' bytelatent/configs/ofat/O1.yaml; mkqg1 O1
mkg1 O2 2000; sed -i 's#^  attn_impl:.*#  attn_impl: "sdpa"#' bytelatent/configs/ofat/O2.yaml; mkqg1 O2
mkg1 O3 2000; sed -i 's#^  attn_impl:.*#  attn_impl: "flex_attention"#' bytelatent/configs/ofat/O3.yaml; mkqg1 O3
mkg1 O4 2000; sed -i '/^distributed:/a\  float8_recipe: rowwise' bytelatent/configs/ofat/O4.yaml; mkqg1 O4
mkg1 O6a 1000; sed -i 's#^  batch_size:.*#  batch_size: 8#' bytelatent/configs/ofat/O6a.yaml; mkqg1 O6a
mkg1 O6b 1000; sed -i 's#^  batch_size:.*#  batch_size: 16#' bytelatent/configs/ofat/O6b.yaml; mkqg1 O6b
mkg1 O7 2000; sed -i -e 's#^  seq_len:.*#  seq_len: 16384#' -e 's#^  max_encoder_seq_length:.*#  max_encoder_seq_length: 16384#' -e 's#^  max_seqlen:.*#  max_seqlen: 16384#' bytelatent/configs/ofat/O7.yaml; mkqg1 O7

# O8: no_shard on node_h (batch8 base, 2GPU)
cp $NH bytelatent/configs/ofat/O8.yaml
sed -i -e "s#^dump_dir:.*#dump_dir: $DD/O8#" -e "s#^name:.*#name: \"ofat_O8\"#" -e "s#^steps:.*#steps: 1000#" -e 's#^  fsdp_type:.*#  fsdp_type: no_shard#' bytelatent/configs/ofat/O8.yaml
cat > scripts/ofat/qsub_O8.sh <<EOF
#!/bin/bash
#\$ -cwd
#\$ -l node_h=1
#\$ -l h_rt=0:30:00
#\$ -N ofat_O8
#\$ -o $DD/O8.out.log
#\$ -e $DD/O8.err.log
source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt
torchrun --nproc_per_node=2 -m bytelatent.train config=bytelatent/configs/ofat/O8.yaml
python scripts/ofat/summarize.py $DD/O8 O8
echo DONE
EOF

echo "=== 生成 config ==="; ls bytelatent/configs/ofat/
echo "=== 生成 qsub ==="; ls scripts/ofat/
