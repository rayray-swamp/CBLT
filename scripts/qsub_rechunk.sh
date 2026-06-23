#!/bin/bash
# 1チャンク化: 冗長データ回収 → stage1+2+3 を1ファイル連結 → Arrow化
# （3チャンク=stage は BLT のチャンク並列(world_size%n_chunks==0)と非互換のため1チャンク化）
# qsub -g tga-RLA scripts/qsub_rechunk.sh
#$ -cwd
#$ -l cpu_8=1
#$ -l h_rt=2:00:00
#$ -N rechunk
#$ -o /gs/bs/tga-RLA/yoshida/blt_runs/rechunk.out.log
#$ -e /gs/bs/tga-RLA/yoshida/blt_runs/rechunk.err.log

source /gs/bs/tga-RLA/yoshida/activate_blt.sh
cd /gs/bs/tga-RLA/yoshida/blt
export OMP_NUM_THREADS=1
LC=/gs/bs/tga-RLA/yoshida/blt_data/largecorpus

echo "=== df 開始 ==="; df -h /gs/bs/tga-RLA | tail -1
echo "=== 冗長回収: wiki生jsonl(stages取込済) + 旧3チャンクarrow(作り直し) 削除 ==="
rm -f  $LC/wiki/*.jsonl
rm -rf $LC/preprocess/fullcorpus $LC/fullcorpus
echo "=== df 回収後 ==="; df -h /gs/bs/tga-RLA | tail -1

echo "=== 1チャンク連結（stage1++2++3、順序保持）==="
mkdir -p $LC/corpus
cat $LC/stage1/stage1.chunk.00.jsonl \
    $LC/stage2/stage2.chunk.00.jsonl \
    $LC/stage3/stage3.chunk.00.jsonl > $LC/corpus/corpus.chunk.00.jsonl
ls -lh $LC/corpus/corpus.chunk.00.jsonl

echo "=== Arrow化（placeholder）==="
python scripts/make_placeholder_arrow.py "$LC/corpus/corpus.chunk.00.jsonl" "$LC/preprocess"
ls -lh $LC/preprocess/corpus/transformer_100m/

echo "=== df 終了 ==="; df -h /gs/bs/tga-RLA | tail -1
echo "DONE"
