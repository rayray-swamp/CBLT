#!/bin/bash
# checkpoint 間引き: step%STRIDE==0 の粗い履歴 + 最新KEEPN個 を残し、他を削除。
# AUC選定用の履歴を粗く保ちつつストレージを抑える。書き込み中の最新は KEEPN で保護。
# usage: thin_checkpoints.sh <checkpoints_dir> [STRIDE=10000] [KEEPN=3]
set -u
DIR="${1:?checkpoints dir required}"
STRIDE="${2:-10000}"
KEEPN="${3:-3}"
[ -d "$DIR" ] || { echo "no dir: $DIR"; exit 0; }
cd "$DIR" || exit 0
mapfile -t ALL < <(ls -d [0-9]*/ 2>/dev/null | sed 's#/##' | sort -n)
N=${#ALL[@]}
(( N == 0 )) && { echo "no checkpoints"; exit 0; }
del=0
for i in "${!ALL[@]}"; do
  d="${ALL[$i]}"
  step=$((10#$d))
  # 残す条件: STRIDE の倍数 / 最新 KEEPN 個
  if (( step % STRIDE == 0 )) || (( i >= N - KEEPN )); then continue; fi
  rm -rf "./$d" && del=$((del+1))
done
kept=$(ls -d [0-9]*/ 2>/dev/null | wc -l)
echo "$(date '+%H:%M:%S') thinned: deleted=$del kept=$kept ($(du -sh . 2>/dev/null | cut -f1)) | df $(df -h /gs/bs/tga-RLA 2>/dev/null | tail -1 | awk '{print $4" free"}')"
