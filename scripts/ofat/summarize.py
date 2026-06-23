#!/usr/bin/env python
# OFAT 各 variant の steady 指標を自分の res_{label}.csv に1行書き出す。
# 各ジョブが終了時に呼ぶ（並行書き込み衝突を避けるため variant 毎に別ファイル）。
# usage: summarize.py <dump_dir> <label>
import sys, json, os
dump_dir, label = sys.argv[1], sys.argv[2]
OFAT = "/gs/bs/tga-RLA/yoshida/blt_runs/ofat"
mpath = os.path.join(dump_dir, "metrics.jsonl")
out = os.path.join(OFAT, f"res_{label}.csv")

def avg(rows, k):
    v = [r[k] for r in rows if k in r and isinstance(r[k], (int, float))]
    return sum(v)/len(v) if v else float("nan")

try:
    rows = [json.loads(l) for l in open(mpath) if l.strip()]
    if len(rows) < 5:
        line = f"{label},INCOMPLETE,,,,,,,steps={len(rows)}"
    else:
        st = rows[-30:] if len(rows) >= 30 else rows  # steady tail (warmup除外)
        mfu = 100*avg(st, "speed/FLOPS")/989e12
        wps = avg(st, "speed/wps")
        it  = avg(st, "speed/curr_iter_time")*1000
        mem = avg(st, "memory/max_reserved_pct")
        bpb = avg(st, "bpb/interval_across_gpus")
        loss= avg(st, "loss/interval_across_gpu")
        ooms= sum(r.get("memory/num_ooms", 0) for r in rows)
        s0, s1 = rows[0]["global_step"], rows[-1]["global_step"]
        line = f"{label},OK,{mfu:.1f},{wps:.3e},{it:.0f},{mem:.0f},{bpb:.3f},{loss:.3f},steps{s0}-{s1}|ooms{ooms}"
except Exception as e:
    line = f"{label},ERROR,,,,,,,{str(e)[:60]}"

with open(out, "w") as f:
    f.write(line + "\n")
print("SUMMARY:", line)
