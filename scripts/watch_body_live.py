"""本体run の loss/bpb/bytes-per-patch ライブモニタ（再起動不要・metrics.jsonl を読む）。
cblt vs byte を横並び表示し ΔBPB(byte-cblt)・bpp_cum(累積bytes/patch・目標6.0/byte≈cblt判定) も出す。
ジョブ名照合なので resume 後も追従。
使い方:  python scripts/watch_body_live.py [interval_sec=15]
         （Ctrl-C で終了。1回だけ見るなら: python scripts/watch_body_live.py once）
"""
import json, os, sys, time, subprocess, glob

BODY = "/gs/bs/tga-RLA/yoshida/blt_runs/body"
EPOCH = 560000  # 1 epoch step (2GPU)
# (表示名, ジョブ名, dump_dir)
RUNS = [
    ("cblt", "body_leadcblt", f"{BODY}/lead_blt_body_cblt_r6"),
    ("byte", "body_leadbyte", f"{BODY}/lead_blt_body_byte_r6"),
]


def qstate(jobname):
    try:
        out = subprocess.run(["qstat"], capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return "?"
    for ln in out.splitlines():
        f = ln.split()
        if len(f) >= 5 and f[2][:10] == jobname[:10]:
            return f"{f[0]}:{f[4]}"  # jobID:state
    return "—"


def tail_metrics(dump_dir, n=8):
    p = os.path.join(dump_dir, "metrics.jsonl")
    if not os.path.exists(p):
        return []
    try:
        with open(p, errors="ignore") as f:
            lines = f.readlines()[-n:]
        return [json.loads(x) for x in lines if x.strip()]
    except Exception:
        return []


def fmt(v, f="{:.3f}"):
    return f.format(v) if isinstance(v, (int, float)) else "—"


def snapshot():
    rows = {}
    for name, job, dd in RUNS:
        m = tail_metrics(dd)
        st = qstate(job)
        if not m:
            rows[name] = dict(state=st, step=None)
            continue
        last = m[-1]
        bpb_hist = [d.get("bpb/interval_across_gpus") for d in m if d.get("bpb/interval_across_gpus")]
        bppc_hist = [d.get("patch/bpp_cumulative") for d in m if d.get("patch/bpp_cumulative")]
        rows[name] = dict(
            state=st,
            step=last.get("global_step"),
            loss=last.get("loss/interval_across_gpu"),
            bpb=last.get("bpb/interval_across_gpus"),
            lr=last.get("optim/lr"),
            wps=last.get("speed/wps"),
            it=last.get("speed/curr_iter_time"),
            bpp_inst=last.get("patch/bpp_inst"),
            bpp_cum=last.get("patch/bpp_cumulative"),
            trend=bpb_hist[-6:],
            bppc_trend=bppc_hist[-6:],
        )
    return rows


def render(rows):
    c, b = rows["cblt"], rows["byte"]
    L = []
    L.append(f"═══ CBLT body live  {time.strftime('%Y-%m-%d %H:%M:%S')} ═══")
    L.append(f"{'':12}{'cblt':>18}{'byte':>18}")
    L.append(f"{'job/state':12}{c['state']:>18}{b['state']:>18}")
    if c["step"] is None and b["step"] is None:
        L.append("\n  (まだ metrics 無し＝build中。学習開始まで待機…)")
        return "\n".join(L)

    def line(label, key, f="{:.3f}"):
        cv = fmt(c.get(key), f) if c["step"] is not None else "—"
        bv = fmt(b.get(key), f) if b["step"] is not None else "—"
        return f"{label:12}{cv:>18}{bv:>18}"

    def pct(r):
        return f"{r['step']/EPOCH*100:.1f}%" if r["step"] else "—"
    cs = f"{c['step']:,}" if c["step"] else "—"
    bs = f"{b['step']:,}" if b["step"] else "—"
    L.append(f"{'step':12}{cs:>18}{bs:>18}")
    L.append(f"{'progress':12}{pct(c):>18}{pct(b):>18}   (/560k=1epoch)")
    L.append(line("loss", "loss"))
    L.append(line("bpb", "bpb"))
    L.append(line("lr", "lr", "{:.2e}"))
    L.append(line("wps", "wps", "{:.2e}"))
    L.append(line("iter(s)", "it", "{:.3f}"))
    L.append(line("bpp_inst", "bpp_inst"))
    L.append(line("bpp_cum", "bpp_cum"))

    def trend(r, key="trend"):
        t = r.get(key) or []
        if len(t) < 2:
            return "—"
        arr = "↓" if t[-1] < t[0] else ("↑" if t[-1] > t[0] else "→")
        return f"{t[0]:.3f}→{t[-1]:.3f} {arr}"
    L.append(f"{'bpb trend':12}{trend(c):>18}{trend(b):>18}")
    L.append(f"{'bpp_cum tr':12}{trend(c, 'bppc_trend'):>18}{trend(b, 'bppc_trend'):>18}")
    L.append("─" * 48)
    if c["step"] and b["step"] and c.get("bpb") and b.get("bpb"):
        d = b["bpb"] - c["bpb"]
        tag = "cblt優位 ✓" if d > 0 else ("byte優位" if d < 0 else "同等")
        L.append(f"  ΔBPB (byte−cblt) = {d:+.4f}   {tag}")
    TARGET = 6.0
    if c.get("bpp_cum") and b.get("bpp_cum"):
        dd = b["bpp_cum"] - c["bpp_cum"]
        match = "✓matched" if abs(dd) < 0.2 else "⚠ 差大"
        L.append(f"  bytes/patch累積: cblt {c['bpp_cum']:.3f} / byte {b['bpp_cum']:.3f}"
                 f"  |Δ|={abs(dd):.3f} {match}  (目標{TARGET})")
    L.append(f"  1epoch={EPOCH:,}step ~2.2日(resume2回)   refresh / Ctrl-C で終了")
    return "\n".join(L)


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "15"
    if arg == "once":
        print(render(snapshot()))
        return
    interval = float(arg)
    try:
        while True:
            os.system("clear")
            print(render(snapshot()), flush=True)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n終了")


if __name__ == "__main__":
    main()
