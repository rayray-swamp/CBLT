"""byte-BLT vs CBLT-√ℓ 内在比較【FLORES+レート整合版】。
交絡除去: corpus較正θだとFLORES+で方式間の bytes/patch がズレる→ per言語×per方式でθをFLORES+上で
再較正し、両方式を同じ realized bytes/patch に揃える。これで (B)F1・(C)surprisal が compute整合。
指標: (A)mid_char_split_rate (B)boundary_F1+morph_rate(c=0除外) (C)BPB proxy(内部/先頭surprisal,示唆的)
      (D)bytes/patch・chars/patch・1字patch率。出力 *_matched.csv + ℓ別要約。"""
import math, json, csv, torch, typer
import numpy as np
from collections import defaultdict
from bytelatent.entropy_model import load_entropy_model
from bytelatent.tokenizers.build_tokenizer import TokenizerArgs
from bytelatent.data.patcher import calculate_entropies

app = typer.Typer()
TOK = "/gs/bs/tga-RLA/yoshida/blt_data/tokenizer/tokenizer.model"
FP = "/gs/bs/tga-RLA/yoshida/blt_data/floresplus"
LANGS = ["en", "ru", "ja", "ar", "zh", "ko", "hi", "th"]
LANG_ELL = {"en": 1, "ru": 2, "ar": 2, "ja": 3, "zh": 3, "ko": 3, "hi": 3, "th": 3}
GOLD = {"en": "ws", "ru": "ws", "ar": "ws", "ko": "ws", "hi": "ws", "ja": "mecab", "zh": "jieba", "th": "pythainlp"}
RATES = [4.0, 5.0, 6.0, 7.0, 8.0]

def gold_spans(text, gtype, tagger):
    spans = []
    if gtype == "ws":
        st = None
        for i, c in enumerate(text):
            if c.isspace():
                if st is not None: spans.append((st, i)); st = None
            elif st is None: st = i
        if st is not None: spans.append((st, len(text)))
    elif gtype == "mecab":
        base = 0
        for line in text.split("\n"):
            pos = 0
            for w in tagger(line):
                spans.append((base + pos, base + pos + len(w.surface))); pos += len(w.surface)
            base += len(line) + 1
    elif gtype == "jieba":
        import jieba
        pos = 0
        for w in jieba.cut(text, cut_all=False): spans.append((pos, pos + len(w))); pos += len(w)
    elif gtype == "pythainlp":
        from pythainlp import word_tokenize
        pos = 0
        for w in word_tokenize(text, keep_whitespace=True): spans.append((pos, pos + len(w))); pos += len(w)
    return spans

def calib_theta(dH, tot_bytes, n_sent, rate):
    if len(dH) == 0: return 0.0
    lo, hi = float(dH.min()) - 1, float(dH.max()) + 1
    for _ in range(50):
        mid = (lo + hi) / 2
        bpp = tot_bytes / (n_sent + int((dH > mid).sum()))
        if bpp < rate: lo = mid
        else: hi = mid
    return (lo + hi) / 2

def patch_metrics(sd, method, th):
    a = defaultdict(float)
    for s in sd:
        eb, raw, lead, cob, Hs, gsp, goldb, n, C = s["eb"], s["raw"], s["lead"], s["cob"], s["H"], s["gsp"], s["goldb"], s["n"], s["C"]
        if method == "byte":
            B = [0] + [b for b in range(1, n) if eb[b] - eb[b - 1] > th]
            starts_b = B; spans = list(zip(B, B[1:] + [n])); pred_char = set(cob[b] for b in B)
        else:
            Bc = [0] + [c for c in range(1, C) if Hs[c] - Hs[c - 1] > th]
            starts_b = [lead[c] for c in Bc]
            ends_c = Bc[1:] + [C]; spans = [(lead[cs], (lead[ce] if ce < C else n)) for cs, ce in zip(Bc, ends_c)]
            pred_char = set(Bc)
        a["npatch"] += len(spans); a["nbytes"] += n; a["nchars"] += C
        a["midsplit"] += sum(1 for b in starts_b if b > 0 and (raw[b] & 0xC0) == 0x80)
        pc = pred_char - {0}
        a["tp"] += len(pc & goldb); a["npred"] += len(pc); a["ngold"] += len(goldb)
        for bs, be in spans:
            cs = cob[bs]; ce = cob[be - 1] + 1
            if (cs, ce) in gsp: a["morph"] += 1
            if sum(1 for c in range(cs, ce) if bs <= lead[c] < be) == 1: a["onechar"] += 1
            a["start_s"] += eb[bs]; a["nstart"] += 1
            if be - bs > 1: a["interior_s"] += sum(eb[bs + 1:be]); a["ninterior"] += (be - bs - 1)
    return a

@app.command()
def main(ckpt_dir: str = typer.Option(...), out_dir: str = typer.Option(...), device: str = typer.Option("cuda")):
    model, _ = load_entropy_model(ckpt_dir, f"{ckpt_dir}/consolidated.pth", device=device); model = model.eval()
    tok = TokenizerArgs(name="blt", init_kwargs={"bpe_tokenizer_path": TOK}).build()
    import fugashi; tagger = fugashi.Tagger()
    ln = math.log(2)
    rows = []
    for lang in LANGS:
        gtype = GOLD[lang]
        sd = []; dHb = []; dHc = []; tot_bytes = 0; n_sent = 0
        for d in (json.loads(l) for l in open(f"{FP}/floresplus_{lang}_devtest.jsonl")):
            text = d["text"]
            if len(text) < 2: continue
            toks = tok.encode(text)
            if len(toks) > 8192: toks = toks[:8192]
            e = calculate_entropies(torch.tensor(toks, dtype=torch.long, device=device).unsqueeze(0), model, 1, device)[0].float()[0]
            raw = text.encode("utf-8"); n = min(len(raw), e.shape[0]); raw = raw[:n]
            eb = [float(e[b]) for b in range(n)]
            chars = []; bi = 0
            while bi < n:
                el = 1
                while bi + el < n and (raw[bi + el] & 0xC0) == 0x80: el += 1
                chars.append((bi, el)); bi += el
            C = len(chars)
            if C < 2: continue
            lead = [k for k, _ in chars]; cob = [0] * n
            for ci, (k, el) in enumerate(chars):
                for b in range(k, k + el): cob[b] = ci
            H = [sum(eb[k:k + el]) / math.sqrt(el) for k, el in chars]
            gsp = set(gold_spans(text, gtype, tagger)); goldb = set(s for s, _ in gsp) - {0}
            sd.append(dict(eb=eb, raw=raw, lead=lead, cob=cob, H=H, gsp=gsp, goldb=goldb, n=n, C=C))
            dHb.extend(eb[b] - eb[b - 1] for b in range(1, n))
            dHc.extend(H[c] - H[c - 1] for c in range(1, C))
            tot_bytes += n; n_sent += 1
        dHb = np.array(dHb); dHc = np.array(dHc)
        for rate in RATES:
            thb = calib_theta(dHb, tot_bytes, n_sent, rate)
            thc = calib_theta(dHc, tot_bytes, n_sent, rate)
            for method, th in [("byte", thb), ("sqrt", thc)]:
                a = patch_metrics(sd, method, th); npat = max(1.0, a["npatch"])
                P = a["tp"] / a["npred"] if a["npred"] else 0.0
                R = a["tp"] / a["ngold"] if a["ngold"] else 0.0
                F1 = 2 * P * R / (P + R) if P + R else 0.0
                rows.append(dict(lang=lang, ell=LANG_ELL[lang], method="byte-BLT" if method == "byte" else "CBLT-sqrtL",
                                 op=f"{rate:.0f}", theta=th, bytes_per_patch=a["nbytes"] / npat, chars_per_patch=a["nchars"] / npat,
                                 one_char_patch_rate=a["onechar"] / npat, mid_char_split_rate=a["midsplit"] / npat,
                                 boundary_F1=F1, morph_rate=a["morph"] / npat,
                                 interior_surprisal_bits=(a["interior_s"] / a["ninterior"] / ln) if a["ninterior"] else 0.0,
                                 start_surprisal_bits=(a["start_s"] / a["nstart"] / ln) if a["nstart"] else 0.0))
        print(f"[OK] {lang} (rate整合: byte/cblt 同一bytes/patch)")
    cols = ["lang", "ell", "method", "op", "theta", "bytes_per_patch", "chars_per_patch", "one_char_patch_rate",
            "mid_char_split_rate", "boundary_F1", "morph_rate", "interior_surprisal_bits", "start_surprisal_bits"]
    with open(f"{out_dir}/intrinsic_byte_vs_cblt_matched.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow({k: (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k]) for k in cols})
    # ℓ別 Δ(byte−CBLT) 要約(動作点平均)
    idx = {(r["lang"], r["method"], r["op"]): r for r in rows}
    metrics = ["mid_char_split_rate", "boundary_F1", "morph_rate", "interior_surprisal_bits", "start_surprisal_bits", "bytes_per_patch"]
    summary = []
    for met in metrics:
        for ell in [1, 2, 3]:
            ls = [l for l in LANGS if LANG_ELL[l] == ell]
            bv = [idx[(l, "byte-BLT", f"{r:.0f}")][met] for l in ls for r in RATES]
            cv = [idx[(l, "CBLT-sqrtL", f"{r:.0f}")][met] for l in ls for r in RATES]
            bm, cm = sum(bv) / len(bv), sum(cv) / len(cv)
            summary.append([met, ell, ",".join(ls), f"{bm:.4f}", f"{cm:.4f}", f"{bm - cm:+.4f}"])
    with open(f"{out_dir}/intrinsic_ell_summary_matched.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["metric", "ell", "langs", "byte_mean", "cblt_mean", "delta_byte_minus_cblt"]); w.writerows(summary)
    print("=== ℓ別 Δ(byte−CBLT) [レート整合] ===")
    for s in summary:
        if s[0] in ("mid_char_split_rate", "boundary_F1", "morph_rate", "interior_surprisal_bits", "bytes_per_patch"):
            print(f"  {s[0]:24} ℓ={s[1]} byte={s[3]} cblt={s[4]} Δ={s[5]}")
    print("[OK] intrinsic_byte_vs_cblt_matched.csv / intrinsic_ell_summary_matched.csv")

if __name__ == "__main__":
    app()
