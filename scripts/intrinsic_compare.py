"""byte-BLT vs CBLT-√ℓ 内在比較（本体GO前 effect size 事前推定, Chat確定仕様）。
FLORES+ 8言語 × 動作点{4,5,6,7,8}、両 monotonic、共通θ=本データ較正値。per-byte e は凍結285k。
指標(言語×方式×動作点):
 (A) mid_char_split_rate: patch境界が多バイト文字の内部(継続バイト)に落ちる割合。byte-BLTのみ>0、CBLT=0。
 (B) boundary_F1 + morph_rate vs gold: byte境界は含む文字のleadへ写像→文字単位。c=0除外(R2)。
 (C) BPB proxy(示唆的・本体BPBでない): patch内部(非先頭バイト)平均surprisal(bits) + patch先頭surprisal(bits)。
 (D) patch分布: bytes/patch・chars/patch・1字patch率。
出力: intrinsic_byte_vs_cblt.csv + ℓ別要約 intrinsic_ell_summary.csv。"""
import math, json, csv, torch, typer
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
# 本データ較正θ (nats), rate 4/5/6/7/8
THETA = {"byte": dict(zip(RATES, [0.3008, 0.5000, 0.6367, 0.7344, 0.8516])),
         "sqrt": dict(zip(RATES, [-0.0035, 0.0067, 0.0937, 0.2439, 0.3934]))}

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

@app.command()
def main(ckpt_dir: str = typer.Option(...), out_dir: str = typer.Option(...), device: str = typer.Option("cuda")):
    model, _ = load_entropy_model(ckpt_dir, f"{ckpt_dir}/consolidated.pth", device=device); model = model.eval()
    tok = TokenizerArgs(name="blt", init_kwargs={"bpe_tokenizer_path": TOK}).build()
    import fugashi; tagger = fugashi.Tagger()
    ln = math.log(2)
    # acc[(lang,method,rate)] = dict of running sums
    acc = defaultdict(lambda: defaultdict(float))
    for lang in LANGS:
        gtype = GOLD[lang]
        for d in (json.loads(l) for l in open(f"{FP}/floresplus_{lang}_devtest.jsonl")):
            text = d["text"]
            if len(text) < 2: continue
            toks = tok.encode(text)
            if len(toks) > 8192: toks = toks[:8192]
            e = calculate_entropies(torch.tensor(toks, dtype=torch.long, device=device).unsqueeze(0), model, 1, device)[0].float()[0]
            raw = text.encode("utf-8"); n = min(len(raw), e.shape[0]); raw = raw[:n]
            eb = [float(e[b]) for b in range(n)]   # nats
            # 文字分割
            chars = []; bi = 0
            while bi < n:
                ell = 1
                while bi + ell < n and (raw[bi + ell] & 0xC0) == 0x80: ell += 1
                chars.append((bi, ell)); bi += ell
            C = len(chars)
            if C < 2: continue
            lead = [k for k, _ in chars]                       # 各文字の lead byte index
            char_of_byte = [0] * n
            for ci, (k, el) in enumerate(chars):
                for b in range(k, k + el): char_of_byte[b] = ci
            H = {"sqrt": [sum(eb[k:k + el]) / math.sqrt(el) for k, el in chars]}
            gsp = set(gold_spans(text, gtype, tagger)); gstart = set(s for s, _ in gsp)
            goldb = gstart - {0}
            for method in ["byte", "sqrt"]:
                for rate in RATES:
                    th = THETA[method][rate]
                    if method == "byte":
                        B = [0] + [b for b in range(1, n) if eb[b] - eb[b - 1] > th]   # byte境界
                        starts_b = B                                                   # byte開始位置
                        pred_char = set(char_of_byte[b] for b in B)
                        # patches(byte-span)
                        ends = B[1:] + [n]
                        spans_byte = list(zip(B, ends))
                    else:
                        Hc = H["sqrt"]
                        Bc = [0] + [c for c in range(1, C) if Hc[c] - Hc[c - 1] > th]   # char境界
                        starts_b = [lead[c] for c in Bc]
                        pred_char = set(Bc)
                        ends_c = Bc[1:] + [C]
                        spans_byte = [(lead[cs], (lead[ce] if ce < C else n)) for cs, ce in zip(Bc, ends_c)]
                    a = acc[(lang, method, rate)]
                    npat = len(spans_byte)
                    a["npatch"] += npat; a["nbytes"] += n; a["nchars"] += C
                    # (A) mid_char_split: 境界が継続バイト
                    a["midsplit"] += sum(1 for b in starts_b if b > 0 and (raw[b] & 0xC0) == 0x80)
                    # (B) F1 (c=0除外) + morph
                    pc = pred_char - {0}
                    a["tp"] += len(pc & goldb); a["npred"] += len(pc); a["ngold"] += len(goldb)
                    for bs, be in spans_byte:
                        cs = char_of_byte[bs]; ce = char_of_byte[be - 1] + 1
                        if (cs, ce) in gsp: a["morph"] += 1
                        # (D) 1字patch: span内の文字lead数==1
                        leads_in = sum(1 for c in range(cs, ce) if lead[c] >= bs and lead[c] < be)
                        if leads_in == 1: a["onechar"] += 1
                        # (C) 内部 surprisal(非先頭バイト) + 先頭 surprisal
                        a["start_s"] += eb[bs]; a["nstart"] += 1
                        if be - bs > 1:
                            a["interior_s"] += sum(eb[bs + 1:be]); a["ninterior"] += (be - bs - 1)
        print(f"[OK] {lang}")

    # CSV 出力
    rows = []
    for lang in LANGS:
        for method in ["byte", "sqrt"]:
            for rate in RATES:
                a = acc[(lang, method, rate)]; npat = max(1.0, a["npatch"])
                P = a["tp"] / a["npred"] if a["npred"] else 0.0
                R = a["tp"] / a["ngold"] if a["ngold"] else 0.0
                F1 = 2 * P * R / (P + R) if P + R else 0.0
                rows.append({
                    "lang": lang, "ell": LANG_ELL[lang], "method": "byte-BLT" if method == "byte" else "CBLT-sqrtL",
                    "op": f"{rate:.0f}", "bytes_per_patch": a["nbytes"] / npat, "chars_per_patch": a["nchars"] / npat,
                    "one_char_patch_rate": a["onechar"] / npat, "mid_char_split_rate": a["midsplit"] / npat,
                    "boundary_F1": F1, "morph_rate": a["morph"] / npat,
                    "interior_surprisal_bits": (a["interior_s"] / a["ninterior"] / ln) if a["ninterior"] else 0.0,
                    "start_surprisal_bits": (a["start_s"] / a["nstart"] / ln) if a["nstart"] else 0.0})
    cols = ["lang", "ell", "method", "op", "bytes_per_patch", "chars_per_patch", "one_char_patch_rate",
            "mid_char_split_rate", "boundary_F1", "morph_rate", "interior_surprisal_bits", "start_surprisal_bits"]
    with open(f"{out_dir}/intrinsic_byte_vs_cblt.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow({k: (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k]) for k in cols})

    # ℓ別要約: 各指標の Δ(byte−CBLT) を ℓ=1/2/3 で平均(動作点平均)
    metrics = ["mid_char_split_rate", "boundary_F1", "morph_rate", "interior_surprisal_bits", "start_surprisal_bits", "bytes_per_patch"]
    idx = {(r["lang"], r["method"], r["op"]): r for r in rows}
    summary = []
    for met in metrics:
        for ell in [1, 2, 3]:
            langs = [l for l in LANGS if LANG_ELL[l] == ell]
            bvals, cvals = [], []
            for l in langs:
                for rate in RATES:
                    bvals.append(idx[(l, "byte-BLT", f"{rate:.0f}")][met])
                    cvals.append(idx[(l, "CBLT-sqrtL", f"{rate:.0f}")][met])
            bm = sum(bvals) / len(bvals); cm = sum(cvals) / len(cvals)
            summary.append([met, ell, ",".join(langs), f"{bm:.4f}", f"{cm:.4f}", f"{bm - cm:+.4f}"])
    with open(f"{out_dir}/intrinsic_ell_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["metric", "ell", "langs", "byte_mean", "cblt_mean", "delta_byte_minus_cblt"])
        w.writerows(summary)
    print("=== ℓ別 Δ(byte−CBLT) 要約(動作点平均) ===")
    for s in summary:
        if s[0] in ("mid_char_split_rate", "boundary_F1", "interior_surprisal_bits"):
            print(f"  {s[0]:24} ℓ={s[1]} byte={s[3]} cblt={s[4]} Δ={s[5]}")
    print("[OK] intrinsic_byte_vs_cblt.csv / intrinsic_ell_summary.csv")

if __name__ == "__main__":
    app()
