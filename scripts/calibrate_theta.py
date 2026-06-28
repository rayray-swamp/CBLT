"""本データ(largecorpus)で全方式の patching θ を較正。Chat依頼。
方式(全て monotonic ΔH>θ, 各doc先頭は強制patch):
  byte-BLT : per-byte  ΔH_byte(b)=e[b]-e[b-1]
  CBLT-√ℓ  : 文字境界 H(c)=Σe[k:k+ℓ]/√ℓ (規約B) の ΔH
  Sum      : H(c)=Σe[k:k+ℓ]
  Avg      : H(c)=Σe[k:k+ℓ]/ℓ
e[b]=byte b の surprisal (nats, 生)。θ も nats(Chat既存CBLT θ_nats と整合)。
共通θ二分探索 → pooled bytes/patch = {4,5,6,7,8}。出力: 方式×レート → θ・実現pooled bpp・per言語bpp。
+ CBLT-√ℓ の既存FLORES+θ {-0.10,0.00,0.12,0.29,0.46} を本データで検証(ドリフト)。"""
import math, json, csv, random, torch, typer
import numpy as np
from collections import defaultdict
from bytelatent.entropy_model import load_entropy_model
from bytelatent.tokenizers.build_tokenizer import TokenizerArgs

app = typer.Typer()
TOK = "/gs/bs/tga-RLA/yoshida/blt_data/tokenizer/tokenizer.model"
CORPUS = "/gs/bs/tga-RLA/yoshida/blt_data/largecorpus/corpus/corpus.chunk.00.jsonl"
LANGS = ["en", "ru", "ja", "ar", "zh", "ko", "hi", "th"]
RATES = [4.0, 5.0, 6.0, 7.0, 8.0]
CBLT_EXISTING_THETA = [-0.10, 0.00, 0.12, 0.29, 0.46]  # FLORES+較正(nats)

def sample_corpus(n_per_lang, max_chars, seed=42, line_cap=400000):
    buf = defaultdict(list)
    need = set(LANGS)
    for i, line in enumerate(open(CORPUS)):
        if i >= line_cap or not need: break
        try: d = json.loads(line)
        except Exception: continue
        lg = d.get("lang")
        if lg in need:
            t = d.get("text", "")[:max_chars]
            if len(t) >= 20: buf[lg].append(t)
            if len(buf[lg]) >= n_per_lang: need.discard(lg)
    return {lg: buf[lg] for lg in LANGS if buf[lg]}

def per_byte_nats(text, model, tok, device, boff=1):
    from bytelatent.data.patcher import calculate_entropies
    toks = tok.encode(text)
    if len(toks) > 8192: toks = toks[:8192]   # entropy_model.max_seqlen=8192 (sliding_window=512)
    tokens = torch.tensor(toks, dtype=torch.long, device=device).unsqueeze(0)
    e = calculate_entropies(tokens, model, 1, device)[0].float()[0]  # nats(生), token index
    raw = text.encode("utf-8"); n = len(raw)
    eb = [float(e[b]) for b in range(min(n, e.shape[0]))]   # e[b]=byte b の surprisal
    return raw[:len(eb)], eb

def char_H(raw, eb, mode):
    """文字境界で集約した H 系列。返り: (H list, ell list)。"""
    n = len(eb); H = []; ells = []; bi = 0
    while bi < n:
        ell = 1
        while bi + ell < n and (raw[bi + ell] & 0xC0) == 0x80: ell += 1
        s = sum(eb[bi:bi + ell])
        H.append(s / math.sqrt(ell) if mode == "sqrt" else (s if mode == "sum" else s / ell))
        ells.append(ell); bi += ell
    return H, ells

def calib(dH_per_lang, bytes_per_lang, ndoc_per_lang, target):
    """共通θ二分探索: pooled bytes/patch = target。返り θ, realized pooled bpp, per-lang bpp。"""
    alld = np.concatenate([dH_per_lang[lg] for lg in dH_per_lang]) if dH_per_lang else np.array([0.0])
    TB = sum(bytes_per_lang.values()); ND = sum(ndoc_per_lang.values())
    def bpp(th): return TB / (ND + int((alld > th).sum()))
    lo, hi = float(alld.min()) - 1, float(alld.max()) + 1
    for _ in range(50):
        mid = (lo + hi) / 2
        if bpp(mid) < target: lo = mid
        else: hi = mid
    th = (lo + hi) / 2
    perlang = {}
    for lg in dH_per_lang:
        npat = ndoc_per_lang[lg] + int((dH_per_lang[lg] > th).sum())
        perlang[lg] = bytes_per_lang[lg] / max(1, npat)
    return th, bpp(th), perlang

@app.command()
def main(ckpt_dir: str = typer.Option(...), out_dir: str = typer.Option(...),
         n_per_lang: int = typer.Option(150), max_chars: int = typer.Option(4000), device: str = typer.Option("cuda")):
    model, _ = load_entropy_model(ckpt_dir, f"{ckpt_dir}/consolidated.pth", device=device); model = model.eval()
    tok = TokenizerArgs(name="blt", init_kwargs={"bpe_tokenizer_path": TOK}).build()
    ln = math.log(2)
    samp = sample_corpus(n_per_lang, max_chars)
    print("サンプル: " + " ".join(f"{lg}={len(samp[lg])}" for lg in samp))
    methods = ["byte", "sqrt", "sum", "avg"]  # byte=byte-BLT
    dH = {m: defaultdict(list) for m in methods}
    nbytes = defaultdict(int); ndoc = defaultdict(int)
    for lg, texts in samp.items():
        for text in texts:
            raw, eb = per_byte_nats(text, model, tok, device)
            if len(eb) < 3: continue
            nbytes[lg] += len(eb); ndoc[lg] += 1
            # byte-BLT: ΔH per byte (b>=1)
            dH["byte"][lg].extend(eb[b] - eb[b - 1] for b in range(1, len(eb)))
            # char-level methods
            for m in ["sqrt", "sum", "avg"]:
                H, _ = char_H(raw, eb, m)
                dH[m][lg].extend(H[c] - H[c - 1] for c in range(1, len(H)))
    for m in methods:
        for lg in dH[m]: dH[m][lg] = np.array(dH[m][lg])

    rows = []
    name = {"byte": "byte-BLT", "sqrt": "CBLT-sqrtL", "sum": "Sum", "avg": "Avg"}
    for m in methods:
        for T in RATES:
            th, rb, pl = calib(dH[m], nbytes, ndoc, T)
            rows.append([name[m], f"{T:.1f}", f"{th:.4f}", f"{th/ln:.4f}", f"{rb:.3f}"] +
                        [f"{pl.get(lg, float('nan')):.2f}" for lg in LANGS])
        print(f"[OK] {name[m]} 較正完了")
    with open(f"{out_dir}/theta_calibration.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["method", "target_bpp", "theta_nats", "theta_bits", "realized_pooled_bpp"] + [f"bpp_{lg}" for lg in LANGS])
        w.writerows(rows)
    # CBLT-√ℓ 既存θ(FLORES+)を本データで検証(ドリフト)
    drift = []
    alld = np.concatenate([dH["sqrt"][lg] for lg in dH["sqrt"]]); TB = sum(nbytes.values()); ND = sum(ndoc.values())
    for th in CBLT_EXISTING_THETA:
        rb = TB / (ND + int((alld > th).sum())); drift.append([f"{th:.4f}", f"{rb:.3f}"])
    with open(f"{out_dir}/cblt_theta_drift.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["existing_theta_nats(FLORES+)", "realized_bpp_on_corpus"]); w.writerows(drift)
    print("=== CBLT-√ℓ 既存θ→本データ bytes/patch (ドリフト) ===")
    for r in drift: print(f"  θ={r[0]} → bpp={r[1]}")
    print(f"[OK] theta_calibration.csv / cblt_theta_drift.csv")

if __name__ == "__main__":
    app()
