"""FLORES-200 devtest で en/ru/ja の境界AUC（1/2/3バイト梯子）。Chat次タスク。
- float32修正: aggregate前に ent.float() → out も float32 → cross-check ~0
- 起動時 golden test（規約B / BOS除外 / AUC両端 / り=27/√3）
- 毎sentence cross-check（実aggregate_char_entropy vs inline規約B, max|Δ|）
- 言語別 pooled AUC（standalone文・per-文で文頭除外, R2）
- gold: en/ru=空白語境界 / ja=MeCab形態素
- 長さスイープ: url文書グルーピング → 文脈長binでAUC
出力: 結果サマリを stdout + ファイル。"""
import math, json, sys, unicodedata
import numpy as np, torch, typer, fugashi
from collections import defaultdict
from bytelatent.entropy_model import load_entropy_model
from bytelatent.tokenizers.build_tokenizer import TokenizerArgs
from bytelatent.data.patcher import calculate_entropies, aggregate_char_entropy

app = typer.Typer()
TOK = "/gs/bs/tga-RLA/yoshida/blt_data/tokenizer/tokenizer.model"
FLORES = "/gs/bs/tga-RLA/yoshida/blt_data/flores"

# ---------- AUC (rank-based, efficient) ----------
def auc_rank(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, np.int8)
    npos = int(y.sum()); nneg = len(y) - npos
    if npos == 0 or nneg == 0: return float("nan")
    order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    i = 0; ss = s[order]
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]: j += 1
        if j > i: ranks[order[i:j + 1]] = (ranks[order[i]] + ranks[order[j]]) / 2
        i = j + 1
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)

# ---------- gold ----------
def gold_whitespace(text):
    bset = set(); newword = True
    for i, c in enumerate(text):
        if c.isspace(): newword = True
        elif newword: bset.add(i); newword = False
        else: newword = False
    return bset

def gold_mecab(text, tagger):
    # 複数行(文書連結)対応: 行ごとにMeCab→\nぶんオフセット。\nをMeCabがスキップしてもズレない
    bset = set(); base = 0
    for line in text.split("\n"):
        pos = 0
        for w in tagger(line): bset.add(base + pos); pos += len(w.surface)
        base += len(line) + 1  # +1 = \n
    return bset

# ---------- per-char H via 実aggregate_char_entropy (float32) + cross-check ----------
def per_char_H(text, model, tok, device, ln, boff=1, stats=None):
    """返り: list[(cidx, H_bits, ctx_char_pos)], gold判定用に char数も。cidx=0は除外しない(呼び側で)。"""
    tokens = torch.tensor(tok.encode(text), dtype=torch.long, device=device).unsqueeze(0)
    ent = calculate_entropies(tokens, model, 1, device)[0].float()   # ★float32化
    out = aggregate_char_entropy(ent, tokens, "sqrt")[0]             # out も float32
    e = ent[0]; raw = text.encode("utf-8"); n = len(raw)
    chars = []; bi = 0
    while bi < n:
        ell = 1
        while bi + ell < n and (raw[bi + ell] & 0xC0) == 0x80: ell += 1
        chars.append((bi, ell)); bi += ell
    res = []
    for cidx, (k, ell) in enumerate(chars):
        idx = boff + k - 1
        if idx < 0 or idx >= out.shape[0]: continue
        H = float(out[idx]) / ln
        if stats is not None and boff + k + ell - 1 <= e.shape[0]:   # cross-check
            Hi = float(e[boff + k - 1: boff + k + ell - 1].sum()) / math.sqrt(ell) / ln
            stats[0] = max(stats[0], abs(H - Hi))
        if math.isfinite(H): res.append((cidx, H))
    return res

def golden_test(model, tok, device, ln):
    # AUC両端
    assert auc_rank([2, 1], [1, 0]) == 1.0 and auc_rank([1, 2], [1, 0]) == 0.0 and auc_rank([1, 1], [1, 0]) == 0.5
    # 規約B: 合成entropy[t]=t で 実aggregate == inline(e[k:k+ℓ]/√ℓ) を全文字照合(bos込み, boff=1→out[k])
    from bytelatent.data.patcher import OFFSET
    bs = list("I have a りんご!".encode("utf-8"))
    tk = torch.tensor([1] + [b + OFFSET for b in bs] + [2], dtype=torch.long).unsqueeze(0)
    ent = torch.arange(tk.shape[1], dtype=torch.float32).unsqueeze(0)
    out = aggregate_char_entropy(ent, tk, "sqrt")[0]; e = ent[0]
    n = len(bs); bi = 0; maxd = 0.0; nchar = 0
    while bi < n:
        ell = 1
        while bi + ell < n and (bs[bi + ell] & 0xC0) == 0x80: ell += 1
        exp = float(e[bi:bi + ell].sum()) / math.sqrt(ell)   # 規約B: entropies[k:k+ℓ]/√ℓ
        maxd = max(maxd, abs(exp - float(out[bi]))); nchar += 1; bi += ell
    assert maxd < 1e-4, f"golden synthetic aggregate!=inline max|Δ|={maxd}"
    return f"PASS (AUC両端 / synthetic aggregate==inline {nchar}文字 max|Δ|={maxd:.1e})"

@app.command()
def main(ckpt_dir: str = typer.Option(...), out_file: str = typer.Option(...), device: str = typer.Option("cuda")):
    model, _ = load_entropy_model(ckpt_dir, f"{ckpt_dir}/consolidated.pth", device=device); model = model.eval()
    tok = TokenizerArgs(name="blt", init_kwargs={"bpe_tokenizer_path": TOK}).build()
    tagger = fugashi.Tagger(); ln = math.log(2); boff = 1
    L = open(out_file, "w")
    def log(s): print(s); L.write(s + "\n"); L.flush()

    log(f"=== golden test: {golden_test(model, tok, device, ln)} ===")
    GOLD = {"en": gold_whitespace, "ru": gold_whitespace, "ja": lambda t: gold_mecab(t, tagger)}
    cc = [0.0]

    # ---------- (a) 言語別 pooled AUC (standalone文) ----------
    log("\n=== (a) 言語別 pooled AUC (FLORES devtest, standalone文, R2文頭除外) ===")
    perlang = {}
    for lang in ["en", "ru", "ja"]:
        docs = [json.loads(l) for l in open(f"{FLORES}/flores_{lang}_devtest.jsonl")]
        H_all, g_all = [], []
        for d in docs:
            text = unicodedata.normalize("NFC", d["text"])
            if len(text) < 2: continue
            gset = GOLD[lang](text)
            for cidx, H in per_char_H(text, model, tok, device, ln, boff, cc):
                if cidx == 0: continue            # R2
                H_all.append(H); g_all.append(1 if cidx in gset else 0)
        a = auc_rank(H_all, g_all); nb = sum(g_all); n = len(g_all)
        posH = np.mean([h for h, g in zip(H_all, g_all) if g]); negH = np.mean([h for h, g in zip(H_all, g_all) if not g])
        bytelen = {"en": 1, "ru": 2, "ja": 3}[lang]
        perlang[lang] = a
        log(f"  {lang}(≈{bytelen}byte/字): AUC={a:.4f}  n={n} 境界率={nb/n:.2f}  境界H={posH:.2f}/非境界H={negH:.2f}")
    log(f"  → 梯子: en={perlang['en']:.3f} / ru={perlang['ru']:.3f} / ja={perlang['ja']:.3f}")

    # ---------- (b) cross-check ----------
    log(f"\n=== (b) cross-check (float32, 実aggregate vs inline規約B) max|Δ| = {cc[0]:.3e} bits "
        f"({'PASS <1e-6' if cc[0] < 1e-6 else 'まだ>1e-6'}) ===")

    # ---------- (c) 長さスイープ (url文書グルーピング, 文脈長bin) ----------
    log("\n=== (c) 長さスイープ: url文書内の文脈長(char)別 AUC ===")
    bins = [(0, 64), (64, 256), (256, 1024), (1024, 10 ** 9)]
    for lang in ["en", "ru", "ja"]:
        docs = [json.loads(l) for l in open(f"{FLORES}/flores_{lang}_devtest.jsonl")]
        byurl = defaultdict(list)
        for d in docs: byurl[d.get("url") or d["id"]].append(d["text"])
        binned = {b: ([], []) for b in bins}
        for url, sents in byurl.items():
            doctext = unicodedata.normalize("NFC", "\n".join(sents))
            if len(doctext) < 2: continue
            gset = GOLD[lang](doctext)
            for cidx, H in per_char_H(doctext, model, tok, device, ln, boff, cc):
                if cidx == 0: continue
                for (lo, hi) in bins:
                    if lo <= cidx < hi:
                        binned[(lo, hi)][0].append(H); binned[(lo, hi)][1].append(1 if cidx in gset else 0); break
        row = " / ".join(f"[{lo}-{hi if hi < 10**8 else '∞'}]:{auc_rank(*binned[(lo,hi)]):.3f}(n={len(binned[(lo,hi)][1])})" for (lo, hi) in bins)
        log(f"  {lang}: {row}")
    log(f"\n(cross-check 累積 max|Δ| = {cc[0]:.3e})")
    L.close()

if __name__ == "__main__":
    app()
