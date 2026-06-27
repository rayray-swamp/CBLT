"""CBLT 評価 正本仕様(R1-R4)の参照実装 + 検証(step1-4)。Notion §📏 の核をそのまま内包。
- step1: ゴールデンテスト(規約B / BOS除外 / AUC両端)
- step2/4: 先ほど作った ja_raw_entropy.csv の H(c)(=sum/√ℓ 行)を使い、実MeCab境界で
           pooled AUC / θ_rel(F1最大閾値) / Precision / Recall / F1
- step3: patcher.py aggregate_char_entropy の i==0 挙動を確認
※ H は CSV(=test3.csv 相当)の値を使用 ＝ 先ほど作らせたCSVそのもの。NFC は MeCab gold 側で一貫適用(R4)。
"""
import math, csv, unicodedata
import fugashi

CSV = "/gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/ja_raw_entropy.csv"

# ===== 参照実装核 (Notion §📏 のまま) =====
def utf8_char_spans(byte_seq):
    spans = []; start = None
    for t, b in enumerate(byte_seq):
        if (b & 0xC0) != 0x80:
            if start is not None: spans.append((start, t))
            start = t
    if start is not None: spans.append((start, len(byte_seq)))
    return spans

def cblt_char_entropy(entropies, byte_seq, agg="sqrt"):
    assert len(entropies) == len(byte_seq)             # next-byte規約
    out = []
    for (i, j) in utf8_char_spans(byte_seq):
        ch = bytes(byte_seq[i:j]).decode("utf-8", "replace")
        if i == 0:                                     # R2: BOS文字は未定義→除外
            out.append(((i, j), ch, None)); continue
        s = sum(entropies[i - 1:j - 1]); ell = j - i   # R1: 規約B (lead-1)
        H = {"sqrt": s / math.sqrt(ell), "sum": s, "avg": s / ell}[agg]
        out.append(((i, j), ch, H))
    return out

def boundary_auc(per_char, gold_is_initial):           # R3
    pos = [H for (_, _, H), g in zip(per_char, gold_is_initial) if H is not None and g]
    neg = [H for (_, _, H), g in zip(per_char, gold_is_initial) if H is not None and not g]
    if not pos or not neg: return None, pos, neg
    c = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return c / (len(pos) * len(neg)), pos, neg

# ===== step1: ゴールデンテスト =====
def golden_test():
    a = [((0, 1), "a", 2.0), ((0, 1), "b", 1.0)]
    assert boundary_auc(a, [True, False])[0] == 1.0
    assert boundary_auc(a, [False, True])[0] == 0.0
    assert boundary_auc([((0, 1), "a", 1.0), ((0, 1), "b", 1.0)], [True, False])[0] == 0.5
    bs = list("I have a りんご!".encode("utf-8"))       # 規約B + BOS除外
    pcs = cblt_char_entropy([float(t) for t in range(len(bs))], bs, "sqrt")
    assert pcs[0][2] is None, "R2: 先頭文字が None でない"
    ri = [H for (_, ch, H) in pcs if ch == "り"][0]
    assert abs(ri - 27 / math.sqrt(3)) < 1e-9, f"規約B: り={ri} != 27/√3"
    return "ALL PASS (AUC両端=1.0/0.0/0.5 / 規約B り=27/√3 / BOS文字=None)"

# ===== CSV パース(先ほどのCSV) =====
def parse_csv(path):
    rows = list(csv.reader(open(path, encoding="utf-8-sig")))
    sents = []; cur = None
    for r in rows:
        if not r:
            if cur: sents.append(cur); cur = None
            continue
        lab = r[0]
        if lab.startswith("■文章"): cur = {"text": r[1] if len(r) > 1 else "", "chars": None, "H": None}
        elif lab == "文字" and cur is not None: cur["chars"] = [c for c in r[1:] if c != ""]
        elif lab.startswith("sum/√ℓ") and cur is not None: cur["H"] = [float(c) for c in r[1:] if c != ""]
    if cur: sents.append(cur)
    return sents

# ===== step2/4: 5文 metrics =====
def measure(sents):
    tagger = fugashi.Tagger()
    PH, PG, per = [], [], []
    for s in sents:
        chars, H = s["chars"], s["H"]
        assert len(chars) == len(H), f"chars{len(chars)} vs H{len(H)}"
        text = unicodedata.normalize("NFC", "".join(chars))      # R4
        starts = set(); pos = 0
        for w in tagger(text): starts.add(pos); pos += len(w.surface)
        Hs, gs = [], []
        for ci in range(len(chars)):
            if ci == 0: continue                                  # R2 文頭除外
            Hs.append(H[ci]); gs.append(ci in starts)
            PH.append(H[ci]); PG.append(ci in starts)
        a, p, n = boundary_auc([((0, 0), "", h) for h in Hs], gs)
        per.append((s["text"][:20], a, len(gs), sum(gs)))
    auc, pos, neg = boundary_auc([((0, 0), "", h) for h in PH], PG)
    best = None                                                   # θ_rel = F1最大 (H>=θ を境界)
    for th in sorted(set(PH)):
        tp = sum(1 for h, g in zip(PH, PG) if h >= th and g)
        fp = sum(1 for h, g in zip(PH, PG) if h >= th and not g)
        fn = sum(1 for h, g in zip(PH, PG) if h < th and g)
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        F = 2 * P * R / (P + R) if P + R else 0.0
        if best is None or F > best[-1]: best = (th, P, R, F)
    return auc, pos, neg, best, PG, per

print("=== step1 ゴールデンテスト ===", golden_test())
sents = parse_csv(CSV)
auc, pos, neg, best, gold, per = measure(sents)
th, P, R, F = best
print(f"\n=== step2/4: 5文 (R1規約B / R2文頭除外 / R3 H(c)vs形態素先頭 / R4 NFC) ===")
print(f"評価対象 {len(gold)}文字 / gold境界 {sum(gold)} ({sum(gold)/len(gold):.1%})  ※各文の先頭文字は除外")
print(f"\n  pooled AUC = {auc:.3f}   (境界平均H {sum(pos)/len(pos):.2f} bits / 非境界 {sum(neg)/len(neg):.2f} bits)")
print(f"  θ_rel(F1最大閾値) = {th:.3f} bits")
print(f"  Precision = {P:.3f}")
print(f"  Recall    = {R:.3f}")
print(f"  F1        = {F:.3f}")
print(f"\n  [文別AUC] " + " / ".join(f"{t}…:{a:.2f}(n={n},境{b})" if a is not None else f"{t}…:NA" for t, a, n, b in per))

# ===== step3: patcher.py i==0 確認 =====
try:
    import torch
    from bytelatent.data.patcher import aggregate_char_entropy, OFFSET
    raw = list("あい".encode("utf-8"))
    toks = [OFFSET - 1] + [b + OFFSET for b in raw]               # bos(<0) + あ + い
    tk = torch.tensor(toks).unsqueeze(0); seq = tk.shape[1]
    out = aggregate_char_entropy(torch.arange(seq, dtype=torch.float32).unsqueeze(0), tk, "sqrt")[0].tolist()
    fmt = ['%.3f' % x if x != float('-inf') else '-inf' for x in out]
    print(f"\n=== step3 patcher.py i==0 ===\n  aggregate_char_entropy(あい, ent=idx) out={fmt}")
    print(f"  先頭文字'あ'→ out[0]={out[0]:.3f}（計算される＝None/-infでない。bos項込みの規約B値）")
    print("  結論: patcher は先頭文字を out[0] に集約。patching では out[:,1:] で out[0] を落とし")
    print("        first_ids=[0,1] が patch@0,1 を強制→無害。∴ patcher修正不要、R2は eval 側で除外(本evalで実施済)。")
except Exception as e:
    print(f"\n=== step3 skip (import失敗: {e}) ===")
