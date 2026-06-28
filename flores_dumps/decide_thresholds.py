#!/usr/bin/env python3
"""
decide_thresholds.py — CBLT monotonic patching の閾値(θ)を動作点ごとに確定する証拠スクリプト.

背景 / なぜこの設計か
---------------------------------
* patching 方式 = **monotonic** (`ΔH(c)=H(c)-H(c-1) > θ`, 各文 c=0 は強制パッチ開始).
  - global(`H>θ`) と同粒度比較で全言語 F1 優位, hybrid(`(ΔH>t_add)∧(H>floor)`) は
    粒度拘束下で monotonic に縮退 or 悪化 → 絶対floorは無効. 探索軸は ΔH 一本.
* θ は **全言語共通の単一値**. 計算予算 = bytes/patch を動作点として決める.
  動作点 = {4.0, 5.0, 6.0, 7.0, 8.0} (等間隔, 粗い側=本番域 6-8 を厚く).
* 各動作点の θ は「**FLORES+ 全8言語を pool した bytes/patch が目標値**になる単一 θ」
  を二分探索で確定 (再現可能・citable). 単一θなので per-言語の実現 bpp はばらつく(報告する).
* F1/morph_rate は文脈情報(各θで境界がどれだけ語/形態素に一致するか). 
  **最終的にどの動作点を採るかは下流BPB**(matched compute)で決める. F1最大化で粗い側を
  選ぶのは罠(粗いと境界数が減り precision が機械的に上がる)ため.

入力 : flores_dumps/flores_dump_<lang>.csv (per-char dump, BOS/EOS込, float32)
出力 : flores_dumps/threshold_decision.csv (動作点 × 言語 の確定θと実現値)

gold境界: 空白=en/ru/ar/ko/hi, MeCab(fugashi)=ja, jieba=zh, pythainlp(newmm)=th.
規約: H/θ とも bits. F1/P/R は c=0 を除外(R2). bpp は c=0 patch も含む.
"""
import csv, math, os
from collections import OrderedDict
import fugashi, jieba
from pythainlp.tokenize import word_tokenize as th_tok

DUMP_DIR = os.path.join(os.path.dirname(__file__), "..", "flores_dumps")
LANGS = ["en", "ru", "ar", "ja", "zh", "ko", "hi", "th"]
WHITESPACE = {"en", "ru", "ar", "ko", "hi"}
TARGET_BPP = [4.0, 5.0, 6.0, 7.0, 8.0]
# CulturaX 学習ミックス重み (entropy_d1.yaml) — 本番compute参照用の mix-weighted bpp に使用
MIX_W = {"en":100, "ru":200, "ar":200, "ja":300, "zh":300, "ko":300, "hi":300, "th":300}

_tagger = fugashi.Tagger()

def gold_tokens(lang, text):
    """言語別 gold 分割 → トークン表層のリスト(境界算出と morph 完全一致に使う)."""
    if lang in WHITESPACE:
        return text.split()                       # 空白語
    if lang == "ja":
        return [w.surface for w in _tagger(text)]
    if lang == "zh":
        return [w for w in jieba.cut(text, cut_all=False) if w.strip()]
    if lang == "th":
        return [w for w in th_tok(text, engine="newmm") if w.strip()]
    raise ValueError(lang)

def gold_starts(lang, chars):
    """gold境界(=各トークン先頭)の文字index集合."""
    text = "".join(chars)
    starts, pos = set(), 0
    for tok in gold_tokens(lang, text):
        idx = text.find(tok, pos)
        if idx < 0:
            pos += 0; continue
        starts.add(idx); pos = idx + len(tok)
    return starts

def load(lang):
    sents = OrderedDict(); cur = None
    with open(os.path.join(DUMP_DIR, f"flores_dump_{lang}.csv"), encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f); next(r)
        for row in r:
            if len(row) < 5: continue
            sid, ch = row[0], row[1]
            if ch == "<BOS>":
                cur = {"ch": [], "H": [], "nb": []}; sents[sid] = cur; continue
            if ch == "<EOS>" or cur is None: continue
            try: H = float(row[4])
            except ValueError: continue
            cur["ch"].append(ch); cur["H"].append(H); cur["nb"].append(len(ch.encode("utf-8")))
    for s in sents.values():
        H = s["H"]
        s["dH"] = [9e9] + [H[i] - H[i - 1] for i in range(1, len(H))]  # c=0 強制
        s["gold"] = gold_starts(lang, s["ch"])
        s["gtoks"] = set(gold_tokens(lang, "".join(s["ch"])))
    return sents

def patch_starts(s, th):
    return {0} | {c for c in range(1, len(s["H"])) if s["dH"][c] > th}

def lang_metrics(sents, th):
    tp = fp = fn = 0; tb = tpat = 0; mm = 0; tot = 0
    for s in sents.values():
        st = patch_starts(s, th)
        pred, gold = st - {0}, s["gold"] - {0}      # R2: c=0 除外
        tp += len(pred & gold); fp += len(pred - gold); fn += len(gold - pred)
        ss = sorted(st); tpat += len(ss); tb += sum(s["nb"])
        for k, b in enumerate(ss):                  # morph_rate: patch表層 == gold1トークン
            e = ss[k + 1] if k + 1 < len(ss) else len(s["ch"]); tot += 1
            surf = "".join(s["ch"][b:e]).lstrip(" ") if True else "".join(s["ch"][b:e])
            if surf in s["gtoks"]: mm += 1
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return dict(bpp=tb / tpat, F1=F, P=P, R=R, morph=mm / tot, bytes=tb, patches=tpat,
                tp=tp, fp=fp, fn=fn)

DATA = {lg: load(lg) for lg in LANGS}

def pooled_bpp(th):
    tb = sum(sum(sum(s["nb"]) for s in DATA[lg].values()) for lg in LANGS)
    tp = sum(len(patch_starts(s, th)) for lg in LANGS for s in DATA[lg].values())
    return tb / tp

def tune_theta(target_bpp):
    lo, hi = -5.0, 5.0                              # 負θも許す(ja等の自然床より細かい動作点用)
    for _ in range(60):
        mid = (lo + hi) / 2
        if pooled_bpp(mid) < target_bpp: lo = mid   # 細すぎ→θ上げ
        else: hi = mid
    return (lo + hi) / 2

# ---- 確定 ----
rows = []
print(f"{'bpp目標':>6} {'θ_bits':>8} {'θ_nats':>8} | " +
      " | ".join(f"{lg} F1/bpp" for lg in LANGS) + " | pooledF1 mixbpp")
LN2 = math.log(2)
for T in TARGET_BPP:
    th = tune_theta(T)
    per = {lg: lang_metrics(DATA[lg], th) for lg in LANGS}
    tp = sum(per[lg]["tp"] for lg in LANGS); fp = sum(per[lg]["fp"] for lg in LANGS)
    fn = sum(per[lg]["fn"] for lg in LANGS)
    pP = tp / (tp + fp); pR = tp / (tp + fn); pooledF1 = 2 * pP * pR / (pP + pR)
    # mix-weighted bpp = (Σ w) / (Σ w/bpp_lang)  (byte比率を学習ミックス重みで近似)
    mixbpp = sum(MIX_W.values()) / sum(MIX_W[lg] / per[lg]["bpp"] for lg in LANGS)
    realized_pooled = pooled_bpp(th)
    cells = " | ".join(f"{per[lg]['F1']:.3f}/{per[lg]['bpp']:.1f}" for lg in LANGS)
    print(f"{T:6.1f} {th:8.4f} {th*LN2:8.4f} | {cells} | {pooledF1:.3f}   {mixbpp:.2f}")
    for lg in LANGS:
        m = per[lg]
        rows.append([T, round(realized_pooled, 3), round(th, 4), round(th * LN2, 4),
                     round(mixbpp, 3), lg, round(m["bpp"], 3), round(m["F1"], 4),
                     round(m["P"], 4), round(m["R"], 4), round(m["morph"], 4),
                     m["patches"], round(pooledF1, 4)])

OUT = os.path.join(DUMP_DIR, "threshold_decision.csv")
with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["target_bpp", "realized_pooled_bpp", "theta_bits", "theta_nats",
                "mix_weighted_bpp", "lang", "lang_bpp", "F1", "precision", "recall",
                "morph_rate", "n_patches", "pooled_F1"])
    w.writerows(rows)
print(f"\n-> {OUT}")
