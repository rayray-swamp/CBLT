"""コーパス再測（本来の step4）。5文と同じ R1〜R4 を held-out 全docに適用。
H(c)=実 aggregate_char_entropy(規約B)。R2=各docの先頭文字のみ除外。R4=NFC。gold=MeCab形態素先頭。
出力: pooled AUC / θ_rel(F1最大) / Precision / Recall / F1（効率版: rank-AUC + F1 sweep）。"""
import math, json, torch, typer, unicodedata
import numpy as np, fugashi
from bytelatent.entropy_model import load_entropy_model
from bytelatent.tokenizers.build_tokenizer import TokenizerArgs
from bytelatent.data.patcher import calculate_entropies, aggregate_char_entropy

app = typer.Typer()
TOK = "/gs/bs/tga-RLA/yoshida/blt_data/tokenizer/tokenizer.model"

def auc_rank(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, np.int8)
    npos = int(y.sum()); nneg = len(y) - npos
    if npos == 0 or nneg == 0: return float("nan")
    order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    i = 0; ss = s[order]
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]: j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2
            ranks[order[i:j + 1]] = avg
        i = j + 1
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)

def f1_sweep(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, np.int8); P = int(y.sum())
    if P == 0: return None
    order = np.argsort(-s, kind="mergesort"); ys = y[order]; ss = s[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    prec = tp / (tp + fp); rec = tp / P
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    k = int(np.argmax(f1))
    return float(ss[k]), float(prec[k]), float(rec[k]), float(f1[k])

@app.command()
def main(ckpt_dir: str = typer.Option(...), jsonl: str = typer.Option(...),
         device: str = typer.Option("cuda"), max_chars: int = typer.Option(120000),
         min_toks: int = typer.Option(64), lang_prefix: str = typer.Option("ja")):
    model, _ = load_entropy_model(ckpt_dir, f"{ckpt_dir}/consolidated.pth", device=device); model = model.eval()
    tok = TokenizerArgs(name="blt", init_kwargs={"bpe_tokenizer_path": TOK}).build()
    tagger = fugashi.Tagger(); ln = math.log(2); boff = 1
    H_all, g_all, ndoc = [], [], 0
    for line in open(jsonl):
        if len(g_all) >= max_chars: break
        d = json.loads(line)
        if not d.get("lang", "").startswith(lang_prefix): continue   # ja / ja_wiki 両対応
        text = unicodedata.normalize("NFC", d["text"])              # R4
        toks = tok.encode(text)[:8192]
        if len(toks) < min_toks: continue
        tokens = torch.tensor(toks, dtype=torch.long, device=device).unsqueeze(0)
        ent, _ = calculate_entropies(tokens, model, 1, device)
        ent = ent.float()                                          # float32化(cross-check~0 / bf16丸め回避)
        out = aggregate_char_entropy(ent, tokens, "sqrt")[0]        # R1 規約B（実patcher）
        raw = text.encode("utf-8")
        starts = set(); base = 0                                       # gold 形態素先頭(行ごと=\n対応)
        for line in text.split("\n"):
            pos = 0
            for w in tagger(line): starts.add(base + pos); pos += len(w.surface)
            base += len(line) + 1  # +1 = \n
        bi = 0; cidx = 0
        while bi < len(raw):
            ell = 1
            while bi + ell < len(raw) and (raw[bi + ell] & 0xC0) == 0x80: ell += 1
            idx = boff + bi - 1
            if cidx >= 1 and 0 <= idx < out.shape[0]:                # R2 先頭文字除外
                h = float(out[idx]) / ln
                if math.isfinite(h):
                    H_all.append(h); g_all.append(1 if cidx in starts else 0)
            cidx += 1; bi += ell
        ndoc += 1
    auc = auc_rank(H_all, g_all); th, P, R, F = f1_sweep(H_all, g_all)
    nb, n = sum(g_all), len(g_all)
    posH = [h for h, g in zip(H_all, g_all) if g]; negH = [h for h, g in zip(H_all, g_all) if not g]
    print("=== コーパス再測 (R1-R4 / 先頭文字除外 / NFC / MeCab gold) ===")
    print(f"docs={ndoc}  評価={n}文字  gold境界={nb} ({nb/n:.1%})")
    print(f"  pooled AUC = {auc:.4f}   (境界平均H {np.mean(posH):.2f} / 非境界 {np.mean(negH):.2f} bits)")
    print(f"  θ_rel(F1最大) = {th:.3f} bits")
    print(f"  Precision = {P:.3f}   Recall = {R:.3f}   F1 = {F:.3f}")

if __name__ == "__main__":
    app()
