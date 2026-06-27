"""CBLT エントロピー可視化（standalone, 規約B）。
渡された UI の色付き表示を活かしつつ、(1)モデルは calculate_entropies、(2)CBLTアライメントは規約B
（char_i ← その文字"自身"のバイトを予測した entropy = e[bi:bi+ell]、bos で lead-1 が織り込まれる）に修正。
出力: 自己完結 HTML（ブラウザ/OnDemand で開く）。日本語は MeCab 形態素境界を ◀ で併記。
usage: python cblt_viz.py --ckpt-dir <consolidated> --text "..." --out out.html
"""
import math, io, base64, datetime
from html import escape as he
import torch, typer
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False
from bytelatent.entropy_model import load_entropy_model
from bytelatent.tokenizers.build_tokenizer import TokenizerArgs
from bytelatent.data.patcher import calculate_entropies

app = typer.Typer()
TOK = "/gs/bs/tga-RLA/yoshida/blt_data/tokenizer/tokenizer.model"

def per_char(text, model, tok, device):
    """規約B per-char H(c)。返り: triples[(char,H_bits,elist_bits)], byte_entropy(bits)."""
    toks = tok.encode(text)[:8192]
    tokens = torch.tensor(toks, dtype=torch.long, device=device).unsqueeze(0)
    ent, _ = calculate_entropies(tokens, model, 1, device)
    e = ent[0]; raw = text.encode("utf-8"); boff = 1; ln = math.log(2)
    triples = []; bi = 0
    while bi < len(raw):
        ell = 1
        while bi + ell < len(raw) and (raw[bi + ell] & 0xC0) == 0x80: ell += 1
        # 規約B: char(自身)のバイトを予測した entropy = e[boff+bi-1 : boff+bi+ell-1] = e[bi : bi+ell]
        if boff + bi + ell - 1 > e.shape[0]: break
        seg = e[boff + bi - 1: boff + bi + ell - 1]
        elist = [float(x) / ln for x in seg.tolist()]
        H = (sum(elist) / math.sqrt(ell)) if elist else 0.0
        triples.append((raw[bi:bi + ell].decode("utf-8", "replace"), H, elist))
        bi += ell
    byte_e = [float(x) / ln for x in e.tolist()[:len(raw) + 1]]
    return triples, byte_e

def gold_boundaries(text):
    """日本語(CJK)=MeCab形態素境界 / それ以外(en/ru)=空白語境界。char-index集合を返す。"""
    has_cjk = any("぀" <= c <= "ヿ" or "一" <= c <= "鿿" for c in text)
    bset = set()
    if has_cjk:
        try:
            import fugashi
            tagger = fugashi.Tagger(); pos = 0
            for w in tagger(text):
                bset.add(pos); pos += len(w.surface)
            return bset
        except Exception:
            return None
    newword = True
    for i, c in enumerate(text):
        if c.isspace():
            newword = True
        elif newword:
            bset.add(i); newword = False
        else:
            newword = False
    return bset

def chart_png(byte_e):
    if not _HAS_MPL:
        return None
    fig = plt.figure(figsize=(8, 2.2))
    plt.plot(range(len(byte_e)), byte_e)
    plt.title("byte-position entropy (bits)"); plt.xlabel("byte pos"); plt.ylabel("bits")
    plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130, bbox_inches="tight"); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def heatmap(triples, bset, show_list, precision=2):
    vals = [h for _, h, _ in triples] or [0.0]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9: hi = lo + 1.0
    def color(h):
        t = (h - lo) / (hi - lo); return f"hsl(215,70%,{96 - 40 * t:.1f}%)"
    chunks = []
    for k, (ch, h, el) in enumerate(triples):
        b = (bset is not None and k in bset)
        border = "border:2px solid #d33;" if b else "border:1px solid rgba(0,0,0,0.08);"
        style = (f"background:{color(h)};{border}padding:1px 5px;border-radius:6px;margin:0 1px 4px 0;"
                 f"display:inline-block;line-height:1.9;font-family:ui-monospace,Menlo,Consolas,monospace;")
        badge = f"<sup style='opacity:.6;font-size:10px;margin-left:3px;'>{h:.{precision}f}</sup>"
        lst = ""
        if show_list:
            lst = "<span style='opacity:.8;margin-left:4px;font-size:11px;'>[" + ",".join(f"{x:.{precision}f}" for x in el) + "]</span>"
        mk = "<span style='color:#d33;font-size:10px;'>◀</span>" if b else ""
        chunks.append(f"<span style='{style}'>{he(ch)}{badge}{lst}{mk}</span>")
    legend = (f"<div style='font-size:12px;opacity:.7;margin-top:6px;'>entropy bits: min={lo:.2f} max={hi:.2f} "
              f"(濃い=高H) / 規約B(自文字) / ◀=境界(日:MeCab/英:語境界)</div>")
    return ("<div style='padding:10px;border:1px solid rgba(0,0,0,.12);border-radius:10px;background:#fafafa;"
            "white-space:normal;word-break:break-word;'>" + "".join(chunks) + "</div>" + legend)

@app.command()
def main(ckpt_dir: str = typer.Option(...), text: str = typer.Option("I have a りんご!"),
         out: str = typer.Option("/gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/cblt_viz.html"),
         device: str = typer.Option("cuda")):
    model, _ = load_entropy_model(ckpt_dir, f"{ckpt_dir}/consolidated.pth", device=device); model = model.eval()
    tok = TokenizerArgs(name="blt", init_kwargs={"bpe_tokenizer_path": TOK}).build()
    triples, byte_e = per_char(text, model, tok, device)
    bset = gold_boundaries(text)
    png = chart_png(byte_e)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    chart_html = (f'<h3>byte-position entropy</h3><img src="data:image/png;base64,{png}" '
                  f'style="max-width:100%;border:1px solid #eee;border-radius:8px;"/>' if png else "")
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>CBLT viz</title></head>
<body style="font-family:system-ui,'Segoe UI',sans-serif;margin:18px;background:#fff;">
<h2>CBLT entropy 可視化 (規約B)</h2>
<div style="opacity:.7;">ckpt: {he(ckpt_dir)} / {now}</div>
<div style="margin:10px 0;padding:8px 10px;border:1px solid #ddd;border-radius:8px;">text: <b>{he(text)}</b></div>
{chart_html}
<h3>CBLT: 文字ごと平均 H(c)</h3>{heatmap(triples, bset, show_list=False)}
<h3>CBLT: 平均 + 割当 entropy リスト</h3>{heatmap(triples, bset, show_list=True)}
</body></html>"""
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[OK] saved {out}  ({len(triples)} chars)")
    # コンソールにも要約（先頭20字）
    print("char  H(bits)  境界")
    for k, (ch, h, _) in enumerate(triples[:20]):
        print(f"  {ch}  {h:5.2f}  {'◀' if (bset and k in bset) else ''}")

if __name__ == "__main__":
    app()
