"""日本語の生バイト×エントロピー×CBLT集約 CSV。各文章6行:
  1 文字          (各文字は先頭バイト列、継続バイトは空セル＝3倍幅に整列)
  2 バイト(hex)
  3 エントロピー(bits)   = e[t] 各バイト位置の出力エントロピー(次バイト予測, 空白高)
  4 文字パッチ番号        = byte p は「次バイト p+1 が属する文字」の番号(規約B/lead-1シフト)
                          → 文字cのパッチ = cを"予測した"バイト範囲 [lead-1 .. lead+ℓ-2]
  5 sum                  = そのパッチ範囲の e[t] 合計 (= Σ entropies[lead-1:lead+ℓ-1])
  6 sum/√ℓ (=H)          = CBLT集約値。実 aggregate_char_entropy と一致(規約B)
sum/H はパッチ先頭バイト(=文字lead-1, 先頭文字は byte0)に配置。
※ 先頭文字のパッチには見えない bos→1byte目の項が入る(byte列に出ない)。
※ 最後のバイトは eos 予測なのでどの文字にも属さず空欄。5文章を1ファイル(utf-8-sig)。"""
import math, csv, torch, typer
from bytelatent.entropy_model import load_entropy_model
from bytelatent.tokenizers.build_tokenizer import TokenizerArgs
from bytelatent.data.patcher import calculate_entropies

app = typer.Typer()
TOK = "/gs/bs/tga-RLA/yoshida/blt_data/tokenizer/tokenizer.model"
SENTENCES = [
    "参議院議員選挙の投票日は来週の日曜日です。",
    "人工知能の研究が急速に進歩し社会を大きく変えつつある。",
    "昨日の夜、彼は図書館で経済学の専門書を静かに読んでいた。",
    "新しい新幹線の駅が来年の春に開業する予定だと発表された。",
    "桜の花が満開になり、公園は大勢の花見客で賑わっていた。",
]

@app.command()
def main(ckpt_dir: str = typer.Option(...), out: str = typer.Option(...), device: str = typer.Option("cuda")):
    model, _ = load_entropy_model(ckpt_dir, f"{ckpt_dir}/consolidated.pth", device=device); model = model.eval()
    tok = TokenizerArgs(name="blt", init_kwargs={"bpe_tokenizer_path": TOK}).build()
    ln = math.log(2); boff = 1  # bos
    rows = []
    for si, text in enumerate(SENTENCES, 1):
        toks = tok.encode(text)
        tokens = torch.tensor(toks, dtype=torch.long, device=device).unsqueeze(0)
        ent, _ = calculate_entropies(tokens, model, 1, device); e = ent[0]
        raw = text.encode("utf-8"); n = len(raw)
        # 文字分割: (lead byte index, ℓ)
        chars = []; bi = 0
        while bi < n:
            ell = 1
            while bi + ell < n and (raw[bi + ell] & 0xC0) == 0x80: ell += 1
            chars.append((bi, ell)); bi += ell
        row_c = [""] * n; row_b = [f"{b:02X}" for b in raw]
        row_e = [""] * n; row_p = [""] * n; row_s = [""] * n; row_h = [""] * n
        # 3 エントロピー: byte p ← e[boff+p] (出力位置=byte p, 次byte予測)
        for p in range(n):
            idx = boff + p
            row_e[p] = f"{float(e[idx])/ln:.3f}" if idx < e.shape[0] else ""
        # 文字index/byte と 1 文字行
        charidx = [0] * n
        for ci, (k, ell) in enumerate(chars):
            row_c[k] = raw[k:k + ell].decode("utf-8", "replace")
            for t in range(k, k + ell): charidx[t] = ci
        # 4 パッチ番号: byte p ← 次byte p+1 の文字index (最後は空=eos予測)
        for p in range(n - 1):
            row_p[p] = str(charidx[p + 1])
        # 5,6 sum / sum/√ℓ: 文字cの予測範囲 = e[k:k+ℓ] (token slice, 規約B), パッチ先頭に配置
        for ci, (k, ell) in enumerate(chars):
            s = float(e[k:k + ell].sum()) / ln
            H = s / math.sqrt(ell)
            pos = max(k - 1, 0)  # パッチ先頭バイト(=lead-1, 先頭文字は0)
            row_s[pos] = f"{s:.3f}"; row_h[pos] = f"{H:.3f}"
        rows += [
            [f"■文章{si}", text],
            ["文字"] + row_c,
            ["バイト(hex)"] + row_b,
            ["エントロピー(bits)"] + row_e,
            ["文字パッチ番号"] + row_p,
            ["sum"] + row_s,
            ["sum/√ℓ (=H)"] + row_h,
            [],
        ]
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(rows)
    print(f"[OK] saved {out}  ({len(SENTENCES)} 文章 × 6行)")
    # 文章1 をコンソール確認(先頭3文字=9バイト)
    t = SENTENCES[0]; raw = t.encode("utf-8")
    toks = tok.encode(t); tk = torch.tensor(toks, dtype=torch.long, device=device).unsqueeze(0)
    e = calculate_entropies(tk, model, 1, device)[0][0]
    print(f"確認 '{t[:3]}' :")
    for ci, (k, ell) in enumerate([(0, 3), (3, 3), (6, 3)]):
        ch = raw[k:k + ell].decode("utf-8")
        s = float(e[k:k + ell].sum()) / ln; H = s / math.sqrt(ell)
        rng = list(range(k, k + ell))
        print(f"  パッチ{ci} '{ch}': 予測token位置{rng} sum={s:.3f} /√{ell} = H={H:.3f}")

if __name__ == "__main__":
    app()
