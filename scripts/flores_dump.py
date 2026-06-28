"""FLORES devtest 各言語1012文の実数値ダンプ(en/ru/ja)。1文字=1行のCSV。
列: sent_id, 文字, バイト(hex), エントロピー(bits), H(=Σ/√ℓ), 備考
- エントロピー = その文字の各バイトを"予測した"surprisal e[k..k+ℓ-1]（規約B, 空白で区切り）。
  これらを足して /√ℓ が H（自己整合）。先頭文字の1番目は BOS→1バイト目 の予測値=BOSを含む。
- 各文の先頭に <BOS> 行(e[0]), 末尾に <EOS> 行(e[n]=末バイト→EOS) を明示。
- 丸めず float32 を6桁表示（モデルはbf16出力→float32castは値不変）。"""
import math, json, csv, torch, typer
from bytelatent.entropy_model import load_entropy_model
from bytelatent.tokenizers.build_tokenizer import TokenizerArgs
from bytelatent.data.patcher import calculate_entropies

app = typer.Typer()
TOK = "/gs/bs/tga-RLA/yoshida/blt_data/tokenizer/tokenizer.model"
FLORES = "/gs/bs/tga-RLA/yoshida/blt_data/flores"

@app.command()
def main(ckpt_dir: str = typer.Option(...), out_dir: str = typer.Option(...), device: str = typer.Option("cuda")):
    model, _ = load_entropy_model(ckpt_dir, f"{ckpt_dir}/consolidated.pth", device=device); model = model.eval()
    tok = TokenizerArgs(name="blt", init_kwargs={"bpe_tokenizer_path": TOK}).build()
    ln = math.log(2); boff = 1
    for lang in ["en", "ru", "ja"]:
        sents = [json.loads(l) for l in open(f"{FLORES}/flores_{lang}_devtest.jsonl")]
        out = f"{out_dir}/flores_dump_{lang}.csv"
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["sent_id", "文字", "バイト(hex)", "エントロピー(bits)", "H(=sum/√ℓ)", "備考"])
            for d in sents:
                sid = d["id"]; text = d["text"]
                tokens = torch.tensor(tok.encode(text), dtype=torch.long, device=device).unsqueeze(0)
                e = calculate_entropies(tokens, model, 1, device)[0].float()[0]   # [seq] float32
                raw = text.encode("utf-8"); n = len(raw)
                # <BOS> 行: e[0] = BOS位置の出力(=1文字目1バイト目を予測)
                w.writerow([sid, "<BOS>", "", f"{float(e[0])/ln:.6f}", "", "BOS出力＝先頭バイトを予測(先頭文字Hに算入)"])
                bi = 0; cidx = 0
                while bi < n:
                    ell = 1
                    while bi + ell < n and (raw[bi + ell] & 0xC0) == 0x80: ell += 1
                    surps = [float(e[bi + i]) / ln for i in range(ell)]   # 規約B: byte b の surprisal = e[b]
                    H = sum(surps) / math.sqrt(ell)
                    hexs = " ".join(f"{raw[bi + i]:02X}" for i in range(ell))
                    ents = " ".join(f"{s:.6f}" for s in surps)
                    note = "先頭文字: 1番目entは<BOS>行(e[0])と同値＝BOS項" if cidx == 0 else ""
                    w.writerow([sid, raw[bi:bi + ell].decode("utf-8", "replace"), hexs, ents, f"{H:.6f}", note])
                    bi += ell; cidx += 1
                # <EOS> 行: e[n] = 末バイト→EOS
                if n < e.shape[0]:
                    w.writerow([sid, "<EOS>", "", f"{float(e[n])/ln:.6f}", "", "末バイト→EOS の予測"])
                w.writerow([])
        print(f"[OK] {lang}: {len(sents)}文 → {out}")

if __name__ == "__main__":
    app()
