"""θ再検証(Chat指定): 修正後の実コード経路で realized bytes/patch が {4,5,6,7,8} 近傍か。
byte: find_entropy_patch_start_ids(monotonicity=True)。cblt: aggregate_char_entropy + find_cblt_monotonic_patch_start_ids。
凍結285k・corpus サンプル。較正θ(per-char ΔH)が実コードでも狙いrateを出すか確認。"""
import math, json, torch, typer
from collections import defaultdict
from bytelatent.entropy_model import load_entropy_model
from bytelatent.tokenizers.build_tokenizer import TokenizerArgs
from bytelatent.data.patcher import (
    calculate_entropies, aggregate_char_entropy,
    find_entropy_patch_start_ids, find_cblt_monotonic_patch_start_ids,
)

app = typer.Typer()
TOK = "/gs/bs/tga-RLA/yoshida/blt_data/tokenizer/tokenizer.model"
CORPUS = "/gs/bs/tga-RLA/yoshida/blt_data/largecorpus/corpus/corpus.chunk.00.jsonl"
LANGS = ["en", "ru", "ja", "ar", "zh", "ko", "hi", "th"]
RATES = [4, 5, 6, 7, 8]
BYTE_TH = dict(zip(RATES, [0.3008, 0.5000, 0.6367, 0.7344, 0.8516]))
CBLT_TH = dict(zip(RATES, [-0.0035, 0.0067, 0.0937, 0.2439, 0.3934]))

def n_patches(ids, L):
    v = ids[0]
    return int(torch.unique(v[v < L]).numel())  # 実start(byte位置<L)の distinct 数

@app.command()
def main(ckpt_dir: str = typer.Option(...), n_per_lang: int = typer.Option(80),
         max_chars: int = typer.Option(3000), device: str = typer.Option("cuda")):
    model, _ = load_entropy_model(ckpt_dir, f"{ckpt_dir}/consolidated.pth", device=device); model = model.eval()
    tok = TokenizerArgs(name="blt", init_kwargs={"bpe_tokenizer_path": TOK}).build()
    # サンプル
    buf = defaultdict(list); need = set(LANGS)
    for i, line in enumerate(open(CORPUS)):
        if i >= 400000 or not need: break
        try: d = json.loads(line)
        except Exception: continue
        lg = d.get("lang")
        if lg in need:
            t = d.get("text", "")[:max_chars]
            if len(t) >= 20: buf[lg].append(t)
            if len(buf[lg]) >= n_per_lang: need.discard(lg)
    print("サンプル:", {lg: len(buf[lg]) for lg in LANGS})
    # 各θで pooled bytes/patch
    acc = {("byte", r): [0, 0] for r in RATES}; acc.update({("cblt", r): [0, 0] for r in RATES})
    for lg in LANGS:
        for text in buf[lg]:
            toks = tok.encode(text)
            if len(toks) > 8192: toks = toks[:8192]
            tokens = torch.tensor(toks, dtype=torch.long, device=device).unsqueeze(0)
            ent = calculate_entropies(tokens, model, 1, device)[0].float()
            L = tokens.shape[1]
            sc = aggregate_char_entropy(ent, tokens, "sqrt")
            for r in RATES:
                idb = find_entropy_patch_start_ids(ent, threshold=BYTE_TH[r], monotonicity=True)
                acc[("byte", r)][0] += L; acc[("byte", r)][1] += n_patches(idb, L)
                idc = find_cblt_monotonic_patch_start_ids(sc, CBLT_TH[r])
                acc[("cblt", r)][0] += L; acc[("cblt", r)][1] += n_patches(idc, L)
    print("\n=== realized pooled bytes/patch (実コード経路) ===")
    print(f"{'rate':4} | {'byte θ':>8} {'byte bpp':>9} | {'cblt θ':>8} {'cblt bpp':>9}")
    okb = okc = True
    for r in RATES:
        bb = acc[("byte", r)][0] / max(1, acc[("byte", r)][1])
        cc = acc[("cblt", r)][0] / max(1, acc[("cblt", r)][1])
        flagb = "" if abs(bb - r) <= 1.0 else " ⚠"
        flagc = "" if abs(cc - r) <= 1.0 else " ⚠"
        if flagb: okb = False
        if flagc: okc = False
        print(f"{r:4} | {BYTE_TH[r]:8.4f} {bb:9.3f}{flagb} | {CBLT_TH[r]:8.4f} {cc:9.3f}{flagc}")
    print(f"\nbyte rate整合(±1): {'OK' if okb else 'ズレあり→要θ再較正'}  cblt rate整合(±1): {'OK' if okc else 'ズレあり→要θ再較正'}")

if __name__ == "__main__":
    app()
