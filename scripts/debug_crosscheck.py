"""aggregate_char_entropy(out[k]) と inline規約B(e[k:k+ell]/√ell) が食い違う文字を特定する。
tokenizer のバイト復元が text.encode と一致するか、bos/eos位置、worst char の文脈を出す。"""
import math, json, random, torch, typer
from bytelatent.entropy_model import load_entropy_model
from bytelatent.tokenizers.build_tokenizer import TokenizerArgs
from bytelatent.data.patcher import calculate_entropies, aggregate_char_entropy, OFFSET

app = typer.Typer()
TOK = "/gs/bs/tga-RLA/yoshida/blt_data/tokenizer/tokenizer.model"
WIKI = "/gs/bs/tga-RLA/yoshida/blt_data/eval/heldout_jawiki.jsonl"
CORP = "/gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/heldout_v3.jsonl"

def samp(j, n, seed, lo=15, hi=45):
    s = []
    for line in open(j):
        d = json.loads(line)
        if not d.get("lang", "").startswith("ja"): continue
        for x in d["text"].replace("\n", "").split("。"):
            x = x.strip()
            if lo <= len(x) <= hi: s.append(x + "。")
    random.Random(seed).shuffle(s); return s[:n]

@app.command()
def main(ckpt_dir: str = typer.Option(...), device: str = typer.Option("cuda")):
    model, _ = load_entropy_model(ckpt_dir, f"{ckpt_dir}/consolidated.pth", device=device); model = model.eval()
    tok = TokenizerArgs(name="blt", init_kwargs={"bpe_tokenizer_path": TOK}).build()
    ln = math.log(2); boff = 1
    sents = samp(WIKI, 25, 42) + samp(CORP, 25, 42)
    worst = (0.0, None)
    byte_mismatch = 0
    for text in sents:
        toks = tok.encode(text)
        tokens = torch.tensor(toks, dtype=torch.long, device=device).unsqueeze(0)
        # tokenizer のバイト復元 vs text.encode
        rec = bytes([t - OFFSET for t in toks if t >= OFFSET])
        raw = text.encode("utf-8")
        if rec != raw: byte_mismatch += 1
        ent = calculate_entropies(tokens, model, 1, device)[0]
        out = aggregate_char_entropy(ent, tokens, "sqrt")[0]; e = ent[0]
        n = len(raw); bi = 0
        while bi < n:
            ell = 1
            while bi + ell < n and (raw[bi + ell] & 0xC0) == 0x80: ell += 1
            He = float(out[bi]) / ln
            Hi = float(e[bi:bi + ell].sum()) / math.sqrt(ell) / ln
            d = abs(He - Hi)
            if d > worst[0]:
                worst = (d, dict(text=text, ch=raw[bi:bi+ell].decode("utf-8","replace"),
                                 k=bi, ell=ell, He=He, Hi=Hi,
                                 len_toks=len(toks), len_raw=n, rec_eq_raw=(rec==raw),
                                 first_tok=toks[0], last_tok=toks[-1], OFFSET=OFFSET,
                                 e_slice=[round(float(e[t])/ln,3) for t in range(bi,bi+ell)],
                                 out_k=round(float(out[bi])/ln,3)))
            bi += ell
    print(f"=== バイト復元 != text.encode の文数: {byte_mismatch}/{len(sents)} ===")
    print(f"=== worst |Δ| = {worst[0]:.4f} bits ===")
    for k, v in worst[1].items(): print(f"  {k}: {v}")

if __name__ == "__main__":
    app()
