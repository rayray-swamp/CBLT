"""wiki(heldout_jawiki)とコーパス(heldout_v3)からランダム抽出した文で、6行CSVを生成。
★H(c)は cblt_eval_corpus.py と同一の実 aggregate_char_entropy で計算（=AUCに入る値そのもの）。
 さらに inline規約B式との cross-check max|Δ| を表示（実eval経路と一致することの証明）。
各文6行: 文字 / バイト(hex) / エントロピー(bits)=e[t] / 文字パッチ番号(規約B lead-1) / sum / sum/√ℓ(=H)。
独立系列(各文に自前bos)。NFC無し(生バイトをそのまま見せる)。"""
import math, csv, json, random, torch, typer
from bytelatent.entropy_model import load_entropy_model
from bytelatent.tokenizers.build_tokenizer import TokenizerArgs
from bytelatent.data.patcher import calculate_entropies, aggregate_char_entropy

app = typer.Typer()
TOK = "/gs/bs/tga-RLA/yoshida/blt_data/tokenizer/tokenizer.model"
WIKI = "/gs/bs/tga-RLA/yoshida/blt_data/eval/heldout_jawiki.jsonl"
CORP = "/gs/bs/tga-RLA/yoshida/blt_runs/auc_eval/heldout_v3.jsonl"

def sample_sents(jsonl, n, seed, lo=15, hi=45):
    sents = []
    for line in open(jsonl):
        d = json.loads(line)
        if not d.get("lang", "").startswith("ja"): continue
        for s in d["text"].replace("\n", "").split("。"):
            s = s.strip()
            if lo <= len(s) <= hi: sents.append(s + "。")
    random.Random(seed).shuffle(sents)
    return sents[:n]

def gen_rows(text, model, tok, ln, boff, label, stats):
    dev = next(model.parameters()).device
    tokens = torch.tensor(tok.encode(text), dtype=torch.long, device=dev).unsqueeze(0)
    ent = calculate_entropies(tokens, model, 1, dev)[0]           # [1, seq]
    out = aggregate_char_entropy(ent, tokens, "sqrt")[0]          # ★実eval関数(cblt_eval_corpusと同一)
    e = ent[0]; raw = text.encode("utf-8"); n = len(raw)
    chars = []; bi = 0
    while bi < n:
        ell = 1
        while bi + ell < n and (raw[bi + ell] & 0xC0) == 0x80: ell += 1
        chars.append((bi, ell)); bi += ell
    row_c = [""] * n; row_b = [f"{b:02X}" for b in raw]; row_e = [""] * n
    row_p = [""] * n; row_s = [""] * n; row_h = [""] * n
    for p in range(n):
        idx = boff + p
        row_e[p] = f"{float(e[idx])/ln:.3f}" if idx < e.shape[0] else ""
    charidx = [0] * n
    for ci, (k, ell) in enumerate(chars):
        row_c[k] = raw[k:k + ell].decode("utf-8", "replace")
        for t in range(k, k + ell): charidx[t] = ci
    for p in range(n - 1): row_p[p] = str(charidx[p + 1])
    for ci, (k, ell) in enumerate(chars):
        H_eval = float(out[k]) / ln                               # eval関数の出力(out[i_tok-1]=out[k])
        s_bits = float(e[k:k + ell].sum()) / ln
        stats[0] = max(stats[0], abs(H_eval - s_bits / math.sqrt(ell)))  # cross-check
        pos = max(k - 1, 0)
        row_s[pos] = f"{s_bits:.3f}"; row_h[pos] = f"{H_eval:.3f}"  # H は実eval関数の値
    return [[label, text], ["文字"] + row_c, ["バイト(hex)"] + row_b, ["エントロピー(bits)"] + row_e,
            ["文字パッチ番号"] + row_p, ["sum"] + row_s, ["sum/√ℓ (=H)"] + row_h, []]

@app.command()
def main(ckpt_dir: str = typer.Option(...), out: str = typer.Option(...),
         n: int = typer.Option(25), seed: int = typer.Option(42), device: str = typer.Option("cuda")):
    model, _ = load_entropy_model(ckpt_dir, f"{ckpt_dir}/consolidated.pth", device=device); model = model.eval()
    tok = TokenizerArgs(name="blt", init_kwargs={"bpe_tokenizer_path": TOK}).build()
    ln = math.log(2); boff = 1; stats = [0.0]
    wiki = sample_sents(WIKI, n, seed); corp = sample_sents(CORP, n, seed)
    rows = []
    for i, s in enumerate(wiki, 1): rows += gen_rows(s, model, tok, ln, boff, f"■wiki-{i}", stats)
    for i, s in enumerate(corp, 1): rows += gen_rows(s, model, tok, ln, boff, f"■corpus-{i}", stats)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(rows)
    print(f"[OK] {out}  wiki{len(wiki)}文 + corpus{len(corp)}文 (計{len(wiki)+len(corp)})")
    print(f"[cross-check] 実aggregate_char_entropy vs inline規約B の max|Δ| = {stats[0]:.2e}  "
          f"({'PASS <1e-6 ＝ CSVのH は cblt_eval_corpus.py と同一経路' if stats[0] < 1e-6 else 'FAIL'})")

if __name__ == "__main__":
    app()
