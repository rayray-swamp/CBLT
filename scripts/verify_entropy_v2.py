"""
追加検証 v2: フルコンテキスト + warm-up 除外 + 多数文書統計

改善点:
- 各文書を最大 seq_len まで使い「コンテキストいっぱい」で評価
- 先頭 WARMUP(=512, sliding_window 相当) 位置を除外（初期位置のエントロピー乱れを排除）
- 英30+ / 日30+ の held-out 文書で統計（top-1 次バイト正解率 / mean H(c) bits）
- 文書ごと結果 + 言語別 aggregate を出力（healthy/degenerate を別実行で対比）

実 calculate_entropies 経路。GPU 必須。
使い方:
  python scripts/verify_entropy_v2.py --ckpt-dir <consolidated> --label healthy --jsonl <heldout_v2.jsonl>
"""
import math, json
import typer
import torch

app = typer.Typer()
TOKENIZER = "/gs/bs/tga-RLA/yoshida/blt_data/tokenizer/tokenizer.model"
OFFSET = 3


@app.command()
def main(
    ckpt_dir: str = typer.Option(...),
    label: str = typer.Option("model"),
    jsonl: str = typer.Option("/gs/bs/tga-RLA/yoshida/blt_data/eval/heldout_v2.jsonl"),
    seq_len: int = typer.Option(8192),
    warmup: int = typer.Option(512, help="除外する先頭位置数（初期乱れ）"),
    device: str = typer.Option("cuda"),
):
    from bytelatent.entropy_model import load_entropy_model
    from bytelatent.tokenizers.build_tokenizer import TokenizerArgs
    from bytelatent.data.patcher import calculate_entropies

    model, _ = load_entropy_model(ckpt_dir, f"{ckpt_dir}/consolidated.pth", device=device)
    model = model.eval()
    tok = TokenizerArgs(name="blt", init_kwargs={"bpe_tokenizer_path": TOKENIZER}).build()

    docs = [json.loads(l) for l in open(jsonl)]
    ln = math.log(2)
    per_lang = {}  # lang -> dict of accumulators
    rows = []

    for idx, d in enumerate(docs):
        lang = d.get("lang", "?")
        text = d["text"]
        toks = tok.encode(text)[:seq_len]
        if len(toks) <= warmup + 64:
            continue
        tokens = torch.tensor(toks, dtype=torch.long, device=device).unsqueeze(0)
        entropies, preds = calculate_entropies(tokens, model, 1, device)
        vocab = preds.shape[1] // tokens.shape[1]
        logits = preds.reshape(1, tokens.shape[1], vocab)[0]
        tk = tokens[0]
        L = tk.shape[0]

        # warm-up 除外して [warmup, L-1) を集計
        argmax_next = logits.argmax(-1)
        correct = 0; total = 0; ent_sum = 0.0; ent_n = 0
        for i in range(warmup, L - 1):
            tgt = tk[i + 1].item()
            if tgt < OFFSET:
                continue
            total += 1
            if argmax_next[i].item() == tgt:
                correct += 1
        # entropy（warmup以降の全位置、bits）
        e = entropies[0][warmup:]
        ent_bits = (e.sum().item() / max(1, e.numel())) / ln
        top1 = correct / total if total else 0.0

        rows.append((idx, lang, len(toks), top1, ent_bits))
        acc = per_lang.setdefault(lang, {"top1": [], "ent": [], "bytes": 0})
        acc["top1"].append(top1); acc["ent"].append(ent_bits); acc["bytes"] += total

    print(f"\n################## label={label} ckpt={ckpt_dir} (warmup={warmup}) ##################")
    print("idx,lang,tokens,top1_acc,mean_Hc_bits")
    for idx, lang, nt, t1, eb in rows:
        print(f"{idx},{lang},{nt},{t1:.4f},{eb:.4f}")

    print("\n--- 言語別 aggregate ---")
    print("lang,n_docs,mean_top1,mean_Hc_bits,std_Hc")
    import statistics as st
    for lang, acc in sorted(per_lang.items()):
        n = len(acc["top1"])
        mt = sum(acc["top1"]) / n
        me = sum(acc["ent"]) / n
        se = st.pstdev(acc["ent"]) if n > 1 else 0.0
        print(f"{lang},{n},{mt:.4f},{me:.4f},{se:.4f}")


if __name__ == "__main__":
    app()
