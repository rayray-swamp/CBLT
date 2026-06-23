"""
凍結前エントロピーモデルの held-out / train-sample bpb 測定（リーク切り分け）

学習と同一の計算でクロス比較する:
  - model.eval() + torch.no_grad()
  - model(x) が内部で学習と同じ local_block_causal + sliding_window マスクを構築
    （transformer.py:119 create_causal_mask、attn_bias_type/sliding_window は params.json 由来）
  - bpb = Σ(tok_loss_nats) / ln(2) / n_bytes  （train.py:585 と同式）
  - tok_loss は per-token CE、pad 位置を mask 除外（compute_loss と同じ）

GPU 必須（qrsh gpu_1）。consolidate 済み consolidated.pth を読む。

使い方:
  python scripts/eval_entropy_bpb.py \
    --ckpt-dir /gs/bs/tga-RLA/yoshida/blt_runs/entropy_real/checkpoints/0000102000/consolidated \
    --jsonl /gs/bs/tga-RLA/yoshida/blt_data/eval/heldout.jsonl
"""
import json, math
import typer
import torch
import torch.nn.functional as F

app = typer.Typer()

TOKENIZER = "/gs/bs/tga-RLA/yoshida/blt_data/tokenizer/tokenizer.model"
PAD_ID = 0  # BLT bytes tokenizer pad


@app.command()
def main(
    ckpt_dir: str = typer.Option(..., help="consolidated ディレクトリ（params.json がある）"),
    jsonl: str = typer.Option(..., help="評価する jsonl（text フィールド）"),
    seq_len: int = typer.Option(8192, help="学習と同じ seq_len"),
    device: str = typer.Option("cuda"),
    max_docs: int = typer.Option(0, help="0=全件"),
):
    from bytelatent.entropy_model import load_entropy_model
    from bytelatent.tokenizers.build_tokenizer import TokenizerArgs

    state_dict_path = f"{ckpt_dir}/consolidated.pth"
    print(f"Loading model: {ckpt_dir}")
    model, _ = load_entropy_model(ckpt_dir, state_dict_path, device=device)
    model = model.eval()

    tok = TokenizerArgs(name="blt", init_kwargs={"bpe_tokenizer_path": TOKENIZER}).build()

    # 全 doc をトークン化（学習と同じ add_bos/add_eos=True）して連結
    stream = []
    n_docs = 0
    with open(jsonl) as f:
        for line in f:
            text = json.loads(line)["text"]
            stream.extend(tok.encode(text))
            n_docs += 1
            if max_docs and n_docs >= max_docs:
                break

    # seq_len チャンクに分割（byte packing 相当）
    total_loss_nats = 0.0
    total_bytes = 0
    n_chunks = 0
    with torch.no_grad():
        for start in range(0, len(stream) - 1, seq_len):
            chunk = stream[start : start + seq_len]
            if len(chunk) < 2:
                continue
            x = torch.tensor(chunk, dtype=torch.long, device=device).unsqueeze(0)
            # y = 次バイト（学習と同じシフト）。最終位置は pad。
            y = torch.full_like(x, PAD_ID)
            y[0, : x.shape[1] - 1] = x[0, 1:]
            mask = torch.zeros_like(x, dtype=torch.bool)
            mask[0, : x.shape[1] - 1] = True  # 最終位置(pad予測)は除外
            logits = model(x)  # 内部で local_block_causal + SWA マスク構築
            tok_loss = F.cross_entropy(
                logits.flatten(0, 1).float(), y.flatten(0, 1), reduction="none"
            )
            m = mask.flatten(0, 1)
            total_loss_nats += (tok_loss * m).sum().item()
            total_bytes += int(m.sum().item())
            n_chunks += 1

    bpb = total_loss_nats / math.log(2) / total_bytes
    avg_nats = total_loss_nats / total_bytes
    print(f"\n=== {jsonl} ===")
    print(f"docs={n_docs}  chunks={n_chunks}  eval_bytes={total_bytes:,}")
    print(f"avg loss (nats/byte) = {avg_nats:.4f}")
    print(f"**bpb = {bpb:.4f}**")


if __name__ == "__main__":
    app()
