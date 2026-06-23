"""
凍結前 最終検証: エントロピーモデルが本当に予測器か（健全 vs degenerate 対比）

検証1: 予測文の再構成（teacher-forcing, 1-shift）
  - 実バイト入力、各位置で次バイトを argmax（autoregressive 生成はしない）
  - 出力: 次バイト top-1 正解率、予測文 vs 実文、不一致ハイライト
検証2: CBLT √ℓ エントロピー & 境界スパイク
  - 実 patching 経路と同じ calculate_entropies（per-byte Shannon entropy）
  - per-char H(c) = (Σ e_i)/√ℓ （patcher.py の √ℓ 集約と同一式）
  - 出力: 文字ごとの H(c) 表

新規（held-out）文で実施。GPU 必須。

使い方:
  python scripts/verify_entropy_model.py --ckpt-dir <consolidated dir> --label healthy
"""
import math
import typer
import torch

app = typer.Typer()
TOKENIZER = "/gs/bs/tga-RLA/yoshida/blt_data/tokenizer/tokenizer.model"
OFFSET = 3

# 新規テスト文（generic・学習に verbatim では入らない）。En3 + Ja3
TESTS = [
    ("en", "The quick brown fox jumps over the lazy dog."),
    ("en", "Machine learning models process text efficiently."),
    ("en", "She walked to the station early in the morning."),
    ("ja", "東京都に住む友人と京都へ旅行した。"),
    ("ja", "自然言語処理の研究は難しいが面白い。"),
    ("ja", "今日は朝から雨が降っている。"),
]


@app.command()
def main(
    ckpt_dir: str = typer.Option(..., help="consolidated ディレクトリ"),
    label: str = typer.Option("model"),
    device: str = typer.Option("cuda"),
):
    from bytelatent.entropy_model import load_entropy_model
    from bytelatent.tokenizers.build_tokenizer import TokenizerArgs
    from bytelatent.data.patcher import calculate_entropies

    model, _ = load_entropy_model(ckpt_dir, f"{ckpt_dir}/consolidated.pth", device=device)
    model = model.eval()
    tok = TokenizerArgs(name="blt", init_kwargs={"bpe_tokenizer_path": TOKENIZER}).build()

    print(f"\n################## label={label}  ckpt={ckpt_dir} ##################")

    for lang, text in TESTS:
        raw = text.encode("utf-8")
        tokens = torch.tensor(tok.encode(text), dtype=torch.long, device=device).unsqueeze(0)
        entropies, preds = calculate_entropies(tokens, model, 1, device)
        vocab = preds.shape[1] // tokens.shape[1]
        logits = preds.reshape(1, tokens.shape[1], vocab)

        # ---- 検証1: teacher-forcing top-1 ----
        argmax_next = logits[0].argmax(-1)  # [L] 予測 byte[i+1]
        tk = tokens[0]
        # 位置 i で argmax_next[i] が tk[i+1] と一致するか（実トークン位置のみ）
        correct = 0; total = 0; pred_bytes = []
        for i in range(tk.shape[0]-1):
            tgt = tk[i+1].item()
            pr = argmax_next[i].item()
            # 実バイト（特殊トークン OFFSET 以上）のみ集計
            if tgt >= OFFSET:
                total += 1
                if pr == tgt:
                    correct += 1
                pred_bytes.append(pr - OFFSET if pr >= OFFSET else 0x3f)  # '?'
        top1 = correct/total if total else 0
        # 予測文を復元（先頭バイトは入力 tk[1] から、以降 argmax）
        try:
            pred_text = bytes(b if 0<=b<256 else 0x3f for b in pred_bytes).decode("utf-8","replace")
        except Exception:
            pred_text = "<decode err>"

        print(f"\n[{lang}] 実文: {text}")
        print(f"  検証1 top-1 次バイト正解率 = {top1:.1%} ({correct}/{total})")
        print(f"  予測文(teacher-forcing): {pred_text[:80]}")

        # ---- 検証2: CBLT √ℓ per-char H(c) ----
        e = entropies[0]  # nats per byte
        # UTF-8 文字単位で √ℓ 集約（patcher.py と同一式 H(c)=Σe/√ℓ）。bits 換算
        hc = []; i = 0; chars = []
        # tokens は bos + bytes + eos。bos/eos を除いて raw に対応させる
        # tok.encode = [BOS] + bytes + [EOS] 前提で、byte 部分を取り出す
        # ここでは raw のバイト境界で集約（entropies は tokens 全体に対応）
        # bos が先頭1つある前提でオフセット
        boff = 1  # BOS 分
        bi = 0
        while bi < len(raw):
            b0 = raw[bi]
            ell = 1
            while bi+ell < len(raw) and (raw[bi+ell] & 0xC0) == 0x80:
                ell += 1
            seg = e[boff+bi: boff+bi+ell]
            h = (seg.sum().item())/math.sqrt(ell)/math.log(2)  # bits
            ch = raw[bi:bi+ell].decode("utf-8","replace")
            chars.append(ch); hc.append(h)
            bi += ell
        # 表示（H(c) を小数2桁、上位スパイク位置をマーク）
        mean_h = sum(hc)/len(hc)
        line_ch = " ".join(f"{c}" for c in chars)
        line_h  = " ".join(f"{h:.1f}" for h in hc)
        spikes = [chars[k] for k in range(len(hc)) if hc[k] > mean_h*1.3]
        print(f"  検証2 per-char H(c) bits (√ℓ集約, 平均{mean_h:.2f}):")
        print(f"    文字: {line_ch}")
        print(f"    H(c): {line_h}")
        print(f"    スパイク(>1.3×平均)の文字: {spikes}")


if __name__ == "__main__":
    app()
