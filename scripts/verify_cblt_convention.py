"""CBLT集約が「その文字を"予測した"バイト範囲で振り分け→√ℓ集約」になっているかの決定的検証。
合成エントロピー entropies[t]=t を入れ、aggregate_char_entropy がどの位置を合計するかを可視化する
（モデル不要・純粋ロジック）。"参議院" で 議 が byte 82,E8,AD の位置を合計するはず。
さらに out への配置→find_entropy_patch_start_ids でパッチ境界が「文字の先頭」に立つかも確認。"""
import math, torch
from bytelatent.tokenizers.build_tokenizer import TokenizerArgs
from bytelatent.data.patcher import (
    aggregate_char_entropy, find_entropy_patch_start_ids,
    patch_lengths_from_start_ids, OFFSET,
)

TOK = "/gs/bs/tga-RLA/yoshida/blt_data/tokenizer/tokenizer.model"
tok = TokenizerArgs(name="blt", init_kwargs={"bpe_tokenizer_path": TOK}).build()

text = "参議院"
toks = tok.encode(text)
tokens = torch.tensor(toks, dtype=torch.long).unsqueeze(0)
seq = tokens.shape[1]
rb = (tokens[0] - OFFSET)  # raw byte値 (負=特殊token)

# 合成エントロピー: entropies[t] = t  →  合計された位置が値で丸わかり
ent = torch.arange(seq, dtype=torch.float32).unsqueeze(0)
out = aggregate_char_entropy(ent, tokens, "sqrt")[0]

print(f"text={text!r}  seq_len={seq}  (entropies[t]=t を投入)")
print("\n[トークン列]  pos | byte | lead? | ent[t]=t | out[t]=集約結果")
def bstr(b): return f"{b:02X}" if b >= 0 else f"SPECIAL({b})"
for t in range(seq):
    b = int(rb[t]); lead = (b & 0xC0) != 0x80
    o = out[t].item(); os = f"{o:.4f}" if math.isfinite(o) else "-inf"
    mark = " <== 集約値あり" if math.isfinite(o) else ""
    print(f"   {t:2d} | {bstr(b):>11} | {str(bool(lead)):5} | {t:5d} | {os:>7}{mark}")

# 文字ごとに「予測したバイト範囲」を明示
print("\n[文字ごとの検算]  ※ entropies[i-1:j-1] = その文字の各バイトを予測した位置")
i = 0; cidx = 0
while i < seq:
    b = int(rb[i])
    if (b & 0xC0) == 0x80:  # 継続バイト
        i += 1; continue
    if b < 0:  # 特殊token
        i += 1; continue
    j = i + 1
    while j < seq and (int(rb[j]) & 0xC0) == 0x80:
        j += 1
    ell = j - i
    if i > 0:
        rng = list(range(i - 1, j - 1))                 # 予測した位置
        rng_bytes = [bstr(int(rb[t])) for t in rng]     # その位置のバイト
        e_sum = float(ent[0, i - 1:j - 1].sum())
        H = e_sum / math.sqrt(ell)
        ch = bytes(int(rb[t]) for t in range(i, j)).decode("utf-8", "replace")
        print(f"  文字{cidx} '{ch}' (バイトpos {i}..{j-1}, ℓ={ell}): "
              f"予測位置={rng}→バイト{rng_bytes}  Σ={e_sum:.1f} /√{ell} = H={H:.4f} → out[{i-1}]")
    cidx += 1; i = j

# 配置 → パッチ境界が文字先頭に立つか（議 lead-1=pos3, 院 lead-1=pos6 を高スコアに）
print("\n[配置→境界]  out[3](議のlead-1)と out[6](院のlead-1)を高スコアにして境界位置を確認")
out2 = torch.full((1, seq), float("-inf"))
out2[0, 3] = 100.0; out2[0, 6] = 100.0
psi = find_entropy_patch_start_ids(out2, patch_size=4.5, threshold=1.0, include_next_token=False)
pl = patch_lengths_from_start_ids(psi, seq)
print(f"  patch_start_ids = {psi[0].tolist()}  (first_ids=[0,1]強制 + 高スコア由来)")
print(f"  → 議のlead(pos4)・院のlead(pos7)に境界が立てば 規約通り。patch_lengths={pl[0].tolist()}")
