"""自己完結版CBLT集約検証（xformers不要）。aggregate_char_entropy を patcher.py から逐語コピー。
OFFSET=4（bytelatent/tokenizers/constants.py より）。entropy[i]=i で足された index を露出。"""
import math, torch
OFFSET = 4

# ---- ↓↓↓ bytelatent/data/patcher.py の aggregate_char_entropy を逐語コピー ↓↓↓ ----
def aggregate_char_entropy(entropies, tokens, mode="sqrt"):
    raw_bytes = tokens - OFFSET
    is_lead = (raw_bytes & 0xC0) != 0x80
    out = torch.full_like(entropies, float("-inf"))
    bs, seq_len = tokens.shape
    for b in range(bs):
        i = 0
        while i < seq_len:
            if not is_lead[b, i]:
                i += 1
                continue
            if raw_bytes[b, i] < 0:
                if i > 0:
                    out[b, i - 1] = entropies[b, i]
                i += 1
                continue
            j = i + 1
            while j < seq_len and not is_lead[b, j]:
                j += 1
            ell = j - i
            e_sum = entropies[b, i:j].sum()
            if mode == "sqrt":
                h = e_sum / (ell ** 0.5)
            elif mode == "sum":
                h = e_sum
            else:
                h = e_sum / ell
            if i > 0:
                out[b, i - 1] = h
            i = j
    return out
# ---- ↑↑↑ 逐語コピーここまで ↑↑↑ ----

text = "I have a りんご!"
raw = list(text.encode("utf-8"))
tokens = torch.tensor([[b + OFFSET for b in raw]])
ent = torch.tensor([[float(i) for i in range(len(raw))]])
out = aggregate_char_entropy(ent, tokens, "sqrt")[0]

print(f"text='{text}'  ({len(text)}文字/{len(raw)}バイト)  OFFSET={OFFSET}")
print("\n[per-byte] idx hex  is_lead entropy(=idx)")
for i, b in enumerate(raw):
    print(f"           {i:2d}  {b:#04x} {str((b&0xC0)!=0x80):5}   {float(ent[0,i]):.0f}")

i=0; spans=[]
while i < len(raw):
    j=i+1
    while j<len(raw) and (raw[j]&0xC0)==0x80: j+=1
    spans.append((i,j)); i=j

print("\n[per-char] 実コード out[i-1] vs 規約A(Σ[i:j]) / 規約B(Σ[i-1:j-1])")
print(f"{'char':4}{'span':9}{'ell':4}{'out[i-1]':>10}{'規約A':>9}{'規約B':>9}  足したidx(実コード=A)")
for (i,j) in spans:
    ch=bytes(raw[i:j]).decode('utf-8','replace'); ell=j-i
    A=sum(range(i,j))/math.sqrt(ell)
    B=(sum(range(i-1,j-1))/math.sqrt(ell)) if i>0 else float('nan')
    ov=float(out[i-1]) if i>0 else float('nan')
    print(f"{ch:4}[{i:2d}:{j:2d}]  {ell:<4}{ov:>10.3f}{A:>9.3f}{B:>9.3f}  {list(range(i,j))}")

print("\n=== 「り」(bytes 9,10,11) 焦点 ===")
print(f"実コード out[8]              = {float(out[8]):.4f}")
print(f"規約A entropies[9,10,11]=30  → 30/√3 = {30/math.sqrt(3):.4f}")
print(f"規約B entropies[8,9,10]=27   → 27/√3 = {27/math.sqrt(3):.4f}")
v=float(out[8])
verdict = "規約A(自位置 entropies[i:j])" if abs(v-30/math.sqrt(3))<1e-3 else ("規約B(lead-1)" if abs(v-27/math.sqrt(3))<1e-3 else f"不明({v})")
print(f"→ 実コード判定: {verdict}")
