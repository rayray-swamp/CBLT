"""CBLT 集約の index アライメント実証: 規約A(自位置 entropies[i:j]) か 規約B(lead-1 entropies[i-1:j-1]) か。
entropy[i]=i を入れ、Σ の値で足された index を可視化する。実関数 aggregate_char_entropy を使用。"""
import math, torch
from bytelatent.data.patcher import aggregate_char_entropy
from bytelatent.tokenizers.constants import OFFSET

text = "I have a りんご!"
raw = list(text.encode("utf-8"))                       # 19 bytes, bos無し→index=バイト位置
tokens = torch.tensor([[b + OFFSET for b in raw]])     # raw_byte = token - OFFSET で復元される
ent = torch.tensor([[float(i) for i in range(len(raw))]])  # entropy[i]=i → Σ が足した index を露出
out = aggregate_char_entropy(ent, tokens, "sqrt")[0]

print(f"text='{text}'  ({len(text)}文字/{len(raw)}バイト)  OFFSET={OFFSET}")
print("\n[per-byte]  idx  hex   is_lead  entropy(=idx)")
for i, b in enumerate(raw):
    print(f"            {i:2d}  {b:#04x}  {(b&0xC0)!=0x80!s:5}    {float(ent[0,i]):.0f}")

# char span 復元
i=0; spans=[]
while i < len(raw):
    j=i+1
    while j<len(raw) and (raw[j]&0xC0)==0x80: j+=1
    spans.append((i,j)); i=j

print("\n[per-char] 実コード out[] と 規約A/B の比較")
print(f"{'char':5}{'span':10}{'ell':4}{'out[i-1]実値':>12}{'規約A Σ[i:j]/√ℓ':>16}{'規約B Σ[i-1:j-1]/√ℓ':>20}")
for (i,j) in spans:
    ch=bytes(raw[i:j]).decode('utf-8','replace'); ell=j-i
    A=sum(range(i,j))/math.sqrt(ell)                    # entropies[i:j]
    B=(sum(range(i-1,j-1))/math.sqrt(ell)) if i>0 else float('nan')  # entropies[i-1:j-1]
    ov=float(out[i-1]) if i>0 else float('nan')
    print(f"{ch:5}[{i:2d}:{j:2d}]   {ell:<4}{ov:>12.3f}{A:>16.3f}{B:>20.3f}")

print("\n=== 「り」(bytes 9,10,11) 焦点 ===")
print(f"実コード out[8]            = {float(out[8]):.4f}")
print(f"規約A entropies[9,10,11]=30 → 30/√3 = {30/math.sqrt(3):.4f}")
print(f"規約B entropies[8,9,10]=27  → 27/√3 = {27/math.sqrt(3):.4f}")
v=float(out[8])
verdict = "規約A(自位置 entropies[i:j])" if abs(v-30/math.sqrt(3))<1e-3 else ("規約B(lead-1)" if abs(v-27/math.sqrt(3))<1e-3 else "不明")
print(f"→ 実コードの判定: {verdict}")
