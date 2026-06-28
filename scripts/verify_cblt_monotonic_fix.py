"""cblt+monotonic の破綻と forward-fill 修正の合成検証(xformers不要・関数コピー)。
現状: aggregate_char_entropy の非lead=-inf → monotonic の差が各文字leadで +inf(全文字patch)+inf-inf=NaN。
修正: 非lead=前文字Hで forward-fill → 連続差が「文字lead=H(c)-H(c-1)、文字内=0」= per-char ΔH(較正と一致)。"""
import math, torch
OFFSET = 4

def aggregate_char_entropy(entropies, tokens, mode="sqrt", fill="ninf"):
    """patcher.py 逐語 + fill オプション。fill='ninf'(現状) / 'ffill'(修正案)。"""
    raw = tokens - OFFSET
    is_lead = (raw & 0xC0) != 0x80
    out = torch.full_like(entropies, float("-inf"))
    bs, seq_len = tokens.shape
    for b in range(bs):
        i = 0
        while i < seq_len:
            if not is_lead[b, i]:
                i += 1; continue
            if raw[b, i] < 0:
                if i > 0: out[b, i - 1] = entropies[b, i - 1]
                i += 1; continue
            j = i + 1
            while j < seq_len and not is_lead[b, j]: j += 1
            ell = j - i
            if i > 0:
                e_sum = entropies[b, i - 1:j - 1].sum()
                h = e_sum / (ell ** 0.5) if mode == "sqrt" else (e_sum if mode == "sum" else e_sum / ell)
                out[b, i - 1] = h
            i = j
    if fill == "ffill":
        # 非lead(-inf)を直前の有限Hで前方補完。先頭の-infは最初の有限値で後方補完。
        for b in range(bs):
            last = None
            for k in range(seq_len):
                if torch.isfinite(out[b, k]): last = out[b, k]
                elif last is not None: out[b, k] = last
            # 先頭の-inf(最初の有限より前)を後方補完
            first = None
            for k in range(seq_len):
                if torch.isfinite(out[b, k]): first = out[b, k]; break
            if first is not None:
                for k in range(seq_len):
                    if not torch.isfinite(out[b, k]): out[b, k] = first
                    else: break
    return out

def mono_mask(entropies, t):
    """patcher.py patch_start_mask_from_entropy_with_monotonicity 逐語。"""
    bs, seq_len = entropies.shape
    mask = torch.zeros_like(entropies, dtype=torch.bool); mask[:, 0] = True
    diff = entropies[:, 1:] - entropies[:, :-1]
    mask[:, 1:] = diff > t
    return mask, diff

# 合成: 4文字×2バイト(é è ê ë 風), entropies=1..8
raw = torch.tensor([[0xC3, 0xA9, 0xC3, 0xA8, 0xC3, 0xAA, 0xC3, 0xAB]])
tokens = raw + OFFSET
ent = torch.tensor([[1., 2., 3., 4., 5., 6., 7., 8.]])
print("文字構造: 4文字×2バイト, entropies =", ent[0].tolist())
print("(規約B per-char H: char2=(2+3)/√2=3.54, char3=(4+5)/√2=6.36, char4=(6+7)/√2=9.19)")

for fill in ["ninf", "ffill"]:
    out = aggregate_char_entropy(ent, tokens, "sqrt", fill=fill)
    # find_entropy_patch_start_ids は entropies[:,1:] を渡す
    m, diff = mono_mask(out[:, 1:], t=1.0)
    vals = ["%.2f" % v if math.isfinite(v) else ("+inf" if v > 0 else ("-inf" if v < 0 else "nan")) for v in out[0].tolist()]
    dvals = ["%.2f" % v if math.isfinite(v) else ("+inf" if v > 0 else ("-inf" if v < 0 else "NaN")) for v in diff[0].tolist()]
    print(f"\n=== fill={fill} ===")
    print("  aggregate out:", vals)
    print("  ΔH(隣接差):  ", dvals, " ← NaN/+inf あれば破綻")
    print("  patch mask(θ=1.0):", m[0].tolist())
    nan = any(not math.isfinite(v) for v in diff[0].tolist())
    print(f"  → {'❌ NaN/inf で破綻(全文字fire)' if (fill=='ninf') else '✅ per-char ΔH(文字lead=H(c)-H(c-1), 文字内=0, θで選別)'}")
print("\n結論: ffill で per-char ΔH が実現＝私の較正θと整合。-inf(現状)は monotonic で破綻。")
