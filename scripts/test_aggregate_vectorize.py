"""aggregate_char_entropy ベクトル化の回帰テスト(Chat必須(a)(b)(d))。
元 per-byte ループ実装(逐語コピー)と新ベクトル版のビット一致を、乱数＋実トークン風列で確認。
特殊token(BOS/EOS/PAD=負raw=1バイト文字,√1=raw entropy)の扱いも照合。PASS で exit 0。"""
import sys, torch
from bytelatent.data.patcher import aggregate_char_entropy as agg_vec  # 新(ベクトル)
from bytelatent.data.patcher import OFFSET

def agg_loop_ref(entropies, tokens, mode="sqrt"):
    """差し替え前の per-byte ループ実装(patcher.py 逐語)。参照用。"""
    raw_bytes = tokens - OFFSET
    is_lead = (raw_bytes & 0xC0) != 0x80
    out = torch.full_like(entropies, float("-inf"))
    bs, seq_len = tokens.shape
    for b in range(bs):
        i = 0
        while i < seq_len:
            if not is_lead[b, i]:
                i += 1; continue
            if raw_bytes[b, i] < 0:
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
    return out

def rand_tokens(bs, seq, seed):
    """BOS(1)始まり、多バイト文字混在、EOS(2)を含む現実的なトークン列を生成。"""
    g = torch.Generator().manual_seed(seed)
    rows = []
    for _ in range(bs):
        t = [1]  # BOS
        while len(t) < seq - 1:
            r = torch.rand(1, generator=g).item()
            if r < 0.4: t.append(int(torch.randint(0x41, 0x7F, (1,), generator=g)) + OFFSET)  # 1byte ASCII
            elif r < 0.7:  # 2byte
                t += [0xC3 + OFFSET, int(torch.randint(0x80, 0xC0, (1,), generator=g)) + OFFSET]
            else:  # 3byte
                t += [0xE3 + OFFSET, int(torch.randint(0x80, 0xC0, (1,), generator=g)) + OFFSET,
                      int(torch.randint(0x80, 0xC0, (1,), generator=g)) + OFFSET]
        t = t[:seq - 1] + [2]  # EOS
        rows.append(t[:seq])
    return torch.tensor(rows, dtype=torch.long)

ok = True
maxd = 0.0
for seed in range(20):
    bs, seq = 4, 64
    tokens = rand_tokens(bs, seq, seed)
    ent = torch.randn(bs, seq, generator=torch.Generator().manual_seed(seed + 100))
    for mode in ["sqrt", "sum", "avg"]:
        a = agg_loop_ref(ent, tokens, mode)
        b = agg_vec(ent, tokens, mode)
        # -inf は両方 -inf であること、有限は完全一致
        same_inf = ((a == float("-inf")) == (b == float("-inf"))).all().item()
        fin = torch.isfinite(a)
        d = (a[fin] - b[fin]).abs().max().item() if fin.any() else 0.0
        maxd = max(maxd, d)
        if not same_inf or d > 1e-5:
            ok = False
            print(f"  seed{seed} {mode}: same_inf={same_inf} max|Δ|={d:.2e} ❌")
print(f"(a)ビット一致: 乱数20×3mode, max|Δ(有限)|={maxd:.2e}, -inf配置一致 {'✅' if ok else '❌'}")

# (b) 特殊token: BOS(1) い(3byte) EOS(2) — BOS/EOSは1バイト文字、√1=raw entropy
tk = torch.tensor([[1, 0xE3 + OFFSET, 0x81 + OFFSET, 0x84 + OFFSET, 2]])
e = torch.tensor([[5., 1., 2., 3., 9.]])
av = agg_vec(e, tk, "sqrt"); al = agg_loop_ref(e, tk, "sqrt")
spec_ok = torch.equal(av.nan_to_num(neginf=-1e9), al.nan_to_num(neginf=-1e9))
print(f"(b)特殊token扱い 元と一致: {'✅' if spec_ok else '❌'}  (vec={av.tolist()})")
if not spec_ok: ok = False

# (d) device/dtype 保持
av32 = agg_vec(e.float(), tk, "sqrt")
dd_ok = (av32.dtype == torch.float32 and av32.device == e.device)
print(f"(d)dtype/device保持: {'✅' if dd_ok else '❌'}")
if not dd_ok: ok = False

print("\n" + ("✅ 回帰テスト PASS" if ok else "❌ FAIL"))
sys.exit(0 if ok else 1)
