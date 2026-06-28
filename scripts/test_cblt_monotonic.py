"""cblt+monotonic 修正の必須ユニットテスト(Chat指定4点)。実patcher関数を使用。
①cblt+monotonic の patch境界が全て文字lead(mid_char_split=0) ②per-char ΔH>θ 参照と一致
③NaN/inf・空パッチ(size0) 無し ④byte arm/絶対閾値cblt の回帰(forward_fill=False で-inf不変)。
PASS で exit 0。"""
import sys, math, torch
from bytelatent.data.patcher import (
    aggregate_char_entropy, find_entropy_patch_start_ids,
    find_cblt_monotonic_patch_start_ids, patch_lengths_from_start_ids, OFFSET,
)

def build(raw_chars, ents):
    """raw_chars: list[bytes(1文字分)]、ents: 各バイトのentropy。tokens/entropies/char_leads を返す。"""
    raw = b"".join(raw_chars)
    tokens = torch.tensor([[b + OFFSET for b in raw]], dtype=torch.long)
    entropies = torch.tensor([ents], dtype=torch.float32)
    leads = []; p = 0
    for c in raw_chars: leads.append(p); p += len(c)
    return tokens, entropies, leads, len(raw)

def ref_per_char_starts(entropies, leads, n, theta, mode="sqrt"):
    """私の較正と同じ per-char ΔH 参照: H(c)=√ℓ集約(規約B), patch開始=文字c(ΔH>θ) + 先頭強制。"""
    H = []
    for ci, k in enumerate(leads):
        j = leads[ci + 1] if ci + 1 < len(leads) else n
        ell = j - k
        if ci == 0:  # 先頭文字はfind側 first_ids=[0,1]で強制、集約スキップ
            H.append(None); continue
        e_sum = float(entropies[0, k - 1:j - 1].sum())
        H.append(e_sum / (ell ** 0.5) if mode == "sqrt" else (e_sum if mode == "sum" else e_sum / ell))
    starts = {leads[0]}  # 先頭文字lead強制
    if len(leads) > 1: starts.add(leads[1])  # find_entropy_patch_start_ids first_ids=[0,1]
    for ci in range(2, len(leads)):
        if H[ci] is not None and H[ci - 1] is not None and (H[ci] - H[ci - 1]) > theta:
            starts.add(leads[ci])
    return starts

def run_cblt(tokens, entropies, theta, monotonic):
    scores = aggregate_char_entropy(entropies, tokens, "sqrt")
    if monotonic:
        ids = find_cblt_monotonic_patch_start_ids(scores, theta)
    else:
        ids = find_entropy_patch_start_ids(scores, threshold=theta, monotonicity=False)
    return scores, ids

# 合成: A(1) あ(3) い(3) é(2) B(1) = 10バイト
tokens, ent, leads, n = build(
    [b"\x41", b"\xe3\x81\x82", b"\xe3\x81\x84", b"\xc3\xa9", b"\x42"],
    [10., 1., 2., 3., 20., 4., 5., 30., 6., 7.],
)
is_lead = [( (tokens[0, i].item() - OFFSET) & 0xC0) != 0x80 for i in range(n)]
print(f"文字lead byte位置: {leads}  (n={n})")

ok = True
for theta in [-0.5, 0.0, 1.0, 3.0]:
    scores, ids = run_cblt(tokens, ent, theta, monotonic=True)
    starts = sorted(set(int(x) for x in ids[0].tolist() if 0 <= int(x) < n))
    # ① mid_char_split: 全 start が文字lead か（文字内境界ゼロ）
    mids = [s for s in starts if not is_lead[s]]
    # ② per-char ΔH>θ 参照と一致
    ref = sorted(ref_per_char_starts(ent, leads, n, theta))
    match = (starts == ref)
    # ③ NaN無し・patch_lengths が valid(assert通過)・空パッチ無し
    no_nan = not bool(torch.isnan(scores).any())
    try:
        plens = patch_lengths_from_start_ids(ids, n)
        valid = bool((plens >= 0).all()) and ids.shape[1] > 0
    except Exception as e:
        valid = False
    print(f"θ={theta:+.1f}: starts={starts} ref={ref} | ①mid_split={len(mids)} ②match={match} ③no_nan={no_nan}&valid={valid}")
    if mids or not match or not no_nan or not valid:
        ok = False

# ④ 回帰: aggregate_char_entropy は原実装と不変（H は lead_c-1 に finite, 他は -inf）
#    ＋ cblt 絶対閾値(monotonicity=False) と byte(entropy) 経路が動く
expected_h_pos = set(leads[ci] - 1 for ci in range(1, len(leads)))  # 文字c(c≥1)の H 配置位置
scores_def = aggregate_char_entropy(ent, tokens, "sqrt")
reg_agg = all(
    (math.isfinite(scores_def[0, i].item()) if i in expected_h_pos
     else scores_def[0, i].item() == float("-inf"))
    for i in range(n)
)
_, ids_abs = run_cblt(tokens, ent, 5.0, monotonic=False)            # cblt 絶対閾値
ids_byte = find_entropy_patch_start_ids(ent, threshold=2.0, monotonicity=True)  # byte arm
reg_paths = ids_abs.shape[1] > 0 and ids_byte.shape[1] > 0
print(f"④回帰: aggregate不変={reg_agg}  cblt絶対&byte経路動作={reg_paths}")
if not reg_agg or not reg_paths:
    ok = False

print("\n" + ("✅ 全テスト PASS（Chat必須4点）" if ok else "❌ FAIL"))
sys.exit(0 if ok else 1)
