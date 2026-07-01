"""⑥ t1: byte予算窓化（sequence_iterator）の不変条件テスト（合成patch列・GPU不要）。
Chat Opt A の核: 窓 ≤ byte_budget・パッチ分割なし・破棄ゼロ（ストリーム消費==窓合算）・順序保存。
使い方: python scripts/test_byte_budget_window.py
"""
import numpy as np
from bytelatent.data.iterators.sequence_iterator import SequenceIterator, SequencePackingArgs


class MockExample:
    def __init__(self, tokens, patch_lengths):
        self.tokens = tokens
        self.mask = [True] * len(tokens)
        self.patch_lengths = patch_lengths


class MockPreproc:
    add_patches = True

    def __init__(self, examples):
        self.examples = examples

    def create_iter(self):
        for e in self.examples:   # 有限ストリーム（一意id）＝破棄/順序を厳密検証
            yield e

    def get_state(self):
        return None


def main():
    budget = 4096
    rng = np.random.default_rng(0)
    examples = []
    gid = 0
    MAXPL = 11
    for _ in range(12000):   # 2000窓分（~8.2Mbyte）に十分な有限ストリーム
        npatch = int(rng.integers(50, 200))
        pls = rng.integers(1, MAXPL + 1, size=npatch).tolist()  # patch長 1..11 byte
        ntok = sum(pls)
        tokens = list(range(gid, gid + ntok)); gid += ntok      # 一意連番＝順序/破棄検証用
        examples.append(MockExample(tokens, pls))

    si = SequenceIterator(
        MockPreproc(examples), rng_state=None,
        sequence_packing_args=SequencePackingArgs(output_seq_len=999, buffer_size=8, byte_budget=budget),
    )
    it = si.create_iter()
    windows = [next(it) for _ in range(2000)]

    fails = []
    all_tokens = []
    for w in windows:
        pl, tok = w.patch_lengths, w.tokens
        if sum(pl) != len(tok): fails.append(f"sum(pl)={sum(pl)} != len(tok)={len(tok)}")
        if len(tok) > budget: fails.append(f"窓 {len(tok)} > budget {budget}")
        if pl and max(pl) > MAXPL: fails.append(f"分割の疑い: patch長 {max(pl)} > 元最大 {MAXPL}")
        if pl and pl[0] <= 0: fails.append(f"先頭patch {pl[0]} <=0")
        if tok and (max(tok) - min(tok) + 1) != len(tok): fails.append("窓内tokenが非連続（分割/欠落）")
        all_tokens.extend(tok)

    # 破棄ゼロ＆順序: yieldされた全tokenの集合が 0..K の連続（重複・欠落なし）
    st = sorted(all_tokens)
    contiguous = (st == list(range(st[0], st[0] + len(st))))
    no_dup = (len(set(all_tokens)) == len(all_tokens))
    if st[0] != 0: fails.append(f"先頭token {st[0]} != 0")
    if not contiguous: fails.append("全token集合が非連続（=窓間で破棄/gap）")
    if not no_dup: fails.append("token重複（=二重消費）")

    wl = [len(w.tokens) for w in windows]
    print(f"=== ⑥ t1 byte予算窓 不変条件（{len(windows)}窓・budget={budget}）===")
    print(f"窓バイト長: min={min(wl)} p50={int(np.median(wl))} max={max(wl)} (全て ≤{budget})")
    print(f"窓patch数: min={min(len(w.patch_lengths) for w in windows)} "
          f"max={max(len(w.patch_lengths) for w in windows)} (可変)")
    print(f"総消費token {len(all_tokens):,} = 連番 0..{st[-1]:,}（破棄ゼロ・重複なし・順序保存）")
    if fails:
        print("❌ FAIL:")
        for f in set(fails): print(f"   - {f}")
        raise SystemExit(1)
    print("✅ t1 PASS: 窓≤budget・分割なし・破棄ゼロ・順序保存・先頭patch>0")


class LoopMock:
    """LoopingIterator 相当（無限周回）。yield したバイト数を数える。"""
    add_patches = True

    def __init__(self, examples):
        self.examples = examples
        self.yielded_bytes = 0

    def create_iter(self):
        while True:
            for e in self.examples:
                self.yielded_bytes += len(e.tokens)
                yield e

    def get_state(self):
        return None


def t4():
    """周回境界で破棄ゼロ: mock が短い有限データを何周もループ。窓合算バイト == mock yield
    バイト − 残バッファ(<budget) を確認（境界で1チャンク落ちれば差≥budgetになる）。"""
    budget = 4096
    rng = np.random.default_rng(1)
    examples = []
    for _ in range(80):  # 短い（何周もループ＝境界を多数通過）
        pls = rng.integers(1, 12, size=int(rng.integers(30, 120))).tolist()
        examples.append(MockExample(list(range(sum(pls))), pls))
    mock = LoopMock(examples)
    si = SequenceIterator(
        mock, rng_state=None,
        sequence_packing_args=SequencePackingArgs(output_seq_len=999, buffer_size=8, byte_budget=budget),
    )
    it = si.create_iter()
    win_bytes = 0
    nwin = 0
    for _ in range(3000):  # 短データを多数周回
        w = next(it)
        win_bytes += len(w.tokens)
        nwin += 1
    diff = mock.yielded_bytes - win_bytes  # 残バッファ（まだ窓化されていない分）
    print(f"\n=== ⑥ t4 周回境界 破棄ゼロ（{nwin}窓・周回多数）===")
    print(f"mock yield {mock.yielded_bytes:,}B / 窓合算 {win_bytes:,}B / 残バッファ {diff}B (期待 0≤diff<{budget})")
    if 0 <= diff < budget:
        print("✅ t4 PASS: 周回境界でも破棄ゼロ（消費==窓化＋残<budget）")
    else:
        print(f"❌ t4 FAIL: diff={diff} が [0,{budget}) 外＝境界で破棄/重複")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
    t4()
