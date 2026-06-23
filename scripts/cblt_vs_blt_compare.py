"""
byte-BLT vs CBLT パッチ境界比較スクリプト

CBLT の UTF-8 文字境界保護が正しく動作しているかを確認する。
同じエントロピーを entropy モード (byte-BLT) と cblt モードに与えて
パッチ境界の違いを可視化する。

使い方:
  python scripts/cblt_vs_blt_compare.py
  python scripts/cblt_vs_blt_compare.py --text "任意のテキスト" --threshold 0.5 --seed 42
"""
import typer
import torch
from bytelatent.data.patcher import PatcherArgs, PatchingModeEnum

OFFSET = 3  # bytelatent/tokenizers/constants.py

app = typer.Typer()


def get_char_lead_positions(raw: bytes) -> list[int]:
    """UTF-8 文字のリードバイト位置一覧を返す。"""
    return [i for i, b in enumerate(raw) if (b & 0xC0) != 0x80]


def get_patch_starts(patch_lengths: list[int], seq_len: int) -> list[int]:
    starts = [0]
    pos = 0
    for l in patch_lengths:
        pos += l
        if pos < seq_len:
            starts.append(pos)
    return starts


def count_mid_char_breaks(starts: list[int], char_leads: list[int], skip_first: int = 2) -> int:
    """先頭 skip_first 個の強制境界を除いた文字内カット数を返す。"""
    return sum(1 for s in starts[skip_first:] if s not in char_leads)


@app.command()
def main(
    text: str = typer.Option("BLTは日本語テストABC", help="入力テキスト"),
    threshold: float = typer.Option(0.5, help="エントロピー閾値"),
    seed: int = typer.Option(42, help="乱数シード (ダミーエントロピー用)"),
    aggregation: str = typer.Option("sqrt", help="CBLT 集約方式 (sqrt|sum|avg)"),
):
    raw = text.encode("utf-8")
    tokens = torch.tensor([[b + OFFSET for b in raw]], dtype=torch.long)
    torch.manual_seed(seed)
    entropies = torch.rand(1, len(raw))

    print(f"入力テキスト : {repr(text)}")
    print(f"バイト列 ({len(raw)} B): {list(raw)}")
    print(f"threshold={threshold}, seed={seed}, cblt_aggregation={aggregation}")
    print()

    char_leads = get_char_lead_positions(raw)
    print(f"文字リードバイト位置: {char_leads}")
    print()

    # byte-BLT (entropy モード)
    blt_patcher = PatcherArgs(
        patching_mode=PatchingModeEnum.entropy,
        threshold=threshold,
        realtime_patching=False,
    ).build()
    pl_blt, _ = blt_patcher.patch(tokens, entropies=entropies)
    starts_blt = get_patch_starts(pl_blt[0].tolist(), len(raw))

    # CBLT
    cblt_patcher = PatcherArgs(
        patching_mode=PatchingModeEnum.cblt,
        threshold=threshold,
        realtime_patching=False,
        cblt_aggregation=aggregation,
    ).build()
    pl_cblt, _ = cblt_patcher.patch(tokens, entropies=entropies)
    starts_cblt = get_patch_starts(pl_cblt[0].tolist(), len(raw))

    print("=== byte-BLT (entropy) ===")
    print(f"  patch_lengths : {pl_blt[0].tolist()}")
    print(f"  patch_starts  : {starts_blt}")
    blt_breaks = count_mid_char_breaks(starts_blt, char_leads)
    print(f"  文字内カット数 (先頭2除く): {blt_breaks}")
    print()

    print(f"=== CBLT ({aggregation}) ===")
    print(f"  patch_lengths : {pl_cblt[0].tolist()}")
    print(f"  patch_starts  : {starts_cblt}")
    cblt_breaks = count_mid_char_breaks(starts_cblt, char_leads)
    print(f"  文字内カット数 (先頭2除く): {cblt_breaks}")
    print()

    # 可視化: どのバイトでパッチが切れるか
    print("=== バイト境界可視化 (| = パッチ境界, * = 文字境界) ===")
    for label, starts in [("byte-BLT", starts_blt), ("CBLT   ", starts_cblt)]:
        vis = ""
        for i, b in enumerate(raw):
            if i in starts:
                vis += "|"
            elif i in char_leads:
                vis += "*"
            else:
                vis += " "
            try:
                vis += chr(b) if 0x20 <= b <= 0x7E else f"\\x{b:02x}"
            except Exception:
                vis += "?"
        print(f"  {label}: {vis}")

    print()
    if cblt_breaks == 0:
        print("✅ CBLT: 文字境界保護が正しく動作（先頭2強制境界を除き文字内カットなし）")
        print("   ※ position 1 の強制カット（first_ids=[0,1] によるBLT仕様）は byte-BLT と共通のアーティファクト")
    else:
        print(f"⚠️  CBLT: {cblt_breaks} 件の文字内カットが発生（要調査）")


if __name__ == "__main__":
    app()
