"""
日本語 smoke データ生成スクリプト (タスクB)
HuggingFace datasets から Wikipedia ja をストリーミングして JSONL に書き出す。
ディスクには最終 JSONL だけが残り、大量 DL は行わない。

使い方:
  python scripts/make_smoke_ja.py
  python scripts/make_smoke_ja.py --n-docs 5000 --out-dir /path/to/smoke_ja
"""
import json
import os
import sys

import typer

app = typer.Typer()


@app.command()
def main(
    n_docs: int = typer.Option(20_000, help="取得するドキュメント数"),
    out_dir: str = typer.Option(
        "/gs/bs/tga-RLA/yoshida/blt_data/smoke_ja",
        help="出力ディレクトリ",
    ),
    dataset_name: str = typer.Option(
        "wikimedia/wikipedia", help="HuggingFace dataset name"
    ),
    dataset_config: str = typer.Option(
        "20231101.ja", help="HuggingFace dataset config"
    ),
    min_chars: int = typer.Option(200, help="最低文字数（短すぎる記事をスキップ）"),
):
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: datasets パッケージが必要です: pip install datasets", file=sys.stderr)
        raise typer.Exit(1)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "wikipedia_ja_smoke.chunk.00.jsonl")

    print(f"ストリーミング: {dataset_name}/{dataset_config}")
    print(f"目標: {n_docs} docs -> {out_path}")

    ds = load_dataset(dataset_name, dataset_config, split="train", streaming=True)

    written = 0
    skipped = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for doc in ds:
            text = doc.get("text", "").strip()
            if len(text) < min_chars:
                skipped += 1
                continue
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            written += 1
            if written % 1000 == 0:
                print(f"  {written}/{n_docs} docs written (skipped {skipped})", flush=True)
            if written >= n_docs:
                break

    print(f"完了: {written} docs -> {out_path}")
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"ファイルサイズ: {size_mb:.1f} MB")


if __name__ == "__main__":
    app()
