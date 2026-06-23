"""
多言語 CulturaX データ取得スクリプト（タスク2）
各言語を配分B（100M文字均等）でストリーミング取得し JSONL に書き出す。

配分B (各言語100M文字):
  en: 100M chars → ~100MB  (1 byte/char)
  ru: 100M chars → ~200MB  (2 byte/char)
  ar: 100M chars → ~200MB  (2 byte/char)
  ja: 100M chars → ~300MB  (3 byte/char)
  zh: 100M chars → ~300MB  (3 byte/char)
  ko: 100M chars → ~300MB  (3 byte/char)
  hi: 100M chars → ~300MB  (3 byte/char)
  th: 100M chars → ~300MB  (3 byte/char)
  合計: ~2.0B bytes

使い方:
  python scripts/make_multilingual_data.py
  python scripts/make_multilingual_data.py --lang ja --target-chars 1000000  # 個別テスト
"""
import json
import os
import sys
import typer
from typing import Optional

app = typer.Typer()

# 配分B: 各言語100M文字
LANG_CONFIGS = {
    "en": {"target_chars": 100_000_000, "dataset_config": "en"},
    "ru": {"target_chars": 100_000_000, "dataset_config": "ru"},
    "ar": {"target_chars": 100_000_000, "dataset_config": "ar"},
    "ja": {"target_chars": 100_000_000, "dataset_config": "ja"},
    "zh": {"target_chars": 100_000_000, "dataset_config": "zh"},
    "ko": {"target_chars": 100_000_000, "dataset_config": "ko"},
    "hi": {"target_chars": 100_000_000, "dataset_config": "hi"},
    "th": {"target_chars": 100_000_000, "dataset_config": "th"},
}


@app.command()
def main(
    out_dir: str = typer.Option(
        "/gs/bs/tga-RLA/yoshida/blt_data/multilingual",
        help="出力ディレクトリ (言語別サブディレクトリを作成)",
    ),
    lang: Optional[str] = typer.Option(None, help="特定言語のみ取得 (未指定で全8言語)"),
    target_chars: Optional[int] = typer.Option(None, help="文字数上書き (テスト用)"),
    min_chars: int = typer.Option(200, help="最低文字数（短い記事をスキップ）"),
    log_every: int = typer.Option(10_000, help="進捗ログ間隔（文字数）"),
):
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: datasets パッケージが必要です", file=sys.stderr)
        raise typer.Exit(1)

    langs_to_process = [lang] if lang else list(LANG_CONFIGS.keys())

    for lg in langs_to_process:
        cfg = LANG_CONFIGS[lg]
        n_chars_target = target_chars if target_chars is not None else cfg["target_chars"]

        # ディレクトリ名 = source 名 (culturax_{lg})。train.py が root_dir/{source}
        # の存在を確認するため、source 名とディレクトリ名を一致させる必要がある
        lang_dir = os.path.join(out_dir, f"culturax_{lg}")
        os.makedirs(lang_dir, exist_ok=True)
        out_path = os.path.join(lang_dir, f"culturax_{lg}.chunk.00.jsonl")

        print(f"\n=== {lg} ===")
        print(f"目標: {n_chars_target:,} 文字 → {out_path}")

        ds = load_dataset(
            "uonlp/CulturaX",
            cfg["dataset_config"],
            split="train",
            streaming=True,
        )

        total_chars = 0
        total_bytes = 0
        n_docs = 0
        n_skipped = 0

        with open(out_path, "w", encoding="utf-8") as f:
            for doc in ds:
                text = doc.get("text", "").strip()
                if len(text) < min_chars:
                    n_skipped += 1
                    continue
                f.write(json.dumps({"text": text, "url": doc.get("url", "")}, ensure_ascii=False) + "\n")
                total_chars += len(text)
                total_bytes += len(text.encode("utf-8"))
                n_docs += 1
                if total_chars >= n_chars_target and n_chars_target > 0:
                    break
                if n_docs % log_every == 0:
                    print(f"  {lg}: {total_chars:,} chars / {total_bytes/1e6:.1f} MB ({n_docs} docs)", flush=True)

        size_mb = os.path.getsize(out_path) / 1e6
        bpc = total_bytes / total_chars if total_chars else 0
        print(f"完了: {n_docs} docs, {total_chars:,} chars, {total_bytes/1e6:.1f} MB")
        print(f"  B/char={bpc:.2f}, ファイル={size_mb:.1f} MB, スキップ={n_skipped}")

    # HF datasets のストリーミングスレッドがインタープリタ終了時に
    # PyGILState_Release で core dump する既知問題を回避するため、
    # Python のクリーンアップをスキップして即時終了する。
    # (ファイルは with ブロックで既に flush/close 済みなので安全)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    app()
