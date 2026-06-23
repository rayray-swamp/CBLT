"""
多言語 JSONL → placeholder Arrow 一括変換スクリプト
make_multilingual_data.py で生成した各言語 JSONL を Arrow 化する。

使い方:
  python scripts/make_multilingual_arrows.py
  python scripts/make_multilingual_arrows.py --lang ja  # 1言語のみ
"""
import os
import re
import sys
import typer
import jsonlines
import numpy as np
import pyarrow as pa
from typing import Optional

app = typer.Typer()

ENTROPY_MODEL_NAME = "transformer_100m"
SCHEMA = pa.schema([
    pa.field("sample_id", pa.string(), nullable=False),
    pa.field("text", pa.string(), nullable=False),
    pa.field("entropies", pa.list_(pa.float16()), nullable=False),
])
PLACEHOLDER_ENTROPIES = np.array([], dtype=np.float16)

LANGS = ["en", "ru", "ar", "ja", "zh", "ko", "hi", "th"]


def convert_jsonl(jsonl_path: str, preprocess_dir: str, batch_size: int = 2000):
    basename = os.path.basename(jsonl_path)
    parts = re.match(r"(.+)\.chunk\.[0-9]+\.jsonl", basename)
    assert parts, f"ファイル名が *.chunk.*.jsonl パターンに合わない: {basename}"
    dataset = parts.group(1)

    data_dir = os.path.join(preprocess_dir, dataset, ENTROPY_MODEL_NAME)
    os.makedirs(data_dir, exist_ok=True)
    output_file = os.path.join(data_dir, f"{basename}.shard_0.arrow")
    complete_file = f"{output_file}.complete"

    if os.path.exists(complete_file):
        print(f"  スキップ（完了済み）: {output_file}")
        return

    print(f"  読み込み: {jsonl_path}")
    print(f"  書き出し: {output_file}")

    id_buf, text_buf, entropy_buf = [], [], []
    total = 0

    with open(output_file, "wb") as sink:
        with pa.ipc.new_file(sink, SCHEMA) as writer:
            with jsonlines.open(jsonl_path) as reader:
                for i, doc in enumerate(reader):
                    text = doc.get("text", doc.get("content", ""))
                    id_buf.append(str(i))
                    text_buf.append(text)
                    entropy_buf.append(PLACEHOLDER_ENTROPIES)
                    total += 1
                    if len(id_buf) == batch_size:
                        writer.write(pa.record_batch(
                            {"sample_id": id_buf, "text": text_buf, "entropies": entropy_buf},
                            schema=SCHEMA,
                        ))
                        id_buf, text_buf, entropy_buf = [], [], []
                if id_buf:
                    writer.write(pa.record_batch(
                        {"sample_id": id_buf, "text": text_buf, "entropies": entropy_buf},
                        schema=SCHEMA,
                    ))

    open(complete_file, "w").close()
    size_mb = os.path.getsize(output_file) / 1e6
    print(f"  完了: {total} records, {size_mb:.1f} MB")


@app.command()
def main(
    data_dir: str = typer.Option(
        "/gs/bs/tga-RLA/yoshida/blt_data/multilingual",
        help="make_multilingual_data.py の出力先",
    ),
    preprocess_dir: str = typer.Option(
        "/gs/bs/tga-RLA/yoshida/blt_data/multilingual/preprocess",
        help="Arrow 出力先",
    ),
    lang: Optional[str] = typer.Option(None, help="特定言語のみ (未指定で全8言語)"),
):
    langs = [lang] if lang else LANGS
    for lg in langs:
        jsonl_glob = os.path.join(data_dir, f"culturax_{lg}", f"culturax_{lg}.chunk.00.jsonl")
        if not os.path.exists(jsonl_glob):
            print(f"{lg}: JSONL なし → スキップ ({jsonl_glob})")
            continue
        print(f"\n=== {lg} ===")
        convert_jsonl(jsonl_glob, preprocess_dir)


if __name__ == "__main__":
    app()
