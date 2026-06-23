"""
held-out eval 用データ生成（リーク切り分けのクロス比較用）

2種類の JSONL を作る:
  1. heldout.jsonl    — 学習で見ていない CulturaX doc（学習 arrow に含まれない doc）
  2. train_sample.jsonl — 学習に使った arrow から数十 doc 抽出

held-out は「学習 arrow の text 集合に含まれない」ことを membership で厳密確認。
ログインノード（HF アクセス可）で実行する。

使い方:
  python scripts/make_heldout_eval_data.py
"""
import json, os, hashlib, sys
import typer
import pyarrow as pa

app = typer.Typer()

BASE = "/gs/bs/tga-RLA/yoshida/blt_data/multilingual"
OUT = "/gs/bs/tga-RLA/yoshida/blt_data/eval"


def text_hash(t: str) -> str:
    return hashlib.md5(t.encode("utf-8")).hexdigest()


@app.command()
def main(
    langs: str = typer.Option("en,ja,zh,th", help="対象言語（カンマ区切り）"),
    heldout_chars_per_lang: int = typer.Option(80_000, help="held-out 各言語の目標文字数"),
    train_sample_docs_per_lang: int = typer.Option(30, help="train-sample 各言語 doc 数"),
    max_stream: int = typer.Option(300_000, help="ストリーミング最大走査 doc 数（安全弁）"),
    min_chars: int = typer.Option(200, help="学習と同じ最低文字数フィルタ"),
):
    from datasets import load_dataset

    os.makedirs(OUT, exist_ok=True)
    lang_list = langs.split(",")
    ho_path = os.path.join(OUT, "heldout.jsonl")
    ts_path = os.path.join(OUT, "train_sample.jsonl")

    ho_f = open(ho_path, "w", encoding="utf-8")
    ts_f = open(ts_path, "w", encoding="utf-8")

    for lg in lang_list:
        # 学習 arrow の text ハッシュ集合
        arrow = f"{BASE}/preprocess/culturax_{lg}/transformer_100m/culturax_{lg}.chunk.00.jsonl.shard_0.arrow"
        train_hashes = set()
        train_texts = []
        with pa.ipc.open_file(arrow) as r:
            t = r.read_all()
            for x in t.column("text"):
                s = x.as_py()
                train_hashes.add(text_hash(s))
                if len(train_texts) < train_sample_docs_per_lang:
                    train_texts.append(s)
        # train-sample 書き出し
        for s in train_texts:
            ts_f.write(json.dumps({"text": s, "lang": lg}, ensure_ascii=False) + "\n")

        # held-out: stream して membership で非学習 doc を収集
        ds = load_dataset("uonlp/CulturaX", lg, split="train", streaming=True)
        got_chars = 0
        scanned = 0
        ho_docs = 0
        for doc in ds:
            scanned += 1
            if scanned > max_stream:
                break
            s = doc.get("text", "").strip()
            if len(s) < min_chars:
                continue
            if text_hash(s) in train_hashes:
                continue  # 学習に含まれる → スキップ
            ho_f.write(json.dumps({"text": s, "lang": lg}, ensure_ascii=False) + "\n")
            got_chars += len(s)
            ho_docs += 1
            if got_chars >= heldout_chars_per_lang:
                break
        print(f"{lg}: held-out {ho_docs} docs / {got_chars:,} chars (scanned {scanned}), "
              f"train-sample {len(train_texts)} docs", flush=True)

    ho_f.close()
    ts_f.close()
    print(f"\nheld-out  -> {ho_path}")
    print(f"train-sample -> {ts_path}")
    sys.stdout.flush()
    os._exit(0)  # HF streaming スレッドの終了時 core dump 回避


if __name__ == "__main__":
    app()
