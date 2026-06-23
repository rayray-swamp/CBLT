"""
追加検証 v2 用データ: 長い held-out 文書を英35+日35（学習非含有・membership確認）。
コンテキストいっぱい検証のため min_chars を大きめに。
ログインノード（HFアクセス可・スレッド制限下）で実行。
"""
import json, os, hashlib, sys
import pyarrow as pa

BASE = "/gs/bs/tga-RLA/yoshida/blt_data/multilingual"
OUT = "/gs/bs/tga-RLA/yoshida/blt_data/eval"


def th(t): return hashlib.md5(t.encode("utf-8")).hexdigest()


def main():
    from datasets import load_dataset
    os.makedirs(OUT, exist_ok=True)
    out = open(f"{OUT}/heldout_v2.jsonl", "w", encoding="utf-8")
    for lg, target in [("en", 35), ("ja", 35)]:
        arrow = f"{BASE}/preprocess/culturax_{lg}/transformer_100m/culturax_{lg}.chunk.00.jsonl.shard_0.arrow"
        train_hashes = set()
        with pa.ipc.open_file(arrow) as r:
            for x in r.read_all().column("text"):
                train_hashes.add(th(x.as_py()))
        ds = load_dataset("uonlp/CulturaX", lg, split="train", streaming=True)
        got = 0; scanned = 0
        for doc in ds:
            scanned += 1
            if scanned > 400_000: break
            s = doc.get("text", "").strip()
            if len(s) < 1500:        # 長文のみ（コンテキストを埋める）
                continue
            if th(s) in train_hashes:
                continue
            out.write(json.dumps({"text": s, "lang": lg}, ensure_ascii=False) + "\n")
            got += 1
            if got >= target:
                break
        print(f"{lg}: {got} docs (scanned {scanned})", flush=True)
    out.close()
    n = sum(1 for _ in open(f"{OUT}/heldout_v2.jsonl"))
    print(f"heldout_v2.jsonl: {n} docs -> {OUT}/heldout_v2.jsonl")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
