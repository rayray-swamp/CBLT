"""
段階2-A: 8言語 Wikipedia 実DL（英語のみ記事数で半減）+ held-out membership除外
出力: blt_data/largecorpus/wiki/{lang}.jsonl  （{"text","lang","source":"wiki"}）

確定方針(§10): 各言語 Wikipedia 全量 + 英語のみ半減（6.41M→3.20M ランダム抽出）。
membership: 既存 held-out eval（heldout.jsonl / heldout_v2.jsonl）と sha1 で disjoint。
CPU のみ（ストリーミングDL + JSONL書き出し）。計算ノード(qsub)で実行。
"""
import json, os, sys, hashlib, random

LANGS = ["en", "ru", "ar", "ja", "zh", "ko", "hi", "th"]
EN_KEEP_PROB = 0.5            # 英語は記事数で半減
MIN_CHARS = 50               # 空・リダイレクト的スタブのみ除外（実質全量）
OUT = "/gs/bs/tga-RLA/yoshida/blt_data/largecorpus/wiki"
EVAL = "/gs/bs/tga-RLA/yoshida/blt_data/eval"


def sha(t): return hashlib.md5(t.encode("utf-8")).hexdigest()


def load_heldout_hashes():
    hs = set()
    for fn in ["heldout.jsonl", "heldout_v2.jsonl"]:
        p = os.path.join(EVAL, fn)
        if os.path.exists(p):
            for line in open(p):
                try:
                    hs.add(sha(json.loads(line)["text"]))
                except Exception:
                    pass
    return hs


def main():
    from datasets import load_dataset
    os.makedirs(OUT, exist_ok=True)
    held = load_heldout_hashes()
    print(f"held-out membership: {len(held)} hashes")
    rng = random.Random(777)

    summary = []
    for lg in LANGS:
        ds = load_dataset("wikimedia/wikipedia", f"20231101.{lg}", split="train", streaming=True)
        out_path = os.path.join(OUT, f"{lg}.jsonl")
        n_docs = 0; n_bytes = 0; n_skip_short = 0; n_skip_half = 0; n_skip_member = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for doc in ds:
                text = (doc.get("text") or "").strip()
                if len(text) < MIN_CHARS:
                    n_skip_short += 1; continue
                if lg == "en" and rng.random() >= EN_KEEP_PROB:
                    n_skip_half += 1; continue          # 英語半減
                if sha(text) in held:
                    n_skip_member += 1; continue          # held-out 除外
                f.write(json.dumps({"text": text, "lang": lg, "source": "wiki"}, ensure_ascii=False) + "\n")
                n_docs += 1
                n_bytes += len(text.encode("utf-8"))
                if n_docs % 200000 == 0:
                    print(f"  {lg}: {n_docs:,} docs / {n_bytes/1e9:.2f} GB", flush=True)
        sz = os.path.getsize(out_path)
        summary.append((lg, n_docs, n_bytes, sz, n_skip_short, n_skip_half, n_skip_member))
        print(f"{lg}: docs={n_docs:,} text_bytes={n_bytes:,} ({n_bytes/1e9:.2f}GB) file={sz/1e9:.2f}GB "
              f"skip(short={n_skip_short},half={n_skip_half},member={n_skip_member})", flush=True)

    print("\n===== Wikipedia 実測サマリ =====")
    print(f"{'lang':4} {'docs':>10} {'text_GB':>9} {'file_GB':>9}")
    tb = 0
    for lg, nd, nb, sz, ss, sh, sm in summary:
        print(f"{lg:4} {nd:>10,} {nb/1e9:>9.2f} {sz/1e9:>9.2f}")
        tb += nb
    print(f"{'計':4} {'':>10} {tb/1e9:>9.2f}")
    sys.stdout.flush()
    os._exit(0)  # HF streaming スレッド終了時 core dump 回避


if __name__ == "__main__":
    main()
