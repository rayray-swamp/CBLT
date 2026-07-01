"""実コーパス(largecorpus_clean)で byte_r6 vs cblt_r6 の realized bytes/patch を測定。
8 shard 全部から sampling、precompute済 entropy で実patcher経路を回す。pooled＋言語別。
公平性チェック: byte と cblt が rate-matched(両~6) か、どの言語で patch 長が乖離するか。
使い方: python scripts/measure_bytes_per_patch.py [n_per_shard=150]
"""
import sys, glob, unicodedata
import torch, pyarrow as pa
from bytelatent.tokenizers.build_tokenizer import TokenizerArgs
from bytelatent.data.patcher import (
    aggregate_char_entropy, find_entropy_patch_start_ids,
    find_cblt_monotonic_patch_start_ids, OFFSET,
)

TOK = "/gs/bs/tga-RLA/yoshida/blt_data/tokenizer/tokenizer.model"
SHARDS = sorted(glob.glob(
    "/gs/bs/tga-RLA/yoshida/blt_data/largecorpus_clean/preprocess/corpus/transformer_100m/*.arrow"))
BYTE_TH, CBLT_TH = 0.6367, 0.0937  # blt_body_{byte,cblt}_r6.yaml


def lang_of(text):
    """支配的 script で言語推定（バケツ用の粗い判定）。"""
    cnt = {}
    for ch in text[:2000]:
        if ch.isspace() or not ch.isalpha():
            continue
        o = ord(ch)
        if 0x3040 <= o <= 0x309F: s = "ja"      # hiragana → 確実に日本語
        elif 0x30A0 <= o <= 0x30FF: s = "ja"     # katakana
        elif 0xAC00 <= o <= 0xD7A3: s = "ko"
        elif 0x0400 <= o <= 0x04FF: s = "ru"
        elif 0x0600 <= o <= 0x06FF: s = "ar"
        elif 0x0900 <= o <= 0x097F: s = "hi"
        elif 0x0E00 <= o <= 0x0E7F: s = "th"
        elif 0x4E00 <= o <= 0x9FFF: s = "han"    # 漢字（ja/zh 兼用）
        elif o < 0x80: s = "lat"                 # ASCII → en
        else: s = "other"
        cnt[s] = cnt.get(s, 0) + 1
    if not cnt: return "other"
    if cnt.get("ja", 0) > 0: return "ja"          # かなあれば日本語
    if cnt.get("han", 0) > max(cnt.get("lat", 0), 1): return "zh"  # 漢字主体→中国語
    dom = max(cnt, key=cnt.get)
    return {"lat": "en", "han": "zh"}.get(dom, dom)


def n_starts(ids, L):
    return len(sorted(set(int(x) for x in ids[0].tolist() if 0 <= int(x) < L)))


def main():
    n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    tok = TokenizerArgs(name="blt", init_kwargs={"bpe_tokenizer_path": TOK}).build()
    # acc[lang][method] = [tot_bytes, tot_patches, n_docs]
    acc = {}
    def add(lang, method, b, p):
        acc.setdefault(lang, {"byte": [0, 0, 0], "cblt": [0, 0, 0]})
        acc[lang][method][0] += b; acc[lang][method][1] += p
        if method == "byte": acc[lang][method][2] += 1
    nseen = 0
    for sp in SHARDS:
        r = pa.ipc.open_file(sp); b0 = r.get_batch(0)
        texts, ents = b0.column("text"), b0.column("entropies")
        step = max(1, b0.num_rows // n_per)
        for i in range(0, min(b0.num_rows, n_per * step), step):
            text = texts[i].as_py(); elist = ents[i].as_py()
            if len(text) < 20 or len(elist) < 5: continue
            lang = lang_of(text)
            tokens = torch.tensor(tok.encode(text), dtype=torch.long).unsqueeze(0)
            ent = torch.tensor(elist, dtype=torch.float32).unsqueeze(0)
            L = min(tokens.shape[1], ent.shape[1])
            tokens, ent = tokens[:, :L], ent[:, :L]
            # byte
            idb = find_entropy_patch_start_ids(ent, threshold=BYTE_TH, monotonicity=True)
            add(lang, "byte", L, n_starts(idb, L)); add("ALL", "byte", L, n_starts(idb, L))
            # cblt
            sc = aggregate_char_entropy(ent, tokens, "sqrt")
            idc = find_cblt_monotonic_patch_start_ids(sc, CBLT_TH)
            add(lang, "cblt", L, n_starts(idc, L)); add("ALL", "cblt", L, n_starts(idc, L))
            nseen += 1
    # 出力
    order = ["ALL", "en", "ru", "ja", "zh", "ar", "ko", "hi", "th", "other"]
    print(f"\n=== realized bytes/patch  ({nseen} docs, largecorpus_clean 実entropy・実patcher) ===")
    print(f"{'lang':6}{'n_doc':>7}{'byte b/p':>11}{'cblt b/p':>11}{'Δ(cblt-byte)':>14}")
    for lang in order:
        if lang not in acc: continue
        a = acc[lang]
        bb = a["byte"][0] / max(1, a["byte"][1])
        cb = a["cblt"][0] / max(1, a["cblt"][1])
        nd = a["byte"][2]
        print(f"{lang:6}{nd:>7}{bb:>11.3f}{cb:>11.3f}{cb-bb:>+14.3f}")
    print("\n注: b/p=bytes per patch。rate-matched なら byte≈cblt≈6。"
          "cblt b/p が小さい言語=cblt の方が patch 数多い（=その言語で global tx 細粒度）。")


if __name__ == "__main__":
    main()
