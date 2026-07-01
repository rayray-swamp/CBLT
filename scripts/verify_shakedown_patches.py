"""shakedown patch検証(Chat③最重要・degenerate でないこと)。precompute済 slice Arrow の
保存entropyで実patcher経路を回し、cblt/byte の patch構造を検証。
cblt: mid_char_split=0・bytes/patch≈6・1字パッチ率低・n_patches≪n_chars。
byte: 「1パッチ/文字」でない(=空fallbackでなく実entropyで切れている)。NaN/inf無し。"""
import math, torch, typer
import pyarrow as pa
from bytelatent.tokenizers.build_tokenizer import TokenizerArgs
from bytelatent.data.patcher import (
    aggregate_char_entropy, find_entropy_patch_start_ids,
    find_cblt_monotonic_patch_start_ids, OFFSET,
)
app = typer.Typer()
TOK = "/gs/bs/tga-RLA/yoshida/blt_data/tokenizer/tokenizer.model"
import os
ARROW = os.environ.get("SHAKE_ARROW", "/gs/bs/tga-RLA/yoshida/blt_data/shakedown/preprocess/corpus/transformer_100m/corpus.chunk.00.jsonl.shard_0.arrow")
CBLT_TH6, BYTE_TH6 = 0.0937, 0.6367

def stats(ids, tokens, L):
    raw = (tokens[0] - OFFSET)
    is_lead = ((raw & 0xC0) != 0x80).tolist()  # lead or special
    starts = sorted(set(int(x) for x in ids[0].tolist() if 0 <= int(x) < L))
    n_pat = len(starts)
    n_chars = sum(1 for v in is_lead if v)
    mid = sum(1 for s in starts if not is_lead[s])         # 文字内に落ちた境界
    ends = starts[1:] + [L]
    one = 0
    for s, e in zip(starts, ends):
        leads_in = sum(1 for p in range(s, e) if is_lead[p])
        if leads_in == 1: one += 1
    return dict(n_pat=n_pat, n_chars=n_chars, nbytes=L, mid=mid, one=one)

@app.command()
def main(n_docs: int = typer.Option(200)):
    tok = TokenizerArgs(name="blt", init_kwargs={"bpe_tokenizer_path": TOK}).build()
    r = pa.ipc.open_file(ARROW); b = r.get_batch(0)
    texts = b.column("text"); ents = b.column("entropies")
    print(f"Arrow: {b.num_rows} docs（先頭{n_docs}検証）, entropies実長(doc0)={len(ents[0].as_py())}")
    acc = {"cblt": dict(n_pat=0, n_chars=0, nbytes=0, mid=0, one=0),
           "byte": dict(n_pat=0, n_chars=0, nbytes=0, mid=0, one=0)}
    nan_seen = 0; n_used = 0
    for i in range(min(n_docs, b.num_rows)):
        text = texts[i].as_py(); elist = ents[i].as_py()
        if len(text) < 2 or len(elist) < 3: continue
        tokens = torch.tensor(tok.encode(text), dtype=torch.long).unsqueeze(0)
        ent = torch.tensor(elist, dtype=torch.float32).unsqueeze(0)
        L = min(tokens.shape[1], ent.shape[1])
        tokens = tokens[:, :L]; ent = ent[:, :L]
        if torch.isnan(ent).any(): nan_seen += 1
        n_used += 1
        # cblt
        sc = aggregate_char_entropy(ent, tokens, "sqrt")
        idc = find_cblt_monotonic_patch_start_ids(sc, CBLT_TH6)
        sc_nan = bool(torch.isnan(sc[torch.isfinite(sc)]).any()) if torch.isfinite(sc).any() else False
        for k, v in stats(idc, tokens, L).items(): acc["cblt"][k] += v
        # byte
        idb = find_entropy_patch_start_ids(ent, threshold=BYTE_TH6, monotonicity=True)
        for k, v in stats(idb, tokens, L).items(): acc["byte"][k] += v
    print(f"\n=== patch検証（{n_used}doc, slice precompute済entropy使用）===")
    for m in ["byte", "cblt"]:
        a = acc[m]; npat = max(1, a["n_pat"])
        bpp = a["nbytes"] / npat
        chars_per_pat = a["n_chars"] / npat
        print(f"{m:5}: bytes/patch={bpp:.2f}  chars/patch={chars_per_pat:.2f}  "
              f"mid_char_split={a['mid']}  1字patch率={a['one']/npat:.3f}  "
              f"n_patches={a['n_pat']} vs n_chars={a['n_chars']}")
    print(f"\n判定:")
    c = acc["cblt"]; cb = max(1, c["n_pat"])
    print(f"  cblt mid_char_split=0: {'✅' if c['mid']==0 else '❌ '+str(c['mid'])}")
    print(f"  cblt bytes/patch≈6: {'✅' if abs(c['nbytes']/cb-6)<=1.5 else '⚠'} ({c['nbytes']/cb:.2f})")
    print(f"  cblt patches≪chars(degenerateでない): {'✅' if c['n_pat'] < c['n_chars']*0.8 else '❌'} ({c['n_pat']}/{c['n_chars']})")
    by = acc["byte"]; bb = max(1, by["n_pat"])
    deg = by["n_pat"] >= by["n_chars"] * 0.9  # 1パッチ/文字なら degenerate
    print(f"  byte 「1パッチ/文字」でない: {'✅' if not deg else '❌ degenerate'} (patches {by['n_pat']} vs chars {by['n_chars']})")
    print(f"  NaN entropies doc数: {nan_seen} {'✅' if nan_seen==0 else '⚠'}")

if __name__ == "__main__":
    app()
