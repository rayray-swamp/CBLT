"""
段階2-B: Wikipedia + CulturaX を文書単位で言語横断シャッフル → 8GB チャンク3分割

確定方針(§10): 全文書を言語横断で文書単位シャッフル → 8GB単位に切る
（各段=縮小版の全体=同分布、catastrophic forgetting 回避）。
  段1 = 8GB / 段2 = +8GB(累計16GB) / 段3 = 残り全部

入力:
  wiki:     blt_data/largecorpus/wiki/{lang}.jsonl           （{"text","lang","source"}）
  culturax: blt_data/multilingual/culturax_{lang}/culturax_{lang}.chunk.00.jsonl（{"text","url"}、held-outとは構築上disjoint）
出力（BLT data layout）:
  blt_data/largecorpus/stage{N}/stage{N}.chunk.00.jsonl  （N=1,2,3）
メモリ安全: オフセット索引方式（全文をRAMに載せない）。8GB境界はtextバイトで判定。
CPU のみ。
"""
import json, os, sys, random
import numpy as np

LANGS = ["en", "ru", "ar", "ja", "zh", "ko", "hi", "th"]
BASE = "/gs/bs/tga-RLA/yoshida/blt_data"
OUT = f"{BASE}/largecorpus"
STAGE_BYTES = 8_000_000_000   # 8GB text/段（段1,2）。段3=残り

def main():
    files = []   # (path, lang, source)
    for lg in LANGS:
        files.append((f"{OUT}/wiki/{lg}.jsonl", lg, "wiki"))
    for lg in LANGS:
        files.append((f"{BASE}/multilingual/culturax_{lg}/culturax_{lg}.chunk.00.jsonl", lg, "culturax"))

    # ---- Pass 1: 索引構築（file_id, byte_offset, text_bytes） ----
    fids=[]; offs=[]; tbs=[]
    file_meta=[]
    print("Pass1: 索引構築", flush=True)
    for fid,(path,lg,src) in enumerate(files):
        file_meta.append((path,lg,src))
        if not os.path.exists(path):
            print(f"  miss {path}"); continue
        n=0
        with open(path,"rb") as f:
            off=f.tell()
            line=f.readline()
            while line:
                try:
                    tb=len(json.loads(line)["text"].encode("utf-8"))
                except Exception:
                    tb=0
                if tb>0:
                    fids.append(fid); offs.append(off); tbs.append(tb); n+=1
                off=f.tell(); line=f.readline()
        print(f"  [{fid}] {os.path.basename(path)} ({lg}/{src}): {n:,} docs", flush=True)
    N=len(fids)
    fids=np.array(fids,dtype=np.int16); offs=np.array(offs,dtype=np.int64); tbs=np.array(tbs,dtype=np.int64)
    total=int(tbs.sum())
    print(f"総doc={N:,} 総textバイト={total:,} ({total/1e9:.2f}GB)", flush=True)

    # ---- シャッフル ----
    rng=np.random.default_rng(777)
    perm=rng.permutation(N)

    # ---- Pass 2: シャッフル順に seek して 8GB ごとに stage へ書く ----
    # 段境界: cum<8GB→1, <16GB→2, それ以外→3
    handles=[open(p,"rb") for (p,_,_) in file_meta]
    os.makedirs(OUT,exist_ok=True)
    stage_paths={s:f"{OUT}/stage{s}/stage{s}.chunk.00.jsonl" for s in (1,2,3)}
    for s in (1,2,3): os.makedirs(f"{OUT}/stage{s}",exist_ok=True)
    outs={s:open(stage_paths[s],"wb") for s in (1,2,3)}
    # 集計: stage -> lang -> [docs,bytes], stage -> source -> bytes
    from collections import defaultdict
    stat=defaultdict(lambda: defaultdict(lambda:[0,0]))     # stat[stage][lang]=[docs,bytes]
    src_stat=defaultdict(lambda: defaultdict(int))            # src_stat[stage][source]=bytes
    cum=0
    print("Pass2: シャッフル書き出し", flush=True)
    for k,i in enumerate(perm):
        fid=int(fids[i]); off=int(offs[i]); tb=int(tbs[i])
        stage = 1 if cum<STAGE_BYTES else (2 if cum<2*STAGE_BYTES else 3)
        h=handles[fid]; h.seek(off); line=h.readline()
        outs[stage].write(line)
        _,lg,src=file_meta[fid]
        stat[stage][lg][0]+=1; stat[stage][lg][1]+=tb; src_stat[stage][src]+=tb
        cum+=tb
        if k%2_000_000==0: print(f"  {k:,}/{N:,} cum={cum/1e9:.1f}GB", flush=True)
    for h in handles: h.close()
    for s in outs: outs[s].close()

    # ---- レポート ----
    print("\n===== 段階分割サマリ（text bytes）=====")
    for s in (1,2,3):
        tot=sum(v[1] for v in stat[s].values()); docs=sum(v[0] for v in stat[s].values())
        print(f"\n[stage{s}] {docs:,} docs / {tot/1e9:.2f} GB")
        print(f"  言語内訳GB: "+", ".join(f"{lg}:{stat[s][lg][1]/1e9:.2f}" for lg in LANGS))
        print(f"  ソース内訳GB: "+", ".join(f"{k}:{src_stat[s][k]/1e9:.2f}" for k in ("wiki","culturax")))
        print(f"  ファイル: {stage_paths[s]}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
