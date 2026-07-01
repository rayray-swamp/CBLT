"""データ量重み global θ 再較正（全コーパス厳密）。Chat Opt1（2026-06-30）。
largecorpus_clean の precompute済 entropy（=学習データそのもの）を全docパスし、各method
(byte/sqrt/sum/avg)の per-doc ΔH を全docぶんヒストグラム化 → **実コーパスのデータ量重み**で
pooled bytes/patch = {4,5,6,7,8} になる global θ を二分探索。サンプリング誤差なし。
patch数モデル: n_patches = n_docs(先頭強制) + #{ΔH>θ}（calibrate_theta.py と同一定義のデータ量重み版）。
出力で realized=target を再確認（必須）。全method×全rateのθ＋per言語実効レートをCSV化。
真の実効レートは別途 measure_bytes_per_patch.py（実patcher・packing）で②ゲート検証する。
使い方: python scripts/calibrate_theta_dataweighted.py --out-csv ... [--shards all|N]
"""
import sys, glob, math
import numpy as np, torch, pyarrow as pa, typer
from collections import defaultdict
from bytelatent.tokenizers.build_tokenizer import TokenizerArgs
from bytelatent.data.patcher import aggregate_char_entropy

app = typer.Typer()
TOK = "/gs/bs/tga-RLA/yoshida/blt_data/tokenizer/tokenizer.model"
SHARDS = sorted(glob.glob(
    "/gs/bs/tga-RLA/yoshida/blt_data/largecorpus_clean/preprocess/corpus/transformer_100m/*.arrow"))
LANGS = ["en", "ru", "ja", "ar", "zh", "ko", "hi", "th", "other"]
RATES = [4.0, 5.0, 6.0, 7.0, 8.0]
METHODS = ["byte", "sqrt", "sum", "avg"]
HMIN, HMAX, NBIN = -30.0, 30.0, 600000          # ΔH ヒストグラム（分解能 1e-4）
EDGES = np.linspace(HMIN, HMAX, NBIN + 1)


def lang_of(text):
    cnt = {}
    for ch in text[:2000]:
        if not ch.isalpha(): continue
        o = ord(ch)
        if 0x3040 <= o <= 0x30FF: cnt["ja"] = cnt.get("ja", 0) + 1
        elif 0xAC00 <= o <= 0xD7A3: cnt["ko"] = cnt.get("ko", 0) + 1
        elif 0x0400 <= o <= 0x04FF: cnt["ru"] = cnt.get("ru", 0) + 1
        elif 0x0600 <= o <= 0x06FF: cnt["ar"] = cnt.get("ar", 0) + 1
        elif 0x0900 <= o <= 0x097F: cnt["hi"] = cnt.get("hi", 0) + 1
        elif 0x0E00 <= o <= 0x0E7F: cnt["th"] = cnt.get("th", 0) + 1
        elif 0x4E00 <= o <= 0x9FFF: cnt["han"] = cnt.get("han", 0) + 1
        elif o < 0x80: cnt["lat"] = cnt.get("lat", 0) + 1
    if not cnt: return "other"
    if cnt.get("ja", 0) > 0: return "ja"
    if cnt.get("han", 0) > max(cnt.get("lat", 0), 1): return "zh"
    dom = max(cnt, key=cnt.get)
    return {"lat": "en", "han": "zh"}.get(dom, dom)


def add_hist(h, vals):
    if vals.size:
        idx = np.clip(((vals - HMIN) / (HMAX - HMIN) * NBIN).astype(np.int64), 0, NBIN - 1)
        h += np.bincount(idx, minlength=NBIN)


def count_gt(h, theta):
    """#{ΔH > theta} を累積ヒストグラムから。"""
    b = int((theta - HMIN) / (HMAX - HMIN) * NBIN)
    b = max(0, min(NBIN, b))
    return int(h[b:].sum())


@app.command()
def main(out_csv: str = typer.Option(...), shards: str = typer.Option("all"),
         report_every: int = typer.Option(100000), max_docs: int = typer.Option(0),
         stride: int = typer.Option(1)):
    tok = TokenizerArgs(name="blt", init_kwargs={"bpe_tokenizer_path": TOK}).build()
    hist = {m: {lg: np.zeros(NBIN, dtype=np.int64) for lg in LANGS} for m in METHODS}
    nbytes = defaultdict(int); ndoc = defaultdict(int); n = 0
    shard_list = SHARDS if shards == "all" else SHARDS[:int(shards)]
    stop = False
    for sp in shard_list:
        r = pa.ipc.open_file(sp)
        for bi in range(r.num_record_batches):
            bt = r.get_batch(bi); texts, ents = bt.column("text"), bt.column("entropies")
            for i in range(0, bt.num_rows, stride):   # stride間引き＝データ量重みサンプル
                text = texts[i].as_py(); elist = ents[i].as_py()
                if len(text) < 2 or len(elist) < 3: continue
                lg = lang_of(text)
                tokens = torch.tensor(tok.encode(text), dtype=torch.long).unsqueeze(0)
                ent = torch.tensor(elist, dtype=torch.float32).unsqueeze(0)
                L = min(tokens.shape[1], ent.shape[1]); tokens = tokens[:, :L]; ent = ent[:, :L]
                nbytes[lg] += L; ndoc[lg] += 1; n += 1
                # byte: per-token ΔH
                eb = ent[0].numpy()
                add_hist(hist["byte"][lg], eb[1:] - eb[:-1])
                # char methods（実patcher集約・規約B）: 文字順 H 系列の隣接差
                for m in ["sqrt", "sum", "avg"]:
                    sc = aggregate_char_entropy(ent, tokens, m)[0]
                    Hf = sc[torch.isfinite(sc)].numpy()
                    if Hf.size >= 2: add_hist(hist[m][lg], Hf[1:] - Hf[:-1])
                if n % report_every == 0:
                    print(f"  {n:,} docs 処理", flush=True)
                if max_docs and n >= max_docs: stop = True; break
            if stop: break
        if stop: break
    TB = sum(nbytes.values()); ND = sum(ndoc.values())
    print(f"=== 全{n:,} docs / 総bytes {TB/1e9:.3f}B / データ量重み(実コーパス比) ===", flush=True)
    print("言語別doc数: " + " ".join(f"{lg}={ndoc[lg]:,}" for lg in LANGS if ndoc[lg]))

    def calib(method, target):
        ph = sum(hist[method][lg] for lg in LANGS)   # pooled = データ量重み
        def bpp(th): return TB / (ND + count_gt(ph, th))
        lo, hi = HMIN, HMAX
        for _ in range(60):
            mid = (lo + hi) / 2
            if bpp(mid) < target: lo = mid     # θ大→patch減→bpp増
            else: hi = mid
        th = (lo + hi) / 2
        pl = {}
        for lg in LANGS:
            if ndoc[lg]:
                pl[lg] = nbytes[lg] / (ndoc[lg] + count_gt(hist[method][lg], th))
        return th, bpp(th), pl

    rows = []; name = {"byte": "byte-BLT", "sqrt": "CBLT-sqrtL", "sum": "Sum", "avg": "Avg"}
    ln2 = math.log(2)
    print("\n=== 再較正θ（データ量重み）＋ realized=target 再確認 ===", flush=True)
    for m in METHODS:
        for T in RATES:
            th, rb, pl = calib(m, T)
            ok = "✅" if abs(rb - T) < 0.01 else "⚠"
            print(f"{name[m]:11} target={T:.0f} θ={th:+.4f}(nats) realized_pooled={rb:.4f} {ok}", flush=True)
            rows.append([name[m], f"{T:.1f}", f"{th:.4f}", f"{th/ln2:.4f}", f"{rb:.4f}"] +
                        [f"{pl.get(lg, float('nan')):.3f}" for lg in LANGS])
    import csv
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["method", "target_bpp", "theta_nats", "theta_bits", "realized_pooled_bpp_DATAWEIGHTED"]
                   + [f"bpp_{lg}" for lg in LANGS])
        w.writerows(rows)
    print(f"\n[OK] 出力: {out_csv}", flush=True)


if __name__ == "__main__":
    app()
