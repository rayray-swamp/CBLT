"""②信頼性ゲート（Chat Opt1）: 再較正θで**実patcher**を回し、実コーパスのデータ量重みで
各rateの実効 bytes/patch を測定 → byte≈cblt≈target を確認。較正(per-docモデル)を実patcher＋
packing相当(forced 2/doc)で裏取り。新θは theta_calibration_dataweighted.csv から読む。
使い方: python scripts/gate_rate_check.py [--stride N]
"""
import sys, glob, csv
import torch, pyarrow as pa, typer
from collections import defaultdict
from bytelatent.tokenizers.build_tokenizer import TokenizerArgs
from bytelatent.data.patcher import (
    aggregate_char_entropy, find_entropy_patch_start_ids,
    find_cblt_monotonic_patch_start_ids, OFFSET,
)
app = typer.Typer()
TOK = "/gs/bs/tga-RLA/yoshida/blt_data/tokenizer/tokenizer.model"
SHARDS = sorted(glob.glob(
    "/gs/bs/tga-RLA/yoshida/blt_data/largecorpus_clean/preprocess/corpus/transformer_100m/*.arrow"))
CSV = "/gs/bs/tga-RLA/yoshida/blt/flores_dumps/theta_calibration_dataweighted.csv"
RATES = [4, 5, 6, 7, 8]


def load_theta():
    th = {"byte": {}, "sqrt": {}, "sum": {}, "avg": {}}
    key = {"byte-BLT": "byte", "CBLT-sqrtL": "sqrt", "Sum": "sum", "Avg": "avg"}
    for row in csv.DictReader(open(CSV, encoding="utf-8-sig")):
        m = key[row["method"]]; r = int(float(row["target_bpp"]))
        th[m][r] = float(row["theta_nats"])
    return th


def n_starts(ids, L):
    return len(sorted(set(int(x) for x in ids[0].tolist() if 0 <= int(x) < L)))


@app.command()
def main(stride: int = typer.Option(400)):
    th = load_theta()
    tok = TokenizerArgs(name="blt", init_kwargs={"bpe_tokenizer_path": TOK}).build()
    # 集計: acc[(method,rate)] = [tot_bytes, tot_patches]
    acc = defaultdict(lambda: [0, 0]); n = 0; gi = 0
    specs = [("byte", r) for r in RATES] + [("sqrt", r) for r in RATES] + [("sum", 6), ("avg", 6)]
    for sp in SHARDS:
        r = pa.ipc.open_file(sp)
        for bi in range(r.num_record_batches):     # 全 record batch を走査（バグ修正）
            b = r.get_batch(bi)
            texts, ents = b.column("text"), b.column("entropies")
            for i in range(b.num_rows):
                gi += 1
                if gi % stride: continue            # 全docグローバル stride＝データ量重み
                text = texts[i].as_py(); el = ents[i].as_py()
                if len(text) < 20 or len(el) < 5: continue
                tokens = torch.tensor(tok.encode(text), dtype=torch.long).unsqueeze(0)
                ent = torch.tensor(el, dtype=torch.float32).unsqueeze(0)
                L = min(tokens.shape[1], ent.shape[1]); tokens = tokens[:, :L]; ent = ent[:, :L]
                sc = {m: aggregate_char_entropy(ent, tokens, m) for m in ["sqrt", "sum", "avg"]}
                for m, rate in specs:
                    if m == "byte":
                        ids = find_entropy_patch_start_ids(ent, threshold=th["byte"][rate], monotonicity=True)
                    else:
                        ids = find_cblt_monotonic_patch_start_ids(sc[m], th[m][rate])
                    a = acc[(m, rate)]; a[0] += L; a[1] += n_starts(ids, L)
                n += 1
    print(f"\n=== ②ゲート: 実patcher 実効 bytes/patch（{n} docs・データ量重み・新θ）===")
    print(f"{'rate':>5} | {'byte b/p':>9} {'cblt b/p':>9} {'|Δ|':>6} | {'sum':>7} {'avg':>7}")
    allok = True
    for rate in RATES:
        bb = acc[("byte", rate)][0] / max(1, acc[("byte", rate)][1])
        cb = acc[("sqrt", rate)][0] / max(1, acc[("sqrt", rate)][1])
        d = abs(bb - cb)
        extra = ""
        if rate == 6:
            sm = acc[("sum", 6)][0] / max(1, acc[("sum", 6)][1])
            av = acc[("avg", 6)][0] / max(1, acc[("avg", 6)][1])
            extra = f"| {sm:>7.3f} {av:>7.3f}"
        ok = abs(bb - rate) < 0.3 and abs(cb - rate) < 0.3 and d < 0.3
        allok &= ok
        print(f"{rate:>5} | {bb:>9.3f} {cb:>9.3f} {d:>6.3f} {extra}  {'✅' if ok else '⚠'}")
    print(f"\n判定（byte≈cblt≈target, 全rate |Δ|<0.3 & |b/p−target|<0.3）: {'✅ PASS' if allok else '⚠ 要確認'}")
    print("注: 実patcherはforced 2/doc込み→packing(2/seq)より僅かに低めだが byte/cblt 同条件。")


if __name__ == "__main__":
    app()
