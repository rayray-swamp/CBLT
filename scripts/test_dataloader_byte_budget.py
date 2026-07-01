"""⑥ t3: 実dataloader（新byte予算窓）エンドツーエンド検証。GPU必須。
truncate発火0・全行 sum(patch_lengths)==max_encoder+1・x幅==max_encoder・real bytes≤max・crash無し。
stream消費==学習バイトは「破棄ゼロ設計＋truncate発火0」で構成的に保証。
使い方: python scripts/test_dataloader_byte_budget.py --config blt_body_cblt_r6 --n-batches 100
"""
import numpy as np, typer
from omegaconf import OmegaConf
import bytelatent.data.iterators.packing_iterator as PI
from bytelatent.config_parser import parse_args_to_pydantic_model
from bytelatent.args import TrainArgs

app = typer.Typer()
FIRE = {"n": 0, "maxwin": 0}
_orig = PI.truncate_batch


def patched(*a, **k):
    batch = a[0]
    ml = a[1] if len(a) > 1 else k.get("max_length")
    sl = np.asarray(batch.patch_lengths).sum(axis=1)
    FIRE["n"] += int((sl > ml + 1).sum())
    FIRE["maxwin"] = max(FIRE["maxwin"], int(sl.max()))
    return _orig(*a, **k)


PI.truncate_batch = patched


@app.command()
def main(config: str = typer.Option("blt_body_cblt_r6"), n_batches: int = typer.Option(100)):
    cli = OmegaConf.from_dotlist([f"config=bytelatent/configs/{config}.yaml", "data.load_async=false"])
    args = parse_args_to_pydantic_model(TrainArgs, cli_args=cli)
    ml = args.data.max_encoder_seq_length
    dl = args.data.build_from_rank(0, 1)
    it = dl.create_iter()
    fails, nrow, realbytes, npatch = [], 0, [], []
    for _ in range(n_batches):
        b = next(it)
        pl = np.asarray(b.patch_lengths)
        s = pl.sum(axis=1)
        if not np.all(s == ml + 1):
            fails.append(f"sum(patch_lengths)!={ml+1} (例 {sorted(set(s.tolist()))[:3]})")
        if b.x.shape[1] != ml:
            fails.append(f"x幅 {b.x.shape[1]} != max_encoder {ml}")
        if not np.all(pl[:, 0] == 1):
            fails.append("先頭patchが1でない行あり")
        nrow += b.x.shape[0]
        npatch.append(pl.shape[1])
        if b.mask is not None:
            realbytes.extend(b.mask.sum(axis=1).tolist())
    rb = np.array(realbytes) if realbytes else np.array([0])
    print(f"\n=== ⑥ t3 実dataloader: {config} ({n_batches}batch, {nrow}行) ===", flush=True)
    print(f"truncate発火: {FIRE['n']} 行 (期待0)  / truncation直前 最大窓バイト={FIRE['maxwin']} (≤{ml+1})")
    print(f"x幅==max_encoder({ml}): {'✅' if not any('x幅' in f for f in fails) else '❌'}")
    print(f"sum(patch_lengths)==max+1({ml+1}) 全行: {'✅' if not any('sum' in f for f in fails) else '❌'}")
    print(f"real bytes/行(mask非pad): min={rb.min()} p50={int(np.median(rb))} max={rb.max()} (≤{ml})")
    print(f"patch次元(バッチ毎・動的): min={min(npatch)} max={max(npatch)}")
    ok = FIRE["n"] == 0 and not fails
    print(f"判定: {'✅ t3 PASS（truncate発火0・全行整合・crash無し）' if ok else '❌ FAIL: '+'; '.join(sorted(set(fails)))}")


if __name__ == "__main__":
    app()
