"""① 実証（Chat診断確認・2026-07-02）: sequence_iterator の窓が「4096パッチ」単位で作られ、
truncate_batch(max_length=4096 byte) が全行を先頭4096Bで切断＝残りを破棄している、を実測。
truncate_batch を monkeypatch し、truncation 直前の窓バイト長 = patch_lengths.sum(axis=1) と
発火率を集計。~24.6kB(r6)・発火100% なら診断確定。GPU必須（patching cuda）。
使い方: python scripts/verify_window_truncation.py --config blt_body_cblt_r6 --n-batches 100
"""
import numpy as np, typer
from omegaconf import OmegaConf
import bytelatent.data.iterators.packing_iterator as PI
from bytelatent.config_parser import parse_args_to_pydantic_model
from bytelatent.args import TrainArgs

app = typer.Typer()
REC = {"winbytes": [], "nfire": 0, "nrow": 0}
_orig = PI.truncate_batch


def patched(*args, **kwargs):
    batch = args[0] if args else kwargs.get("batch")
    max_length = args[1] if len(args) > 1 else kwargs.get("max_length")
    sl = np.asarray(batch.patch_lengths).sum(axis=1)   # truncation直前の窓バイト長（行ごと）
    REC["winbytes"].extend(sl.tolist())
    REC["nfire"] += int((sl > max_length + 1).sum())
    REC["nrow"] += len(sl)
    return _orig(*args, **kwargs)


PI.truncate_batch = patched


@app.command()
def main(config: str = typer.Option("blt_body_cblt_r6"), n_batches: int = typer.Option(100)):
    cli = OmegaConf.from_dotlist([f"config=bytelatent/configs/{config}.yaml", "data.load_async=false"])
    args = parse_args_to_pydantic_model(TrainArgs, cli_args=cli)
    max_enc = args.data.max_encoder_seq_length
    seqlen = args.data.seq_len
    dl = args.data.build_from_rank(0, 1)
    it = dl.create_iter()
    for _ in range(n_batches):
        next(it)
    wb = np.array(REC["winbytes"])
    print(f"\n=== ① 窓truncation 実証: {config} ({n_batches}batch, {REC['nrow']}行) ===", flush=True)
    print(f"config: data.seq_len(=窓パッチ数)={seqlen}  max_encoder_seq_length(=byte上限)={max_enc}")
    print(f"truncation直前 窓バイト長: min={wb.min():.0f} p50={np.median(wb):.0f} "
          f"mean={wb.mean():.0f} max={wb.max():.0f}")
    print(f"truncate発火行: {REC['nfire']}/{REC['nrow']} = {REC['nfire']/REC['nrow']*100:.1f}%")
    print(f"窓 ≈ {wb.mean()/max_enc:.2f}× max_encoder → 学習される割合 ≈ {max_enc/wb.mean()*100:.1f}%  "
          f"(残り ≈ {(1-max_enc/wb.mean())*100:.1f}% は消費済だが破棄)")
    exp = seqlen * (wb.mean()/wb.mean())  # noop
    print(f"判定: 窓が『{seqlen}パッチ』単位＝1行 ~{wb.mean()/1000:.1f}kB・全行切断 → 診断{'確定 ✅' if REC['nfire']/max(1,REC['nrow'])>0.95 and wb.mean()>max_enc*2 else '要再確認 ⚠'}")


if __name__ == "__main__":
    app()
