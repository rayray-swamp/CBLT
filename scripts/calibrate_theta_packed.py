"""packing経路の実 dataloader で packed bytes/patch を測り θ較正（Chat Opt1 正版・2026-07-01）。
per-doc 較正は packing(doc連結→4096byte窓→強制patch)を ~5%過大評価していた（③ログで実 cblt5.65/byte5.80 判明）。
本スクリプトは **実 dataloader(build_from_rank, load_async=false)** を θごとに構築し、③と同じ
`n_bytes / (patch_lengths>0).sum()` で **packed bpp** を測定。θをスイープ→各目標rateの θ を補間で確定。
GPU必須（patching cuda）。使い方: python scripts/calibrate_theta_packed.py --config blt_body_cblt_r6 \
    --thetas "0.30,0.40,0.50,0.60,0.70,0.80" --targets "4,5,6,7,8" --n-batches 150
"""
import sys, typer
import numpy as np
from omegaconf import OmegaConf
from bytelatent.config_parser import parse_args_to_pydantic_model
from bytelatent.args import TrainArgs
app = typer.Typer()


def measure_bpp(config_name, theta, n_batches):
    cli = OmegaConf.from_dotlist([
        f"config=bytelatent/configs/{config_name}.yaml",
        f"data.patcher_args.threshold={theta}",
        f"model.patching_threshold={theta}",
        "data.load_async=false",
    ])
    args = parse_args_to_pydantic_model(TrainArgs, cli_args=cli)
    dl = args.data.build_from_rank(0, 1)
    it = dl.create_iter()
    tb = tp = 0
    for _ in range(n_batches):
        batch = next(it)
        tb += int(batch.y.size)
        tp += int((batch.patch_lengths > 0).sum())
    return tb / tp


@app.command()
def main(config: str = typer.Option(...), thetas: str = typer.Option(...),
         targets: str = typer.Option("4,5,6,7,8"), n_batches: int = typer.Option(150)):
    th_list = [float(x) for x in thetas.split(",")]
    tgt_list = [float(x) for x in targets.split(",")]
    print(f"=== packed bpp 測定: {config}（実dataloader・{n_batches}batch/θ）===", flush=True)
    xs, ys = [], []
    for th in th_list:
        bpp = measure_bpp(config, th, n_batches)
        xs.append(th); ys.append(bpp)
        print(f"  θ={th:+.4f} → packed bpp={bpp:.4f}", flush=True)
    # bpp は θ について単調増。各目標rateの θ を線形補間
    xs, ys = np.array(xs), np.array(ys)
    order = np.argsort(ys)
    ys_s, xs_s = ys[order], xs[order]
    print(f"\n=== {config} 補間θ（packed=target）===", flush=True)
    for T in tgt_list:
        if T < ys_s.min() or T > ys_s.max():
            print(f"  rate {T:.0f}: スイープ範囲外（bpp {ys_s.min():.2f}〜{ys_s.max():.2f}）⚠", flush=True)
            continue
        th = float(np.interp(T, ys_s, xs_s))
        # 検証: その θ で実測
        chk = measure_bpp(config, th, n_batches)
        ok = "✅" if abs(chk - T) < 0.1 else "⚠"
        print(f"  rate {T:.0f}: θ={th:+.4f} → 実測 packed bpp={chk:.4f} {ok}", flush=True)


if __name__ == "__main__":
    app()
