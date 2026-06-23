"""
C-3: データローダの系列再供給（重複）直接監査

仮説: checkpoint 毎の MultiprocessIterator.get_state()/refresh、または
SequenceIterator のシャッフルバッファ未保存により、同一系列が再供給され
実効データが縮小 → メモリ化（train bpb 0.02 / held-out 5.19）。

2モード:
  --mode continuous : MPなし(load_async=False)で連続供給。コア配線の重複を見る
  --mode checkpoint : 学習と同じ checkpoint refresh サイクルを再現。
                      K batch 毎に get_state_and_refresh して重複を見る

各 batch の x 系列を hash し、供給回数分布を出す。
CPU のみ・GPU不要。計算ノード(qrsh -l cpu_*)で実行推奨（ログインノードは
スレッド上限が低い）。

使い方（計算ノード）:
  python scripts/audit_loader_dup.py --mode continuous --n-batches 3000
  python scripts/audit_loader_dup.py --mode checkpoint --n-batches 3000 --ckpt-every 500
"""
import hashlib, collections
import typer
import numpy as np

app = typer.Typer()


def build_loader(load_async: bool):
    from bytelatent.args import DataloaderArgs
    from bytelatent.data.patcher import PatcherArgs
    from bytelatent.tokenizers.build_tokenizer import TokenizerArgs
    return DataloaderArgs(
        root_dir="/gs/bs/tga-RLA/yoshida/blt_data/multilingual",
        preprocess_dir="/gs/bs/tga-RLA/yoshida/blt_data/multilingual/preprocess",
        entropy_model_name="transformer_100m",
        sources={"culturax_en":100,"culturax_ru":200,"culturax_ar":200,"culturax_ja":300,
                 "culturax_zh":300,"culturax_ko":300,"culturax_hi":300,"culturax_th":300},
        batch_size=2, seq_len=8192, max_encoder_seq_length=8192,
        load_async=load_async, add_patches=False, seed=777,
        buffer_size=64, prefetch_size=64,
        patcher_args=PatcherArgs(patching_mode="byte"),
        tokenizer_args=TokenizerArgs(name="blt", init_kwargs={"bpe_tokenizer_path":"/gs/bs/tga-RLA/yoshida/blt_data/tokenizer/tokenizer.model"}),
    )


def hash_row(row) -> str:
    return hashlib.md5(np.ascontiguousarray(row).tobytes()).hexdigest()


def report(seen: collections.Counter, n_seq: int):
    uniq = len(seen)
    dup = n_seq - uniq
    mult = collections.Counter(seen.values())
    print(f"\n供給系列 {n_seq} / unique {uniq} / 重複供給 {dup} ({dup/n_seq:.2%})")
    print(f"実効エポック(供給/unique) = {n_seq/uniq:.3f}")
    print(f"供給回数:系列数 = {dict(sorted(mult.items()))}")
    print(f"最多供給 top5 = {[c for _,c in seen.most_common(5)]}")


@app.command()
def main(
    mode: str = typer.Option("continuous", help="continuous | checkpoint"),
    n_batches: int = typer.Option(3000),
    ckpt_every: int = typer.Option(500, help="checkpoint モードの refresh 間隔(batch)"),
):
    import torch, pyarrow
    torch.set_num_threads(1)
    pyarrow.set_cpu_count(1)
    pyarrow.set_io_thread_count(1)

    seen = collections.Counter()
    n_seq = 0

    if mode == "continuous":
        dl = build_loader(load_async=False)
        it = dl.build_from_rank(0, 1)
        pyit = it.create_iter()
        for i, batch in enumerate(pyit):
            if i >= n_batches:
                break
            for row in batch.x:
                seen[hash_row(row)] += 1
                n_seq += 1
        report(seen, n_seq)

    elif mode == "checkpoint":
        # 学習と同じ refresh サイクルを再現
        from bytelatent.data.iterators.abstract_iterator import get_state_and_refresh
        dl = build_loader(load_async=True)
        data_loader = dl.build_from_rank(0, 1)
        pyit = data_loader.create_iter()
        i = 0
        while i < n_batches:
            batch = next(pyit)
            for row in batch.x:
                seen[hash_row(row)] += 1
                n_seq += 1
            i += 1
            if i % ckpt_every == 0:
                # 学習の checkpoint 保存と同じ操作
                _, data_loader, pyit = get_state_and_refresh(data_loader)
                print(f"  [checkpoint refresh @ batch {i}]  unique so far={len(seen)}", flush=True)
        report(seen, n_seq)
        if hasattr(data_loader, "shutdown"):
            data_loader.shutdown()
    else:
        raise ValueError(mode)


if __name__ == "__main__":
    app()
