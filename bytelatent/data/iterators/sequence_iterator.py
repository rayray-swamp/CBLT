# Copyright (c) Meta Platforms, Inc. and affiliates.
from logging import getLogger
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from bytelatent.data.data_types import BltSequence
from bytelatent.data.iterators.abstract_iterator import (
    PydanticIteratorState,
    StatefulIterator,
)
from bytelatent.data.iterators.arrow_iterator import ArrowFileIterator
from bytelatent.data.iterators.limit_iterator import LimitIterator
from bytelatent.data.iterators.looping_iterator import LoopingIterator
from bytelatent.data.iterators.preprocess_iterator import (
    PreprocessIterator,
    PreprocessIteratorState,
)

logger = getLogger()


class SequencePackingArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    output_seq_len: int
    buffer_size: int
    # byte-budget windowing (Chat Opt A, 2026-07-02): 窓を「バイト予算」で構成する。
    # None なら従来の「output_seq_len パッチ固定」窓（後方互換・eval等）。
    # 学習では max_encoder_seq_length を渡す → 1窓 ≤ byte_budget バイト・パッチ丸ごと・破棄ゼロ。
    byte_budget: int | None = None


class SequenceIteratorState(PydanticIteratorState):
    model_config = ConfigDict(extra="forbid")
    sequence_packing_args: SequencePackingArgs
    preprocess_iterator_state: PreprocessIteratorState
    # If None, rng is disabled.
    rng_state: dict[str, Any] | None

    def build(self):
        preprocess_iterator = self.preprocess_iterator_state.build()
        return SequenceIterator(
            preprocess_iterator,
            sequence_packing_args=self.sequence_packing_args,
            rng_state=self.rng_state,
        )


def get_datafile(
    iterator: PreprocessIterator | ArrowFileIterator | LoopingIterator | LimitIterator,
):
    if isinstance(iterator, ArrowFileIterator):
        return f"file={iterator.file_path} n_shards={len(iterator.dataset_files) if iterator.dataset_files is not None else None}"
    elif isinstance(iterator, PreprocessIterator):
        return get_datafile(iterator.arrow_iterator)
    elif isinstance(iterator, LoopingIterator):
        return get_datafile(iterator.file_iterator)
    elif isinstance(iterator, LimitIterator):
        return get_datafile(iterator.base_iterator)
    else:
        # ログ用途のみ。未知の iterator（テストのモック等）でクラッシュさせない。
        return f"<{type(iterator).__name__}>"


class SequenceIterator(StatefulIterator):
    def __init__(
        self,
        preprocess_iterator: PreprocessIterator,
        *,
        rng_state: dict[str, Any] | None,
        sequence_packing_args: SequencePackingArgs,
    ):
        self.preprocess_iterator = preprocess_iterator
        self.sequence_packing_args = sequence_packing_args
        self.output_seq_len = sequence_packing_args.output_seq_len
        self.buffer_size = sequence_packing_args.buffer_size
        self.byte_budget = sequence_packing_args.byte_budget
        if rng_state is None:
            self.rng = None
        else:
            self.rng = np.random.default_rng()
            self.rng.bit_generator.state = rng_state

    def get_state(self):
        # TODO: need to also perist the current shuffle buffer
        return SequenceIteratorState(
            sequence_packing_args=self.sequence_packing_args,
            preprocess_iterator_state=self.preprocess_iterator.get_state(),
            rng_state=None if self.rng is None else self.rng.bit_generator.state,
        )

    def create_iter(self):
        if self.byte_budget is not None:
            yield from self._create_iter_byte_budget()
            return
        example_iter = self.preprocess_iterator.create_iter()
        n_buffer_patches = self.buffer_size * self.output_seq_len

        patch_lengths: list[int] = []
        tokens: list[int] = []
        mask: list[bool] = []
        first = True
        logger.info(
            "Starting first buffer for: %s",
            get_datafile(self.preprocess_iterator),
        )
        for example in example_iter:
            assert example.tokens is not None
            assert example.mask is not None
            if self.preprocess_iterator.add_patches:
                assert example.patch_lengths is not None
                assert len(example.tokens) == sum(example.patch_lengths)
            else:
                assert example.patch_lengths is None
            assert len(example.tokens) != 0
            assert len(example.mask) != 0
            assert len(example.tokens) == len(example.mask)

            tokens.extend(example.tokens)
            mask.extend(example.mask)
            if self.preprocess_iterator.add_patches:
                patch_lengths.extend(example.patch_lengths)
            else:
                # This lets the rest of the code work as expected and just yield byte seqs
                patch_lengths.extend([1] * len(example.tokens))

            while len(patch_lengths) >= n_buffer_patches:
                if first:
                    first = False
                    logger.info(
                        "First buffer complete for: %s",
                        get_datafile(self.preprocess_iterator),
                    )

                x_patches = np.array(patch_lengths[:n_buffer_patches]).reshape(
                    self.buffer_size, self.output_seq_len
                )
                seq_tokens = []
                seq_mask = []
                start_id = 0
                # We fix the number of patches and therefore global steps per batch
                # so we have a variable number of tokens we need to account for
                for num_tokens in x_patches.sum(axis=-1):
                    seq_tokens.append(tokens[start_id : start_id + num_tokens])
                    seq_mask.append(mask[start_id : start_id + num_tokens])
                    start_id += num_tokens

                assert start_id == x_patches.sum()

                # Remove what we just added from the buffer
                patch_lengths = patch_lengths[n_buffer_patches:]
                tokens = tokens[x_patches.sum() :]
                mask = mask[x_patches.sum() :]

                seq_patch_lengths: list[list[int]] = x_patches.tolist()
                assert len(seq_patch_lengths) == self.buffer_size
                if self.rng is None:
                    permutations = list(range(len(seq_patch_lengths)))
                else:
                    permutations = self.rng.permutation(len(seq_patch_lengths))

                for idx in permutations:
                    assert len(seq_patch_lengths[idx]) == self.output_seq_len
                    assert (
                        sum(seq_patch_lengths[idx])
                        == len(seq_tokens[idx])
                        == len(seq_mask[idx])
                    ), f"{sum(seq_patch_lengths[idx])}, {len(seq_tokens[idx])} {len(seq_mask[idx])}, idx={idx}"
                    assert seq_patch_lengths[idx][0] > 0, f"{seq_patch_lengths[idx]}"
                    if self.preprocess_iterator.add_patches:
                        yield BltSequence(
                            tokens=seq_tokens[idx],
                            mask=seq_mask[idx],
                            patch_lengths=seq_patch_lengths[idx],
                        )
                    else:
                        yield BltSequence(
                            tokens=seq_tokens[idx],
                            mask=seq_mask[idx],
                            patch_lengths=None,
                        )

    def _create_iter_byte_budget(self):
        """バイト予算窓（Chat Opt A・2026-07-02）: 窓を byte_budget バイト以内でパッチを
        丸ごと詰める。パッチは絶対に分割しない（cblt の文字保護 mid_char_split=0 を維持）。
        入り切らないパッチ（と対応 tokens/mask）は次窓の先頭へ繰越＝破棄ゼロ（ストリーム消費
        バイト == 学習バイト）。窓の実バイト ≈ byte_budget − 端数。patch 数は窓ごとに可変。
        LoopingIterator の周回境界でも buffer が継続するので不変条件を保つ。"""
        example_iter = self.preprocess_iterator.create_iter()
        budget = self.byte_budget
        add_patches = self.preprocess_iterator.add_patches
        pls: list[int] = []
        toks: list[int] = []
        msks: list[bool] = []
        win: list = []
        first = True
        logger.info(
            "Starting first buffer (byte-budget=%d) for: %s",
            budget,
            get_datafile(self.preprocess_iterator),
        )

        def cut_windows():
            i = 0  # 消費済 patch 数
            b = 0  # 消費済 byte 数
            npl = len(pls)
            while i < npl:
                cum = 0
                n = 0
                while i + n < npl and cum + pls[i + n] <= budget:
                    cum += pls[i + n]
                    n += 1
                forced = False
                if n == 0:
                    # 単一パッチが budget 超（極稀・分割禁止なので単独窓に）
                    n = 1
                    cum = pls[i]
                    forced = True
                if i + n < npl or forced:
                    win.append((pls[i : i + n], toks[b : b + cum], msks[b : b + cum]))
                    i += n
                    b += cum
                else:
                    # i+n == npl かつ 非forced: 次パッチで budget 到達しうる → 待つ
                    break
            if i:
                del pls[:i]
                del toks[:b]
                del msks[:b]

        for example in example_iter:
            assert example.tokens is not None and example.mask is not None
            assert len(example.tokens) == len(example.mask) and len(example.tokens) != 0
            if add_patches:
                assert example.patch_lengths is not None
                assert len(example.tokens) == sum(example.patch_lengths)
                pls.extend(example.patch_lengths)
            else:
                pls.extend([1] * len(example.tokens))
            toks.extend(example.tokens)
            msks.extend(example.mask)
            cut_windows()
            while len(win) >= self.buffer_size:
                if first:
                    first = False
                    logger.info(
                        "First buffer complete for: %s",
                        get_datafile(self.preprocess_iterator),
                    )
                batch = win[: self.buffer_size]
                del win[: self.buffer_size]
                if self.rng is None:
                    order = list(range(len(batch)))
                else:
                    order = self.rng.permutation(len(batch))
                for idx in order:
                    w_pl, w_tok, w_msk = batch[idx]
                    assert sum(w_pl) == len(w_tok) == len(w_msk)
                    assert w_pl[0] > 0
                    yield BltSequence(
                        tokens=w_tok,
                        mask=w_msk,
                        patch_lengths=(w_pl if add_patches else None),
                    )
