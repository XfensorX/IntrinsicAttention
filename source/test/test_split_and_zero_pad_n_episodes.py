# test_split_and_zero_pad_n_episodes.py
from typing import List, Tuple

import numpy as np
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from ray.rllib.utils.postprocessing.zero_padding import (
    split_and_zero_pad_n_episodes as split_np,
)

from brainstorming.learners.pytorch_differentiable_funcs import (
    split_and_zero_pad_n_episodes as split_torch,
)

# ---------- helpers for episode lengths ----------


def _normalize_to_sum(xs: List[int], target: int) -> List[int]:
    s = sum(xs)
    if s == target:
        return xs
    if s == 0:
        return [target] + [0] * (len(xs) - 1)
    scaled = [int((x / s) * target) for x in xs]
    diff = target - sum(scaled)
    if scaled:
        idx = max(range(len(scaled)), key=lambda i: (xs[i], i))
        scaled[idx] += diff
    return scaled


def _partitions_of(n: int) -> st.SearchStrategy[List[int]]:
    if n == 0:
        return st.one_of(
            st.just([]),
            st.lists(st.just(0), min_size=1, max_size=5),
        )

    def make_parts(k: int) -> st.SearchStrategy[List[int]]:
        return st.lists(
            st.integers(min_value=0, max_value=n),
            min_size=k,
            max_size=k,
        ).map(lambda xs: _normalize_to_sum(xs, n))

    return st.integers(min_value=1, max_value=min(8, n)).flatmap(make_parts)


# ---------- strategies ----------

feature_shape_strategy = st.lists(
    st.integers(min_value=1, max_value=4),  # positive feature dims
    min_size=0,
    max_size=3,
).map(tuple)

total_len_strategy = st.integers(min_value=0, max_value=50)
max_seq_len_strategy = st.integers(min_value=1, max_value=12)
dtype_strategy = st.sampled_from([np.float32, np.int64, np.bool_])

float_values = st.floats(
    allow_nan=False, allow_infinity=False, width=32, min_value=-1000, max_value=1000
)
int_values = st.integers(min_value=-1000, max_value=1000)


# ---------- main property test ----------


@settings(deadline=None, max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(
    data=st.data(),
    total_len=total_len_strategy,
    feature_shape=feature_shape_strategy,
    episode_lens=st.integers(min_value=0, max_value=50).flatmap(_partitions_of),
    max_seq_len=max_seq_len_strategy,
    dtype=dtype_strategy,
)
def test_numpy_and_torch_results_match(
    data, total_len, feature_shape, episode_lens, max_seq_len, dtype
):
    # Ensure episode_lens sums to total_len (keep zero-length episodes)
    s = sum(episode_lens)
    if s != total_len:
        if not episode_lens:
            episode_lens = [total_len]
        else:
            episode_lens = list(episode_lens)
            episode_lens[-1] += total_len - s
            if episode_lens[-1] < 0:
                episode_lens[-1] = 0
                if len(episode_lens) > 1:
                    episode_lens[0] += total_len - sum(episode_lens)

    shape: Tuple[int, ...] = (total_len,) + feature_shape

    # Draw the NumPy array using hypothesis.extra.numpy (no .example())
    if dtype is np.float32:
        np_arr = data.draw(hnp.arrays(np.float32, shape, elements=float_values))
    elif dtype is np.int64:
        np_arr = data.draw(hnp.arrays(np.int64, shape, elements=int_values))
    else:  # bool
        np_arr = data.draw(hnp.arrays(np.bool_, shape))

    # Torch tensor with identical content
    t_tensor = torch.from_numpy(np_arr.copy())

    # Run both implementations
    try:
        np_chunks = split_np(np_arr, episode_lens, max_seq_len)
    except IndexError:
        return
    torch_chunks = split_torch(t_tensor, episode_lens, max_seq_len)

    # Structure
    assert isinstance(np_chunks, list)
    assert isinstance(torch_chunks, list)
    assert len(np_chunks) == len(torch_chunks), "Different number of chunks returned"

    # Compare each chunk
    for i, (na, ta) in enumerate(zip(np_chunks, torch_chunks)):
        assert isinstance(na, np.ndarray)
        assert isinstance(ta, torch.Tensor)
        ta_np = ta.detach().cpu().numpy()

        assert na.shape == ta_np.shape, (
            f"Chunk {i}: shapes differ: {na.shape} vs {ta_np.shape}"
        )
        assert na.shape[0] == max_seq_len, (
            f"Chunk {i}: unexpected batch length {na.shape[0]}"
        )
        assert na.shape[1:] == feature_shape, f"Chunk {i}: feature shape changed"

        if na.dtype == np.bool_:
            assert np.array_equal(na.astype(bool), ta_np.astype(bool)), (
                f"Chunk {i}: bool values differ"
            )
        else:
            assert np.array_equal(na, ta_np), f"Chunk {i}: values differ"
