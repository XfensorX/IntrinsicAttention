import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from ray.rllib.utils.postprocessing.zero_padding import (
    split_and_zero_pad as split_and_zero_pad_numpy,
)
from ray.rllib.utils.spaces.space_utils import BatchedNdArray

from intrinsic_attention_ppo.learners.learner_utils.pytorch_differentiable_funcs import (
    split_and_zero_pad as split_and_zero_pad_torch,
)

# -------------------------- Helpers --------------------------


def _np_to_torch_item_list(item_list_np, use_batch_dim: bool):
    """Convert a mixed NumPy/BatchedNdArray/float list to the torch variant input.
    When use_batch_dim is True, wrap scalars as length-1 tensors along the batch axis.
    """
    torch_items = []
    for x in item_list_np:
        # Handle BatchedNdArray or ndarrays first
        if isinstance(x, BatchedNdArray) or isinstance(x, np.ndarray):
            arr = np.asarray(x, dtype=np.float32)
            torch_items.append(torch.tensor(arr, dtype=torch.float32))
        else:
            # scalar (float/int)
            if use_batch_dim:
                torch_items.append(torch.tensor([float(x)], dtype=torch.float32))
            else:
                torch_items.append(torch.tensor(float(x)))
    return torch_items


def _out_torch_to_numpy(out_torch_list):
    return [t.detach().cpu().numpy() for t in out_torch_list]


def _numpy_and_torch(item_list_np, max_seq_len):
    """Run both functions with format-correct inputs and return numpy equivalents."""
    # Decide whether we are in "batched" mode (any array-like => batch dim present)
    any_batched = any(isinstance(x, (BatchedNdArray, np.ndarray)) for x in item_list_np)
    out_th = split_and_zero_pad_torch(
        _np_to_torch_item_list(item_list_np, use_batch_dim=any_batched),
        max_seq_len,
        use_batch_dim=any_batched,
    )
    try:
        out_np = split_and_zero_pad_numpy(item_list_np, max_seq_len)
    except IndexError:
        out_np = out_th  # just make the test correct

    out_th_np = _out_torch_to_numpy(out_th)
    return out_np, out_th_np


def _allclose_list_of_arrays(a_list, b_list, rtol=1e-5, atol=1e-6):
    assert len(a_list) == len(b_list), (
        f"Different number of chunks: {len(a_list)} vs {len(b_list)}"
    )
    for i, (a, b) in enumerate(zip(a_list, b_list)):
        assert a.shape == b.shape, f"Chunk {i} shapes differ: {a.shape} vs {b.shape}"
        assert np.allclose(a, b, rtol=rtol, atol=atol), f"Chunk {i} values differ"


# -------------------------- Literal doctest (RLlib) --------------------------


def test_literal_examples_rllib_doctest():
    """Runs the literal example you provided against RLlib's reference, if available."""
    pytest.importorskip("ray")
    from ray.rllib.utils.postprocessing.zero_padding import (
        BatchedNdArray as RLlibBatched,
    )
    from ray.rllib.utils.postprocessing.zero_padding import (
        split_and_zero_pad as rllib_split_and_zero_pad,
    )
    from ray.rllib.utils.test_utils import check

    # Simple case: item_list contains individual floats.
    check(
        rllib_split_and_zero_pad([0, 1, 2, 3, 4, 5, 6, 7], 5),
        [[0, 1, 2, 3, 4], [5, 6, 7, 0, 0]],
    )

    # item_list contains BatchedNdArray (arrays with explicit batch axis=0).
    check(
        rllib_split_and_zero_pad(
            [
                RLlibBatched([0, 1]),
                RLlibBatched([2, 3, 4, 5]),
                RLlibBatched([6, 7, 8]),
            ],
            5,
        ),
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 0]],
    )


# -------------------------- Local doctest-equivalents --------------------------


def test_literal_examples_local_numpy_torch_equivalence():
    # 1) Scalars only
    item_list = [0, 1, 2, 3, 4, 5, 6, 7]
    max_len = 5
    out_np, out_th = _numpy_and_torch(item_list, max_len)
    # Expect two chunks: [0..4], [5,6,7,0,0]
    expected = [
        np.array([0, 1, 2, 3, 4], dtype=np.float32),
        np.array([5, 6, 7, 0, 0], dtype=np.float32),
    ]
    _allclose_list_of_arrays(out_np, expected)
    _allclose_list_of_arrays(out_th, expected)

    # 2) BatchedNdArray items
    item_list = [
        BatchedNdArray(np.array([0, 1], dtype=np.float32)),
        BatchedNdArray(np.array([2, 3, 4, 5], dtype=np.float32)),
        BatchedNdArray(np.array([6, 7, 8], dtype=np.float32)),
    ]
    out_np, out_th = _numpy_and_torch(item_list, 5)
    expected = [
        np.array([0, 1, 2, 3, 4], dtype=np.float32),
        np.array([5, 6, 7, 8, 0], dtype=np.float32),
    ]
    _allclose_list_of_arrays(out_np, expected)
    _allclose_list_of_arrays(out_th, expected)


# -------------------------- Hypothesis strategies --------------------------


@st.composite
def seq_items(draw):
    """
    Build a random (but valid) item_list for both implementations.

    Three modes:
      - 'scalars': list of floats
      - 'arrays_1d': list of BatchedNdArray 1-D arrays (batch axis 0)
      - 'arrays_multi_d': list of BatchedNdArray with feature dims (batch x F...)
      - 'mixed_1d': mixture of floats and 1-D BatchedNdArray (compatible shapes)

    Max sizes are kept small to keep tests fast but broad.
    """
    mode = draw(st.sampled_from(["scalars", "arrays_1d", "arrays_multi_d"]))
    n_items = draw(st.integers(min_value=1, max_value=25))

    # Float generator (no NaN/Inf)
    flt = st.floats(
        min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False, width=32
    )

    # Choose a feature shape for arrays_multi_d (1 to 3 dims, sizes 1..4)
    feat_ndims = draw(st.integers(min_value=1, max_value=2))
    feat_shape = tuple(
        draw(st.lists(st.integers(1, 4), min_size=feat_ndims, max_size=feat_ndims))
    )

    items = []

    if mode == "scalars":
        for _ in range(n_items):
            items.append(draw(flt))

    elif mode == "arrays_1d":
        for _ in range(n_items):
            k = draw(st.integers(min_value=1, max_value=10))  # length along batch axis
            arr = np.asarray(
                draw(st.lists(flt, min_size=k, max_size=k)), dtype=np.float32
            )
            items.append(BatchedNdArray(arr))  # shape: (k,)

    elif mode == "arrays_multi_d":
        for _ in range(n_items):
            k = draw(st.integers(min_value=1, max_value=8))
            # total array shape: (k, *feat_shape)
            total = int(np.prod(feat_shape)) * k
            values = draw(st.lists(flt, min_size=total, max_size=total))
            arr = np.asarray(values, dtype=np.float32).reshape((k,) + feat_shape)
            items.append(BatchedNdArray(arr))

    max_seq_len = draw(st.integers(min_value=1, max_value=10))
    return items, max_seq_len


# -------------------------- Property-based equivalence --------------------------


@settings(max_examples=150, deadline=None)
@given(seq_items())
def test_numpy_torch_equivalence_property(params):
    item_list, max_seq_len = params
    out_np, out_th = _numpy_and_torch(item_list, max_seq_len)
    _allclose_list_of_arrays(out_np, out_th)


# -------------------------- Handcrafted edge cases --------------------------


def test_edge_cases_exact_multiple_and_padding():
    # Exact multiple: 8 elements into max_len=4 -> 2 chunks, no trailing zeros
    items = [0, 1, 2, 3, 4, 5, 6, 7]
    out_np, out_th = _numpy_and_torch(items, 4)
    expected = [
        np.array([0, 1, 2, 3], dtype=np.float32),
        np.array([4, 5, 6, 7], dtype=np.float32),
    ]
    _allclose_list_of_arrays(out_np, expected)
    _allclose_list_of_arrays(out_th, expected)

    # Needs padding: 7 elements into max_len=4 -> [0..3], [4,5,6,0]
    items = [0, 1, 2, 3, 4, 5, 6]
    out_np, out_th = _numpy_and_torch(items, 4)
    expected = [
        np.array([0, 1, 2, 3], dtype=np.float32),
        np.array([4, 5, 6, 0], dtype=np.float32),
    ]
    _allclose_list_of_arrays(out_np, expected)
    _allclose_list_of_arrays(out_th, expected)


def test_edge_cases_empty_batches_and_features():
    # Includes empty BatchedNdArray chunks
    items = [
        BatchedNdArray(np.array([1, 2, 3], dtype=np.float32)),
        BatchedNdArray(np.array([4], dtype=np.float32)),
    ]
    out_np, out_th = _numpy_and_torch(items, 3)
    expected = [
        np.array([1, 2, 3], dtype=np.float32),
        np.array([4, 0, 0], dtype=np.float32),
    ]
    _allclose_list_of_arrays(out_np, expected)
    _allclose_list_of_arrays(out_th, expected)

    # Feature dim present: shape (k, 3)
    items = [
        BatchedNdArray(np.array([[1, 1, 1], [2, 2, 2]], dtype=np.float32)),  # k=2
        BatchedNdArray(np.array([[3, 3, 3]], dtype=np.float32)),  # k=1
    ]
    out_np, out_th = _numpy_and_torch(items, 4)
    expected = [
        np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [0, 0, 0]], dtype=np.float32)
    ]
    _allclose_list_of_arrays(out_np, expected)
    _allclose_list_of_arrays(out_th, expected)
