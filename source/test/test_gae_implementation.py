import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from ray.rllib.utils.postprocessing.value_predictions import (
    compute_value_targets as compute_value_targets_rllib,
)

from brainstorming.learners.pytorch_differentiable_funcs import (
    compute_value_targets,
    unpad_data_if_necessary,
)


def test_unpad_data_if_necessary():
    # Case 1: simple right-side zero-padding removal
    data = torch.tensor(
        [
            [2, 4, 5, 3, 0, 0, 0, 0],
            [-1, 3, 0, 0, 0, 0, 0, 0],
        ]
    )
    unpadded = unpad_data_if_necessary(
        episode_lens=[4, 2],
        data=data,
    )
    expected = torch.tensor([2, 4, 5, 3, -1, 3])
    assert torch.equal(unpadded, expected), f"Expected {expected}, got {unpadded}"

    # Case 2: first episode length is shorter
    data = torch.tensor(
        [
            [2, 0, 0, 0, 0],
            [-1, -2, -3, -4, -5],
        ]
    )
    unpadded = unpad_data_if_necessary(
        episode_lens=[1, 5],
        data=data,
    )
    expected = torch.tensor([2, -1, -2, -3, -4, -5])
    assert torch.equal(unpadded, expected), f"Expected {expected}, got {unpadded}"


# ---- Helpers -----------------------------------------------------------------


def _device_id(d):
    # nicer test IDs in the output
    return str(d)


def _to_torch(x):
    # We always pass float32 to the torch version for stable, comparable math.
    return torch.tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32)


def _call_numpy(values, rewards, terminateds, truncateds, gamma, lambda_):
    return compute_value_targets_rllib(
        np.asarray(values, dtype=np.float32),
        np.asarray(rewards, dtype=np.float32),
        np.asarray(terminateds, dtype=np.float32),
        np.asarray(truncateds, dtype=np.float32),
        float(gamma),
        float(lambda_),
    )


def _call_torch(values, rewards, terminateds, truncateds, gamma, lambda_):
    out = compute_value_targets(
        _to_torch(values),
        _to_torch(rewards),
        _to_torch(terminateds),
        _to_torch(truncateds),
        float(gamma),
        float(lambda_),
    )
    # Move back to CPU/NumPy for comparison
    return out.detach().cpu().numpy()


# ---- Hypothesis strategies ---------------------------------------------------


@st.composite
def seq_inputs(draw):
    # Sequence length ≥ 1 (the numpy implementation errors on length 0)
    n = draw(st.integers(min_value=1, max_value=300))

    # Use float32-range-ish values; no NaN/Inf in the main property test
    flt = st.floats(
        min_value=-1e5,
        max_value=1e5,
        allow_nan=False,
        allow_infinity=False,
        width=32,
    )

    values = draw(st.lists(flt, min_size=n, max_size=n))
    rewards = draw(st.lists(flt, min_size=n, max_size=n))

    term = draw(st.lists(st.integers(min_value=0, max_value=1), min_size=n, max_size=n))
    trunc = draw(
        st.lists(st.integers(min_value=0, max_value=1), min_size=n, max_size=n)
    )

    # Can't be terminal AND truncated at the same time step.
    trunc = [trunc[i] if term[i] == 0 else 0 for i in range(n)]

    # Include the endpoints 0.0 and 1.0
    gamma = draw(
        st.floats(
            min_value=0.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
            width=32,
        )
    )
    lambda_ = draw(
        st.floats(
            min_value=0.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
            width=32,
        )
    )

    return (
        np.asarray(values, dtype=np.float32),
        np.asarray(rewards, dtype=np.float32),
        np.asarray(term, dtype=np.float32),
        np.asarray(trunc, dtype=np.float32),
        float(gamma),
        float(lambda_),
    )


# ---- Property-based test: broad input coverage -------------------------------


@settings(max_examples=200, deadline=None)
@given(seq_inputs())
def test_numpy_torch_match_property(x):
    values, rewards, terminateds, truncateds, gamma, lambda_ = x

    np_out = _call_numpy(values, rewards, terminateds, truncateds, gamma, lambda_)
    th_out = _call_torch(values, rewards, terminateds, truncateds, gamma, lambda_)

    assert np_out.shape == th_out.shape
    # NumPy version returns float32; allow small floating diffs
    assert np.allclose(np_out, th_out, rtol=1e-4, atol=1e-5), (np_out, th_out)


# ---- Handcrafted edge cases --------------------------------------------------


@pytest.mark.parametrize(
    "values,rewards,terminateds,truncateds,gamma,lambda_",
    [
        # No terminals, no truncations
        ([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [0, 0, 0], [0, 0, 0], 0.99, 0.95),
        # Terminal at last step
        ([1.5, -0.5, 10.0], [1.0, 2.0, 3.0], [0, 0, 1], [0, 0, 0], 1.0, 1.0),
        # Terminal in the middle
        (
            [0.0, 5.0, 5.0, 5.0],
            [1.0, -1.0, 2.0, -2.0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            0.9,
            0.8,
        ),
        # Truncation in the middle (forces bootstrap from orig_values at that step)
        (
            [3.0, 4.0, 5.0, 6.0],
            [0.5, 0.5, 0.5, 0.5],
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            0.95,
            0.7,
        ),
        # Last step truncated (common in partial episodes / time limits)
        ([10.0, 0.0, -10.0], [1.0, 1.0, 1.0], [0, 0, 0], [0, 0, 1], 0.0, 0.0),
        # All steps "terminated" (degenerate but legal)
        ([5.0, 5.0, 5.0], [7.0, 7.0, 7.0], [1, 1, 1], [0, 0, 0], 0.5, 0.5),
        # Extreme gamma/lambda boundaries
        ([1.0, 2.0], [3.0, 4.0], [0, 1], [0, 0], 0.0, 1.0),
        ([1.0, 2.0], [3.0, 4.0], [0, 0], [0, 1], 1.0, 0.0),
    ],
)
def test_handcrafted_agree(values, rewards, terminateds, truncateds, gamma, lambda_):
    np_out = _call_numpy(values, rewards, terminateds, truncateds, gamma, lambda_)
    th_out = _call_torch(values, rewards, terminateds, truncateds, gamma, lambda_)

    assert np_out.shape == th_out.shape
    assert np.allclose(np_out, th_out, rtol=1e-5)


# ---- NaN propagation parity --------------------------------------------------


def test_nan_inputs_propagate_equally():
    # Inject NaNs; both implementations should propagate them identically.
    values = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    rewards = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    terminateds = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    truncateds = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    gamma, lambda_ = 0.99, 0.95

    np_out = _call_numpy(values, rewards, terminateds, truncateds, gamma, lambda_)
    th_out = _call_torch(values, rewards, terminateds, truncateds, gamma, lambda_)

    assert np_out.shape == th_out.shape
    # Use equal_nan=True to ensure NaN positions are considered equal
    assert np.allclose(np_out, th_out, rtol=1e-5, equal_nan=True)


# ---- Zero-length behavior parity --------------------------------------------


def test_zero_length_raises_same_kind_of_error():
    values = np.array([], dtype=np.float32)
    rewards = np.array([], dtype=np.float32)
    terminateds = np.array([], dtype=np.float32)
    truncateds = np.array([], dtype=np.float32)
    gamma, lambda_ = 0.9, 0.9

    # NumPy implementation raises on np.stack([]); ensure both raise *something*.
    with pytest.raises(Exception):
        _call_numpy(values, rewards, terminateds, truncateds, gamma, lambda_)

    with pytest.raises(Exception):
        _call_torch(values, rewards, terminateds, truncateds, gamma, lambda_)
