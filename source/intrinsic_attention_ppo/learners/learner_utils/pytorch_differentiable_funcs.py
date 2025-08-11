from typing import List, Sequence, Union

from collections import deque

from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


def unpad_data_if_necessary(
    episode_lens: List[int],
    data: torch.Tensor,
) -> torch.Tensor:
    """
    Pytorch Version of https://github.com/ray-project/ray/blob/8bdc75c09318dfb319df44265ffadbcb4c806e4f/rllib/utils/postprocessing/zero_padding.py#L203
    """

    # If data des NOT have time dimension, return right away.
    if len(data.shape) == 1:
        return data

    # Assert we only have B and T dimensions (meaning this function only operates
    # on single-float data, such as value function predictions, advantages, or rewards).
    assert len(data.shape) == 2

    new_data = []
    row_idx = 0

    T = data.shape[1]
    for len_ in episode_lens:
        # Calculate how many full rows this array occupies and how many elements are
        # in the last, potentially partial row.
        num_rows, col_idx = divmod(
            int(len_), T
        )  # TODO; check that this int-cast does not break differentiability

        # If the array spans multiple full rows, fully include these rows.
        for i in range(num_rows):
            new_data.append(data[row_idx])
            row_idx += 1

        # If there are elements in the last, potentially partial row, add this
        # partial row as well.
        if col_idx > 0:
            new_data.append(data[row_idx, :col_idx])

            # Move to the next row for the next array (skip the zero-padding zone).
            row_idx += 1

    return torch.cat(new_data)


def compute_value_targets(
    values: torch.Tensor,
    rewards: torch.Tensor,
    terminateds: torch.Tensor,
    truncateds: torch.Tensor,
    gamma: float,
    lambda_: float,
):
    """Computes value function (vf) targets given vf predictions and rewards.

    Note that advantages can then easily be computed via the formula:
    advantages = targets - vf_predictions
    Pytorch version of https://github.com/ray-project/ray/blob/8bdc75c09318dfb319df44265ffadbcb4c806e4f/rllib/utils/postprocessing/value_predictions.py#L7
    Preserves differentiability
    w.r.t. tensor inputs (values, rewards, gamma, lambda_) as long as they
    are provided as tensors with requires_grad=True.
    Shapes are assumed to be 1D over time (T,).
    """
    # Ensure scalars live on the same device/dtype and remain differentiable if tensors.
    device, dtype = values.device, values.dtype

    continues = 1.0 - terminateds.to(
        dtype
    )  # continue mask (1.0 where not terminated, 0.0 where terminated)

    # Match original: mask values where terminated
    orig_values = values * continues
    flat_values = orig_values

    # Append a trailing 0.0 (constant; no gradient needed for that element).
    flat_values = torch.cat(
        [flat_values, torch.zeros(1, dtype=dtype, device=device)], dim=0
    )

    # One-step “intermediate” targets
    intermediates = rewards + gamma * (1.0 - lambda_) * flat_values[1:]

    # Backward scan to compute returns with λ
    Rs = []
    last = flat_values[-1]  # this is the appended 0.0
    T = intermediates.shape[0]

    # Ensure truncateds is boolean for control via torch.where (keeps graph intact)
    truncateds_bool = truncateds.to(torch.bool)

    for t in reversed(range(T)):
        last = intermediates[t] + continues[t] * gamma * lambda_ * last
        # If truncated at t, reset last to the original (masked) value at t.
        # Using torch.where keeps everything in the graph (no .item()).
        Rs.append(last)
        last = torch.where(truncateds_bool[t], orig_values[t], last)

    # Reverse to chronological order and cast to float32 like the original code
    value_targets = torch.stack(Rs[::-1], dim=0).to(torch.float32)
    return value_targets


def split_and_zero_pad(
    item_list: List[Union[torch.Tensor, float]], max_seq_len: int, use_batch_dim: bool
) -> List[torch.Tensor]:
    """
    PythonVersion (for differentiability) of :

    https://github.com/ray-project/ray/blob/93f765d04ee055ebbd8437825ca197ab54bace4a/rllib/utils/postprocessing/zero_padding.py#L50


        Split `item_list` into chunks of length `max_seq_len` along time dim (dim 0),
        right-padding the last chunk with zeros if needed.

        Args:
            item_list: List of tensors.
                - If use_batch_dim=False: each tensor is a *single time step* with
                  feature shape `F` (any rank). We'll add a leading time dim of size 1.
                - If use_batch_dim=True: each tensor already has a leading time dim,
                  i.e., shape (T_i, *F).
            max_seq_len: Desired time length of each returned chunk.
            use_batch_dim: See above.

        Returns:
            List[torch.Tensor], each of shape (max_seq_len, *F).
    """
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be > 0")
    if not item_list:
        return []

    q = deque(item_list)
    ret: List[torch.Tensor] = []
    buffer: List[torch.Tensor] = []
    cur_len = 0

    # Infer feature shape/device/dtype from the first non-empty tensor encountered.
    def to_time_major(x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("All items must be torch.Tensors.")
        return x if use_batch_dim else x.unsqueeze(0)

    while q:
        raw = q.popleft()
        x = to_time_major(raw)
        if x.numel() == 0 or x.shape[0] == 0:
            continue  # skip empties safely

        remaining = max_seq_len - cur_len
        take = x[:remaining]
        buffer.append(take)
        cur_len += take.shape[0]

        # If leftover time steps remain, push them back to process next.
        if x.shape[0] > remaining:
            q.appendleft(x[remaining:])

        # Flush a full chunk.
        if cur_len == max_seq_len:
            ret.append(torch.cat(buffer, dim=0))
            buffer.clear()
            cur_len = 0

    # Tail + zero pad
    if cur_len > 0:
        current = torch.cat(buffer, dim=0) if len(buffer) > 1 else buffer[0]
        pad_len = max_seq_len - cur_len
        pad_shape = (pad_len,) + current.shape[1:]
        pad = torch.zeros(pad_shape, dtype=current.dtype, device=current.device)
        ret.append(torch.cat([current, pad], dim=0))

    return ret


def split_and_zero_pad_n_episodes(
    t: torch.Tensor, episode_lens: Sequence[int], max_seq_len: int
):
    ret = []

    cursor = 0
    episode_lens = [int(x) for x in episode_lens]
    for episode_len in episode_lens:
        items = t[cursor : cursor + episode_len]
        ret.extend(split_and_zero_pad([items], int(max_seq_len), True))
        cursor += episode_len

    return ret


def compute_gae(
    module,
    gamma: float,
    lambda_: float,
    vf_preds: torch.Tensor,
    rewards: torch.Tensor,
    terminateds: torch.Tensor,
    truncateds: torch.Tensor,
    real_seq_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    !!! Real Seq Lens have to be Adapted first for real episode lengths
                (e.g. if learner connector AddOneTsToEpisodesAndTruncate is used, 1 has ot be subtracted)
    """
    if vf_preds.dim() > 2 and vf_preds.shape[-1] == 1:
        vf_preds = vf_preds.squeeze(-1)

    vf_preds = unpad_data_if_necessary(real_seq_lens, vf_preds)

    value_targets = compute_value_targets(
        values=vf_preds,
        rewards=unpad_data_if_necessary(real_seq_lens, rewards),
        terminateds=unpad_data_if_necessary(real_seq_lens, terminateds),
        truncateds=unpad_data_if_necessary(real_seq_lens, truncateds),
        gamma=gamma,
        lambda_=lambda_,
    )
    assert value_targets.shape[0] == sum(real_seq_lens)

    advantages = value_targets - vf_preds
    advantages = (advantages - advantages.mean()) / advantages.std(
        unbiased=False
    ).clamp_min(1e-4)
    if module.is_stateful():
        advantages = torch.stack(
            split_and_zero_pad_n_episodes(
                advantages,
                episode_lens=real_seq_lens,
                max_seq_len=int(module.model_config["max_seq_len"]),
            ),
            dim=0,
        )
        value_targets = torch.stack(
            split_and_zero_pad_n_episodes(
                value_targets,
                episode_lens=real_seq_lens,
                max_seq_len=int(module.model_config["max_seq_len"]),
            ),
            dim=0,
        )

    return advantages, value_targets
