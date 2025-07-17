from ray.rllib.utils.framework import try_import_torch

from source.proposal.models.ReluMlp import ReluMlp

torch, nn = try_import_torch()


# For better Interpretabilty, could be more if it is not working
NUM_HEADS = 1


class IntrinsicAttention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        v_dim: int,
        qk_dim: int,
    ):
        super().__init__()

        self.attention_layer = nn.MultiheadAttention(
            embed_dim=input_dim,
            kdim=qk_dim,
            vdim=v_dim,
            num_heads=NUM_HEADS,
            batch_first=True,
        )

        self.reward_layer = ReluMlp([v_dim, v_dim // 2, 1], output_layer=nn.Tanh)

    def forward(
        self,
        inputs: torch.Tensor,
        need_weights: bool = False,
    ):
        """
        !! Observations have to be padded to same length already
        input_dim should be some kind of mixture of actions and observations

        Args:
            inputs (torch.Tensor): shape BATCH x max_trajectory_length x input_dim

        Return:
            rewards:
                shape: BATCH x max_trajectory_length x 1
            attn_weights: from the attn layer add the dimensions
                shape: BATCH x max_trajectory_length x max_trajectory_length
                 IF NEED_WEIGHTS is TRUE!!, else None
        """

        attn_out, attn_weights = self.attention_layer(inputs, need_weights=need_weights)
        rewards = self.reward_layer(attn_out).squeeze()

        return rewards, attn_weights
