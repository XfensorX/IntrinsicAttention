from ray.rllib.utils.framework import try_import_torch

from source.brainstorming.base_models.ReluMlp import ReluMlp

torch, nn = try_import_torch()


class GRUBase(nn.Module):
    def __init__(
        self, input_dim: int, hidden_size: int, num_layers: int, output_size: int
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.output_net = ReluMlp(
            [hidden_size, (hidden_size + output_size) // 2, output_size]
        )

    def forward(self, inputs: torch.Tensor, input_states: torch.Tensor):
        """
        !! Observations have to be padded to same length already
        input_dim should be some kind of mixture of actions and observations

        Args:
            inputs (torch.Tensor): shape BATCH x max_trajectory_length x input_dim

        Return:
            # output: (batch x time x hidden_size)
            # h: (num_layers x  batch x hidden_size)
        """
        input_states = input_states.contiguous()
        output, h = self.gru(inputs, input_states)
        output = self.output_net(output)
        return output, h
