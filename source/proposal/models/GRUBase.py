import torch
import torch.nn as nn


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

        output_net_hidden_size = (hidden_size + output_size) // 2
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size, output_net_hidden_size),
            nn.ReLU(),
            nn.Linear(output_net_hidden_size, output_size),
            nn.ReLU(),
        )

    def forward(self, inputs: torch.Tensor, input_states: torch.Tensor):
        """
        !! Observations have to be padded to same length already
        input_dim should be some kind of mixture of actions and observations

        Args:
            inputs (torch.Tensor): shape BATCH x max_trajectory_length x input_dim

        Return:
            # TODO: add shapes
        """

        output, h = self.gru(inputs, input_states.unsqueeze(0))
        output = self.output_net(output)
        return output, h
