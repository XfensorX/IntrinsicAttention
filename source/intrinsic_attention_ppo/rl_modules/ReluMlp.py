from typing import Type

from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class ReluMlp(nn.Module):
    def __init__(
        self,
        hidden_layers: list[int],
        input_size: int,
        output_size: int,
        output_layer: Type[nn.Module] | None = nn.ReLU,
    ) -> None:
        """Simple Module for a multilayer perceptron

        Args:
            sizes (list[int]): The sizes of the layers, starting with input_size, ending with output_size
            output_layer (None | Type[nn.Module], optional): The output-layer activation function to use. Can be None to return logits. Defaults to nn.ReLU.
        """
        super().__init__()
        sizes = [input_size] + hidden_layers + [output_size]

        layers: list[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.ReLU())

        if output_layer:
            layers[-1] = output_layer()
        else:
            layers.pop(-1)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.net(x)
