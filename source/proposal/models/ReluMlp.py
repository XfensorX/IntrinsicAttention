from torch import nn


class ReluMlp(nn.Module):
    def __init__(self, sizes: list[int], output_layer: None | nn.Module = nn.ReLU):
        super().__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.ReLU())

        if output_layer:
            layers[-1] = output_layer()
        else:
            layers.pop(-1)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
