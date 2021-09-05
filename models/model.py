import torch.nn as nn
import torch.nn.functional as f

import torch.optim as opt

import collections


class MLP(nn.ModuleList):

    def __init__(self, in_channels, out_channels, channels, num_layers=None, activation='relu'):
        super().__init__()

        if isinstance(channels, collections.Iterable) and num_layers is None:
            current = in_channels
            channels = channels if isinstance(channels, (tuple, list)) else list(channels)
            for item in channels:
                self.append(nn.Linear(current, item))
                current = item
            self.append(nn.Linear(current, out_channels))

        elif isinstance(channels, int) and isinstance(num_layers, int):
            current = in_channels
            for _ in range(num_layers):
                self.append(nn.Linear(current, channels))
                current = channels
                channels //= 2
            self.append(nn.Linear(current, out_channels))

        else:
            raise TypeError(
                "(channels, num_layers) must be (Iterable, None) or (int, int), "
                "got (%s, %s)." % (type(channels).__name__, type(num_layers).__name__)
            )

        self.activation = str(activation)

    def _get_activation(self):
        if hasattr(f, self.activation):
            return getattr(f, self.activation)

    def forward(self, x):  # noqa
        num_layers = len(self) - 1
        activation = self._get_activation()
        for index in range(num_layers):
            x = activation(self[index](x))
        return self[-1](x)
