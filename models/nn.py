import torch.nn as nn
import torch.nn.functional as f

import collections


class MLP(nn.Module):

    """
    Multi Layer Perceptron.

    Example:

        >>> my_model = MLP(
        ...     in_channels=100, out_channels=4,
        ...     channels=256, num_layers=4, activation='relu'
        ... )
        ...
        >>> my_model
        MLP(
          (0): Linear(in_features=100, out_features=256, bias=True)
          (1): Linear(in_features=256, out_features=128, bias=True)
          (2): Linear(in_features=128, out_features=64, bias=True)
          (3): Linear(in_features=64, out_features=32, bias=True)
          (4): Linear(in_features=32, out_features=4, bias=True)
        )

    """

    def __init__(
            self,
            in_channels,
            out_channels,
            channels,
            num_layers=None,
            dropout_rate=None,
            activation='relu'
    ):

        super().__init__()
        self.layers = nn.ModuleList()

        if isinstance(channels, collections.Iterable) and num_layers is None:
            current = in_channels
            channels = channels if isinstance(channels, (tuple, list)) else list(channels)
            for item in channels:
                self.layers.append(nn.Linear(current, item))
                current = item
            self.last_layer = nn.Linear(current, out_channels)

        elif isinstance(channels, int) and isinstance(num_layers, int):
            current = in_channels
            for _ in range(num_layers):
                self.layers.append(nn.Linear(current, channels))
                current = channels
                channels //= 2
            self.last_layer = nn.Linear(current, out_channels)

        else:
            raise TypeError(
                "(channels, num_layers) must be (Iterable, None) or (int, int), "
                "got (%s, %s)." % (type(channels).__name__, type(num_layers).__name__)
            )

        if dropout_rate is not None:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = lambda x: x

        self.activation = self._get_activation(activation)

    @staticmethod
    def _get_activation(activation):
        if callable(activation):
            return activation
        elif isinstance(activation, str):
            if hasattr(f, activation):
                return getattr(f, activation)
            elif hasattr(nn, activation):
                return getattr(f, activation)()
            else:
                raise TypeError("Wrong activation string: %s" % activation)
        elif activation is None:
            return lambda x: x
        else:
            raise TypeError("Wrong activation function: %r" % activation)

    def extra_repr(self):
        if not isinstance(self.activation, nn.Module):
            return "(non-linearity): %r" % self.activation
        return ""

    def forward(self, x):  # noqa
        for layer in self.layers:
            x = self.dropout(self.activation(layer(x)))
        return self.last_layer(x)


__all__ = ['MLP']
