import torch.nn as nn
import torch.nn.functional as f

import collections


class MLP(nn.ModuleList):

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
            activation='relu'
    ):
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
        if callable(activation):
            self.activation = activation
        elif hasattr(f, activation):
            self.activation = getattr(f, activation)
        elif hasattr(nn, activation):
            self.activation = getattr(f, activation)()
        else:
            raise TypeError("Wrong activation function: %r" % activation)

    def extra_repr(self):
        return "(non-linearity): %r" % self.activation

    def forward(self, x):  # noqa
        num_layers = len(self) - 1
        for index in range(num_layers):
            x = self.activation(self[index](x))
        return self[-1](x)


def get_model(**override):
    import config
    options = dict(
        in_channels=config.IN_CHANNELS,
        out_channels=1,
        channels=config.CHANNELS,
        num_layers=config.NUM_LAYERS,
        activation=config.ACTIVATION
    )
    options.update(override)
    return MLP(**options)
