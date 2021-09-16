from .mlp import MLP

from . import mlp


def get_model(**override):
    import config
    options = dict(
        in_channels=config.IN_CHANNELS,
        out_channels=1,
        channels=config.CHANNELS,
        num_layers=config.NUM_LAYERS,
        dropout_rate=config.DROPOUT_RATE,
        activation=config.ACTIVATION
    )
    options.update(override)
    return MLP(**options)


def get_optimizer_from_config(model):
    import config
    import torch.optim as optim
    return getattr(optim, config.OPTIMIZER)(model.parameters(), **config.OPTIMIZER_OPTIONS)


def get_lr_scheduler_from_config(optimizer):
    import config
    import torch.optim.lr_scheduler as scheduler
    return getattr(scheduler, config.LR_SCHEDULER)(optimizer, **config.LR_SCHEDULER_OPTIONS)
