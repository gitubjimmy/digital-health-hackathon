# ==============================================================================
# Freezes configs to make itself cannot be changed in other modules, by chance.
# By naming each variable with ** CAMEL CASE **,
# just write your configs down in this block:
#

#
# IMPORT
#
import sys
import pathlib

#
# PATH CONFIGS
#

ROOT = pathlib.Path(__file__).resolve().parent

OUTPUT_FILE_PATH = ROOT / 'output.txt'

CHECKPOINT_PATH = ROOT / 'checkpoint'

#
# HYPER PARAMETERS : convertible as json
#

IN_CHANNELS = 311  # type: int

CHANNELS = 512  # type: int

NUM_LAYERS = 5  # type: int

ACTIVATION = 'leaky_relu'  # type: str

DROPOUT_RATE = 0.5  # type: float

BATCH_SIZE = 5  # type: int

NUM_K_FOLD = 5  # type: int

EPOCH_PER_K_FOLD = 200  # type: int

K_FOLD_REPEAT = 15  # type: int

OPTIMIZER = 'Adam'  # type: str

OPTIMIZER_OPTIONS = {

    'lr': 1e-6

}  # type: dict[str, float]

LR_SCHEDULER = 'CosineAnnealingWarmUpRestarts'  # type: str

LR_SCHEDULER_OPTIONS = {

    'T_0': 20, 'T_mult': 2, 'eta_max': 1e-3, 'T_up': 5, 'gamma': 0.5

}  # type: dict[str, [int, float]]

# OPTIMIZER: str = 'Adam'
#
# OPTIMIZER_OPTIONS: dict = {'lr': 1e-3}
#
# LR_SCHEDULER: str = 'CosineAnnealingWarmRestarts'
#
# LR_SCHEDULER_OPTIONS: dict = {'T_0': 10, 'T_mult': 2, 'eta_min': 1e-5, 'last_epoch': -1}

# OPTIMIZER: str = 'Adam'
#
# OPTIMIZER_OPTIONS: dict = {'lr': 1e-3}
#
# LR_SCHEDULER: str =  'StepLR'
#
# LR_SCHEDULER_OPTIONS: dict = {'step_size': 10, 'gamma': 1e-1 ** 0.5}

#
# ==============================================================================


# ==============================================================================
# If a particular config needs to be mutable,
# add the name of the particular setting in the following tuple:
#
__mutable_attributes__ = ('ACTIVATION', )
#
# ==============================================================================


# ==============================================================================
# Do not change scripts below:
#
class FrozenConfig(type(sys)):
    __doc__ = __doc__

    __INITIALIZED = False
    __MUTABLE_ATTRIBUTES = __mutable_attributes__[:]

    def __init__(self, name):
        super().__init__(name)
        self.__package__ = __package__
        for k, v in globals().items():
            if k.isupper():
                super().__setattr__(k, v)  # Avoid redundant checking by super()
        self.__INITIALIZED = True

    def __setattr__(self, k, v):
        if self.__INITIALIZED and not k.startswith('_') and k not in self.__MUTABLE_ATTRIBUTES:
            raise TypeError("can't set {!r} attribute".format(k))
        return super().__setattr__(k, v)

    def __delattr__(self, k):
        if self.__INITIALIZED and not k.startswith('_') and k not in self.__MUTABLE_ATTRIBUTES:
            raise TypeError("can't delete {!r} attribute".format(k))
        return super().__delattr__(k)


sys.modules[__name__] = FrozenConfig(__name__)
#
# ==============================================================================
