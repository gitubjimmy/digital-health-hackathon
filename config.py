# ==============================================================================
# Freezes configs to make itself cannot be changed in other modules, by chance.
# By naming each variable with ** CAMEL CASE **,
# just write your configs down in this block:
#

#
# PATH CONFIGS
#

import pathlib

ROOT = pathlib.Path(__file__).resolve().parent

OUTPUT_FILE_PATH = ROOT / 'output.txt'

PATH = ROOT / 'cifar_net.pth'

#
# HYPER PARAMETERS  todo: hyper-parameter as json?
#

ACTIVATION: str = 'relu'

IN_CHANNELS: int

CHANNELS: int

NUM_LAYERS: int

BATCH_SIZE: int

INPUT_SIZE: ...

EPOCH: ...

#
# ==============================================================================


# ==============================================================================
# If a particular config needs to be mutable,
# add the name of the particular setting in the following tuple:
#
__mutable_attributes__ = ()
#
# ==============================================================================


# ==============================================================================
# Do not change scripts below:
#
import sys
import types


class FrozenConfig(types.ModuleType):
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
