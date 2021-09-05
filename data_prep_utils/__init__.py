from .data_getter import *
from .data_loader import *
from .preprocessor import *

from . import data_getter
from . import data_loader
from . import preprocessor


def get_data_loaders(batch_size=4):

    import torch
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from torchvision.datasets import CIFAR10

    from utils import kill_stderr

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    with kill_stderr():
        train_set = CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    with kill_stderr():
        test_set = CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader
