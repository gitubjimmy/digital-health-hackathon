import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

import pandas as pd

from utils import kill_stderr

import functools


csv_loader = functools.partial(pd.read_csv, index_col=0)


def get_data_dir():
    from config import ROOT
    return ROOT / 'data'


def get_data():
    data_dir = get_data_dir()
    clinical_variables = csv_loader(data_dir / 'Clinical_Variables.csv')
    generic_alterations = csv_loader(data_dir / 'Genetic_alterations.csv')
    survival_time_event = csv_loader(data_dir / 'Survival_time_event.csv')
    treatment = csv_loader(data_dir / 'Treatment.csv')

    return clinical_variables, generic_alterations, survival_time_event, treatment


def get_data_loaders():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    with kill_stderr():
        train_set = CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    with kill_stderr():
        test_set = CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader
