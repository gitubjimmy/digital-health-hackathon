import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

import pandas as pd

from utils import kill_stderr


def get_data():
    clinical_variables = pd.read_csv('./data/Clinical_Variables.csv')
    generic_alterations = pd.read_csv('./data/Genetic_alterations.csv')
    survival_time_event = pd.read_csv('./data/Survival_time_event.csv')
    treatment = pd.read_csv('./data/Treatment.csv')

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
