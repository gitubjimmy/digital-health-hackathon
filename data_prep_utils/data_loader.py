import torch
import torch.utils.data

import pandas as pd

import config


class PandasDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            dataframe: pd.DataFrame,
            label_target: str,
    ) -> None:
        super().__init__()
        self.label_target = label_target
        self.label_input = list(dataframe.columns)
        self.label_input.remove('time')
        # Drop NaN
        dataframe = dataframe.drop(dataframe.index[dataframe[label_target] != dataframe[label_target]])
        dataframe.index = range(len(dataframe))
        self.samples = dataframe

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> ...:
        row = self.samples.iloc[index]
        x = torch.tensor(row[self.label_input])
        y = torch.tensor(row[self.label_target])
        return x, y


def get_loader(dataset, train=True, batch_size=config.BATCH_SIZE):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=2)
