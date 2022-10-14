import numpy as np

import torch
from torch.utils.data import Dataset

class BasicDataset(Dataset):
    def __init__(self, dataSet, labelSet):
        super(BasicDataset, self).__init__()

        self.dataSet = torch.Tensor(np.array(dataSet))
        self.labelSet = torch.Tensor(np.array(labelSet))

    def __len__(self):
        return len(self.labelSet)

    def __getitem__(self, index):
        return self.dataSet[index], self.labelSet[index]
