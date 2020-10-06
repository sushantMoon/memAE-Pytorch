from __future__ import print_function, absolute_import

import torch
from torch.utils.data import Dataset
import numpy as np

# N x C
class VectorDataset(Dataset):
    # Same are TensorDatasets, as given in following link
    # https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#TensorDataset
    # Expects a numpy array of size (BatchSize N x Features C)
    def __init__(self, *tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)