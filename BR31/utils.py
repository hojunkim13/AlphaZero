import numpy as np
import torch


def preprocess(n):
    x = np.eye(31)[n].reshape(-1, 31)
    x = torch.tensor(x, dtype=torch.float)
    return x
