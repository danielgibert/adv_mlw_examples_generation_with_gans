import torch
import numpy as np


def minkowski_distance_torch(P, Q, p=2):
    return torch.pow(torch.sum(torch.pow(torch.abs(torch.subtract(P,Q)),p)),1/p)


def minkowski_distance_np(P, Q, p=2):
    return np.power(np.sum(np.power(np.abs(np.subtract(P,Q)),p)),1/p)

