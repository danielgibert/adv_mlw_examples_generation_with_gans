import torch
import numpy as np


def hamming_distance_torch(P, Q):
    return torch.divide(torch.sum(torch.abs(torch.subtract(P, Q))), P.shape[0])


def hamming_distance_np(P, Q):
    return np.divide(np.sum(np.abs(np.subtract(P,Q))), P.shape[0])

