import torch
import numpy as np


def euclidean_distance_torch(P, Q):
    return torch.sqrt(torch.sum(torch.square(torch.sub(P, Q))))


def euclidean_distance_np(P, Q):
    return np.linalg.norm(P-Q)

