import torch
import numpy as np


def jensen_shannon_distance_torch(P, Q):
    P_mean = torch.mean(P, dim=0)
    Q_mean = torch.mean(Q, dim=0)
    M = torch.div(torch.add(P_mean, Q_mean), 2.0)
    KL_PM = torch.sum(torch.mul(P_mean, torch.log(torch.div(P_mean, M))))
    KL_QM = torch.sum(torch.mul(Q_mean, torch.log(torch.div(Q_mean, M))))
    distance = torch.sqrt(torch.div(torch.add(KL_PM,KL_QM), 2))
    return distance


def jensen_shannon_distance_np(P, Q):
    P_mean = np.mean(P, axis=0)
    Q_mean = np.mean(Q, axis=0)
    M = np.divide(np.add(P_mean, Q_mean), 2.0)
    KL_PM = np.sum(np.multiply(P_mean, np.log(np.divide(P_mean, M))))
    KL_QM = np.sum(np.multiply(Q_mean, np.log(np.divide(Q_mean, M))))
    distance = np.sqrt(np.divide(np.add(KL_PM, KL_QM), 2))
    return distance
