import torch
import os
import numpy as np


class NumpyFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, numpy_filepath:str):
        self.numpy_filepath = numpy_filepath
        self.filenames = os.listdir(self.numpy_filepath)
        self.size = len(self.filenames)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        features_path = os.path.join(self.numpy_filepath, self.filenames[index])
        features = np.load(features_path, allow_pickle=True)["arr_0"]
        X = features[:-1]
        Y = features[-1]
        X = X.astype(np.float32)
        Y = Y.astype(np.float32)

        return X, Y

