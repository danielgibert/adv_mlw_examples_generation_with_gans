import torch
import pandas as pd
import numpy as np


class CSVFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, features_filepath:str):
        self.features_filepath = features_filepath
        self.df = pd.read_csv(self.features_filepath)
        self.Y = self.df["label"]
        self.X = self.df.drop(columns=["sha256", "label"])
        self.size = len(self.df)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.X.iloc[index], self.Y.iloc[index]

#dataset = CSVFeatureDataset("../hashed_imports_detector/data/csv/subset_training_features_2017.csv")
#print(dataset.__getitem__([0,1,2]))

