import pandas as pd
import sys

sys.path.append("../../../../")
import numpy as np
import torch
from src.ml_models.torch_models.inception_detector.byte_histogram_net import ByteHistogramNetwork
from src.gan_implementations.utils import load_json
from numpy import savez_compressed

training_ember_2017_filepath = "/home/kaito/postdoc_projects/CeADAR/appending_bytes_to_bypass_ML_detectors/src/byte_histogram/data/csv/shuffled_train_features_2017.csv"
training_ember_2018_filepath = "/home/kaito/postdoc_projects/CeADAR/appending_bytes_to_bypass_ML_detectors/src/byte_histogram/data/csv/shuffled_train_features_2018.csv"

parameters = load_json("network_parameters/byte_histogram_net_params.json")
model = ByteHistogramNetwork(parameters)
model.load_state_dict(torch.load("models/byte_histogram_model_2017/best/best_model.pt"))
model.eval()

# EMBER 2017

intermediate_features_array = []
with open(training_ember_2017_filepath, "r") as features_file:
    lines = features_file.readlines()
    for line in lines[1:]:
        line = line.strip().split(",")
        sha256 = line[0]
        X = np.array([float(x) for x in line[1:-1]])
        X = torch.from_numpy(X.astype(np.float32))

        Y = np.array(line[-1])
        intermediate_features = model.retrieve_features(X)
        intermediate_features_array.append(intermediate_features.cpu().detach().numpy())

intermediate_features_array = np.array(intermediate_features_array)
savez_compressed("intermediate_features/byte_histogram_2017.npz", intermediate_features_array)

# EMBER 2018
intermediate_features_array = []
with open(training_ember_2018_filepath, "r") as features_file:
    lines = features_file.readlines()
    for line in lines[1:]:
        line = line.strip().split(",")
        sha256 = line[0]
        X = np.array([float(x) for x in line[1:-1]])
        X = torch.from_numpy(X.astype(np.float32))

        Y = np.array(line[-1])
        intermediate_features = model.retrieve_features(X)
        intermediate_features_array.append(intermediate_features.cpu().detach().numpy())

intermediate_features_array = np.array(intermediate_features_array)
savez_compressed("intermediate_features/byte_histogram_2018.npz", intermediate_features_array)


