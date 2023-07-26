import pandas as pd
import numpy as np
from numpy import savez_compressed
import os


training_ember_2017_filepath = "/home/kaito/postdoc_projects/CeADAR/modifying_IAT_to_bypass_ML_detectors/src/hashed_imports_detector/data/csv/clean_training_features_2017.csv"
testing_ember_2017_filepath = "/home/kaito/postdoc_projects/CeADAR/modifying_IAT_to_bypass_ML_detectors/src/hashed_imports_detector/data/csv/clean_testing_features_2017.csv"

training_numpy_2017_filepath = "/mnt/hdd2/inception/imports/2017/training/"
testing_numpy_2017_filepath = "/mnt/hdd2/inception/imports/2017/testing"

training_ember_2018_filepath = "/home/kaito/postdoc_projects/CeADAR/modifying_IAT_to_bypass_ML_detectors/src/hashed_imports_detector/data/csv/clean_training_features_2018.csv"
testing_ember_2018_filepath = "/home/kaito/postdoc_projects/CeADAR/modifying_IAT_to_bypass_ML_detectors/src/hashed_imports_detector/data/csv/clean_testing_features_2018.csv"
training_numpy_2018_filepath = "/mnt/hdd2/inception/imports/2018/training/"
testing_numpy_2018_filepath = "/mnt/hdd2/inception/imports/2018/testing/"


with open(training_ember_2017_filepath, "r") as features_file:
    lines = features_file.readlines()
    for line in lines[1:]:
        line = line.strip().split(",")
        sha256 = line[0]
        X = np.array([float(x) for x in line[1:]])
        savez_compressed(os.path.join(training_numpy_2017_filepath, '{}.npz'.format(sha256)),
                         X)

with open(testing_ember_2017_filepath, "r") as features_file:
    lines = features_file.readlines()
    for line in lines[1:]:
        line = line.strip().split(",")
        sha256 = line[0]
        X = np.array([float(x) for x in line[1:]])
        savez_compressed(os.path.join(testing_numpy_2017_filepath, '{}.npz'.format(sha256)),
                         X)

with open(training_ember_2018_filepath, "r") as features_file:
    lines = features_file.readlines()
    for line in lines[1:]:
        line = line.strip().split(",")
        sha256 = line[0]
        X = np.array([float(x) for x in line[1:]])
        savez_compressed(os.path.join(training_numpy_2018_filepath, '{}.npz'.format(sha256)),
                         X)

with open(testing_ember_2018_filepath, "r") as features_file:
    lines = features_file.readlines()
    for line in lines[1:]:
        line = line.strip().split(",")
        sha256 = line[0]
        X = np.array([float(x) for x in line[1:]])
        savez_compressed(os.path.join(testing_numpy_2018_filepath, '{}.npz'.format(sha256)),
                         X)