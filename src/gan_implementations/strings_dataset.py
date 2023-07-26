import torch
import pandas as pd
import os
import numpy as np
import json
from src.feature_extractors.utils import load_all_strings


class StringsDataset(torch.utils.data.Dataset):
    def __init__(self, strings_features_filepath: str, hashed_strings_features_filepath: str, all_strings_filepath: str,
                 ember_features_filepath_version1: str, ember_features_filepath_version2: str,
                 raw_executables_filepath: str, annotations_filepath: str):
        self.strings_features_filepath = strings_features_filepath
        self.hashed_strings_features_filepath = hashed_strings_features_filepath
        self.all_strings_filepath = all_strings_filepath
        self.ember_features_filepath_version1 = ember_features_filepath_version1
        self.ember_features_filepath_version2 = ember_features_filepath_version2
        self.raw_executables_filepath = raw_executables_filepath
        self.annotations_filepath = annotations_filepath
        self.sha256_labels = pd.read_csv(self.annotations_filepath)

        self.number_of_strings_features_files = len(os.listdir(self.strings_features_filepath))
        self.number_of_hashed_strings_features_files = len(os.listdir(self.hashed_strings_features_filepath))
        self.number_of_all_strings_files = len(os.listdir(self.all_strings_filepath))
        self.number_of_ember_features_files_version1 = len(os.listdir(self.ember_features_filepath_version1))
        self.number_of_ember_features_files_version2 = len(os.listdir(self.ember_features_filepath_version2))

        """ No check! When all features are extracted uncomment next piece of code
        if self.number_of_strings_features_files != self.number_of_hashed_strings_features_files and self.number_of_strings_features_files != self.number_of_all_strings_files and self.number_of_strings_features_files != self.number_of_ember_features_files_version1 and self.number_of_strings_features_files != self.number_of_ember_features_files_version2:
            raise Exception(
                "The number of files in {}, {}, {}, {} and {} must be the same! It is {}, {}, {}, {} and {}".format(
                    self.strings_features_filepath,
                    self.hashed_strings_features_filepath,
                    self.all_strings_filepath,
                    self.ember_features_filepath_version1,
                    self.ember_features_filepath_version2,
                    self.number_of_strings_features_files,
                    self.number_of_hashed_strings_features_files,
                    self.number_of_all_strings_files,
                    self.number_of_ember_features_files_version1,
                    self.number_of_ember_features_files_version2
                ))
        """

    def __len__(self):
        return len(self.sha256_labels.index)  # Remember to create the CSV file with the hashes and labels first


    def __getitem__(self, index):
        if isinstance(index, int):
            # Load strings features
            strings_features_path = os.path.join(self.strings_features_filepath, self.sha256_labels.iloc[index, 0]+".npz")
            strings_features = np.load(strings_features_path, allow_pickle=True)["arr_0"]

            # Load hashed strings features
            hashed_strings_features_path = os.path.join(self.hashed_strings_features_filepath,
                                                        self.sha256_labels.iloc[index, 0] + ".npz")
            hashed_strings_features = np.load(hashed_strings_features_path, allow_pickle=True)["arr_0"]

            # Allstrings filename
            all_strings_path = os.path.join(self.all_strings_filepath,  self.sha256_labels.iloc[index, 0] + ".txt")

            # Load EMBER features
            ember_features_path_version1 = os.path.join(self.ember_features_filepath_version1,
                                                        self.sha256_labels.iloc[index, 0] + ".npz")
            ember_features_version1 = np.load(ember_features_path_version1, allow_pickle=True)["arr_0"]

            ember_features_path_version2 = os.path.join(self.ember_features_filepath_version2,
                                                        self.sha256_labels.iloc[index, 0] + ".npz")
            ember_features_version2 = np.load(ember_features_path_version2, allow_pickle=True)["arr_0"]

            raw_path = os.path.join(self.raw_executables_filepath, self.sha256_labels.iloc[index, 0])
            if not os.path.isfile(raw_path):
                raw_path += ".exe"
            # Get label. In theory, all of them are malicious
            label = self.sha256_labels.iloc[index, 1]

            return strings_features, hashed_strings_features, all_strings_path, ember_features_version1, ember_features_version2, raw_path, label
        else:
            strings_features = []
            hashed_strings_features = []
            all_strings_features = []
            ember_features_version1 = []
            ember_features_version2 = []
            raw_executables_paths = []
            labels = []

            for idx in index:
                # Load strings features
                strings_features_path = os.path.join(self.strings_features_filepath,
                                                     self.sha256_labels.iloc[idx, 0] + ".npz")
                strings_features_array = np.load(strings_features_path, allow_pickle=True)["arr_0"]

                # Load hashed strings features
                hashed_strings_features_path = os.path.join(self.hashed_strings_features_filepath,
                                                            self.sha256_labels.iloc[idx, 0] + ".npz")
                hashed_strings_features_array = np.load(hashed_strings_features_path, allow_pickle=True)["arr_0"]

                all_strings_path = os.path.join(self.all_strings_filepath, self.sha256_labels.iloc[index, 0] + ".txt")

                # Load EMBER features
                ember_features_path_version1 = os.path.join(self.ember_features_filepath_version1,
                                                            self.sha256_labels.iloc[idx, 0] + ".npz")
                ember_features_array_version1 = np.load(ember_features_path_version1, allow_pickle=True)["arr_0"]

                ember_features_path_version2 = os.path.join(self.ember_features_filepath_version2,
                                                            self.sha256_labels.iloc[idx, 0] + ".npz")
                ember_features_array_version2 = np.load(ember_features_path_version2, allow_pickle=True)["arr_0"]
                raw_path = os.path.join(self.raw_executables_filepath, self.sha256_labels.iloc[idx, 0])
                if not os.path.isfile(raw_path):
                    raw_path += ".exe"
                label = self.sha256_labels.iloc[idx, 1]

                strings_features.append(strings_features_array)
                hashed_strings_features.append(hashed_strings_features_array)
                all_strings_features.append(all_strings_path)
                ember_features_version1.append(ember_features_array_version1)
                ember_features_version2.append(ember_features_array_version2)
                raw_executables_paths.append(raw_path)
                labels.append(label)

            strings_features = np.array(strings_features)
            hashed_strings_features = np.array(hashed_strings_features)
            all_strings_features = np.array(all_strings_features)
            ember_features_version1 = np.array(ember_features_version1)
            ember_features_version2 = np.array(ember_features_version2)
            raw_executables_paths = np.array(raw_executables_paths)
            labels = np.array(labels)

            return strings_features, hashed_strings_features, all_strings_features, ember_features_version1, ember_features_version2, raw_executables_paths, labels







