import torch
import pandas as pd
import os
import numpy as np
import json
import copy
import argparse


class ByteHistogramDataset(torch.utils.data.Dataset):
    def __init__(self, histogram_features_filepath: str, ember_features_filepath_version1: str,
                 ember_features_filepath_version2: str, raw_executables_filepath: str,
                 raw_npz_executables_filepath: str, annotations_filepath: str):
        """

        :param histogram_features_filepath: Where the byte histogram features stored as .npz files are located
        :param ember_features_filepath_version1: Where the EMBER features (version1) stored as .npz files are located
        :param ember_features_filepath_version2: Where the EMBER features (version2) stored as .npz files are located
        :param raw_executables_filepath: Where the raw executables are located
        :param annotations_filepath: Annotations filepath
        """
        self.histogram_features_filepath = histogram_features_filepath
        self.ember_features_filepath_version1 = ember_features_filepath_version1
        self.ember_features_filepath_version2 = ember_features_filepath_version2
        self.raw_executables_filepath = raw_executables_filepath
        self.raw_npz_executables_filepath = raw_npz_executables_filepath
        self.annotations_filepath = annotations_filepath
        self.sha256_labels = pd.read_csv(self.annotations_filepath)


        self.number_of_histogram_features_files = len(os.listdir(self.histogram_features_filepath))
        self.number_of_ember_features_files_version1 = len(os.listdir(self.ember_features_filepath_version1))
        self.number_of_ember_features_files_version2 = len(os.listdir(self.ember_features_filepath_version2))

        """
        if self.number_of_histogram_features_files != self.number_of_ember_features_files_version1 and self.number_of_histogram_features_files != self.number_of_ember_features_files_version2:
            raise Exception("The number of files in {}, {} and {} must be the same! It is {}, {} and {}".format(
                self.histogram_features_filepath,
                self.ember_features_filepath_version1,
                self.ember_features_filepath_version2,
                self.number_of_histogram_features_files,
                self.number_of_ember_features_files_version1,
                self.number_of_ember_features_files_version2
            ))
        """


    def __len__(self):
        return len(self.sha256_labels.index) # Remember to create the CSV file with the hashes and labels first

    def __getitem__(self, index):
        if isinstance(index, int):
            # Load byte histogram features
            histogram_features_path = os.path.join(self.histogram_features_filepath, self.sha256_labels.iloc[index, 0]+".npz")
            histogram_features = np.load(histogram_features_path, allow_pickle=True)["arr_0"]

            # Load EMBER features
            ember_features_path_version1 = os.path.join(self.ember_features_filepath_version1, self.sha256_labels.iloc[index, 0]+".npz")
            ember_features_version1 = np.load(ember_features_path_version1, allow_pickle=True)["arr_0"]

            ember_features_path_version2 = os.path.join(self.ember_features_filepath_version2, self.sha256_labels.iloc[index, 0] + ".npz")
            ember_features_version2 = np.load(ember_features_path_version2, allow_pickle=True)["arr_0"]

            # Get label. In theory, all of them are malicious
            raw_path = os.path.join(
                self.raw_executables_filepath, self.sha256_labels.iloc[index, 0])
            if not os.path.isfile(raw_path):
                raw_path += ".exe"
            label = self.sha256_labels.iloc[index, 1]
            return histogram_features, ember_features_version1, ember_features_version2, raw_path, os.path.join(
                self.raw_npz_executables_filepath, self.sha256_labels.iloc[index, 0]),label

        else:
            features = []
            ember_features_version1 = []
            ember_features_version2 = []
            raw_executables_paths = []
            raw_npz_executables_paths = []
            labels = []
            for idx in index:
                # Load byte histogram features
                histogram_features_path = os.path.join(self.histogram_features_filepath, self.sha256_labels.iloc[idx, 0] + ".npz")
                histogram_features_array = np.load(histogram_features_path, allow_pickle=True)["arr_0"]

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
                raw_npz_path = os.path.join(self.raw_npz_executables_filepath, self.sha256_labels.iloc[idx, 0])
                # Get label. In theory, all of them are malicious
                label = self.sha256_labels.iloc[idx, 1]

                features.append(histogram_features_array)
                ember_features_version1.append(ember_features_array_version1)
                ember_features_version2.append(ember_features_array_version2)
                raw_executables_paths.append(raw_path)
                raw_npz_executables_paths.append(raw_npz_path)
                labels.append(label)
            import_features = np.array(features)
            ember_features_version1 = np.array(ember_features_version1)
            ember_features_version2 = np.array(ember_features_version2)
            label = np.array(labels)
            raw_executables_paths = np.array(raw_executables_paths)

            return import_features, ember_features_version1, ember_features_version2, raw_executables_paths, raw_npz_executables_paths, label

"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PE dataset (Byte histogram features)')
    parser.add_argument("histogram_features_filepath",
                        type=str,
                        help="Filepath to import features")
    parser.add_argument("ember_features_filepath_version1",
                        type=str,
                        help="Filepath to EMBER features (Version 1)")
    parser.add_argument("ember_features_filepath_version2",
                        type=str,
                        help="Filepath to EMBER features (Version 2)")
    parser.add_argument("raw_filepath",
                         type=str,
                         help="Filepath to the raw executables")
    parser.add_argument("annotations_filepath",
                        type=str,
                        help="Filepath to annotations (training)")
    parser.add_argument('--sample_idxs', nargs="+", type=int, help="Indices")
    args = parser.parse_args()

    pe_dataset = PEDataset(
        args.histogram_features_filepath,
        args.ember_features_filepath_version1,
        args.ember_features_filepath_version2,
        args.raw_filepath,
        args.annotations_filepath
    )
    print("Indices: {}".format(args.sample_idxs))
    print(pe_dataset[args.sample_idxs])

    from torch.utils.data import DataLoader
    dataloader = DataLoader(pe_dataset, batch_size=4,
                                      shuffle=True, drop_last=True)
    for (d_features, d_ember_features_version1, d_ember_features_version2, d_raw_paths, d_y) in dataloader:
        print("Histogram features: ", d_features)
        print("EMBER features (Version 1): ", d_ember_features_version1)
        print("EMBER features (Version 2): ", d_ember_features_version2)
        print("Executable filepaths: ", d_raw_paths)
        print("Label: ", d_y)
"""

