import torch
import pandas as pd
import os
import numpy as np
import json
import copy
import argparse


class ImportsDataset(torch.utils.data.Dataset):
    def __init__(self, imports_features_filepath: str, hashed_imports_features_filepath: str, imports_filepath: str,
                 ember_features_filepath_version1: str, ember_features_filepath_version2: str, raw_executables_filepath: str, annotations_filepath: str):
        """

        :param imports_features_filepath: Where the import features stored as .npz files are located
        :param hashed_imports_features_filepath: Where the hashed import features stored as .npz files are located
        :param imports_filepath: Where the imports dictionaries stored as .json files are located
        :param ember_features_filepath_version1: Where the EMBER features (version1) stored as .npz files are located
        :param ember_features_filepath_version2: Where the EMBER features (version2) stored as .npz files are located
        :param raw_executables_filepath: Where the raw executables are located
        :param annotations_filepath: Annotations filepath
        """

        self.imports_features_filepath = imports_features_filepath
        self.hashed_imports_features_filepath = hashed_imports_features_filepath
        self.imports_filepath = imports_filepath
        self.ember_features_filepath_version1 = ember_features_filepath_version1
        self.ember_features_filepath_version2 = ember_features_filepath_version2
        self.raw_executables_filepath = raw_executables_filepath
        self.annotations_filepath = annotations_filepath
        self.sha256_labels = pd.read_csv(self.annotations_filepath)


        self.number_of_imports_features_files = len(os.listdir(self.imports_features_filepath))
        self.number_of_hashed_imports_features_files = len(os.listdir(self.hashed_imports_features_filepath))
        self.number_of_imports_files = len(os.listdir(self.imports_filepath))
        self.number_of_ember_features_files_version1 = len(os.listdir(self.ember_features_filepath_version1))
        self.number_of_ember_features_files_version2 = len(os.listdir(self.ember_features_filepath_version2))

        """
        if self.number_of_imports_features_files != self.number_of_hashed_imports_features_files and self.number_of_imports_features_files != self.number_of_imports_files and self.number_of_imports_features_files != self.number_of_ember_features_files_version1 and self.number_of_imports_features_files != self.number_of_ember_features_files_version2:
            raise Exception("The number of files in {}, {}, {}, {} and {} must be the same! It is {}, {}, {}, {} and {}".format(
                self.imports_features_filepath,
                self.hashed_imports_features_filepath,
                self.imports_features_filepath,
                self.ember_features_filepath_version1,
                self.ember_features_filepath_version2,
                self.number_of_imports_features_files,
                self.number_of_hashed_imports_features_files,
                self.number_of_imports_files,
                self.number_of_ember_features_files_version1,
                self.number_of_ember_features_files_version2
            ))
        """

    def __len__(self):
        return len(self.sha256_labels.index) # Remember to create the CSV file with the hashes and labels first

    def __getitem__(self, index):
        if isinstance(index, int):
            # Load import features
            imports_features_path = os.path.join(self.imports_features_filepath, self.sha256_labels.iloc[index, 0]+".npz")
            import_features = np.load(imports_features_path, allow_pickle=True)["arr_0"]

            # Load hashed import features
            hashed_imports_features_path = os.path.join(self.hashed_imports_features_filepath, self.sha256_labels.iloc[index, 0]+".npz")
            hashed_import_features = np.load(hashed_imports_features_path, allow_pickle=True)["arr_0"]

            # Load EMBER features
            ember_features_path_version1 = os.path.join(self.ember_features_filepath_version1, self.sha256_labels.iloc[index, 0]+".npz")
            ember_features_version1 = np.load(ember_features_path_version1, allow_pickle=True)["arr_0"]

            ember_features_path_version2 = os.path.join(self.ember_features_filepath_version2, self.sha256_labels.iloc[index, 0] + ".npz")
            ember_features_version2 = np.load(ember_features_path_version2, allow_pickle=True)["arr_0"]

            raw_path = os.path.join(self.raw_executables_filepath, self.sha256_labels.iloc[index, 0])
            if not os.path.isfile(raw_path):
                raw_path += ".exe"

            label = self.sha256_labels.iloc[index, 1]
            return import_features, hashed_import_features, os.path.join(self.imports_filepath, self.sha256_labels.iloc[index, 0]+".json"), ember_features_version1, ember_features_version2, raw_path, label

        else:
            features = []
            hashed_features = []
            ember_features_version1 = []
            ember_features_version2 = []
            imports_paths = []
            raw_executables_paths = []
            labels = []
            for idx in index:
                # Load import features
                imports_features_path = os.path.join(self.imports_features_filepath, self.sha256_labels.iloc[idx, 0] + ".npz")
                import_features_array = np.load(imports_features_path, allow_pickle=True)["arr_0"]

                # Load hashed import features
                hashed_imports_features_path = os.path.join(self.hashed_imports_features_filepath,
                                                   self.sha256_labels.iloc[idx, 0] + ".npz")
                hashed_import_features_array = np.load(hashed_imports_features_path, allow_pickle=True)["arr_0"]

                # Load EMBER features
                ember_features_path_version1 = os.path.join(self.ember_features_filepath_version1,
                                                   self.sha256_labels.iloc[idx, 0] + ".npz")
                ember_features_array_version1 = np.load(ember_features_path_version1, allow_pickle=True)["arr_0"]

                ember_features_path_version2 = os.path.join(self.ember_features_filepath_version2,
                                                   self.sha256_labels.iloc[idx, 0] + ".npz")
                ember_features_array_version2 = np.load(ember_features_path_version2, allow_pickle=True)["arr_0"]


                # Load imports dictionary
                imports_path = os.path.join(self.imports_filepath, self.sha256_labels.iloc[idx, 0] + ".json")
                raw_path = os.path.join(self.raw_executables_filepath, self.sha256_labels.iloc[idx, 0])
                if not os.path.isfile(raw_path):
                    raw_path += ".exe"

                label = self.sha256_labels.iloc[idx, 1]

                features.append(import_features_array)
                hashed_features.append(hashed_import_features_array)
                ember_features_version1.append(ember_features_array_version1)
                ember_features_version2.append(ember_features_array_version2)
                imports_paths.append(imports_path)
                raw_executables_paths.append(raw_path)
                labels.append(label)
            import_features = np.array(features)
            hashed_import_features = np.array(hashed_features)
            ember_features_version1 = np.array(ember_features_version1)
            ember_features_version2 = np.array(ember_features_version2)
            label = np.array(labels)
            imports_paths = np.array(imports_paths)
            raw_executables_paths = np.array(raw_executables_paths)

            return import_features, hashed_import_features, imports_paths, ember_features_version1, ember_features_version2, raw_executables_paths, label

    def retrieve_imports_dictionaries(self, filepaths):
        imports_dictionaries = []
        for path in filepaths:
            with open(path, "r") as dictionary_file:
                imports_dict = json.load(dictionary_file)
            imports_dictionaries.append(imports_dict)
        return imports_dictionaries

"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MalGAN training')
    parser.add_argument("imports_features_filepath",
                        type=str,
                        help="Filepath to import features")
    parser.add_argument("hashed_imports_features_filepath",
                        type=str,
                        help="Filepath to hashed import features")
    parser.add_argument("ember_features_filepath_version1",
                        type=str,
                        help="Filepath to EMBER features (Version 1)")
    parser.add_argument("ember_features_filepath_version2",
                        type=str,
                        help="Filepath to EMBER features (Version 2)")
    parser.add_argument("imports_filepath",
                        type=str,
                        help="Filepath to import dictionaries")
    parser.add_arguments("raw_filepath",
                         type=str,
                         help="Filepath to the raw executables")
    parser.add_argument("annotations_filepath",
                        type=str,
                        help="Filepath to annotations (training)")
    parser.add_argument('--sample_idxs', nargs="+", type=int, help="Indices")
    args = parser.parse_args()

    pe_dataset = PEImportsDataset(
        args.imports_features_filepath,
        args.hashed_imports_features_filepath,
        args.imports_filepath,
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
    for (d_features, d_hashed_features, d_paths, d_ember_features_version1, d_ember_features_version2, d_raw_paths, d_y) in dataloader:
        print("Import features: ", d_features)
        print("Hashed import features: ", d_hashed_features)
        print("Imports dict: ", d_paths)
        print("EMBER features (Version 1): ", d_ember_features_version1)
        print("EMBER features (Version 2): ", d_ember_features_version2)
        print("Label: ", d_y)

        print(pe_dataset.retrieve_imports_dictionaries(d_paths))
"""
