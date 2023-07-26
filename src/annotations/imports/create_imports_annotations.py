import sys
import os
sys.path.append("../../../")
from src.pe_modifier import PEModifier
from src.feature_extractors.imports_info_extractor import ImportsInfoExtractor
import pandas as pd
import argparse
import json
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extract features to train Imports GAN')
    parser.add_argument("--imports_features_filepath",
                        type=str,
                        help="Filepath where the imports features are stored",
                        default="../../../npz/BODMAS/imports_features/baseline/benign/")
    parser.add_argument("--annotations_filepath",
                        type=str,
                        help="Filepath where the annotations will be stored",
                        default="data/BODMAS_benign.csv")
    parser.add_argument("--ytrue",
                        type=int,
                        help="Y true",
                        default=0)
    args = parser.parse_args()

    annotations_df = pd.DataFrame()
    hashes = []
    labels = []

    filenames = os.listdir(args.imports_features_filepath)
    for i, filename in enumerate(filenames):
        try:
            print("{}/{}: {}".format(i, len(filenames), os.path.join(args.imports_features_filepath, filename)))
            imports_features = np.load(os.path.join(args.imports_features_filepath, filename), allow_pickle=True)["arr_0"]
            print("Shape: {}; Sum: {}".format(imports_features.shape, np.sum(imports_features)))
            if np.sum(imports_features) > 0: # Only use executables that have imported more than 1 API function
                # Extract imports and hashed import features
                hashes.append(filename.replace(".npz", ""))
                labels.append(args.ytrue)
        except Exception as e:
            print(e)

    annotations_df["sha256"] = hashes
    annotations_df["label"] = labels
    annotations_df.to_csv(args.annotations_filepath, index=False)