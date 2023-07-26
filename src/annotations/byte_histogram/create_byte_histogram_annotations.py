import sys
import os
sys.path.append("../../../")
from src.pe_modifier import PEModifier
import pandas as pd
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract features to train Byte Histogram GAN')
    parser.add_argument("--executables_filepath",
                        type=str,
                        help="Filepath where the benign executables are stored",
                        default="../../../data/EMBER/2018/benign/")
    parser.add_argument("--annotations_filepath",
                        type=str,
                        help="Filepath where the annotations will be stored",
                        default="data/benign.csv")
    parser.add_argument("--ytrue",
                        type=int,
                        help="Y true",
                        default=0)
    args = parser.parse_args()


    annotations_df = pd.DataFrame()
    hashes = []
    labels = []

    filenames = os.listdir(args.executables_filepath)
    for i, filename in enumerate(filenames):
        try:
            print(i, os.path.join(args.executables_filepath, filename))
            if os.path.getsize(os.path.join(args.executables_filepath, filename)) < 1048576:
                # Extract imports and hashed import features
                pe_modifier = PEModifier(os.path.join(args.executables_filepath, filename))
                hashes.append(pe_modifier.sha256)
                labels.append(args.ytrue)
        except Exception as e:
            print(e)

    annotations_df["sha256"] = hashes
    annotations_df["label"] = labels
    annotations_df.to_csv(args.annotations_filepath, index=False)