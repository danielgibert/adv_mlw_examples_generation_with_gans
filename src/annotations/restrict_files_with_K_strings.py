import argparse
import os
import numpy as np
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check files with K imported functions')
    parser.add_argument("original_annotations_filepath",
                        type=str,
                        help="Annotations filepath to check")
    parser.add_argument("resulting_annotations_filepath",
                        type=str,
                        help="Annotations filepath to check")
    parser.add_argument("strings_filepath",
                        type=str,
                        help="Imports filepath, where the .npz files are stored")
    parser.add_argument("num_strings",
                        type=int,
                        help="1")
    args = parser.parse_args()

    pd_annotations = pd.read_csv(args.original_annotations_filepath)
    sha256_hashes = list(pd_annotations["sha256"])
    restricted_sha256_hashes = []
    labels = []
    num_strings_per_sample = []

    count_below_K = 0
    for i, sha256_hash in enumerate(sha256_hashes):
        strings_features_path = os.path.join(args.strings_filepath, sha256_hash+".npz")
        strings_features = np.load(strings_features_path, allow_pickle=True)["arr_0"]
        #print(i, sha256_hash, np.sum(import_features))

        if np.sum(strings_features) >= args.num_strings:
            num_strings_per_sample.append(np.sum(strings_features))
            restricted_sha256_hashes.append(sha256_hash)
            labels.append(0)
    print("Len: {}; Mean: {}".format(len(num_strings_per_sample), sum(num_strings_per_sample)/len(num_strings_per_sample)))

    target_pd_annotations = pd.DataFrame()
    target_pd_annotations["sha256"] = restricted_sha256_hashes
    target_pd_annotations["label"] = labels

    target_pd_annotations.to_csv(args.resulting_annotations_filepath, index=False)
