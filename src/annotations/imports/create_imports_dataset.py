import sys
import os
sys.path.append("../../../")
from src.pe_modifier import PEModifier
from src.feature_extractors.imports_info_extractor import ImportsInfoExtractor
from src.feature_extractors.ember_feature_extractor import EmberFeatureExtractor
import pandas as pd
from numpy import savez_compressed
import numpy as np
import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract features to train Imports GAN')
    parser.add_argument("--executables_filepath",
                        type=str,
                        help="Filepath where the executables are stored",
                        default="../../../raw/BODMAS/benign/") # "../../../raw/BODMAS/malicious/"
    parser.add_argument("--imports_features_filepath",
                        type=str,
                        help="Filepath where the imports features will be stored in .npz format",
                        default="../../../npz/BODMAS/imports_features/baseline/benign/") # "../../../npz/BODMAS/imports_features/baseline/malicious/"
    parser.add_argument("--hashed_imports_features_filepath",
                        type=str,
                        help="Filepath where the imports features will be stored in .npz format",
                        default="../../../npz/BODMAS/hashed_imports_features/benign/")  # "../../../npz/BODMAS/hashed_imports_features/malicious/"
    parser.add_argument("--ember_v1_filepath",
                        type=str,
                        help="Filepath where the ember features (v1) will be stored in .npz format",
                        default="../../../npz/BODMAS/ember_features/2017/benign/") # "../../../npz/BODMAS/ember_features/2017/malicious/"
    parser.add_argument("--ember_v2_filepath",
                        type=str,
                        help="Filepath where the ember features (v2) will be stored in .npz format",
                        default="../../../npz/BODMAS/ember_features/2018/benign/") # "../../../npz/BODMAS/ember_features/2018/malicious/"
    parser.add_argument("--imports_filepath",
                        type=str,
                        help="Filepath where the imports dictionary is stored in .json format",
                        default="../../../npz/BODMAS/imports/benign/")  # "../../../npz/BODMAS/imports/malicious/"
    parser.add_argument("--raw_npz_filepath",
                        type=str,
                        help="Filepath where the bytez will be stored in .npz format",
                        default="../../../npz/BODMAS/raw_npz/benign/") # "../../../npz/BODMAS/raw_npz/malicious/"
    parser.add_argument("--annotations_filepath",
                        type=str,
                        help="Filepath where the annotations will be stored",
                        default="data/BODMAS_benign.csv")
    parser.add_argument("--vocabulary_mapping_filepath",
                        type=str,
                        help="Vocabulary mapping",
                        default="../../feature_extractors/imports_vocabulary/baseline/vocabulary/vocabulary_mapping.json")
    parser.add_argument("--inverse_vocabulary_mapping_filepath",
                        type=str,
                        help="Inverse vocabulary mapping",
                        default="../../feature_extractors/imports_vocabulary/baseline/vocabulary/inverse_vocabulary_mapping.json")
    parser.add_argument("--ytrue",
                        type=int,
                        help="Y true",
                        default=0)
    args = parser.parse_args()

    with open(args.vocabulary_mapping_filepath, "r") as input_file:
        vocabulary_mapping = json.load(input_file)

    with open(args.inverse_vocabulary_mapping_filepath, "r") as input_file:
        inverse_vocabulary_mapping = json.load(input_file)

    annotations_df = pd.DataFrame()
    hashes = []
    labels = []

    filename_exceptions = []
    filenames = os.listdir(args.executables_filepath)
    for i, filename in enumerate(filenames):
        try:
            print("{}/{}: {}".format(i, len(filenames), os.path.join(args.executables_filepath, filename)))

            # Extract imports and hashed import features
            pe_modifier = PEModifier(os.path.join(args.executables_filepath, filename))
            bytez_int_array = np.array(pe_modifier.bytez_int_list, dtype=np.int32)

            feature_extractor = ImportsInfoExtractor(vocabulary_mapping, inverse_vocabulary_mapping)

            raw_obj = feature_extractor.raw_features(pe_modifier.bytez, pe_modifier.lief_binary)

            # if sum(list(raw_obj[feature_extractor.name].values())) > 0: # Only use executables that have imported more than 1 API function
            imports_array = feature_extractor.process_raw_features(raw_obj)
            hashed_imports_array = feature_extractor.apply_hashing_trick(raw_obj["hashed_imports"])

            savez_compressed(os.path.join(args.imports_features_filepath, '{}.npz'.format(filename)),
                             imports_array)
            savez_compressed(os.path.join(args.hashed_imports_features_filepath, '{}.npz'.format(filename)),
                             hashed_imports_array)

            with open(os.path.join(args.imports_filepath, '{}.json'.format(filename)), "w") as imports_file:
                json.dump(raw_obj["hashed_imports"], imports_file)

            # Extract EMBER features 2017
            ember_extractor_v1 = EmberFeatureExtractor(feature_version=1)
            ember_features_vector_v1 = ember_extractor_v1.feature_vector(pe_modifier.bytez)
            savez_compressed(os.path.join(args.ember_v1_filepath, '{}.npz'.format(filename)),
                             ember_features_vector_v1)

            # Extract EMBER features 2018
            ember_extractor_v2 = EmberFeatureExtractor(feature_version=2)
            ember_features_vector_v2 = ember_extractor_v2.feature_vector(pe_modifier.bytez)
            savez_compressed(os.path.join(args.ember_v2_filepath, '{}.npz'.format(filename)),
                             ember_features_vector_v2)

            # Copy executable
            #savez_compressed(os.path.join(args.raw_npz_filepath,'{}.npz'.format(filename)), bytez_int_array)

            hashes.append(filename)
            labels.append(args.ytrue)
        except Exception as e:
            filename_exceptions.append(filename)
            print(e)

    annotations_df["sha256"] = hashes
    annotations_df["label"] = labels
    annotations_df.to_csv(args.annotations_filepath, index=False)

    with open("imports_exceptions.debug", "w") as exceptions_file:
        for filename in filename_exceptions:
            exceptions_file.write("{}\n".format(filename))

