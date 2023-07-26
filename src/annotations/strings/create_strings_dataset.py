import argparse
import gc

import pandas as pd
import os
import numpy as np
import sys
from numpy import savez_compressed
import json
sys.path.append("../../../")
from src.pe_modifier import PEModifier
from src.feature_extractors.ember_feature_extractor import EmberFeatureExtractor
from src.feature_extractors.strings_statistics_extractor import StringsStatisticsExtractor
from src.feature_extractors.strings_extractor import StringsExtractor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract features to train Strings GAN')
    parser.add_argument("--executables_filepath",
                        type=str,
                        help="Filepath where the executables are stored",
                        default="../../../raw/BODMAS/benign/")
    parser.add_argument("--strings_filepath",
                        type=str,
                        help="Filepath where the strings features will be stored in .npz format",
                        default="../../../npz/BODMAS/strings_features/benign/")
    parser.add_argument("--hashed_strings_filepath",
                        type=str,
                        help="Filepath where the strings features will be stored in .npz format",
                        default="../../../npz/BODMAS/hashed_strings_features/benign/")
    parser.add_argument("--allstrings_filepath",
                        type=str,
                        help="Filepath where the strings features will be stored in .npz format",
                        default="../../../npz/BODMAS/allstrings/benign/")
    parser.add_argument("--ember_v1_filepath",
                        type=str,
                        help="Filepath where the ember features (v1) will be stored in .npz format",
                        default="../../../npz/BODMAS/ember_features/2017/benign/")  # "../../../npz/BODMAS/ember_features/2017/malicious/"
    parser.add_argument("--ember_v2_filepath",
                        type=str,
                        help="Filepath where the ember features (v2) will be stored in .npz format",
                        default="../../../npz/BODMAS/ember_features/2018/benign/")  # "../../../npz/BODMAS/ember_features/2018/malicious/"
    parser.add_argument("--raw_npz_filepath",
                        type=str,
                        help="Filepath where the bytez will be stored in .npz format",
                        default="../../../npz/BODMAS/raw_npz/benign/") # "../../../npz/BODMAS/raw_npz/malicious/"
    parser.add_argument("--vocabulary_mapping_filepath",
                        type=str,
                        help="Vocabulary mapping",
                        default="../../feature_extractors/strings_vocabulary/vocabulary_mapping_top20000.json")
    parser.add_argument("--inverse_vocabulary_mapping_filepath",
                        type=str,
                        help="Inverse vocabulary mapping",
                        default="../../feature_extractors/strings_vocabulary/inverse_vocabulary_mapping_top20000.json")
    parser.add_argument("--annotations_filepath",
                        type=str,
                        help="Filepath where the annotations will be stored",
                        default="data/BODMAS_benign.csv")
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
            print(i, os.path.join(args.executables_filepath, filename))
            pe_modifier = PEModifier(os.path.join(args.executables_filepath, filename))

            # Extract strings features
            print("Extract String features")
            strings_feature_extractor = StringsExtractor(vocabulary_mapping=vocabulary_mapping, inverse_vocabulary_mapping=inverse_vocabulary_mapping)
            raw_obj = strings_feature_extractor.raw_features(pe_modifier.bytez, pe_modifier.lief_binary)
            strings_array = strings_feature_extractor.process_raw_features(raw_obj)
            print(np.sum(strings_array))
            savez_compressed(os.path.join(args.strings_filepath, '{}.npz'.format(filename)),
                             strings_array)

            # Extract allstrings and store in a npz file
            print("Extract AllStrings features")
            strings_feature_extractor.save_all_strings(
                strings_feature_extractor.get_allstrings(),
                os.path.join(args.allstrings_filepath, '{}.txt'.format(filename))
            )

            #allstrings_array = np.array(strings_feature_extractor.get_allstrings(), dtype=np.str)
            #savez_compressed(os.path.join(args.allstrings_filepath, '{}.txt'.format(filename)),
            #                 allstrings_array)

            # Extract hashed strings features
            print("Extract Hashed Strings features")
            hashed_strings_extractor = StringsStatisticsExtractor()
            raw_obj = hashed_strings_extractor.raw_features(pe_modifier.bytez, pe_modifier.lief_binary)
            hashed_strings_array = hashed_strings_extractor.process_raw_features(raw_obj)
            savez_compressed(os.path.join(args.hashed_strings_filepath, '{}.npz'.format(filename)),
                             hashed_strings_array)

            del raw_obj, strings_feature_extractor, hashed_strings_extractor, hashed_strings_array, strings_array
            gc.collect()
            """
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
            """
            # Npz bytes
            #bytez_int_array = np.array(pe_modifier.bytez_int_list, dtype=np.int32)
            #savez_compressed(os.path.join(args.raw_npz_filepath,'{}.npz'.format(filename)), bytez_int_array)

            # Hashes and labels
            hashes.append(filename)
            labels.append(args.ytrue)
        except Exception as e:
            filename_exceptions.append(filename)
            print(e)

    annotations_df["sha256"] = hashes
    annotations_df["label"] = labels
    annotations_df.to_csv(args.annotations_filepath, index=False)

    with open("strings_exceptions.debug", "w") as exceptions_file:
        for filename in filename_exceptions:
            exceptions_file.write("{}\n".format(filename))
