import gc
import sys
import os
sys.path.append("../../../")
from src.pe_modifier import PEModifier
from src.feature_extractors.bytes_histogram_extractor import ByteHistogramExtractor
from src.feature_extractors.imports_info_extractor import ImportsInfoExtractor
from src.feature_extractors.strings_extractor import StringsExtractor
from src.feature_extractors.strings_statistics_extractor import StringsStatisticsExtractor
from src.feature_extractors.ember_feature_extractor import EmberFeatureExtractor
import pandas as pd
from numpy import savez_compressed
import numpy as np
from shutil import copyfile
import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract features to train Byte Histogram GAN')
    parser.add_argument("--executables_filepath",
                        type=str,
                        help="Filepath where the executables are stored",
                        default="../../../raw/BODMAS/benign/") # "../../../raw/BODMAS/malicious/"
    parser.add_argument("--histogram_filepath",
                        type=str,
                        help="Filepath where the histogram features will be stored in .npz format",
                        default="../../../npz/BODMAS/histogram_features/benign/") # "../../../npz/BODMAS/histogram_features/malicious/"
    parser.add_argument("--ember_v1_filepath",
                        type=str,
                        help="Filepath where the ember features (v1) will be stored in .npz format",
                        default="../../../npz/BODMAS/ember_features/2017/benign/") # "../../../npz/BODMAS/ember_features/2017/malicious/"
    parser.add_argument("--ember_v2_filepath",
                        type=str,
                        help="Filepath where the ember features (v2) will be stored in .npz format",
                        default="../../../npz/BODMAS/ember_features/2018/benign/") # "../../../npz/BODMAS/ember_features/2018/malicious/"
    parser.add_argument("--imports_features_filepath",
                        type=str,
                        help="Filepath where the imports features will be stored in .npz format",
                        default="../../../npz/BODMAS/imports_features/baseline/benign/")  # "../../../npz/BODMAS/imports_features/baseline/malicious/"
    parser.add_argument("--hashed_imports_features_filepath",
                        type=str,
                        help="Filepath where the imports features will be stored in .npz format",
                        default="../../../npz/BODMAS/hashed_imports_features/benign/")  # "../../../npz/BODMAS/hashed_imports_features/malicious/"
    parser.add_argument("--imports_filepath",
                        type=str,
                        help="Filepath where the imports dictionary is stored in .json format",
                        default="../../../npz/BODMAS/imports/benign/")  # "../../../npz/BODMAS/imports/malicious/"
    parser.add_argument("--imports_vocabulary_mapping_filepath",
                        type=str,
                        help="Vocabulary mapping",
                        default="../../feature_extractors/imports_vocabulary/baseline/vocabulary/vocabulary_mapping.json")
    parser.add_argument("--imports_inverse_vocabulary_mapping_filepath",
                        type=str,
                        help="Inverse vocabulary mapping",
                        default="../../feature_extractors/imports_vocabulary/baseline/vocabulary/inverse_vocabulary_mapping.json")
    parser.add_argument("--strings_vocabulary_mapping_filepath",
                        type=str,
                        help="Vocabulary mapping",
                        default="../../feature_extractors/strings_vocabulary/vocabulary_mapping_top20000.json")
    parser.add_argument("--strings_inverse_vocabulary_mapping_filepath",
                        type=str,
                        help="Inverse vocabulary mapping",
                        default="../../feature_extractors/strings_vocabulary/inverse_vocabulary_mapping_top20000.json")
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
    parser.add_argument("--annotations_filepath",
                        type=str,
                        help="Filepath where the annotations will be stored",
                        default="data/BODMAS_benign.csv")
    parser.add_argument("--ytrue",
                        type=int,
                        help="Y true",
                        default=0)
    args = parser.parse_args()

    try:
        os.makedirs(args.histogram_filepath)
    except FileExistsError as e:
        pass
    try:
        os.makedirs(args.ember_v1_filepath)
    except FileExistsError as e:
        pass
    try:
        os.makedirs(args.ember_v2_filepath)
    except FileExistsError as e:
        pass
    try:
        os.makedirs(args.imports_features_filepath)
    except FileExistsError as e:
        pass
    try:
        os.makedirs(args.hashed_imports_features_filepath)
    except FileExistsError as e:
        pass
    try:
        os.makedirs(args.imports_filepath)
    except FileExistsError as e:
        pass
    try:
        os.makedirs(args.strings_filepath)
    except FileExistsError as e:
        pass
    try:
        os.makedirs(args.hashed_strings_filepath)
    except FileExistsError as e:
        pass
    try:
        os.makedirs(args.allstrings_filepath)
    except FileExistsError as e:
        pass

    annotations_df = pd.DataFrame()
    hashes = []
    labels = []

    with open(args.imports_vocabulary_mapping_filepath, "r") as input_file:
        imports_vocabulary_mapping = json.load(input_file)

    with open(args.imports_inverse_vocabulary_mapping_filepath, "r") as input_file:
        imports_inverse_vocabulary_mapping = json.load(input_file)

    with open(args.strings_vocabulary_mapping_filepath, "r") as input_file:
        strings_vocabulary_mapping = json.load(input_file)

    with open(args.strings_inverse_vocabulary_mapping_filepath, "r") as input_file:
        strings_inverse_vocabulary_mapping = json.load(input_file)

    filename_exceptions = []
    filenames = os.listdir(args.executables_filepath)
    for i, filename in enumerate(filenames):
        try:
            print(i, os.path.join(args.executables_filepath, filename))
            #if os.path.getsize(os.path.join(args.executables_filepath, filename)) < 1048576:

            # Extract imports and hashed import features
            pe_modifier = PEModifier(os.path.join(args.executables_filepath, filename))
            bytez_int_array = np.array(pe_modifier.bytez_int_list, dtype=np.int32)
            print(bytez_int_array.shape)

            byte_histogram_feature_extractor = ByteHistogramExtractor()

            raw_obj = byte_histogram_feature_extractor.raw_features(pe_modifier.bytez, pe_modifier.lief_binary)
            histogram_array = byte_histogram_feature_extractor.process_raw_features(raw_obj)

            savez_compressed(os.path.join(args.histogram_filepath, '{}.npz'.format(filename)),
                             histogram_array)
            #savez_compressed(os.path.join(args.raw_npz_filepath,'{}.npz'.format(filename)), bytez_int_array)
            del raw_obj, byte_histogram_feature_extractor, histogram_array
            gc.collect()

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
            del ember_extractor_v1, ember_extractor_v2, ember_features_vector_v1, ember_features_vector_v2
            gc.collect()

            # Copy executable - No need to copy the executables to a directory. We already have all inside the same directory
            # copyfile(os.path.join(args.executables_filepath, filename), os.path.join(args.raw_filepath, filename))

            # Imports features
            imports_feature_extractor = ImportsInfoExtractor(imports_vocabulary_mapping, imports_inverse_vocabulary_mapping)
            raw_obj = imports_feature_extractor.raw_features(pe_modifier.bytez, pe_modifier.lief_binary)
            imports_array = imports_feature_extractor.process_raw_features(raw_obj)
            hashed_imports_array = imports_feature_extractor.apply_hashing_trick(raw_obj["hashed_imports"])

            savez_compressed(os.path.join(args.imports_features_filepath, '{}.npz'.format(filename)),
                             imports_array)
            savez_compressed(os.path.join(args.hashed_imports_features_filepath, '{}.npz'.format(filename)),
                             hashed_imports_array)

            with open(os.path.join(args.imports_filepath, '{}.json'.format(filename)), "w") as imports_file:
                json.dump(raw_obj["hashed_imports"], imports_file)

            del raw_obj, imports_feature_extractor, imports_array, hashed_imports_array
            gc.collect()

            """
            # Extract strings features
            strings_feature_extractor = StringsExtractor(vocabulary_mapping=strings_vocabulary_mapping,
                                                         inverse_vocabulary_mapping=strings_inverse_vocabulary_mapping)
            raw_obj = strings_feature_extractor.raw_features(pe_modifier.bytez, pe_modifier.lief_binary)
            strings_array = strings_feature_extractor.process_raw_features(raw_obj)
            savez_compressed(os.path.join(args.strings_filepath, '{}.npz'.format(filename)),
                             strings_array)

            # Extract allstrings and store in a npz file
            strings_feature_extractor.save_all_strings(
                strings_feature_extractor.get_allstrings(),
                os.path.join(args.allstrings_filepath, '{}.txt'.format(filename))
            )

            # allstrings_array = np.array(strings_feature_extractor.get_allstrings(), dtype=np.str)
            # savez_compressed(os.path.join(args.allstrings_filepath, '{}.txt'.format(filename)),
            #                 allstrings_array)

            # Extract hashed strings features
            hashed_strings_extractor = StringsStatisticsExtractor()
            raw_obj = hashed_strings_extractor.raw_features(pe_modifier.bytez, pe_modifier.lief_binary)
            hashed_strings_array = hashed_strings_extractor.process_raw_features(raw_obj)
            savez_compressed(os.path.join(args.hashed_strings_filepath, '{}.npz'.format(filename)),
                             hashed_strings_array)
            
            del strings_feature_extractor, raw_obj, strings_array, hashed_strings_extractor, hashed_strings_array
            gc.collect()                 
            """

            hashes.append(filename)
            labels.append(args.ytrue)
        except Exception as e:
            filename_exceptions.append(filename)
            print(e)

    annotations_df["sha256"] = hashes
    annotations_df["label"] = labels
    annotations_df.to_csv(args.annotations_filepath, index=False)

    with open("exceptions.debug", "w") as exceptions_file:
        for filename in filename_exceptions:
            exceptions_file.write("{}\n".format(filename))