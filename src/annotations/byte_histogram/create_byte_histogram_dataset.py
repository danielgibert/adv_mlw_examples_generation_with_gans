import sys
import os
sys.path.append("../../../")
from src.pe_modifier import PEModifier
from src.feature_extractors.bytes_histogram_extractor import ByteHistogramExtractor
from src.feature_extractors.ember_feature_extractor import EmberFeatureExtractor
import pandas as pd
from numpy import savez_compressed
import numpy as np
from shutil import copyfile
import argparse

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
    parser.add_argument("--raw_npz_filepath",
                        type=str,
                        help="Filepath where the bytez will be stored in .npz format",
                        default="../../../npz/BODMAS/raw_npz/benign/") # "../../../npz/BODMAS/raw_npz/malicious/"
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

            feature_extractor = ByteHistogramExtractor()

            raw_obj = feature_extractor.raw_features(pe_modifier.bytez, pe_modifier.lief_binary)
            histogram_array = feature_extractor.process_raw_features(raw_obj)

            savez_compressed(os.path.join(args.histogram_filepath, '{}.npz'.format(filename)),
                             histogram_array)
            #savez_compressed(os.path.join(args.raw_npz_filepath,'{}.npz'.format(filename)), bytez_int_array)

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
            # Copy executable - No need to copy the executables to a directory. We already have all inside the same directory
            # copyfile(os.path.join(args.executables_filepath, filename), os.path.join(args.raw_filepath, filename))

            hashes.append(filename)
            labels.append(args.ytrue)
        except Exception as e:
            filename_exceptions.append(filename)
            print(e)

    annotations_df["sha256"] = hashes
    annotations_df["label"] = labels
    annotations_df.to_csv(args.annotations_filepath, index=False)

    with open("byte_exceptions.debug", "w") as exceptions_file:
        for filename in filename_exceptions:
            exceptions_file.write("{}\n".format(filename))