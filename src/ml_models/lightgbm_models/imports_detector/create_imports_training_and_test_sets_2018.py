import pandas as pd
import json
import sys
import csv
sys.path.append("../../../../")
from src.gan_implementations.utils import load_json
from src.feature_extractors.imports_info_extractor import ImportsInfoExtractor
from collections import OrderedDict
import argparse


training_csv_filepath = "../../../../data/EMBER_2018/training.csv"
testing_csv_filepath = "../../../../data/EMBER_2018/testing.csv"

testing_jsonl_filepaths = [
    "../../../../data/EMBER_2018/test_features.jsonl"
]

training_jsonl_filepaths = [
    "../../../../data/EMBER_2018/train_features_0.jsonl",
    "../../../../data/EMBER_2018/train_features_1.jsonl",
    "../../../../data/EMBER_2018/train_features_2.jsonl",
    "../../../../data/EMBER_2018/train_features_3.jsonl",
    "../../../../data/EMBER_2018/train_features_4.jsonl",
    "../../../../data/EMBER_2018/train_features_5.jsonl"
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create Imports training and test sets (EMBER 2018)')
    parser.add_argument("--vocabulary_mapping",
                        type=str,
                        help="Vocabulary mapping",
                        default=None)
    parser.add_argument("--inverse_vocabulary_mapping",
                        type=str,
                        help="Inverse vocabulary mapping",
                        default=None)
    parser.add_argument("--training_output_filepath",
                        type=str,
                        help="Training output filepath",
                        default=None)
    parser.add_argument("--testing_output_filepath",
                        type=str,
                        help="Testing output filepath",
                        default=None)
    args = parser.parse_args()

    df_train = pd.read_csv(training_csv_filepath)
    df_test = pd.read_csv(testing_csv_filepath)

    vocabulary_mapping = load_json(args.vocabulary_mapping)
    inverse_vocabulary_mapping = load_json(args.inverse_vocabulary_mapping)
    feature_extractor = ImportsInfoExtractor(vocabulary_mapping, inverse_vocabulary_mapping)

    features_dict = OrderedDict({"sha256": None})
    features_dict.update(OrderedDict({"f_{}".format(i): 0.0 for i in range(feature_extractor.dim)}))
    features_dict.update(OrderedDict({"label": None}))

    # Training data
    i = 0
    for training_json_filepath in training_jsonl_filepaths:
        # Get name without .jsonl
        output_filename = args.training_output_filepath + "_{}.csv".format(i)
        with open(output_filename, "w") as output_file:
            fieldnames = features_dict.keys()
            output_file.write(",".join(fieldnames) + "\n")
            with open(training_json_filepath, "r") as input_file:
                for line in input_file:
                    data = json.loads(line)
                    feature_vector = feature_extractor.get_imports_features_from_ember_json(data)
                    label = df_train.loc[df_train["sha256"] == data["sha256"]]["label"]
                    print(i, data["sha256"], int(label), feature_extractor.dim)

                    output_file.write("{},{},{}\n".format(
                        data["sha256"],
                        ",".join([str(x) for x in feature_vector]),
                        int(label)
                    ))
                    i += 1

    features_dict = OrderedDict({"sha256": None})
    features_dict.update(OrderedDict({"f_{}".format(i): 0.0 for i in range(feature_extractor.dim)}))
    features_dict.update(OrderedDict({"label": None}))

    i = 0
    for testing_json_filepath in testing_jsonl_filepaths:
        # Get name without .jsonl
        with open(args.testing_output_filepath, "w") as output_file:
            fieldnames = features_dict.keys()
            output_file.write(",".join(fieldnames) + "\n")

            with open(testing_json_filepath, "r") as input_file:
                for line in input_file:
                    data = json.loads(line)
                    feature_vector = feature_extractor.get_imports_features_from_ember_json(data)
                    label = df_test.loc[df_test["sha256"] == data["sha256"]]["label"]
                    print(i, data["sha256"], int(label), feature_extractor.dim)

                    output_file.write("{},{},{}\n".format(
                        data["sha256"],
                        ",".join([str(x) for x in feature_vector]),
                        int(label)
                    ))
                    i += 1
