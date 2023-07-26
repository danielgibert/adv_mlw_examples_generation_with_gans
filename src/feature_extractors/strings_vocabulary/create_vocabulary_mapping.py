import argparse
import csv
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create strings vocabulary mapping (and the inverse one)')
    parser.add_argument("strings_filepath",
                        type=str,
                        help="Filepath containing the top K strings in the dataset")
    parser.add_argument("vocabulary_mapping_filepath",
                        type=str,
                        help="JSON-like file where the vocabulary mapping will be stored")
    parser.add_argument("inverse_vocabulary_mapping_filepath",
                        type=str,
                        help="JSON-like file where the inverse vocabulary mapping will be stored")
    args = parser.parse_args()

    vocabulary_mapping = {}
    inverse_vocabulary_mapping = {}

    with open(args.strings_filepath, "r") as input_file:
        data = json.load(input_file)
        i = 0
        for line in data:
            vocabulary_mapping[line[0]] = i
            inverse_vocabulary_mapping[i] = line[0]
            i += 1


    with open(args.vocabulary_mapping_filepath, "w") as output_file:
        json.dump(vocabulary_mapping, output_file)

    with open(args.inverse_vocabulary_mapping_filepath, "w") as output_file:
        json.dump(inverse_vocabulary_mapping, output_file)
