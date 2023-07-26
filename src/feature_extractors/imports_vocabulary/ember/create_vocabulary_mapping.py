import argparse
import csv
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate barplot')
    parser.add_argument("functions_filepath",
                        type=str,
                        help="CSV-like file containing the functions invoked")
    parser.add_argument("vocabulary_mapping_filepath",
                        type=str,
                        help="JSON-like file where the vocabulary mapping will be stored")
    parser.add_argument("inverse_vocabulary_mapping_filepath",
                        type=str,
                        help="JSON-like file where the inverse vocabulary mapping will be stored")
    args = parser.parse_args()

    vocabulary_mapping = {}
    inverse_vocabulary_mapping = {}

    i = 0
    with open(args.functions_filepath, "r") as input_file:
        reader = csv.DictReader(input_file, fieldnames=["key", "value"])
        reader.__next__()
        for row in reader:
            vocabulary_mapping[row["key"]] = i
            inverse_vocabulary_mapping[i] = row["key"]
            i += 1

    with open(args.vocabulary_mapping_filepath, "w") as output_file:
        json.dump(vocabulary_mapping, output_file)

    with open(args.inverse_vocabulary_mapping_filepath, "w") as output_file:
        json.dump(inverse_vocabulary_mapping, output_file)


