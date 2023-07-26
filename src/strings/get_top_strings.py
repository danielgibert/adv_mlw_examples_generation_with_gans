import argparse
import json
import os
import gc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get top K strings')
    parser.add_argument("strings_filepath",
                        type=str,
                        help="Filepath of the executable")
    parser.add_argument("output_filepath",
                        type=str,
                        help="Output file containing the top K strings found")
    parser.add_argument("--K",
                        type=int,
                        default=20000,
                        help="Default number of strings to consider")
    args = parser.parse_args()

    strings_dictionary = {}

    filenames = os.listdir(args.strings_filepath)
    for i, filename in enumerate(filenames):
        with open(os.path.join(args.strings_filepath, filename), "r") as input_file:
            strings_ordered_list = json.load(input_file)
        strings_ordered_list = strings_ordered_list[:args.K]
        for string in strings_ordered_list:
            try:
                strings_dictionary[string[0]] += string[1]
            except KeyError:
                strings_dictionary[string[0]] = string[1]
        del strings_ordered_list
        gc.collect()

    sorted_strings = sorted(strings_dictionary.items(), key=lambda x: x[1], reverse=True)
    with open(args.output_filepath, "w") as output_file:
        json.dump(sorted_strings[:args.K], output_file)