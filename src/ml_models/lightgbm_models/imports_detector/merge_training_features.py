import pandas as pd
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create Imports training and test sets (EMBER 2017)')
    parser.add_argument("training_filepaths_basename",
                        type=str,
                        help="Training filepaths")
    parser.add_argument("training_output_filepath",
                        type=str,
                        help="Training output filepath",
                        default=None)
    args = parser.parse_args()

    filenames = os.listdir("data/csv/")

    i = 0
    with open(args.training_output_filepath, "w") as output_file:
        for filename in filenames:
            if args.training_filepaths_basename in filename:
                print(i, filename)
                with open(os.path.join("data/csv/", filename), "r") as input_file:
                    lines = input_file.readlines()
                    if i != 0:
                        lines = lines[1:]
                    for line in lines:
                        output_file.write(line)
                i += 1
