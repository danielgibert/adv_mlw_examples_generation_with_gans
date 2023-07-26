import argparse
import pandas as pd
import lightgbm as lgb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Shuffle CSV file rows')
    parser.add_argument("csv_filepath",
                        type=str,
                        help="Input CSV file")
    parser.add_argument("output_filepath",
                        type=str,
                        help="Output CSV file")
    args = parser.parse_args()

    with open(args.output_filepath, "w") as output_file:
        with open(args.csv_filepath, "r") as input_file:
            lines = input_file.readlines()
            output_file.write(lines[0])
            for i, line in enumerate(lines[1:]):
                tokens = line.strip().split(",")
                label = int(tokens[-1])
                if label != -1.0:
                    print(i, label)
                    output_file.write(line)

    #train_data = pd.read_csv(args.csv_filepath)
    #train_data = train_data[train_data.label != -1]

    #train_data.to_csv(args.output_filepath, index=False)