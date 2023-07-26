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

    train_data = pd.read_csv(args.csv_filepath)
    train_data = train_data.sample(frac=1)

    train_data.to_csv(args.output_filepath, index=False)