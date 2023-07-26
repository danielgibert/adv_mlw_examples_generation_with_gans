import pandas as pd
import numpy as np
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split annotations files into training, validation and testing')
    parser.add_argument("malicious_filepath",
                        type=str,
                        help="Malicious CSV filepath")
    parser.add_argument("benign_filepath",
                        type=str,
                        help="Benign CSV filepath")
    parser.add_argument("output_filepath",
                        type=str,
                        help="Output filepath")
    parser.add_argument("--server",
                        type=str,
                        help="local")
    args = parser.parse_args()


    bodmas_malicious_annotations_filepath = args.malicious_filepath
    bodmas_benign_annotations_filepath = args.benign_filepath

    bodmas_malicious_df = pd.read_csv(bodmas_malicious_annotations_filepath)
    bodmas_benign_df = pd.read_csv(bodmas_benign_annotations_filepath)

    bodmas_malicious_train, bodmas_malicious_validation, bodmas_malicious_test = np.split(bodmas_malicious_df.sample(frac=1, random_state=42), [int(.8*len(bodmas_malicious_df)), int(.9*len(bodmas_malicious_df))])
    bodmas_benign_train, bodmas_benign_validation, bodmas_benign_test = np.split(bodmas_benign_df.sample(frac=1, random_state=42), [int(.8*len(bodmas_benign_df)), int(.9*len(bodmas_benign_df))])

    bodmas_benign_train.to_csv(os.path.join(args.output_filepath, "BODMAS_benign_train_{}.csv".format(args.server)), index=False)
    bodmas_benign_validation.to_csv(os.path.join(args.output_filepath, "BODMAS_benign_validation_{}.csv".format(args.server)), index=False)
    bodmas_benign_test.to_csv(os.path.join(args.output_filepath, "BODMAS_benign_test_{}.csv".format(args.server)), index=False)


    bodmas_malicious_train.to_csv(os.path.join(args.output_filepath, "BODMAS_malicious_train_{}.csv".format(args.server)), index=False)
    bodmas_malicious_validation.to_csv(os.path.join(args.output_filepath, "BODMAS_malicious_validation_{}.csv".format(args.server)), index=False)
    bodmas_malicious_test.to_csv(os.path.join(args.output_filepath, "BODMAS_malicious_test_{}.csv".format(args.server)), index=False)