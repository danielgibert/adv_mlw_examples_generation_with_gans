import lightgbm as lgb
import argparse
import pandas as pd
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LightGBM Model Training Script')
    parser.add_argument("training_filepath",
                        type=str,
                        help="Training CSV file")
    parser.add_argument("validation_filepath",
                        type=str,
                        help="Validation CSV file")
    parser.add_argument("hyperparameters_filepath",
                        type=str,
                        help="Hyperparameters filepath")
    parser.add_argument("output_filepath",
                        type=str,
                        help="Resulting LightGBM model")
    args = parser.parse_args()

    train_data = pd.read_csv(args.training_filepath)
    val_data = pd.read_csv(args.validation_filepath)

    # Shuffle the dataframe
    #train_data = train_data.sample(frac=1)
    #val_data = val_data.sample(frac=1)

    # Get labels and hashes
    train_labels = train_data["label"]
    train_sha256 = train_data["sha256"]
    val_labels = val_data["label"]
    val_sha256 = val_data["sha256"]


    train_data = train_data.drop(labels=["sha256", "label"], axis=1)
    val_data = val_data.drop(labels=["sha256", "label"], axis=1)

    print(train_data.info)

    #for col in train_data.columns:
    #    print(col)
    train_data = lgb.Dataset(train_data, label=train_labels)
    print(train_data)
    val_data = lgb.Dataset(val_data, label=val_labels, reference=train_data)
    print(val_data)
    with open(args.hyperparameters_filepath, "r") as hyperparameters_file:
        params = json.load(hyperparameters_file)

    num_rounds = 1000
    early_stopping_rounds = 10
    bst = lgb.train(params, train_data, num_rounds, valid_sets=[val_data], early_stopping_rounds=5)

    bst.save_model(args.output_filepath, num_iteration=bst.best_iteration)

