#!/usr/bin/env bash

# Create datasets for training and testing (EMBER 2017 and EMBER 2018)
python create_training_and_test_sets_2017.py
python create_training_and_test_sets_2018.py

#Move the data to its correct folder
mv *.csv data/csv/

# Merge training and testing features
python merge_training_features_2017.py
python merge_training_features_2018.py

python remove_unclassified_executables.py data/csv/train_features_2017.csv data/csv/clean_train_features_2017.csv
python remove_unclassified_executables.py data/csv/train_features_2018.csv data/csv/clean_train_features_2018.csv
python remove_unclassified_executables.py data/csv/test_features_2017.csv data/csv/clean_test_features_2017.csv
python remove_unclassified_executables.py data/csv/test_features_2018.csv data/csv/clean_test_features_2018.csv

# Shuffle data
python shuffle_sets.py data/csv/clean_train_features_2017.csv data/csv/train_multimodal_features_2017.csv
python shuffle_sets.py data/csv/clean_test_features_2017.csv data/csv/test_multimodal_features_2017.csv
python shuffle_sets.py data/csv/clean_train_features_2018.csv data/csv/train_multimodal_features_2018.csv
python shuffle_sets.py data/csv/clean_test_features_2018.csv data/csv/test_multimodal_features_2018.csv