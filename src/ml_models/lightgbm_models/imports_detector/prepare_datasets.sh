################################################## TOP 150 Imports #####################################################
: << 'COMMENT'
python create_imports_training_and_test_sets_2017.py --vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/vocabulary_mapping_top150.json --inverse_vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/inverse_vocabulary_mapping_top150.json
python create_imports_training_and_test_sets_2018.py --vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/vocabulary_mapping_top150.json --inverse_vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/inverse_vocabulary_mapping_top150.json

#Move the data to its correct folder
mv *.csv data/csv/

# Merge training and testing features
python merge_training_features_2017.py
python merge_training_features_2018.py

# Remove unlabeled executables
python remove_unclassified_executables.py data/csv/train_features_2017.csv data/csv/clean_train_features_2017.csv
python remove_unclassified_executables.py data/csv/train_features_2018.csv data/csv/clean_train_features_2018.csv
python remove_unclassified_executables.py data/csv/test_features_2017.csv data/csv/clean_test_features_2017.csv
python remove_unclassified_executables.py data/csv/test_features_2018.csv data/csv/clean_test_features_2018.csv

# Shuffle data
python shuffle_sets.py data/csv/clean_train_features_2017.csv data/csv/train_top150_imports_features_2017.csv
python shuffle_sets.py data/csv/clean_test_features_2017.csv data/csv/test_top150_imports_features_2017.csv
python shuffle_sets.py data/csv/clean_train_features_2018.csv data/csv/train_top150_imports_features_2018.csv
python shuffle_sets.py data/csv/clean_test_features_2018.csv data/csv/test_top150_imports_features_2018.csv


################################################## TOP 300 Imports #####################################################
python create_imports_training_and_test_sets_2017.py --vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/vocabulary_mapping_top300.json --inverse_vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/inverse_vocabulary_mapping_top300.json
python create_imports_training_and_test_sets_2018.py --vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/vocabulary_mapping_top300.json --inverse_vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/inverse_vocabulary_mapping_top300.json

#Move the data to its correct folder
mv *.csv data/csv/

# Merge training and testing features
python merge_training_features_2017.py
python merge_training_features_2018.py

# Remove unlabeled executables
python remove_unclassified_executables.py data/csv/train_features_2017.csv data/csv/clean_train_features_2017.csv
python remove_unclassified_executables.py data/csv/train_features_2018.csv data/csv/clean_train_features_2018.csv
python remove_unclassified_executables.py data/csv/test_features_2017.csv data/csv/clean_test_features_2017.csv
python remove_unclassified_executables.py data/csv/test_features_2018.csv data/csv/clean_test_features_2018.csv

# Shuffle data
python shuffle_sets.py data/csv/clean_train_features_2017.csv data/csv/train_top300_imports_features_2017.csv
python shuffle_sets.py data/csv/clean_test_features_2017.csv data/csv/test_top300_imports_features_2017.csv
python shuffle_sets.py data/csv/clean_train_features_2018.csv data/csv/train_top300_imports_features_2018.csv
python shuffle_sets.py data/csv/clean_test_features_2018.csv data/csv/test_top300_imports_features_2018.csv


################################################## TOP 500 Imports #####################################################
python create_imports_training_and_test_sets_2017.py --vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/vocabulary_mapping_top500.json --inverse_vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/inverse_vocabulary_mapping_top500.json
python create_imports_training_and_test_sets_2018.py --vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/vocabulary_mapping_top500.json --inverse_vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/inverse_vocabulary_mapping_top500.json

#Move the data to its correct folder
mv *.csv data/csv/

# Merge training and testing features
python merge_training_features_2017.py
python merge_training_features_2018.py

# Remove unlabeled executables
python remove_unclassified_executables.py data/csv/train_features_2017.csv data/csv/clean_train_features_2017.csv
python remove_unclassified_executables.py data/csv/train_features_2018.csv data/csv/clean_train_features_2018.csv
python remove_unclassified_executables.py data/csv/test_features_2017.csv data/csv/clean_test_features_2017.csv
python remove_unclassified_executables.py data/csv/test_features_2018.csv data/csv/clean_test_features_2018.csv

# Shuffle data
python shuffle_sets.py data/csv/clean_train_features_2017.csv data/csv/train_top500_imports_features_2017.csv
python shuffle_sets.py data/csv/clean_test_features_2017.csv data/csv/test_top500_imports_features_2017.csv
python shuffle_sets.py data/csv/clean_train_features_2018.csv data/csv/train_top500_imports_features_2018.csv
python shuffle_sets.py data/csv/clean_test_features_2018.csv data/csv/test_top500_imports_features_2018.csv


################################################## TOP 1000 Imports ####################################################
python create_imports_training_and_test_sets_2017.py --vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/vocabulary_mapping_top1000.json --inverse_vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/inverse_vocabulary_mapping_top1000.json
python create_imports_training_and_test_sets_2018.py --vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/vocabulary_mapping_top1000.json --inverse_vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/inverse_vocabulary_mapping_top1000.json

#Move the data to its correct folder
mv *.csv data/csv/

# Merge training and testing features
python merge_training_features_2017.py
python merge_training_features_2018.py

# Remove unlabeled executables
python remove_unclassified_executables.py data/csv/train_features_2017.csv data/csv/clean_train_features_2017.csv
python remove_unclassified_executables.py data/csv/train_features_2018.csv data/csv/clean_train_features_2018.csv
python remove_unclassified_executables.py data/csv/test_features_2017.csv data/csv/clean_test_features_2017.csv
python remove_unclassified_executables.py data/csv/test_features_2018.csv data/csv/clean_test_features_2018.csv

# Shuffle data
python shuffle_sets.py data/csv/clean_train_features_2017.csv data/csv/train_top1000_imports_features_2017.csv
python shuffle_sets.py data/csv/clean_test_features_2017.csv data/csv/test_top1000_imports_features_2017.csv
python shuffle_sets.py data/csv/clean_train_features_2018.csv data/csv/train_top1000_imports_features_2018.csv
python shuffle_sets.py data/csv/clean_test_features_2018.csv data/csv/test_top1000_imports_features_2018.csv


################################################## TOP 2000 Imports ####################################################
python create_imports_training_and_test_sets_2017.py --vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/vocabulary_mapping_top2000.json --inverse_vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/inverse_vocabulary_mapping_top2000.json
python create_imports_training_and_test_sets_2018.py --vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/vocabulary_mapping_top2000.json --inverse_vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/inverse_vocabulary_mapping_top2000.json

#Move the data to its correct folder
mv *.csv data/csv/

# Merge training and testing features
python merge_training_features_2017.py
python merge_training_features_2018.py

# Remove unlabeled executables
python remove_unclassified_executables.py data/csv/train_features_2017.csv data/csv/clean_train_features_2017.csv
python remove_unclassified_executables.py data/csv/train_features_2018.csv data/csv/clean_train_features_2018.csv
python remove_unclassified_executables.py data/csv/test_features_2017.csv data/csv/clean_test_features_2017.csv
python remove_unclassified_executables.py data/csv/test_features_2018.csv data/csv/clean_test_features_2018.csv

# Shuffle data
python shuffle_sets.py data/csv/clean_train_features_2017.csv data/csv/train_top2000_imports_features_2017.csv
python shuffle_sets.py data/csv/clean_test_features_2017.csv data/csv/test_top2000_imports_features_2017.csv
python shuffle_sets.py data/csv/clean_train_features_2018.csv data/csv/train_top2000_imports_features_2018.csv
python shuffle_sets.py data/csv/clean_test_features_2018.csv data/csv/test_top2000_imports_features_2018.csv
COMMENT

################################################## TOP 5000 Imports ####################################################
#python create_imports_training_and_test_sets_2017_local.py --vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/vocabulary_mapping_top5000.json --inverse_vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/inverse_vocabulary_mapping_top5000.json --training_output_filepath train_top5000_imports_features_2017 --testing_output_filepath test_top5000_imports_features_2017.csv
#python create_imports_training_and_test_sets_2018_local.py --vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/vocabulary_mapping_top5000.json --inverse_vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/inverse_vocabulary_mapping_top5000.json --training_output_filepath train_top5000_imports_features_2018 --testing_output_filepath test_top5000_imports_features_2018.csv

#Move the data to its correct folder
#mv *.csv data/csv/

# Merge training and testing features
#python merge_training_features.py train_top5000_imports_features_2017 data/csv/train_top5000_imports_features_2017.csv
#python merge_training_features.py train_top5000_imports_features_2018 data/csv/train_top5000_imports_features_2018.csv

# Remove unlabeled executables
#python remove_unclassified_executables.py data/csv/train_top5000_imports_features_2017.csv data/csv/clean_train_top5000_imports_features_2017.csv
#python remove_unclassified_executables.py data/csv/test_top5000_imports_features_2017.csv data/csv/clean_test_top5000_imports_features_2017.csv

#python remove_unclassified_executables.py data/csv/train_top5000_imports_features_2018.csv data/csv/clean_train_top5000_imports_features_2018.csv
#python remove_unclassified_executables.py data/csv/test_top5000_imports_features_2018.csv data/csv/clean_test_top5000_imports_features_2018.csv

# Shuffle data
#python shuffle_sets.py data/csv/clean_train_top5000_imports_features_2017.csv data/csv/final_train_top5000_imports_features_2017.csv
#python shuffle_sets.py data/csv/clean_test_top5000_imports_features_2017.csv data/csv/final_test_top5000_imports_features_2017.csv

#python shuffle_sets.py data/csv/clean_train_top5000_imports_features_2018.csv data/csv/final_train_top5000_imports_features_2018.csv
#python shuffle_sets.py data/csv/clean_test_top5000_imports_features_2018.csv data/csv/final_test_top5000_imports_features_2018.csv


################################################## TOP 10000 Imports ###################################################
python create_imports_training_and_test_sets_2017_local.py --vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/vocabulary_mapping_top10000.json --inverse_vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/inverse_vocabulary_mapping_top10000.json --training_output_filepath train_top10000_imports_features_2017 --testing_output_filepath test_top10000_imports_features_2017.csv
python create_imports_training_and_test_sets_2018_local.py --vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/vocabulary_mapping_top10000.json --inverse_vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/inverse_vocabulary_mapping_top10000.json --training_output_filepath train_top10000_imports_features_2018 --testing_output_filepath test_top10000_imports_features_2018.csv

#Move the data to its correct folder
mv *.csv data/csv/

# Merge training and testing features
python merge_training_features.py train_top10000_imports_features_2017 data/csv/train_top10000_imports_features_2017.csv
python merge_training_features.py train_top10000_imports_features_2018 data/csv/train_top10000_imports_features_2018.csv

# Remove unlabeled executables
python remove_unclassified_executables.py data/csv/train_top10000_imports_features_2017.csv data/csv/clean_train_top10000_imports_features_2017.csv
python remove_unclassified_executables.py data/csv/test_top10000_imports_features_2017.csv data/csv/clean_test_top10000_imports_features_2017.csv

python remove_unclassified_executables.py data/csv/train_top10000_imports_features_2018.csv data/csv/clean_train_top10000_imports_features_2018.csv
python remove_unclassified_executables.py data/csv/test_top10000_imports_features_2018.csv data/csv/clean_test_top10000_imports_features_2018.csv

# Shuffle data
#python shuffle_sets.py data/csv/clean_train_top10000_imports_features_2017.csv data/csv/final_train_top10000_imports_features_2017.csv
#python shuffle_sets.py data/csv/clean_test_top10000_imports_features_2017.csv data/csv/final_test_top10000_imports_features_2017.csv

#python shuffle_sets.py data/csv/clean_train_top10000_imports_features_2018.csv data/csv/final_train_top10000_imports_features_2018.csv
#python shuffle_sets.py data/csv/clean_test_top10000_imports_features_2018.csv data/csv/final_test_top10000_imports_features_2018.csv


################################################## TOP 20000 Imports ###################################################
python create_imports_training_and_test_sets_2017_local.py --vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/vocabulary_mapping_top20000.json --inverse_vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/inverse_vocabulary_mapping_top20000.json --training_output_filepath train_top20000_imports_features_2017 --testing_output_filepath test_top20000_imports_features_2017.csv
python create_imports_training_and_test_sets_2018_local.py --vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/vocabulary_mapping_top20000.json --inverse_vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/inverse_vocabulary_mapping_top20000.json --training_output_filepath train_top20000_imports_features_2018 --testing_output_filepath test_top20000_imports_features_2018.csv

#Move the data to its correct folder
mv *.csv data/csv/

# Merge training and testing features
python merge_training_features.py train_top20000_imports_features_2017 data/csv/train_top20000_imports_features_2017.csv
python merge_training_features.py train_top20000_imports_features_2018 data/csv/train_top20000_imports_features_2018.csv

# Remove unlabeled executables
python remove_unclassified_executables.py data/csv/train_top20000_imports_features_2017.csv data/csv/clean_train_top20000_imports_features_2017.csv
python remove_unclassified_executables.py data/csv/test_top20000_imports_features_2017.csv data/csv/clean_test_top20000_imports_features_2017.csv

python remove_unclassified_executables.py data/csv/train_top20000_imports_features_2018.csv data/csv/clean_train_top20000_imports_features_2018.csv
python remove_unclassified_executables.py data/csv/test_top20000_imports_features_2018.csv data/csv/clean_test_top20000_imports_features_2018.csv

# Shuffle data
#python shuffle_sets.py data/csv/clean_train_top20000_imports_features_2017.csv data/csv/final_train_top20000_imports_features_2017.csv
#python shuffle_sets.py data/csv/clean_test_top20000_imports_features_2017.csv data/csv/final_test_top20000_imports_features_2017.csv

#python shuffle_sets.py data/csv/clean_train_top20000_imports_features_2018.csv data/csv/final_train_top20000_imports_features_2018.csv
#python shuffle_sets.py data/csv/clean_test_top20000_imports_features_2018.csv data/csv/final_test_top20000_imports_features_2018.csv

: << 'COMMENT'
################################################## TOP 50000 Imports ###################################################
python create_imports_training_and_test_sets_2017.py --vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/vocabulary_mapping_top50000.json --inverse_vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/inverse_vocabulary_mapping_top50000.json --training_output_filepath train_top50000_imports_features_2017 --testing_output_filepath test_top50000_imports_features_2017.csv
python create_imports_training_and_test_sets_2018.py --vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/vocabulary_mapping_top50000.json --inverse_vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/inverse_vocabulary_mapping_top50000.json --training_output_filepath train_top50000_imports_features_2018 --testing_output_filepath test_top50000_imports_features_2018.csv

#Move the data to its correct folder
mv *.csv data/csv/

# Merge training and testing features
python merge_training_features.py train_top50000_imports_features_2017 data/csv/train_top50000_imports_features_2017.csv
python merge_training_features.py train_top50000_imports_features_2018 data/csv/train_top50000_imports_features_2018.csv

# Remove unlabeled executables
python remove_unclassified_executables.py data/csv/train_top50000_imports_features_2017.csv data/csv/clean_train_top50000_imports_features_2017.csv
python remove_unclassified_executables.py data/csv/test_top50000_imports_features_2017.csv data/csv/clean_test_top50000_imports_features_2017.csv

python remove_unclassified_executables.py data/csv/train_top50000_imports_features_2018.csv data/csv/clean_train_top50000_imports_features_2018.csv
python remove_unclassified_executables.py data/csv/test_top50000_imports_features_2018.csv data/csv/clean_test_top50000_imports_features_2018.csv

# Shuffle data
python shuffle_sets.py data/csv/clean_train_top50000_imports_features_2017.csv data/csv/final_train_top50000_imports_features_2017.csv
python shuffle_sets.py data/csv/clean_test_top50000_imports_features_2017.csv data/csv/final_test_top50000_imports_features_2017.csv

python shuffle_sets.py data/csv/clean_train_top50000_imports_features_2018.csv data/csv/final_train_top50000_imports_features_2018.csv
python shuffle_sets.py data/csv/clean_test_top50000_imports_features_2018.csv data/csv/final_test_top50000_imports_features_2018.csv


################################################## TOP 100000 Imports ###################################################
python create_imports_training_and_test_sets_2017.py --vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/vocabulary_mapping_top100000.json --inverse_vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/inverse_vocabulary_mapping_top100000.json --training_output_filepath train_top100000_imports_features_2017 --testing_output_filepath test_top100000_imports_features_2017.csv
python create_imports_training_and_test_sets_2018.py --vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/vocabulary_mapping_top100000.json --inverse_vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/inverse_vocabulary_mapping_top100000.json --training_output_filepath train_top100000_imports_features_2018 --testing_output_filepath test_top100000_imports_features_2018.csv

#Move the data to its correct folder
mv *.csv data/csv/

# Merge training and testing features
python merge_training_features.py train_top100000_imports_features_2017 data/csv/train_top100000_imports_features_2017.csv
python merge_training_features.py train_top100000_imports_features_2018 data/csv/train_top100000_imports_features_2018.csv

# Remove unlabeled executables
python remove_unclassified_executables.py data/csv/train_top100000_imports_features_2017.csv data/csv/clean_train_top100000_imports_features_2017.csv
python remove_unclassified_executables.py data/csv/test_top100000_imports_features_2017.csv data/csv/clean_test_top100000_imports_features_2017.csv

python remove_unclassified_executables.py data/csv/train_top100000_imports_features_2018.csv data/csv/clean_train_top100000_imports_features_2018.csv
python remove_unclassified_executables.py data/csv/test_top100000_imports_features_2018.csv data/csv/clean_test_top100000_imports_features_2018.csv

# Shuffle data
python shuffle_sets.py data/csv/clean_train_top100000_imports_features_2017.csv data/csv/final_train_top100000_imports_features_2017.csv
python shuffle_sets.py data/csv/clean_test_top100000_imports_features_2017.csv data/csv/final_test_top100000_imports_features_2017.csv

python shuffle_sets.py data/csv/clean_train_top100000_imports_features_2018.csv data/csv/final_train_top100000_imports_features_2018.csv
python shuffle_sets.py data/csv/clean_test_top100000_imports_features_2018.csv data/csv/final_test_top100000_imports_features_2018.csv


################################################## TOP 200000 Imports ###################################################
python create_imports_training_and_test_sets_2017.py --vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/vocabulary_mapping_top200000.json --inverse_vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/inverse_vocabulary_mapping_top200000.json --training_output_filepath train_top200000_imports_features_2017 --testing_output_filepath test_top200000_imports_features_2017.csv
python create_imports_training_and_test_sets_2018.py --vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/vocabulary_mapping_top200000.json --inverse_vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/inverse_vocabulary_mapping_top200000.json --training_output_filepath train_top200000_imports_features_2018 --testing_output_filepath test_top200000_imports_features_2018.csv

#Move the data to its correct folder
mv *.csv data/csv/

# Merge training and testing features
python merge_training_features.py train_top200000_imports_features_2017 data/csv/train_top200000_imports_features_2017.csv
python merge_training_features.py train_top200000_imports_features_2018 data/csv/train_top200000_imports_features_2018.csv

# Remove unlabeled executables
python remove_unclassified_executables.py data/csv/train_top200000_imports_features_2017.csv data/csv/clean_train_top200000_imports_features_2017.csv
python remove_unclassified_executables.py data/csv/test_top200000_imports_features_2017.csv data/csv/clean_test_top200000_imports_features_2017.csv

python remove_unclassified_executables.py data/csv/train_top200000_imports_features_2018.csv data/csv/clean_train_top200000_imports_features_2018.csv
python remove_unclassified_executables.py data/csv/test_top200000_imports_features_2018.csv data/csv/clean_test_top200000_imports_features_2018.csv

# Shuffle data
python shuffle_sets.py data/csv/clean_train_top200000_imports_features_2017.csv data/csv/final_train_top200000_imports_features_2017.csv
python shuffle_sets.py data/csv/clean_test_top200000_imports_features_2017.csv data/csv/final_test_top200000_imports_features_2017.csv

python shuffle_sets.py data/csv/clean_train_top200000_imports_features_2018.csv data/csv/final_train_top200000_imports_features_2018.csv
python shuffle_sets.py data/csv/clean_test_top200000_imports_features_2018.csv data/csv/final_test_top200000_imports_features_2018.csv


################################################## TOP 500000 Imports ###################################################
python create_imports_training_and_test_sets_2017.py --vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/vocabulary_mapping_top500000.json --inverse_vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/inverse_vocabulary_mapping_top500000.json --training_output_filepath train_top500000_imports_features_2017 --testing_output_filepath test_top500000_imports_features_2017.csv
python create_imports_training_and_test_sets_2018.py --vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/vocabulary_mapping_top500000.json --inverse_vocabulary_mapping ../../../feature_extractors/imports_vocabulary/ember/vocabulary/inverse_vocabulary_mapping_top500000.json --training_output_filepath train_top500000_imports_features_2018 --testing_output_filepath test_top500000_imports_features_2018.csv

#Move the data to its correct folder
mv *.csv data/csv/

# Merge training and testing features
python merge_training_features.py train_top500000_imports_features_2017 data/csv/train_top500000_imports_features_2017.csv
python merge_training_features.py train_top500000_imports_features_2018 data/csv/train_top500000_imports_features_2018.csv

# Remove unlabeled executables
python remove_unclassified_executables.py data/csv/train_top500000_imports_features_2017.csv data/csv/clean_train_top500000_imports_features_2017.csv
python remove_unclassified_executables.py data/csv/test_top500000_imports_features_2017.csv data/csv/clean_test_top500000_imports_features_2017.csv

python remove_unclassified_executables.py data/csv/train_top500000_imports_features_2018.csv data/csv/clean_train_top500000_imports_features_2018.csv
python remove_unclassified_executables.py data/csv/test_top500000_imports_features_2018.csv data/csv/clean_test_top500000_imports_features_2018.csv

# Shuffle data
python shuffle_sets.py data/csv/clean_train_top500000_imports_features_2017.csv data/csv/final_train_top500000_imports_features_2017.csv
python shuffle_sets.py data/csv/clean_test_top500000_imports_features_2017.csv data/csv/final_test_top500000_imports_features_2017.csv

python shuffle_sets.py data/csv/clean_train_top500000_imports_features_2018.csv data/csv/final_train_top500000_imports_features_2018.csv
python shuffle_sets.py data/csv/clean_test_top500000_imports_features_2018.csv data/csv/final_test_top500000_imports_features_2018.csv

COMMENT