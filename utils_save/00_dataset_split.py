import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold
import numpy as np

def has_common_elements(list1, list2):
    """Use sets to determine whether two lists contain identical elements."""
    return bool(set(list1) & set(list2))

if __name__ == '__main__':
    # # Input file: data_base\01_midim_db\full_assessment\7-3\train.csv
    # train_dir = r'data_base\\01_midim_db\\full_assessment\\6-4\\train.csv'
    # output_train_dir = r'data_base\\00_minus_db1\\full_assessment\\6-4\\train.csv'
    # df_train = pd.read_csv(train_dir)
    # # Input file: data_base\01_midim_db\full_assessment\7-3\test.csv
    # test_dir = r'data_base\\01_midim_db\\full_assessment\\6-4\\test.csv'
    # output_test_dir = r'data_base\\00_minus_db1\\full_assessment\\6-4\\test.csv'
    # df_test = pd.read_csv(test_dir)
    # # Check whether the `motion_filled` field in `train.csv` equals 1. If it equals 1, modify the corresponding `motion` field to -1; otherwise, leave it unchanged.
    # # Perform operations on df_train
    # df_train.loc[df_train['motion_filled'] == 1, 'motion'] = -1
    # # Perform operations on df_test
    # df_test.loc[df_test['motion_filled'] == 1, 'motion'] = -1
    # # Check if the `language_filled` field in `train.csv` equals 1. If it equals 1, modify the corresponding `language` field to -1; otherwise, leave it unchanged.
    # # Perform operations on df_train
    # df_train.loc[df_train['language_filled'] == 1, 'language'] = -1
    # # Perform operations on df_test
    # df_test.loc[df_test['language_filled'] == 1, 'language'] = -1
    # # Save df_train and df_test separately
    # df_train.to_csv(output_train_dir, index=False)
    # df_test.to_csv(output_test_dir, index=False)
    # exit()
    # #############################Select the GCS-coded items from the test set.###############################
    df_test = pd.read_csv(r'data_base\\00_minus_db1\\full_assessment\\6-4\\test.csv')
    # Filter using the isin method
    columns_to_check = ['open_one\'s_eyes', 'motion', 'language']
    # df_gcs_encoding = df_test[df_test[columns_to_check].isin([-1]).any(axis=1)]
    # Determine whether motion_filled or language_filled is True
    mask_motion_filled = df_test['motion_filled'] == 1
    mask_language_filled = df_test['language_filled'] == 1
    df_gcs_encoding = df_test[mask_motion_filled | mask_language_filled]
    df_gcs_encoding.to_csv(r'data_base\\00_minus_db1\\abnormal_assessment\\6_4_test_filled.csv')
    mask_motion_unfilled = df_test['motion_filled'] == 0
    mask_language_unfilled = df_test['language_filled'] == 0
    df_gcs_encoding_unfilled = df_test[mask_motion_unfilled & mask_language_unfilled]
    df_gcs_encoding_unfilled.to_csv(r'data_base\\00_minus_db1\\normal_assessment\\6_4_test_unfilled.csv')
    exit()
