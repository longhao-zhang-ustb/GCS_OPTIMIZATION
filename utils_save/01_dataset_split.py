import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold
import numpy as np

def has_common_elements(list1, list2):
    """Use sets to determine whether two lists contain identical elements."""
    return bool(set(list1) & set(list2))

if __name__ == '__main__':
    # data_dir = r'zigong_data\\20251227_final_processed_data.csv'
    # output_dir = r'data_base\\01_midim_db\\full_assessment\\6-4\\'
    # test_size = 0.4
    # df = pd.read_csv(data_dir)
    # X = df
    # y = df['consciousness']
    # target_col = 'consciousness'
    # # Divide the training and test sets based on patient IDs, ensuring that data from the same patient does not appear in both the training and test sets.
    # PATIENT_ID and INP_NO maintain a one-to-one correspondence and can be used interchangeably as the grouping basis.
    # # 7:3 ==> (1241, 532), 6:4 ==> (1063, 710)
    # unique_ids = df['INP_NO'].unique()
    # # Retrieve the arousal level labels corresponding to different patient IDs
    # id_to_consciousness = df.groupby('INP_NO')['consciousness'].first().to_dict()
    # # Set the stratify parameter to the arousal level label corresponding to the patient ID.
    # train_ids, test_ids = train_test_split(
    #     unique_ids,
    #     random_state=42,
    #     stratify=[id_to_consciousness[i] for i in unique_ids],
    #     test_size=test_size
    # )
    # print(f'Number of unique patients in training set: {len(train_ids)}')
    # print(f'Number of unique patients in testing set: {len(test_ids)}')
    # # # Based on the assigned training set and test set IDs, retrieve the corresponding training set and test set.
    # train_mask = df['INP_NO'].isin(train_ids)
    # test_mask = df['INP_NO'].isin(test_ids)
    # X_train = X[train_mask]
    # X_test = X[test_mask]
    # y_train = y[train_mask]
    # y_test = y[test_mask]
    # print(f'Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}')
    # ########################################################################
    # # Padding is performed only on the median values of motion and language calculated for the training set.
    # # Maintain two table variables: the first records entries where motion is not -1, and the second records entries where language is not -1.
    # X_train_motion_not_minus1 = X_train[X_train['motion'] != -1]
    # X_train_language_not_minus1 = X_train[X_train['language'] != -1]
    # print(X_train_motion_not_minus1, X_train_language_not_minus1)
    # # Calculate the median of the corresponding motion under different consciousness states and print the results.
    # motion_median = X_train_motion_not_minus1.groupby('consciousness')['motion'].median()
    # print("Under different states of consciousness, the median of motion is:")
    # print(motion_median)
    # # Calculate the median of the corresponding language under different consciousness states and print it.
    # language_median = X_train_language_not_minus1.groupby('consciousness')['language'].median()
    # print("Under different states of consciousness, the median of language is:")
    # print(language_median)
    # # Find records in df where motion is -1. For each such record, fill in the previously calculated median motion value (motion_median) based on the consciousness field. Add a new column to indicate whether the value was filled.
    # for index, row in X_train.iterrows():
    #     if row['motion'] == -1:
    #         consciousness_value = row['consciousness']
    #         X_train.at[index, 'motion'] = motion_median[consciousness_value]
    #         X_train.at[index, 'motion_filled'] = 1
    #     else:
    #         X_train.at[index, 'motion_filled'] = 0
    # # Find records in df where language is -1 and fill in the calculated language_median based on consciousness.
    # for index, row in X_train.iterrows():
    #     if row['language'] == -1:
    #         consciousness_value = row['consciousness']
    #         X_train.at[index, 'language'] = language_median[consciousness_value]
    #         X_train.at[index, 'language_filled'] = 1
    #     else:
    #         X_train.at[index, 'language_filled'] = 0
    # # Count the number of distinct values for language and motion.
    # motion_counts = X_train['motion'].value_counts()
    # language_counts = X_train['language'].value_counts()
    # # Process the test set data according to the correspondence relation.
    # for index, row in X_test.iterrows():
    #     if row['motion'] == -1:
    #         consciousness_value = row['consciousness']
    #         X_test.at[index, 'motion'] = motion_median[consciousness_value]
    #         X_test.at[index, 'motion_filled'] = 1
    #     else:
    #         X_test.at[index, 'motion_filled'] = 0
    # for index, row in X_test.iterrows():
    #     if row['language'] == -1:
    #         consciousness_value = row['consciousness']
    #         X_test.at[index, 'language'] = language_median[consciousness_value]
    #         X_test.at[index, 'language_filled'] = 1
    #     else:
    #         X_test.at[index, 'language_filled'] = 0
    # # Save the fill relation to the CSV file
    # motion_median.to_csv(r'zigong_data\\' + output_dir.split(r"\\")[1] + '_' + output_dir.split(r"\\")[3] +  '_motion_median_filled.csv')
    # language_median.to_csv(r'zigong_data\\' + output_dir.split(r"\\")[1] + '_' + output_dir.split(r"\\")[3] + '_language_median_filled.csv')
    # # Save the training set X_train and the test set X_test
    # X_train = X_train.to_csv(output_dir + 'train.csv', index=False)
    # X_test = X_test.to_csv(output_dir + 'test.csv', index=False)
    #         # # Save the processed data to a new CSV file.
    #         # df.to_csv(r'zigong_data\20251216_final_processed_filled.csv', index=False)
    #         # ########################################################################
    #         # train_data = pd.DataFrame(X_train, columns=X.columns)
    #         # train_data[target_col] = y_train.values
    #         # test_data = pd.DataFrame(X_test, columns=X.columns)
    #         # test_data[target_col] = y_test.values
    #         # train_data.to_csv(output_dir + 'train.csv', index=False)
    #         # test_data.to_csv(output_dir + 'test.csv', index=False)
    # exit()
    # #############################Select the GCS-coded items from the test set.###############################
    df_test = pd.read_csv(r'data_base\\01_midim_db\\full_assessment\\6-4\\test.csv')
    # Filter using the isin method
    columns_to_check = ['open_one\'s_eyes', 'motion', 'language']
    # df_gcs_encoding = df_test[df_test[columns_to_check].isin([-1]).any(axis=1)]
    # Determine whether motion_filled or language_filled is True
    mask_motion_filled = df_test['motion_filled'] == 1
    mask_language_filled = df_test['language_filled'] == 1
    df_gcs_encoding = df_test[mask_motion_filled | mask_language_filled]
    df_gcs_encoding.to_csv(r'data_base\\01_midim_db\\abnormal_assessment\\6_4_test_filled.csv')
    mask_motion_unfilled = df_test['motion_filled'] == 0
    mask_language_unfilled = df_test['language_filled'] == 0
    df_gcs_encoding_unfilled = df_test[mask_motion_unfilled & mask_language_unfilled]
    df_gcs_encoding_unfilled.to_csv(r'data_base\\01_midim_db\\normal_assessment\\6_4_test_unfilled.csv')
    exit()
