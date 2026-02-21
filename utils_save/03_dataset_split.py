import pandas as pd
from sklearn.model_selection import GroupKFold
import numpy as np
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler
import joblib

def has_common_elements(list1, list2):
    """Use sets to determine whether two lists contain identical elements."""
    return bool(set(list1) & set(list2))

if __name__ == '__main__':
    # # Input file: data_base\01_midim_db\full_assessment\7-3\train.csv
    # train_dir = r'data_base\\00_minus_db1\\full_assessment\\7-3\\train.csv'
    # output_train_dir = r'data_base\\03_cos_similarity_db\\full_assessment\\7-3\\train.csv'
    # train_std_scaler = r'data_base\\03_cos_similarity_db\\full_assessment\\7-3\\scaler.pkl'
    # df_train = pd.read_csv(train_dir)
    # # Input file: data_base\01_midim_db\full_assessment\7-3\test.csv
    # test_dir = r'data_base\\00_minus_db1\\full_assessment\\7-3\\test.csv'
    # output_test_dir = r'data_base\\03_cos_similarity_db\\full_assessment\\7-3\\test.csv'
    # df_test = pd.read_csv(test_dir)
    # ######################################################################
    # # Standardize the training set and save the standardizer for the training set.
    # scaler = MinMaxScaler()
    # cols_to_scale = ['heart_rate', 'breathing', 'Blood_oxygen_saturation', 'Blood_pressure_high', 'Blood_pressure_low', 'Left_pupil_size', 'Right_pupil_size']
    # df_train[cols_to_scale] = scaler.fit_transform(df_train[cols_to_scale])
    # joblib.dump(scaler, train_std_scaler)
    # # Find records where both motion_filled and language_filled are 0.
    # df_train_ml_unfilled = df_train[(df_train['motion_filled'] == 0) & (df_train['language_filled'] == 0)]
    # # Apply the custom function to each record in df_train_filled.
    # def fill_motion(row, df_train_ml_unfilled, row_index):
    #     print('row_index:', row_index)
    #     # Remove the following fields: INP_NO, ChartTime, motion_filled, language_filled
    #     motion_filled = row['motion_filled']
    #     language_filled = row['language_filled']
    #     if motion_filled == 0 and language_filled == 0:
    #         return pd.Series([row['motion'], row['language']])
    #     row_copy = row.copy()
    #     # Exclude the patient themselves
    #     df_train_ml_unfilled = df_train_ml_unfilled[df_train_ml_unfilled['INP_NO'] != row['INP_NO']]
    #     row = row.drop(['INP_NO', 'Age', 'SEX', 'ChartTime', 'motion_filled', 'language_filled'])
    #     # Remove INP_NO, ChartTime, motion_filled, and language_filled from df_train_motion_unfilled.
    #     df_train_ml_unfilled = df_train_ml_unfilled.drop(['INP_NO', 'Age', 'SEX', 'ChartTime', 'motion_filled', 'language_filled'], axis=1)
    #     df_train_ml_unfilled_origin = df_train_ml_unfilled.copy()
    #     # Remove rows containing Age, SEX, open_one's_eyes, and consciousness from df_train_ml_unfilled.
    #     # First, filter based on Age, SEX, open_one's_eyes, and consciousness.
    #     # Exclude language indicators and motion indicators from similarity calculations.
    #     if motion_filled == 0:
    #         # Remove the language column from the row and the df_train_ml_unfilled column.
    #         row = row.drop(['language'])
    #         #  Remove the language column from df_train_ml_unfilled
    #         df_train_ml_unfilled = df_train_ml_unfilled.drop(['language'], axis=1)
    #     else:
    #          # Remove the motion column from the row and the df_train_ml_unfilled column.
    #         row = row.drop(['motion'])
    #         # Remove the motion column from df_train_ml_unfilled
    #         df_train_ml_unfilled = df_train_ml_unfilled.drop(['motion'], axis=1)
    #         # Remove the language column from the row and the df_train_ml_unfilled column.
    #         row = row.drop(['language'])
    #         # Remove the language column from df_train_ml_unfilled
    #         df_train_ml_unfilled = df_train_ml_unfilled.drop(['language'], axis=1)
    #     df_train_ml_unfilled_copy = df_train_ml_unfilled.copy()
    #     # If the value in the motion_filled column is not 1
    #     if motion_filled == 0:
    #         df_train_ml_unfilled = df_train_ml_unfilled[df_train_ml_unfilled['open_one\'s_eyes'] == row['open_one\'s_eyes']]
    #         df_train_ml_unfilled = df_train_ml_unfilled[df_train_ml_unfilled['motion'] == row['motion']]
    #         df_train_ml_unfilled = df_train_ml_unfilled[df_train_ml_unfilled['consciousness'] == row['consciousness']]
    #     else:
    #         df_train_ml_unfilled = df_train_ml_unfilled[df_train_ml_unfilled['open_one\'s_eyes'] == row['open_one\'s_eyes']]
    #         df_train_ml_unfilled = df_train_ml_unfilled[df_train_ml_unfilled['consciousness'] == row['consciousness']]
    #     # If the content is empty, set `train_ml_unfilled` to `df_train_ml_unfilled_copy`.
    #     if df_train_ml_unfilled.empty:
    #         df_train_ml_unfilled = df_train_ml_unfilled_copy
    #         # Only consciousness is restricted
    #         df_train_ml_unfilled = df_train_ml_unfilled[df_train_ml_unfilled['consciousness'] == row['consciousness']]
    #     # Calculate the Manhattan distance between two vectors
    #     def manhattan_distance(vec1, vec2):
    #         return distance.cityblock(vec1, vec2)
    #     # Calculate the Manhattan distance between each row in `row` and each record in `df_train_ml_unfilled`.
    #     similarities = df_train_ml_unfilled.apply(lambda x: manhattan_distance(row, x), axis=1)
    #     # Find the record with the highest similarity
    #     most_similar_index = similarities.idxmin()
    #     # Return the motion and language with the highest similarity score
    #     if motion_filled == 1 and language_filled == 0:
    #         return pd.Series([df_train_ml_unfilled_origin.loc[most_similar_index, 'motion'], row_copy['language']])
    #     elif motion_filled == 0 and language_filled == 1:
    #         return pd.Series([row_copy['motion'], df_train_ml_unfilled_origin.loc[most_similar_index, 'language']])
    #     elif motion_filled == 1 and language_filled == 1:
    #         return pd.Series([df_train_ml_unfilled_origin.loc[most_similar_index, 'motion'], df_train_ml_unfilled_origin.loc[most_similar_index, 'language']])
    # df_train[['motion', 'language']] = df_train.apply(lambda row: fill_motion(row, df_train_ml_unfilled, row.name), axis=1, result_type='expand')
    # df_train.to_csv(output_train_dir, index=False)
    # exit()
    #################################################################################################################################
    # # The test set filling strategy mirrors that of the training set, filtering out normally evaluated samples from the training set and samples that cannot be evaluated in the test set, then calculating the Manhattan distance.
    # # Find records where both motion_filled and language_filled are 0.
    # # Load Standardizer
    # df_test_scaler = joblib.load(train_std_scaler)
    # # Columns requiring standardization
    # cols_to_scale = ['heart_rate', 'breathing', 'Blood_oxygen_saturation', 'Blood_pressure_high', 'Blood_pressure_low', 'Left_pupil_size', 'Right_pupil_size']
    # df_test[cols_to_scale] = df_test_scaler.transform(df_test[cols_to_scale])
    # df_train_ml_unfilled = df_train[(df_train['motion_filled'] == 0) & (df_train['language_filled'] == 0)]
    # def fill_motion(row, df_train_ml_unfilled, row_index):
    #     print('row_index:', row_index)
    #     # Remove the following fields: INP_NO, ChartTime, motion_filled, language_filled
    #     motion_filled = row['motion_filled']
    #     language_filled = row['language_filled']
    #     if motion_filled == 0 and language_filled == 0:
    #         return pd.Series([row['motion'], row['language']])
    #     row_copy = row.copy()
    #     # Exclude the patient themselves
    #     df_train_ml_unfilled = df_train_ml_unfilled[df_train_ml_unfilled['INP_NO'] != row['INP_NO']]
    #     row = row.drop(['INP_NO', 'Age', 'SEX', 'ChartTime', 'motion_filled', 'language_filled'])
    #     # Remove INP_NO, ChartTime, motion_filled, and language_filled from df_train_motion_unfilled.
    #     df_train_ml_unfilled = df_train_ml_unfilled.drop(['INP_NO', 'Age', 'SEX', 'ChartTime', 'motion_filled', 'language_filled'], axis=1)
    #     df_train_ml_unfilled_origin = df_train_ml_unfilled.copy()
    #     # Remove rows containing Age, SEX, open_one's_eyes, and consciousness from df_train_ml_unfilled.
    #     # First, filter based on Age, SEX, open_one's_eyes, and consciousness.
    #     # Exclude language indicators and motion indicators from similarity calculations.
    #     if motion_filled == 0:
    #         # Remove the language column from the row and the df_train_ml_unfilled column.
    #         row = row.drop(['language'])
    #         # Remove the language column from df_train_ml_unfilled
    #         df_train_ml_unfilled = df_train_ml_unfilled.drop(['language'], axis=1)
    #     else:
    #          # Remove the motion column from the row and the df_train_ml_unfilled column.
    #         row = row.drop(['motion'])
    #         # Remove the motion column from df_train_ml_unfilled
    #         df_train_ml_unfilled = df_train_ml_unfilled.drop(['motion'], axis=1)
    #         # Remove the language column from the row and the df_train_ml_unfilled column.
    #         row = row.drop(['language'])
    #         # Remove the language column from df_train_ml_unfilled
    #         df_train_ml_unfilled = df_train_ml_unfilled.drop(['language'], axis=1)
    #     df_train_ml_unfilled_copy = df_train_ml_unfilled.copy()
    #     # If the value in the motion_filled column is not 1
    #     if motion_filled == 0:
    #         df_train_ml_unfilled = df_train_ml_unfilled[df_train_ml_unfilled['open_one\'s_eyes'] == row['open_one\'s_eyes']]
    #         df_train_ml_unfilled = df_train_ml_unfilled[df_train_ml_unfilled['motion'] == row['motion']]
    #         df_train_ml_unfilled = df_train_ml_unfilled[df_train_ml_unfilled['consciousness'] == row['consciousness']]
    #     else:
    #         df_train_ml_unfilled = df_train_ml_unfilled[df_train_ml_unfilled['open_one\'s_eyes'] == row['open_one\'s_eyes']]
    #         df_train_ml_unfilled = df_train_ml_unfilled[df_train_ml_unfilled['consciousness'] == row['consciousness']]
    #     # If the content is empty, set `train_ml_unfilled` to `df_train_ml_unfilled_copy`.
    #     if df_train_ml_unfilled.empty:
    #         df_train_ml_unfilled = df_train_ml_unfilled_copy
    #         # Restrict consciousness only
    #         df_train_ml_unfilled = df_train_ml_unfilled[df_train_ml_unfilled['consciousness'] == row['consciousness']]
    #     # Calculate the Manhattan distance between two vectors
    #     def manhattan_distance(vec1, vec2):
    #         return distance.cityblock(vec1, vec2)
    #     # Calculate the Manhattan distance between each row in `row` and each record in `df_train_ml_unfilled`.
    #     similarities = df_train_ml_unfilled.apply(lambda x: manhattan_distance(row, x), axis=1)
    #     # Find the record with the highest similarity
    #     most_similar_index = similarities.idxmin()
    #     # Return the motion and language with the highest similarity score
    #     if motion_filled == 1 and language_filled == 0:
    #         return pd.Series([df_train_ml_unfilled_origin.loc[most_similar_index, 'motion'], row_copy['language']])
    #     elif motion_filled == 0 and language_filled == 1:
    #         return pd.Series([row_copy['motion'], df_train_ml_unfilled_origin.loc[most_similar_index, 'language']])
    #     elif motion_filled == 1 and language_filled == 1:
    #         return pd.Series([df_train_ml_unfilled_origin.loc[most_similar_index, 'motion'], df_train_ml_unfilled_origin.loc[most_similar_index, 'language']])
    # df_test[['motion', 'language']] = df_test.apply(lambda row: fill_motion(row, df_train_ml_unfilled, row.name), axis=1, result_type='expand')
    # df_test.to_csv(output_test_dir, index=False)
    # exit()
    # #############################Select the GCS-coded items from the test set.###############################
    df_test = pd.read_csv(r'data_base\\03_cos_similarity_db\\full_assessment\\6-4\\test.csv')
    # Filter using the isin method
    columns_to_check = ['open_one\'s_eyes', 'motion', 'language']
    # df_gcs_encoding = df_test[df_test[columns_to_check].isin([-1]).any(axis=1)]
    # Determine whether motion_filled or language_filled is True
    mask_motion_filled = df_test['motion_filled'] == 1
    mask_language_filled = df_test['language_filled'] == 1
    df_gcs_encoding = df_test[mask_motion_filled | mask_language_filled]
    df_gcs_encoding.to_csv(r'data_base\\03_cos_similarity_db\\abnormal_assessment\\6_4_test_filled.csv')
    mask_motion_unfilled = df_test['motion_filled'] == 0
    mask_language_unfilled = df_test['language_filled'] == 0
    df_gcs_encoding_unfilled = df_test[mask_motion_unfilled & mask_language_unfilled]
    df_gcs_encoding_unfilled.to_csv(r'data_base\\03_cos_similarity_db\\normal_assessment\\6_4_test_unfilled.csv')
    exit()
