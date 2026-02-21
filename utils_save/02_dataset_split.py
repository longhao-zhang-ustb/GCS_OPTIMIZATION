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
    # output_train_dir = r'data_base\\02_mode_db\\full_assessment\\6-4\\train.csv'
    # df_train = pd.read_csv(train_dir)
    # # Input file: data_base\01_midim_db\full_assessment\7-3\test.csv
    # test_dir = r'data_base\\01_midim_db\\full_assessment\\6-4\\test.csv'
    # output_test_dir = r'data_base\\02_mode_db\\full_assessment\\6-4\\test.csv'
    # df_test = pd.read_csv(test_dir)
    # ##############################Calculate the mode of the training set########################################
    # # 获取df_train中motion_filled为0的内容
    # df_train_motion_unfilled = df_train[df_train['motion_filled'] == 0]
    # # 按照consciousness进行分组，计算每个分组的众数，打印分组结果
    # motion_mode_by_consciousness = df_train_motion_unfilled.groupby('consciousness')['motion'].agg(lambda x: x.mode()[0] if not x.mode().empty else None)
    # # 如果motion_filled为1，则根据其consciousness对应的motion_mode_by_consciousness进行填充
    # df_train.loc[df_train['motion_filled'] == 1, 'motion'] = df_train.loc[df_train['motion_filled'] == 1, 'consciousness'].map(motion_mode_by_consciousness)
    # # 获取df_train中language_filled为0的内容
    # df_train_language_unfilled = df_train[df_train['language_filled'] == 0]
    # # 按照consciousness进行分组，计算每个分组的众数，打印分组结果
    # language_mode_by_consciousness = df_train_language_unfilled.groupby('consciousness')['language'].agg(lambda x: x.mode()[0] if not x.mode().empty else None)
    # # 如果language_filled为1，则根据其consciousness对应的language_mode_by_consciousness进行填充
    # df_train.loc[df_train['language_filled'] == 1, 'language'] = df_train.loc[df_train['language_filled'] == 1, 'consciousness'].map(language_mode_by_consciousness)
    # # # 保存填充关系到csv文件
    # motion_mode_by_consciousness.to_csv(r'zigong_data\\' + output_train_dir.split(r"\\")[1] + '_' + output_train_dir.split(r"\\")[3] +  '_motion_mode_filled.csv')
    # language_mode_by_consciousness.to_csv(r'zigong_data\\' + output_train_dir.split(r"\\")[1] + '_' + output_train_dir.split(r"\\")[3] + '_language_mode_filled.csv')
    # # 根据填充关系对df_test进行填充
    # df_test.loc[df_test['motion_filled'] == 1, 'motion'] = df_test.loc[df_test['motion_filled'] == 1, 'consciousness'].map(motion_mode_by_consciousness)
    # df_test.loc[df_test['language_filled'] == 1, 'language'] = df_test.loc[df_test['language_filled'] == 1, 'consciousness'].map(language_mode_by_consciousness)
    # # 保存df_train和df_test
    # df_train.to_csv(output_train_dir, index=False)
    # df_test.to_csv(output_test_dir, index=False)
    # exit()
    # #############################将测试集中的gcs编码项挑选出来###############################
    df_test = pd.read_csv(r'data_base\\02_mode_db\\full_assessment\\6-4\\test.csv')  # 改4
    # 使用isin方法筛选
    columns_to_check = ['open_one\'s_eyes', 'motion', 'language']
    # df_gcs_encoding = df_test[df_test[columns_to_check].isin([-1]).any(axis=1)]
    # 判断motion_filled或language_filled是否为True
    mask_motion_filled = df_test['motion_filled'] == 1
    mask_language_filled = df_test['language_filled'] == 1
    df_gcs_encoding = df_test[mask_motion_filled | mask_language_filled]
    df_gcs_encoding.to_csv(r'data_base\\02_mode_db\\abnormal_assessment\\6_4_test_filled.csv')  # 改5
    mask_motion_unfilled = df_test['motion_filled'] == 0
    mask_language_unfilled = df_test['language_filled'] == 0
    df_gcs_encoding_unfilled = df_test[mask_motion_unfilled & mask_language_unfilled]
    df_gcs_encoding_unfilled.to_csv(r'data_base\\02_mode_db\\normal_assessment\\6_4_test_unfilled.csv')  # 改5
    exit()
