import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold
import numpy as np

def has_common_elements(list1, list2):
    """使用集合判断两个列表是否有相同元素"""
    return bool(set(list1) & set(list2))

if __name__ == '__main__':
    # # 读取data_base\01_midim_db\full_assessment\7-3\train.csv
    # train_dir = r'data_base\\01_midim_db\\full_assessment\\6-4\\train.csv'
    # output_train_dir = r'data_base\\00_minus_db1\\full_assessment\\6-4\\train.csv'
    # df_train = pd.read_csv(train_dir)
    # # 读取data_base\01_midim_db\full_assessment\7-3\test.csv
    # test_dir = r'data_base\\01_midim_db\\full_assessment\\6-4\\test.csv'
    # output_test_dir = r'data_base\\00_minus_db1\\full_assessment\\6-4\\test.csv'
    # df_test = pd.read_csv(test_dir)
    # # 判断train.csv中motion_filled字段是否=1，等于1，则对应的motion字段修改为-1，否则不变
    # # 对df_train进行操作
    # df_train.loc[df_train['motion_filled'] == 1, 'motion'] = -1
    # # 对df_test进行操作
    # df_test.loc[df_test['motion_filled'] == 1, 'motion'] = -1
    # # 判断train.csv中language_filled字段是否=1，等于1，则对应的language字段修改为-1，否则不变
    # # 对df_train进行操作
    # df_train.loc[df_train['language_filled'] == 1, 'language'] = -1
    # # 对df_test进行操作
    # df_test.loc[df_test['language_filled'] == 1, 'language'] = -1
    # # 分别保存df_train和df_test
    # df_train.to_csv(output_train_dir, index=False)
    # df_test.to_csv(output_test_dir, index=False)
    # exit()
    # #############################将测试集中的gcs编码项挑选出来###############################
    df_test = pd.read_csv(r'data_base\\00_minus_db1\\full_assessment\\6-4\\test.csv')  # 改4
    # 使用isin方法筛选
    columns_to_check = ['open_one\'s_eyes', 'motion', 'language']
    # df_gcs_encoding = df_test[df_test[columns_to_check].isin([-1]).any(axis=1)]
    # 判断motion_filled或language_filled是否为True
    mask_motion_filled = df_test['motion_filled'] == 1
    mask_language_filled = df_test['language_filled'] == 1
    df_gcs_encoding = df_test[mask_motion_filled | mask_language_filled]
    df_gcs_encoding.to_csv(r'data_base\\00_minus_db1\\abnormal_assessment\\6_4_test_filled.csv')  # 改5
    mask_motion_unfilled = df_test['motion_filled'] == 0
    mask_language_unfilled = df_test['language_filled'] == 0
    df_gcs_encoding_unfilled = df_test[mask_motion_unfilled & mask_language_unfilled]
    df_gcs_encoding_unfilled.to_csv(r'data_base\\00_minus_db1\\normal_assessment\\6_4_test_unfilled.csv')  # 改5
    exit()
    #############################5折交叉验证#############################
    data_dir = 'zigong_data/final_processed_data.csv'
    group_kfold = GroupKFold(n_splits=5)
    df = pd.read_csv(data_dir)
    X = df.drop(columns=['consciousness'])
    y = df['consciousness']
    patient_id = df['INP_NO']
    fold_train = []
    fold_test = []
    for fold, (train_idx, test_idx) in enumerate(group_kfold.split(X, y, patient_id)):
        fold_train.append(df['INP_NO'].iloc[train_idx])
        fold_test.append(df['INP_NO'].iloc[test_idx])
    for i, ele in enumerate(fold_train):
        train_mask = patient_id.isin(ele)
        df_train = df[train_mask]
        df_train.to_csv(r'data_base\\5-folder\\' + str(i+1) + '-fold-train.csv')
    for i, ele in enumerate(fold_test):
        test_mask = patient_id.isin(ele)
        df_test = df[test_mask]
        df_test.to_csv(r'data_base\\5-folder\\' + str(i+1) + '-fold-test.csv')
