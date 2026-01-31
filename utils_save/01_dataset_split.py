import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold
import numpy as np

def has_common_elements(list1, list2):
    """使用集合判断两个列表是否有相同元素"""
    return bool(set(list1) & set(list2))

if __name__ == '__main__':
    # data_dir = r'zigong_data\\20251227_final_processed_data.csv' # 改1
    # output_dir = r'data_base\\01_midim_db\\full_assessment\\6-4\\'  # 改2
    # test_size = 0.4  # 改3
    # df = pd.read_csv(data_dir)
    # X = df
    # y = df['consciousness']
    # target_col = 'consciousness'
    # # 根据患者ID划分训练集和测试集，确保同一患者的数据不出现在训练集和测试集中
    # PATIENT_ID和INP_NO是一一对应的关系，可以任选其一作为分组依据
    # # 7:3 ==> (1241, 532), 6:4 ==> (1063, 710)
    # unique_ids = df['INP_NO'].unique()
    # # 获取不同患者ID对应的觉醒度标签
    # id_to_consciousness = df.groupby('INP_NO')['consciousness'].first().to_dict()
    # # 将stratify参数设置为对应患者ID的觉醒度标签
    # train_ids, test_ids = train_test_split(
    #     unique_ids,
    #     random_state=42,
    #     stratify=[id_to_consciousness[i] for i in unique_ids],
    #     test_size=test_size
    # )
    # # # 打印训练集和测试集的患者数量
    # print(f'Number of unique patients in training set: {len(train_ids)}')
    # print(f'Number of unique patients in testing set: {len(test_ids)}')
    # # # 根据划分好的训练集和测试集ID，获取对应的训练集和测试集
    # train_mask = df['INP_NO'].isin(train_ids)
    # test_mask = df['INP_NO'].isin(test_ids)
    # X_train = X[train_mask]
    # X_test = X[test_mask]
    # y_train = y[train_mask]
    # y_test = y[test_mask]
    # print(f'Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}')
    # ########################################################################
    # # 只针对训练集计算motion和language的中位数进行填充
    # # 保留2个表变量，第一个记录motion不为-1的那些记录，第二个记录language不为-1的那些记录
    # X_train_motion_not_minus1 = X_train[X_train['motion'] != -1]
    # X_train_language_not_minus1 = X_train[X_train['language'] != -1]
    # print(X_train_motion_not_minus1, X_train_language_not_minus1)
    # # 统计不同consciousness下，对应的motion的中位数并打印
    # motion_median = X_train_motion_not_minus1.groupby('consciousness')['motion'].median()
    # print("不同consciousness下，motion的中位数为：")
    # print(motion_median)
    # # 统计不同consciousness下，对应的language的中位数并打印
    # language_median = X_train_language_not_minus1.groupby('consciousness')['language'].median()
    # print("不同consciousness下，language的中位数为：")
    # print(language_median)
    # # 查找df中的motion为-1的记录，根据consciousness填充上边计算出的中位数motion_median，增加新的一列，标记是否填充
    # for index, row in X_train.iterrows():
    #     if row['motion'] == -1:
    #         consciousness_value = row['consciousness']
    #         X_train.at[index, 'motion'] = motion_median[consciousness_value]
    #         X_train.at[index, 'motion_filled'] = 1
    #     else:
    #         X_train.at[index, 'motion_filled'] = 0
    # # 查找df中的language为-1的记录，根据consciousness填充上边计算出的中位数language_median
    # for index, row in X_train.iterrows():
    #     if row['language'] == -1:
    #         consciousness_value = row['consciousness']
    #         X_train.at[index, 'language'] = language_median[consciousness_value]
    #         X_train.at[index, 'language_filled'] = 1
    #     else:
    #         X_train.at[index, 'language_filled'] = 0
    # # 统计language和motion不同取值的数量
    # motion_counts = X_train['motion'].value_counts()
    # language_counts = X_train['language'].value_counts()
    # # 根据对应关系处理测试集数据
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
    # # 保存填充关系到csv文件
    # motion_median.to_csv(r'zigong_data\\' + output_dir.split(r"\\")[1] + '_' + output_dir.split(r"\\")[3] +  '_motion_median_filled.csv')
    # language_median.to_csv(r'zigong_data\\' + output_dir.split(r"\\")[1] + '_' + output_dir.split(r"\\")[3] + '_language_median_filled.csv')
    # # 保存训练集X_train和测试集X_test
    # X_train = X_train.to_csv(output_dir + 'train.csv', index=False)
    # X_test = X_test.to_csv(output_dir + 'test.csv', index=False)
    #         # # 将处理后的数据保存到新的csv文件中
    #         # df.to_csv(r'zigong_data\20251216_final_processed_filled.csv', index=False)
    #         # ########################################################################
    #         # train_data = pd.DataFrame(X_train, columns=X.columns)
    #         # train_data[target_col] = y_train.values
    #         # test_data = pd.DataFrame(X_test, columns=X.columns)
    #         # test_data[target_col] = y_test.values
    #         # train_data.to_csv(output_dir + 'train.csv', index=False)
    #         # test_data.to_csv(output_dir + 'test.csv', index=False)
    # exit()
    # #############################将测试集中的gcs编码项挑选出来###############################
    df_test = pd.read_csv(r'data_base\\01_midim_db\\full_assessment\\6-4\\test.csv')  # 改4
    # 使用isin方法筛选
    columns_to_check = ['open_one\'s_eyes', 'motion', 'language']
    # df_gcs_encoding = df_test[df_test[columns_to_check].isin([-1]).any(axis=1)]
    # 判断motion_filled或language_filled是否为True
    mask_motion_filled = df_test['motion_filled'] == 1
    mask_language_filled = df_test['language_filled'] == 1
    df_gcs_encoding = df_test[mask_motion_filled | mask_language_filled]
    df_gcs_encoding.to_csv(r'data_base\\01_midim_db\\abnormal_assessment\\6_4_test_filled.csv')  # 改5
    mask_motion_unfilled = df_test['motion_filled'] == 0
    mask_language_unfilled = df_test['language_filled'] == 0
    df_gcs_encoding_unfilled = df_test[mask_motion_unfilled & mask_language_unfilled]
    df_gcs_encoding_unfilled.to_csv(r'data_base\\01_midim_db\\normal_assessment\\6_4_test_unfilled.csv')  # 改5
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
