import pandas as pd
from sklearn.model_selection import GroupKFold
import numpy as np
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler
import joblib

def has_common_elements(list1, list2):
    """使用集合判断两个列表是否有相同元素"""
    return bool(set(list1) & set(list2))

if __name__ == '__main__':
    # # 读取data_base\01_midim_db\full_assessment\7-3\train.csv
    # train_dir = r'data_base\\00_minus_db1\\full_assessment\\7-3\\train.csv'
    # output_train_dir = r'data_base\\03_cos_similarity_db\\full_assessment\\7-3\\train.csv'
    # train_std_scaler = r'data_base\\03_cos_similarity_db\\full_assessment\\7-3\\scaler.pkl'
    # df_train = pd.read_csv(train_dir)
    # # 读取data_base\01_midim_db\full_assessment\7-3\test.csv
    # test_dir = r'data_base\\00_minus_db1\\full_assessment\\7-3\\test.csv'
    # output_test_dir = r'data_base\\03_cos_similarity_db\\full_assessment\\7-3\\test.csv'
    # df_test = pd.read_csv(test_dir)
    # ##############################对训练集进行众数处理########################################
    # # 训练集标准化，并保存训练集的标准化器
    # scaler = MinMaxScaler()
    # cols_to_scale = ['heart_rate', 'breathing', 'Blood_oxygen_saturation', 'Blood_pressure_high', 'Blood_pressure_low', 'Left_pupil_size', 'Right_pupil_size']
    # df_train[cols_to_scale] = scaler.fit_transform(df_train[cols_to_scale])
    # # 保存标准化器
    # joblib.dump(scaler, train_std_scaler)
    # # 查找其中motion_filled和language_filled都为0的记录
    # df_train_ml_unfilled = df_train[(df_train['motion_filled'] == 0) & (df_train['language_filled'] == 0)]
    # # 对df_train_filled中的每条记录分别应用自定义函数
    # def fill_motion(row, df_train_ml_unfilled, row_index):
    #     print('row_index:', row_index)
    #     # 去掉其中的INP_NO, ChartTime, motion_filled, language_filled
    #     motion_filled = row['motion_filled']
    #     language_filled = row['language_filled']
    #     if motion_filled == 0 and language_filled == 0:
    #         return pd.Series([row['motion'], row['language']])
    #     row_copy = row.copy()
    #     # 排除患者自身
    #     df_train_ml_unfilled = df_train_ml_unfilled[df_train_ml_unfilled['INP_NO'] != row['INP_NO']]
    #     row = row.drop(['INP_NO', 'Age', 'SEX', 'ChartTime', 'motion_filled', 'language_filled'])
    #     # 去掉df_train_motion_unfilled中的INP_NO, ChartTime, motion_filled, language_filled
    #     df_train_ml_unfilled = df_train_ml_unfilled.drop(['INP_NO', 'Age', 'SEX', 'ChartTime', 'motion_filled', 'language_filled'], axis=1)
    #     df_train_ml_unfilled_origin = df_train_ml_unfilled.copy()
    #     # 去掉df_train_ml_unfilled中的Age, SEX, open_one's_eyes和consciousness不同的行
    #     # 先根据Age, SEX, open_one's_eyes和consciousness筛选
    #     # 删除其中的语言指标和运动指标不参与相似度计算
    #     if motion_filled == 0:
    #         # 去掉row中和df_train_ml_unfilled列的language列
    #         row = row.drop(['language'])
    #         # 去掉df_train_ml_unfilled中的language列
    #         df_train_ml_unfilled = df_train_ml_unfilled.drop(['language'], axis=1)
    #     else:
    #          # 去掉row中和df_train_ml_unfilled列的motion列
    #         row = row.drop(['motion'])
    #         # 去掉df_train_ml_unfilled中的motion列
    #         df_train_ml_unfilled = df_train_ml_unfilled.drop(['motion'], axis=1)
    #         # 去掉row中和df_train_ml_unfilled列的language列
    #         row = row.drop(['language'])
    #         # 去掉df_train_ml_unfilled中的language列
    #         df_train_ml_unfilled = df_train_ml_unfilled.drop(['language'], axis=1)
    #     df_train_ml_unfilled_copy = df_train_ml_unfilled.copy()
    #     # 如果motion_filled列值不为1
    #     if motion_filled == 0:
    #         df_train_ml_unfilled = df_train_ml_unfilled[df_train_ml_unfilled['open_one\'s_eyes'] == row['open_one\'s_eyes']]
    #         df_train_ml_unfilled = df_train_ml_unfilled[df_train_ml_unfilled['motion'] == row['motion']]
    #         df_train_ml_unfilled = df_train_ml_unfilled[df_train_ml_unfilled['consciousness'] == row['consciousness']]
    #     else:
    #         df_train_ml_unfilled = df_train_ml_unfilled[df_train_ml_unfilled['open_one\'s_eyes'] == row['open_one\'s_eyes']]
    #         df_train_ml_unfilled = df_train_ml_unfilled[df_train_ml_unfilled['consciousness'] == row['consciousness']]
    #     # 如果内容为空，则将train_ml_unfilled设置为df_train_ml_unfilled_copy
    #     if df_train_ml_unfilled.empty:
    #         df_train_ml_unfilled = df_train_ml_unfilled_copy
    #         # 只限制consciousness
    #         df_train_ml_unfilled = df_train_ml_unfilled[df_train_ml_unfilled['consciousness'] == row['consciousness']]
    #     # 计算两个向量的曼哈顿距离
    #     def manhattan_distance(vec1, vec2):
    #         return distance.cityblock(vec1, vec2)
    #     # 计算row与df_train_ml_unfilled中每条记录的曼哈顿距离
    #     similarities = df_train_ml_unfilled.apply(lambda x: manhattan_distance(row, x), axis=1)
    #     # 找到相似度最高的记录
    #     most_similar_index = similarities.idxmin()
    #     # 返回相似度最高记录的motion和language
    #     if motion_filled == 1 and language_filled == 0:
    #         return pd.Series([df_train_ml_unfilled_origin.loc[most_similar_index, 'motion'], row_copy['language']])
    #     elif motion_filled == 0 and language_filled == 1:
    #         return pd.Series([row_copy['motion'], df_train_ml_unfilled_origin.loc[most_similar_index, 'language']])
    #     elif motion_filled == 1 and language_filled == 1:
    #         return pd.Series([df_train_ml_unfilled_origin.loc[most_similar_index, 'motion'], df_train_ml_unfilled_origin.loc[most_similar_index, 'language']])
    # df_train[['motion', 'language']] = df_train.apply(lambda row: fill_motion(row, df_train_ml_unfilled, row.name), axis=1, result_type='expand')
    # # 训练集写入csv文件
    # df_train.to_csv(output_train_dir, index=False)
    # exit()
    #################################################################################################################################
    # # 测试集填充策略与训练集相同，筛选出训练集正常评估样本及测试集无法评估的样本，计算曼哈顿距离
    # # 查找其中motion_filled和language_filled都为0的记录
    # # 加载标准化器
    # df_test_scaler = joblib.load(train_std_scaler)
    # # 需要标准化的列
    # cols_to_scale = ['heart_rate', 'breathing', 'Blood_oxygen_saturation', 'Blood_pressure_high', 'Blood_pressure_low', 'Left_pupil_size', 'Right_pupil_size']
    # df_test[cols_to_scale] = df_test_scaler.transform(df_test[cols_to_scale])
    # df_train_ml_unfilled = df_train[(df_train['motion_filled'] == 0) & (df_train['language_filled'] == 0)]
    # def fill_motion(row, df_train_ml_unfilled, row_index):
    #     print('row_index:', row_index)
    #     # 去掉其中的INP_NO, ChartTime, motion_filled, language_filled
    #     motion_filled = row['motion_filled']
    #     language_filled = row['language_filled']
    #     if motion_filled == 0 and language_filled == 0:
    #         return pd.Series([row['motion'], row['language']])
    #     row_copy = row.copy()
    #     # 排除患者自身
    #     df_train_ml_unfilled = df_train_ml_unfilled[df_train_ml_unfilled['INP_NO'] != row['INP_NO']]
    #     row = row.drop(['INP_NO', 'Age', 'SEX', 'ChartTime', 'motion_filled', 'language_filled'])
    #     # 去掉df_train_motion_unfilled中的INP_NO, ChartTime, motion_filled, language_filled
    #     df_train_ml_unfilled = df_train_ml_unfilled.drop(['INP_NO', 'Age', 'SEX', 'ChartTime', 'motion_filled', 'language_filled'], axis=1)
    #     df_train_ml_unfilled_origin = df_train_ml_unfilled.copy()
    #     # 去掉df_train_ml_unfilled中的Age, SEX, open_one's_eyes和consciousness不同的行
    #     # 先根据Age, SEX, open_one's_eyes和consciousness筛选
    #     # 删除其中的语言指标和运动指标不参与相似度计算
    #     if motion_filled == 0:
    #         # 去掉row中和df_train_ml_unfilled列的language列
    #         row = row.drop(['language'])
    #         # 去掉df_train_ml_unfilled中的language列
    #         df_train_ml_unfilled = df_train_ml_unfilled.drop(['language'], axis=1)
    #     else:
    #          # 去掉row中和df_train_ml_unfilled列的motion列
    #         row = row.drop(['motion'])
    #         # 去掉df_train_ml_unfilled中的motion列
    #         df_train_ml_unfilled = df_train_ml_unfilled.drop(['motion'], axis=1)
    #         # 去掉row中和df_train_ml_unfilled列的language列
    #         row = row.drop(['language'])
    #         # 去掉df_train_ml_unfilled中的language列
    #         df_train_ml_unfilled = df_train_ml_unfilled.drop(['language'], axis=1)
    #     df_train_ml_unfilled_copy = df_train_ml_unfilled.copy()
    #     # 如果motion_filled列值不为1
    #     if motion_filled == 0:
    #         df_train_ml_unfilled = df_train_ml_unfilled[df_train_ml_unfilled['open_one\'s_eyes'] == row['open_one\'s_eyes']]
    #         df_train_ml_unfilled = df_train_ml_unfilled[df_train_ml_unfilled['motion'] == row['motion']]
    #         df_train_ml_unfilled = df_train_ml_unfilled[df_train_ml_unfilled['consciousness'] == row['consciousness']]
    #     else:
    #         df_train_ml_unfilled = df_train_ml_unfilled[df_train_ml_unfilled['open_one\'s_eyes'] == row['open_one\'s_eyes']]
    #         df_train_ml_unfilled = df_train_ml_unfilled[df_train_ml_unfilled['consciousness'] == row['consciousness']]
    #     # 如果内容为空，则将train_ml_unfilled设置为df_train_ml_unfilled_copy
    #     if df_train_ml_unfilled.empty:
    #         df_train_ml_unfilled = df_train_ml_unfilled_copy
    #         # 只限制consciousness
    #         df_train_ml_unfilled = df_train_ml_unfilled[df_train_ml_unfilled['consciousness'] == row['consciousness']]
    #     # 计算两个向量的曼哈顿距离
    #     def manhattan_distance(vec1, vec2):
    #         return distance.cityblock(vec1, vec2)
    #     # 计算row与df_train_ml_unfilled中每条记录的曼哈顿距离
    #     similarities = df_train_ml_unfilled.apply(lambda x: manhattan_distance(row, x), axis=1)
    #     # 找到相似度最高的记录
    #     most_similar_index = similarities.idxmin()
    #     # 返回相似度最高记录的motion和language
    #     if motion_filled == 1 and language_filled == 0:
    #         return pd.Series([df_train_ml_unfilled_origin.loc[most_similar_index, 'motion'], row_copy['language']])
    #     elif motion_filled == 0 and language_filled == 1:
    #         return pd.Series([row_copy['motion'], df_train_ml_unfilled_origin.loc[most_similar_index, 'language']])
    #     elif motion_filled == 1 and language_filled == 1:
    #         return pd.Series([df_train_ml_unfilled_origin.loc[most_similar_index, 'motion'], df_train_ml_unfilled_origin.loc[most_similar_index, 'language']])
    # df_test[['motion', 'language']] = df_test.apply(lambda row: fill_motion(row, df_train_ml_unfilled, row.name), axis=1, result_type='expand')
    # df_test.to_csv(output_test_dir, index=False)
    # exit()
    # #############################将测试集中的gcs编码项挑选出来###############################
    df_test = pd.read_csv(r'data_base\\03_cos_similarity_db\\full_assessment\\6-4\\test.csv')  # 改4
    # 使用isin方法筛选
    columns_to_check = ['open_one\'s_eyes', 'motion', 'language']
    # df_gcs_encoding = df_test[df_test[columns_to_check].isin([-1]).any(axis=1)]
    # 判断motion_filled或language_filled是否为True
    mask_motion_filled = df_test['motion_filled'] == 1
    mask_language_filled = df_test['language_filled'] == 1
    df_gcs_encoding = df_test[mask_motion_filled | mask_language_filled]
    df_gcs_encoding.to_csv(r'data_base\\03_cos_similarity_db\\abnormal_assessment\\6_4_test_filled.csv')  # 改5
    mask_motion_unfilled = df_test['motion_filled'] == 0
    mask_language_unfilled = df_test['language_filled'] == 0
    df_gcs_encoding_unfilled = df_test[mask_motion_unfilled & mask_language_unfilled]
    df_gcs_encoding_unfilled.to_csv(r'data_base\\03_cos_similarity_db\\normal_assessment\\6_4_test_unfilled.csv')  # 改5
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
