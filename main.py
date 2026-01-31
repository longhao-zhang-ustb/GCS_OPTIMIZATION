import pandas as pd
from train_zigong import train_all_models

if __name__ == '__main__':
    # 注意以下train_zigong.py中的内容
    seed = [42, 230, 4399, 9999, 55000]
    ###########################################################################
    seed = seed[4] # 改1：选择一个种子进行实验
    ###########################################################################
    df_train = pd.read_csv(r'data_base\\03_cos_similarity_db\\full_assessment\\6-4\\train.csv')  # 改2
    df_test = pd.read_csv(r'data_base\\03_cos_similarity_db\\full_assessment\\6-4\\test.csv') # 改3
    model_save_pth = r'model_pth\\exp_2\\02_similarity\\05_motion_language\\6-4\\seed_' + str(seed) + r'\\' # 改4
    ###########################################################################
    target_col = 'consciousness'
    select_columns = ['INP_NO', 'Age', 'SEX', 'heart_rate', 'breathing', 'Blood_oxygen_saturation', 
                      'Blood_pressure_high', 'Blood_pressure_low', 'Left_pupil_size', 'Right_pupil_size',
                      'consciousness', 'ChartTime']
    X_train = df_train.drop(columns=select_columns)
    y_train = df_train[target_col]
    X_test = df_test.drop(columns=select_columns)
    y_test = df_test[target_col]
    X = df_train.drop(columns=select_columns)
    Columns = X.columns
    ###########################################################################
    type_con = ""
    # open_one's_eyes,motion,language,motion_filled,language_filled
    Columns = [Columns[-4], Columns[-3], Columns[-2], Columns[-1]]  # 更换指标时修改
    continuous_cols = []
    categorical_cols = Columns
    ############################################################################
    # 训练特征与测试特征集合
    X_train = X_train[Columns]
    X_test = X_test[Columns]
    # print(Columns, continuous_cols, categorical_cols)
    # print(X_train, X_test)
    # print(y_train, y_test)
    # exit()
    train_all_models(X_train, X_test, y_train, y_test, Columns, continuous_cols, categorical_cols, target_col, model_save_pth, seed, type_con=type_con)
