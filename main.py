import pandas as pd
from train_zigong import train_all_models

if __name__ == '__main__':
    seed = [42, 230, 4399, 9999, 55000]
    ###########################################################################
    seed = seed[4] # Select a seed for experimentation
    ###########################################################################
    df_train = pd.read_csv(r'data_base\\03_cos_similarity_db\\full_assessment\\6-4\\train.csv')  
    df_test = pd.read_csv(r'data_base\\03_cos_similarity_db\\full_assessment\\6-4\\test.csv') 
    model_save_pth = r'model_pth\\exp_2\\02_similarity\\05_motion_language\\6-4\\seed_' + str(seed) + r'\\' 
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
    Columns = [Columns[-4], Columns[-3], Columns[-2], Columns[-1]]  # Modify when changing indicators
    continuous_cols = []
    categorical_cols = Columns
    ############################################################################
    # Training feature set and test feature set
    X_train = X_train[Columns]
    X_test = X_test[Columns]
    # print(Columns, continuous_cols, categorical_cols)
    # print(X_train, X_test)
    # print(y_train, y_test)
    # exit()
    train_all_models(X_train, X_test, y_train, y_test, Columns, continuous_cols, categorical_cols, target_col, model_save_pth, seed, type_con=type_con)
