import pickle
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import numpy as np

if __name__ == '__main__':
    
    seed = [42, 230, 4399, 9999, 55000]
    #############################################################################
    seed = seed[4]  # 改1
    # 评估对GCS编码项的预测结果
    exp_class = r'6-4'  # 改2
    file_path = file_path = r"model_pth\\exp_1\\03_cos_similarity_db\\" + exp_class + r"\\seed_" + str(seed) + r'\\' # 改3
    df_test = pd.read_csv(r'data_base\\03_cos_similarity_db\\abnormal_assessment\\' + exp_class.replace('-', '_') + r'_test_filled.csv', index_col=0) # 改4
    ##############################################################################
    select_columns = ['INP_NO', 'Age', 'SEX', 'heart_rate', 'breathing', 'Blood_oxygen_saturation', 
                      'Blood_pressure_high', 'Blood_pressure_low', 'Left_pupil_size', 'Right_pupil_size',
                      'consciousness', 'ChartTime']
    Columns = df_test.drop(columns=select_columns).columns
    Columns = Columns[:]
    # 标记离散特征
    categorical_cols = Columns[-5:]
    ##############################################################################
    # 选择测试数据
    X_test = df_test[Columns]
    target_col = 'consciousness'
    y_test = df_test[target_col]
    test_data = pd.DataFrame(X_test, columns=Columns)
    test_data[target_col] = y_test.values
    ##############################################################################
    # 保存到文件
    with open('classification_report.txt', 'a+') as f:
        f.write('-----------------' + str(datetime.now()) + '-----------------' + exp_class)
        f.write('\n')
    #####################深度神经网络##########################
    with open(file_path + r'\\DANet.pkl', 'rb') as file:
        model = pickle.load(file)
        y_pred = model.predict(test_data)['consciousness_prediction']
        print('DANet模型实验结果：')
        print(classification_report(y_test, y_pred, digits=4))
        # 保存到文件
        with open('classification_report.txt', 'a+') as f:
            f.write('DANet模型实验结果：')
            f.write(classification_report(y_test, y_pred, digits=4))
            f.write('\n')

    with open(file_path + r'\\TabNet.pkl', 'rb') as file:
        model = pickle.load(file)
        y_pred = model.predict(test_data)['consciousness_prediction']
        print('TabNet模型实验结果：')
        print(classification_report(y_test, y_pred, digits=4))
        # 保存到文件
        with open('classification_report.txt', 'a+') as f:
            f.write('TabNet模型实验结果：')
            f.write(classification_report(y_test, y_pred, digits=4))
            f.write('\n')

    with open(file_path + r'\\FTTransformer.pkl', 'rb') as file:
        model = pickle.load(file)
        y_pred = model.predict(test_data)['consciousness_prediction']
        print('FTTransformer模型实验结果：')
        print(classification_report(y_test, y_pred, digits=4))
        # 保存到文件
        with open('classification_report.txt', 'a+') as f:
            f.write('FTTransformer模型实验结果：')
            f.write(classification_report(y_test, y_pred, digits=4))
            f.write('\n')

    with open(file_path + r'\\TabTransformer.pkl', 'rb') as file:
        model = pickle.load(file)
        y_pred = model.predict(test_data)['consciousness_prediction']
        print('TabTransformer模型实验结果：')
        print(classification_report(y_test, y_pred, digits=4))
        # 保存到文件
        with open('classification_report.txt', 'a+') as f:
            f.write('TabTransformer模型实验结果：')
            f.write(classification_report(y_test, y_pred, digits=4))
            f.write('\n')
     
    #####################机器学习技术##########################
    with open(file_path + r'\\lgb.pkl', 'rb') as file:
        model = pickle.load(file)
        y_pred = model.predict(X_test)
        print('LightGBM模型实验结果：')
        print(classification_report(y_test, y_pred, digits=4)) 
        # 保存到文件
        with open('classification_report.txt', 'a+') as f:
            f.write('LightGBM模型实验结果：')
            f.write(classification_report(y_test, y_pred, digits=4))
            f.write('\n')

    with open(file_path + r'\\xgboost.pkl', 'rb') as file:
        model = pickle.load(file)
        y_pred = model.predict(X_test)
        print('XGBoost模型实验结果：')
        print(classification_report(y_test, y_pred, digits=4)) 
        # 保存到文件
        with open('classification_report.txt', 'a+') as f:
            f.write('XGBoost模型实验结果：')
            f.write(classification_report(y_test, y_pred, digits=4))
            f.write('\n')
            
    with open(file_path + r'\\rf_model.pkl', 'rb') as file:
        model = pickle.load(file)
        y_pred = model.predict(X_test)
        print('RandomForest模型实验结果：')
        print(classification_report(y_test, y_pred, digits=4))
        # 保存到文件
        with open('classification_report.txt', 'a+') as f:
            f.write('RandomForest模型实验结果：')
            f.write(classification_report(y_test, y_pred, digits=4))
            f.write('\n')

    with open(file_path + r'\\catboost.pkl', 'rb') as file:
        X_test_str = X_test.copy()
        for col in categorical_cols:
            X_test_str[col] = X_test_str[col].astype(str)
        model = pickle.load(file)
        y_pred = model.predict(X_test_str)
        print('CatBoost模型实验结果：')
        print(classification_report(y_test, y_pred, digits=4))
        # 保存到文件
        with open('classification_report.txt', 'a+') as f:
            f.write('CatBoost模型实验结果：')
            f.write(classification_report(y_test, y_pred, digits=4))
            f.write('\n')
    
    # 对离散特征先进行序数编码，在进行标准化处理
    # 加载编码器
    ordinal_encoder = pickle.load(open(file_path + r'ordinal_encoder.pkl', 'rb'))
    X_test[categorical_cols] = ordinal_encoder.transform(X_test[categorical_cols])
    # 加载标准化器
    scaler = pickle.load(open(file_path + r'scaler.pkl', 'rb'))
    X_test = scaler.transform(X_test)

    with open(file_path + r'\\mlp.pkl', 'rb') as file:
        model = pickle.load(file)
        y_pred = model.predict(X_test)
        print('MLP模型实验结果：')
        print(classification_report(y_test, y_pred, digits=4))
        # 保存到文件
        with open('classification_report.txt', 'a+') as f:
            f.write('MLP模型实验结果：')
            f.write(classification_report(y_test, y_pred, digits=4))
            f.write('\n')         
