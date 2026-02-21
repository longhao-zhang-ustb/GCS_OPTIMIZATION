import pickle
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import numpy as np

if __name__ == '__main__':
    
    seed = [42, 230, 4399, 9999, 55000]
    #############################################################################
    seed = seed[4]
    # Evaluate the prediction results for GCS coding items
    exp_class = r'6-4'
    file_path = file_path = r"model_pth\\exp_1\\03_cos_similarity_db\\" + exp_class + r"\\seed_" + str(seed) + r'\\'
    df_test = pd.read_csv(r'data_base\\03_cos_similarity_db\\abnormal_assessment\\' + exp_class.replace('-', '_') + r'_test_filled.csv', index_col=0)
    ##############################################################################
    select_columns = ['INP_NO', 'Age', 'SEX', 'heart_rate', 'breathing', 'Blood_oxygen_saturation', 
                      'Blood_pressure_high', 'Blood_pressure_low', 'Left_pupil_size', 'Right_pupil_size',
                      'consciousness', 'ChartTime']
    Columns = df_test.drop(columns=select_columns).columns
    Columns = Columns[:]
    # Mark discrete features
    categorical_cols = Columns[-5:]
    ##############################################################################
    # Select test data
    X_test = df_test[Columns]
    target_col = 'consciousness'
    y_test = df_test[target_col]
    test_data = pd.DataFrame(X_test, columns=Columns)
    test_data[target_col] = y_test.values
    ##############################################################################
    # Save to file
    with open('classification_report.txt', 'a+') as f:
        f.write('-----------------' + str(datetime.now()) + '-----------------' + exp_class)
        f.write('\n')
    #####################DNNs##########################
    with open(file_path + r'\\DANet.pkl', 'rb') as file:
        model = pickle.load(file)
        y_pred = model.predict(test_data)['consciousness_prediction']
        print('DANet experimental results:')
        print(classification_report(y_test, y_pred, digits=4))
        # Save to file
        with open('classification_report.txt', 'a+') as f:
            f.write('DANet experimental results:')
            f.write(classification_report(y_test, y_pred, digits=4))
            f.write('\n')

    with open(file_path + r'\\TabNet.pkl', 'rb') as file:
        model = pickle.load(file)
        y_pred = model.predict(test_data)['consciousness_prediction']
        print('TabNet experimental results:')
        print(classification_report(y_test, y_pred, digits=4))
        # Save to file
        with open('classification_report.txt', 'a+') as f:
            f.write('TabNet experimental results:')
            f.write(classification_report(y_test, y_pred, digits=4))
            f.write('\n')

    with open(file_path + r'\\FTTransformer.pkl', 'rb') as file:
        model = pickle.load(file)
        y_pred = model.predict(test_data)['consciousness_prediction']
        print('FTTransformer experimental results:')
        print(classification_report(y_test, y_pred, digits=4))
        # Save to file
        with open('classification_report.txt', 'a+') as f:
            f.write('FTTransformer experimental results:')
            f.write(classification_report(y_test, y_pred, digits=4))
            f.write('\n')

    with open(file_path + r'\\TabTransformer.pkl', 'rb') as file:
        model = pickle.load(file)
        y_pred = model.predict(test_data)['consciousness_prediction']
        print('TabTransformer experimental results:')
        print(classification_report(y_test, y_pred, digits=4))
        # Save to file
        with open('classification_report.txt', 'a+') as f:
            f.write('TabTransformer experimental results:')
            f.write(classification_report(y_test, y_pred, digits=4))
            f.write('\n')
     
    #####################Machine Learning##########################
    with open(file_path + r'\\lgb.pkl', 'rb') as file:
        model = pickle.load(file)
        y_pred = model.predict(X_test)
        print('LightGBM experimental results:')
        print(classification_report(y_test, y_pred, digits=4)) 
        # Save to file
        with open('classification_report.txt', 'a+') as f:
            f.write('LightGBM experimental results:')
            f.write(classification_report(y_test, y_pred, digits=4))
            f.write('\n')

    with open(file_path + r'\\xgboost.pkl', 'rb') as file:
        model = pickle.load(file)
        y_pred = model.predict(X_test)
        print('XGBoost experimental results:')
        print(classification_report(y_test, y_pred, digits=4)) 
        # Save to file
        with open('classification_report.txt', 'a+') as f:
            f.write('XGBoost experimental results:')
            f.write(classification_report(y_test, y_pred, digits=4))
            f.write('\n')
            
    with open(file_path + r'\\rf_model.pkl', 'rb') as file:
        model = pickle.load(file)
        y_pred = model.predict(X_test)
        print('RandomForest experimental results:')
        print(classification_report(y_test, y_pred, digits=4))
        # Save to file
        with open('classification_report.txt', 'a+') as f:
            f.write('RandomForest experimental results:')
            f.write(classification_report(y_test, y_pred, digits=4))
            f.write('\n')

    with open(file_path + r'\\catboost.pkl', 'rb') as file:
        X_test_str = X_test.copy()
        for col in categorical_cols:
            X_test_str[col] = X_test_str[col].astype(str)
        model = pickle.load(file)
        y_pred = model.predict(X_test_str)
        print('CatBoost experimental results:')
        print(classification_report(y_test, y_pred, digits=4))
        # Save to file
        with open('classification_report.txt', 'a+') as f:
            f.write('CatBoost experimental results:')
            f.write(classification_report(y_test, y_pred, digits=4))
            f.write('\n')
    
    # Perform ordinal encoding on discrete features first, then proceed with normalization processing.
    # Load Encoder
    ordinal_encoder = pickle.load(open(file_path + r'ordinal_encoder.pkl', 'rb'))
    X_test[categorical_cols] = ordinal_encoder.transform(X_test[categorical_cols])
    # Load Standardizer
    scaler = pickle.load(open(file_path + r'scaler.pkl', 'rb'))
    X_test = scaler.transform(X_test)

    with open(file_path + r'\\mlp.pkl', 'rb') as file:
        model = pickle.load(file)
        y_pred = model.predict(X_test)
        print('MLP experimental results:')
        print(classification_report(y_test, y_pred, digits=4))
        # Save to file
        with open('classification_report.txt', 'a+') as f:
            f.write('MLP experimental results:')
            f.write(classification_report(y_test, y_pred, digits=4))
            f.write('\n')         
