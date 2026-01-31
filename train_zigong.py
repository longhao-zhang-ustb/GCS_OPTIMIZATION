import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from imblearn.over_sampling import BorderlineSMOTE, SMOTE, SVMSMOTE
from imblearn.under_sampling import AllKNN, RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from pytorch_tabular import TabularModel
from pytorch_tabular.models import TabTransformerConfig, FTTransformerConfig, TabNetModelConfig, DANetConfig
from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import os
import pickle
from datetime import datetime
import pytorch_tabular as pt
import random
import numpy as np
import torch

os.environ['TF_USE_LEGACY_KERAS'] = '0'

def set_global_seed(seed):
    """全局设置随机种子，覆盖所有随机模块"""
    # Python原生随机模块
    random.seed(seed)
    # Numpy随机模块
    np.random.seed(seed)
    # Pytorch CPU随机种子
    torch.manual_seed(seed)
    # Pytorch GPU随机种子（单卡）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多卡场景
    # 设置环境变量种子
    os.environ["PYTHONHASHSEED"] = str(seed)

def train_all_models(X_train, X_test, y_train, y_test, Columns, continuous_cols, categorical_cols, target_col, model_save_pth, seed, type_con=""):

    # 设置全局随机种子
    set_global_seed(seed)
    # 使用Transformer模型分类consciousness
    continuous_cols = continuous_cols
    categorical_cols = categorical_cols
    target_col = target_col
    # 合并X_train和y_train为一个DataFrame,y_train为consciousness列
    train_data = pd.DataFrame(X_train, columns=Columns)
    train_data[target_col] = y_train.values
    test_data = pd.DataFrame(X_test, columns=Columns)
    test_data[target_col] = y_test.values
    
    # ##############设置保存结果##################
    # 保存到文件
    with open('classification_report.txt', 'a+') as f:
        # exp1，这里的[4] ==> [2]
        f.write('-----------------' + str(datetime.now()) + '-----------------' + type_con + model_save_pth.split('\\')[6])
        f.write('\n')

    # 以下是DANet模型的代码
    data_config = DataConfig(
        target = [target_col],
        continuous_cols = continuous_cols,
        categorical_cols = categorical_cols,
        normalize_continuous_features=True # 标准化连续特征
    )
    # 优化器配置
    optimizer_config = OptimizerConfig()
    # 训练器配置
    trainer_config = TrainerConfig(
        accelerator='gpu',
        auto_select_gpus=True,
        auto_lr_find=True,
        batch_size=2048,
        max_epochs=100,
        load_best=True, # 训练结束后加载最佳模型
        progress_bar="simple", # 进度条类型
        seed=seed
    )
    model_config = DANetConfig(
        task="classification",
        learning_rate=1e-3,
        metrics=["accuracy", "f1_score"],
        seed=seed                        # 设置随机种子以保证可重复性
    )
    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        trainer_config=trainer_config,
        optimizer_config=optimizer_config
    )
    # 训练模型
    tabular_model.fit(
        train=train_data,
        seed=seed
    )
    y_pred = tabular_model.predict(test_data, tta_seed=seed)['consciousness_prediction']
    print('DANet模型实验结果：')
    print(classification_report(y_test, y_pred, digits=4))
    # 保存到文件
    with open('classification_report.txt', 'a+') as f:
        f.write('DANet模型实验结果：')
        f.write(classification_report(y_test, y_pred, digits=4))
        f.write('\n')
    # 保存模型
    pickle.dump(tabular_model, open(model_save_pth + r'DANet.pkl', 'wb'))
    
    # 以下是TabNet模型的代码
    data_config = DataConfig(
        target = [target_col],
        continuous_cols = continuous_cols,
        categorical_cols = categorical_cols,
        normalize_continuous_features=True, # 标准化连续特征
        num_workers=19
    )
    # 优化器配置
    optimizer_config = OptimizerConfig()
    # 训练器配置
    trainer_config = TrainerConfig(
        auto_select_gpus=True,
        auto_lr_find=True,
        batch_size=2048,
        max_epochs=100,
        load_best=True, # 训练结束后加载最佳模型
        progress_bar="simple", # 进度条类型
        seed=seed
    )
    # TabNet模型配置
    model_config = TabNetModelConfig(
        task="classification",
        learning_rate=1e-3,
        n_d = 64, # 决策层宽度
        n_a = 64, # 注意力层宽度
        n_steps = 5, # 决策步骤数
        gamma=1.5, # 缩放系数
        mask_type="sparsemax", # 掩码类型
        n_shared=2, # 共享GLU层数
        n_independent=2, # 独立GLU层数
        virtual_batch_size=2048, # 虚拟批大小
        seed=seed
    )
    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        trainer_config=trainer_config,
        optimizer_config=optimizer_config
    )
    # 训练模型
    tabular_model.fit(
        train=train_data,
        seed=seed
    )
    y_pred = tabular_model.predict(test_data, tta_seed=seed)['consciousness_prediction']
    print('TabNet模型实验结果：')
    print(classification_report(y_test, y_pred, digits=4))
    # 保存模型
    # 保存到文件
    with open('classification_report.txt', 'a+') as f:
        f.write('TabNet模型实验结果：')
        f.write(classification_report(y_test, y_pred, digits=4))
        f.write('\n')
    pickle.dump(tabular_model, open(model_save_pth + r'TabNet.pkl', 'wb'))
    
    # 以下是FTTransformer模型的代码
    data_config = DataConfig(
        target=[target_col],
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        normalize_continuous_features=True # 标准化连续特征
    )
    trainer_config = TrainerConfig(
        auto_select_gpus=True,
        max_epochs=100,
        batch_size=2048,
        auto_lr_find=True,
        progress_bar='simple',
        load_best=True,
        seed=seed
    )
    optimizer_config = OptimizerConfig()
    model_config = FTTransformerConfig(
        task="classification",
        learning_rate=1e-3,
        input_embed_dim=128,
        num_heads=8,
        num_attn_blocks=6,
        attn_dropout=0.1,
        ff_dropout=0.1,
        seed=seed
    )
    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        trainer_config=trainer_config,
        optimizer_config=optimizer_config
    )
    # 训练模型
    tabular_model.fit(
        train=train_data,
        seed=seed
    )
    y_pred = tabular_model.predict(test_data, tta_seed=seed)['consciousness_prediction']
    print('FTTransformer模型实验结果：')
    print(classification_report(y_test, y_pred, digits=4))
    with open('classification_report.txt', 'a+') as f:
        f.write('FTTransformer模型实验结果：')
        f.write(classification_report(y_test, y_pred, digits=4))
        f.write('\n')
    pickle.dump(tabular_model, open(model_save_pth + r'FTTransformer.pkl', 'wb'))
    
    # 以下是TabTransformer模型的代码
    data_config = DataConfig(
        target=[target_col],
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        normalize_continuous_features=True # 标准化连续特征
    )
    
    trainer_config = TrainerConfig(
        auto_select_gpus=True,
        max_epochs=100,
        batch_size=2048,
        auto_lr_find=True,
        progress_bar='simple',
        load_best=True,
        seed=seed
    )
    optimizer_config = OptimizerConfig()
    model_config = TabTransformerConfig(
        task="classification",
        learning_rate=1e-3,
        input_embed_dim=128,
        num_heads=8,
        num_attn_blocks=6,
        attn_dropout=0.1,
        ff_dropout=0.1,
        seed=seed
    )
    # 创建并训练模型
    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        trainer_config=trainer_config,
        optimizer_config=optimizer_config
    )
    # 训练模型
    tabular_model.fit(
        train=train_data,
        seed=seed
    )
    y_pred = tabular_model.predict(test_data, tta_seed=seed)['consciousness_prediction']
    print('TabTransformer模型实验结果：')
    print(classification_report(y_test, y_pred, digits=4))
    with open('classification_report.txt', 'a+') as f:
        f.write('TabTransformer模型实验结果：')
        f.write(classification_report(y_test, y_pred, digits=4))
        f.write('\n')
    pickle.dump(tabular_model, open(model_save_pth + r'TabTransformer.pkl', 'wb'))
    
    ##################机器学习模型#######################
    # 训练lightgbm模型
    lgb = LGBMClassifier(
        num_leaves=32,   # 16 --> 32
        n_estimators=100, # 60 --> 100
        learning_rate=0.1, # 新增
        random_state=seed,
        objective='multiclass',
        bagging_fraction=0.9,
        feature_fraction=0.9,
        bagging_seed=seed,
        feature_fraction_seed=seed,
        max_depth=6, # 新增
        class_weight='balanced'   # 新增
    )
    lgb.fit(X_train, y_train, categorical_feature=categorical_cols)
    # 评估模型
    y_pred = lgb.predict(X_test)
    print('LightGBM模型实验结果：')
    print(classification_report(y_test, y_pred, digits=4))
    with open('classification_report.txt', 'a+') as f:
        f.write('LightGBM模型实验结果：')
        f.write(classification_report(y_test, y_pred, digits=4))
        f.write('\n')
    pickle.dump(lgb, open(model_save_pth + r'lgb.pkl', 'wb'))

    # 训练xgboost模型
    xgb = XGBClassifier(
        n_estimators=200, # 500 --> 200
        learning_rate=0.05,  # 新增
        max_depth=5,  # 新增
        min_child_weight=3,  # 新增
        reg_alpha=0.1,  # 新增
        reg_lambda=2.0,  # 新增
        random_state=seed,
        objective='multi:softprob', # multi:softmax --> multi:softprob
        n_jobs=-1,  # 新增
        subsample=0.9,
        colsample_bytree=0.9,
        colsample_bylevel=0.9,
        seed=seed
    )
    xgb.fit(X_train, y_train)
    # 评估模型
    y_pred = xgb.predict(X_test)
    print('XGBoost模型实验结果：')
    print(classification_report(y_test, y_pred, digits=4))
    with open('classification_report.txt', 'a+') as f:
        f.write('XGBoost模型实验结果：')
        f.write(classification_report(y_test, y_pred, digits=4))
        f.write('\n')
    pickle.dump(xgb, open(model_save_pth + r'xgboost.pkl', 'wb'))
    
    # 训练随机森林模型
    rf_model = RandomForestClassifier(
        n_estimators=150,  # 300 --> 150
        max_depth=10, # 新增
        min_samples_split=20, # 新增
        min_samples_leaf=10, # 新增
        class_weight='balanced', # 新增
        n_jobs=-1, # 新增
        random_state=seed
    )
    rf_model.fit(X_train, y_train)
    # 评估模型
    y_pred = rf_model.predict(X_test)
    print('RandomForest模型实验结果：')
    print(classification_report(y_test, y_pred, digits=4))
    with open('classification_report.txt', 'a+') as f:
        f.write('RandomForest模型实验结果：')
        f.write(classification_report(y_test, y_pred, digits=4))
        f.write('\n')
    pickle.dump(rf_model, open(model_save_pth + r'rf_model.pkl', 'wb'))
    
    # CatBoost模型的代码
    model = CatBoostClassifier(
        bootstrap_type='Bayesian', # 新增
        verbose=50,    # 100 --> 50
        random_seed=seed,
        task_type="GPU",
        n_estimators=500, # 新增
        learning_rate=0.05, # 新增
        max_depth=6 # 新增
    )
    X_train_str = X_train.copy()
    X_test_str = X_test.copy()
    for col in categorical_cols:
        X_train_str[col] = X_train_str[col].astype(str)
        X_test_str[col] = X_test_str[col].astype(str)
    model.fit(X_train_str, y_train, cat_features=categorical_cols)
    y_pred = model.predict(X_test_str)
    print('CatBoost模型实验结果：')
    print(classification_report(y_test, y_pred, digits=4))
    with open('classification_report.txt', 'a+') as f:
        f.write('CatBoost模型实验结果：')
        f.write(classification_report(y_test, y_pred, digits=4))
        f.write('\n')
    pickle.dump(model, open(model_save_pth + r'catboost.pkl', 'wb'))
    
    # 对离散特征先进行序数编码，在进行标准化处理
    ordinal_encoder = OrdinalEncoder()
    X_train[categorical_cols] = ordinal_encoder.fit_transform(X_train[categorical_cols])
    X_test[categorical_cols] = ordinal_encoder.transform(X_test[categorical_cols])
    # 保存编码器
    pickle.dump(ordinal_encoder, open(model_save_pth + r'ordinal_encoder.pkl', 'wb'))
    # 进行标准化处理
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # 保存标准化器
    pickle.dump(scaler, open(model_save_pth + r'scaler.pkl', 'wb'))
    
    # 训练多层感知器模型,并显示训练进度
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(128, 32), # (32, 128) --> (128, 32)
        max_iter=300, # 100 --> 300
        early_stopping=True,
        n_iter_no_change=20, # 新增
        learning_rate='adaptive', # 新增
        batch_size=2048, # 新增
        activation='relu', # 新增
        random_state=seed,
        verbose=True
    )
    mlp_model.fit(X_train, y_train)
    # 评估模型
    y_pred = mlp_model.predict(X_test)
    print('MLP模型实验结果：')
    print(classification_report(y_test, y_pred, digits=4))
    with open('classification_report.txt', 'a+') as f:
        f.write('MLP模型实验结果：')
        f.write(classification_report(y_test, y_pred, digits=4))
        f.write('\n')
    pickle.dump(mlp_model, open(model_save_pth + r'mlp.pkl', 'wb'))
