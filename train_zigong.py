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
    """Set the global random seed to override all random modules."""
    # Python's built-in random module
    random.seed(seed)
    # Numpy random module
    np.random.seed(seed)
    # Pytorch CPU random seed
    torch.manual_seed(seed)
    # Pytorch GPU random seed (single card)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Multi-card scenarios
    # Set environment variable seed
    os.environ["PYTHONHASHSEED"] = str(seed)

def train_all_models(X_train, X_test, y_train, y_test, Columns, continuous_cols, categorical_cols, target_col, model_save_pth, seed, type_con=""):

    # Set the global random seed
    set_global_seed(seed)
    # Using Transformer for classifying consciousness
    continuous_cols = continuous_cols
    categorical_cols = categorical_cols
    target_col = target_col
    # Merge X_train and y_train into a single DataFrame, with y_train as the consciousness column
    train_data = pd.DataFrame(X_train, columns=Columns)
    train_data[target_col] = y_train.values
    test_data = pd.DataFrame(X_test, columns=Columns)
    test_data[target_col] = y_test.values
    
    # save to file
    with open('classification_report.txt', 'a+') as f:
        f.write('-----------------' + str(datetime.now()) + '-----------------' + type_con + model_save_pth.split('\\')[6])
        f.write('\n')

    # Below is the code for the DANet model
    data_config = DataConfig(
        target = [target_col],
        continuous_cols = continuous_cols,
        categorical_cols = categorical_cols,
        normalize_continuous_features=True # Standardized continuous features 
    )
    # Optimizer Configuration
    optimizer_config = OptimizerConfig()
    # Trainer Configuration
    trainer_config = TrainerConfig(
        accelerator='gpu',
        auto_select_gpus=True,
        auto_lr_find=True,
        batch_size=2048,
        max_epochs=100,
        load_best=True, # Load the best model after training
        progress_bar="simple", # Progress bar type
        seed=seed
    )
    model_config = DANetConfig(
        task="classification",
        learning_rate=1e-3,
        metrics=["accuracy", "f1_score"],
        seed=seed                        # Set a random seed to ensure reproducibility
    )
    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        trainer_config=trainer_config,
        optimizer_config=optimizer_config
    )
    # Training model
    tabular_model.fit(
        train=train_data,
        seed=seed
    )
    y_pred = tabular_model.predict(test_data, tta_seed=seed)['consciousness_prediction']
    print('DANet experimental results：')
    print(classification_report(y_test, y_pred, digits=4))
    # Save to file
    with open('classification_report.txt', 'a+') as f:
        f.write('DANet experimental results')
        f.write(classification_report(y_test, y_pred, digits=4))
        f.write('\n')
    # Save model
    pickle.dump(tabular_model, open(model_save_pth + r'DANet.pkl', 'wb'))
    
    # Below is the code for the TabNet model
    data_config = DataConfig(
        target = [target_col],
        continuous_cols = continuous_cols,
        categorical_cols = categorical_cols,
        normalize_continuous_features=True, # Standardized continuous features
        num_workers=19
    )
    # Optimizer configuration
    optimizer_config = OptimizerConfig()
    # Trainer configuration
    trainer_config = TrainerConfig(
        auto_select_gpus=True,
        auto_lr_find=True,
        batch_size=2048,
        max_epochs=100,
        load_best=True, # Load the best model after training
        progress_bar="simple", # Progress bar type
        seed=seed
    )
    # TabNet configuration
    model_config = TabNetModelConfig(
        task="classification",
        learning_rate=1e-3,
        n_d = 64, # Decision-making layer width
        n_a = 64, # Attention layer width
        n_steps = 5, # Number of Decision steps
        gamma=1.5, # scaling factor
        mask_type="sparsemax", # Mask type
        n_shared=2, # Shared GLU layer count
        n_independent=2, # Number of independent GLU layers
        virtual_batch_size=2048,
        seed=seed
    )
    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        trainer_config=trainer_config,
        optimizer_config=optimizer_config
    )

    tabular_model.fit(
        train=train_data,
        seed=seed
    )
    y_pred = tabular_model.predict(test_data, tta_seed=seed)['consciousness_prediction']
    print('TabNet experimental results：')
    print(classification_report(y_test, y_pred, digits=4))
    
    with open('classification_report.txt', 'a+') as f:
        f.write('TabNet experimental results：')
        f.write(classification_report(y_test, y_pred, digits=4))
        f.write('\n')
    pickle.dump(tabular_model, open(model_save_pth + r'TabNet.pkl', 'wb'))
    
    # Below is the code for the FTTransformer model
    data_config = DataConfig(
        target=[target_col],
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        normalize_continuous_features=True # Standardized Continuous Features
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
    tabular_model.fit(
        train=train_data,
        seed=seed
    )
    y_pred = tabular_model.predict(test_data, tta_seed=seed)['consciousness_prediction']
    print('FTTransformer experimental results：')
    print(classification_report(y_test, y_pred, digits=4))
    with open('classification_report.txt', 'a+') as f:
        f.write('FTTransformer experimental results：')
        f.write(classification_report(y_test, y_pred, digits=4))
        f.write('\n')
    pickle.dump(tabular_model, open(model_save_pth + r'FTTransformer.pkl', 'wb'))
    
    # Below is the code for the TabTransformer model
    data_config = DataConfig(
        target=[target_col],
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        normalize_continuous_features=True # Standardized Continuous Features
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
    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        trainer_config=trainer_config,
        optimizer_config=optimizer_config
    )
    tabular_model.fit(
        train=train_data,
        seed=seed
    )
    y_pred = tabular_model.predict(test_data, tta_seed=seed)['consciousness_prediction']
    print('TabTransformer experimental results：')
    print(classification_report(y_test, y_pred, digits=4))
    with open('classification_report.txt', 'a+') as f:
        f.write('TabTransformer experimental results：')
        f.write(classification_report(y_test, y_pred, digits=4))
        f.write('\n')
    pickle.dump(tabular_model, open(model_save_pth + r'TabTransformer.pkl', 'wb'))
    
    ##################Machine Learning models#######################
    # Training a LightGBM model
    lgb = LGBMClassifier(
        num_leaves=32,
        n_estimators=100,
        learning_rate=0.1,
        random_state=seed,
        objective='multiclass',
        bagging_fraction=0.9,
        feature_fraction=0.9,
        bagging_seed=seed,
        feature_fraction_seed=seed,
        max_depth=6,
        class_weight='balanced'
    )
    lgb.fit(X_train, y_train, categorical_feature=categorical_cols)
    # Evaluation model
    y_pred = lgb.predict(X_test)
    print('LightGBM experimental results：')
    print(classification_report(y_test, y_pred, digits=4))
    with open('classification_report.txt', 'a+') as f:
        f.write('LightGBM experimental results：')
        f.write(classification_report(y_test, y_pred, digits=4))
        f.write('\n')
    pickle.dump(lgb, open(model_save_pth + r'lgb.pkl', 'wb'))

    # Training an XGBoost model
    xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=2.0,
        random_state=seed,
        objective='multi:softprob', # multi:softmax --> multi:softprob
        n_jobs=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        colsample_bylevel=0.9,
        seed=seed
    )
    xgb.fit(X_train, y_train)
    # Evaluation model
    y_pred = xgb.predict(X_test)
    print('XGBoost experimental results：')
    print(classification_report(y_test, y_pred, digits=4))
    with open('classification_report.txt', 'a+') as f:
        f.write('XGBoost experimental results：')
        f.write(classification_report(y_test, y_pred, digits=4))
        f.write('\n')
    pickle.dump(xgb, open(model_save_pth + r'xgboost.pkl', 'wb'))
    
    # Training a Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',
        n_jobs=-1,
        random_state=seed
    )
    rf_model.fit(X_train, y_train)
    # Evaluation model
    y_pred = rf_model.predict(X_test)
    print('RandomForest experimental results：')
    print(classification_report(y_test, y_pred, digits=4))
    with open('classification_report.txt', 'a+') as f:
        f.write('RandomForest experimental results：')
        f.write(classification_report(y_test, y_pred, digits=4))
        f.write('\n')
    pickle.dump(rf_model, open(model_save_pth + r'rf_model.pkl', 'wb'))
    
    # CatBoost
    model = CatBoostClassifier(
        bootstrap_type='Bayesian',
        verbose=50,  
        random_seed=seed,
        task_type="GPU",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6
    )
    X_train_str = X_train.copy()
    X_test_str = X_test.copy()
    for col in categorical_cols:
        X_train_str[col] = X_train_str[col].astype(str)
        X_test_str[col] = X_test_str[col].astype(str)
    model.fit(X_train_str, y_train, cat_features=categorical_cols)
    y_pred = model.predict(X_test_str)
    print('CatBoost experimental results：')
    print(classification_report(y_test, y_pred, digits=4))
    with open('classification_report.txt', 'a+') as f:
        f.write('CatBoost experimental results：')
        f.write(classification_report(y_test, y_pred, digits=4))
        f.write('\n')
    pickle.dump(model, open(model_save_pth + r'catboost.pkl', 'wb'))
    
    # Perform ordinal encoding on discrete features first, followed by normalization processing
    ordinal_encoder = OrdinalEncoder()
    X_train[categorical_cols] = ordinal_encoder.fit_transform(X_train[categorical_cols])
    X_test[categorical_cols] = ordinal_encoder.transform(X_test[categorical_cols])
    # Save encoder
    pickle.dump(ordinal_encoder, open(model_save_pth + r'ordinal_encoder.pkl', 'wb'))
    # Perform standardized processing
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Save Standardizer
    pickle.dump(scaler, open(model_save_pth + r'scaler.pkl', 'wb'))
    
    # Train a multilayer perceptron model and display the training progress
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(128, 32),
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=20,
        learning_rate='adaptive',
        batch_size=2048,
        activation='relu',
        random_state=seed,
        verbose=True
    )
    mlp_model.fit(X_train, y_train)
    # Evaluation model
    y_pred = mlp_model.predict(X_test)
    print('MLP model experimental results：')
    print(classification_report(y_test, y_pred, digits=4))
    with open('classification_report.txt', 'a+') as f:
        f.write('MLP model experimental results：')
        f.write(classification_report(y_test, y_pred, digits=4))
        f.write('\n')
    pickle.dump(mlp_model, open(model_save_pth + r'mlp.pkl', 'wb'))
