import pandas as pd
import numpy as np

import lightgbm as lgb

from .params.hyperparameters import PARAMS, K_FOLD

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def model_training(input_path, output_path):
    
    loan_df_merged = pd.read_csv(input_path)
    print(loan_df_merged.head())
    X = loan_df_merged.drop(columns=['target'])
    y = loan_df_merged['target']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    train_data = lgb.Dataset(X_train, label=y_train)
    
    skf = StratifiedKFold(n_splits=K_FOLD, shuffle=True, random_state=42)
    fold_auc_scores = []
    fold = 1

    for train_index, val_index in skf.split(X_train, y_train):
        print(f'Fold : {fold}/{K_FOLD}')

        X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

        train_data = lgb.Dataset(X_tr, label=y_tr)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        lgbm_model = lgb.train(
            PARAMS,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'valid'],
        )

        # Predict on val set
        y_pred = lgbm_model.predict(X_val, num_iteration=lgbm_model.best_iteration)

        # Eval
        auc = roc_auc_score(y_val, y_pred)
        print(f'AUC for fold {fold}: {auc:.4f}')
        fold_auc_scores.append(auc)

        fold += 1
    
    average_auc = np.mean(fold_auc_scores)
    std_auc = np.std(fold_auc_scores)
    print(f'Average AUC over {K_FOLD} folds: {average_auc:.4f}')
    print(f'Std Dev AUC over {K_FOLD} folds: {std_auc:.4f}')
        
    # save the model
    lgbm_model.save_model(output_path)