import pandas as pd
import numpy as np

import lightgbm as lgb

from .params.hyperparameters import PARAMS, K_FOLD, TEST_SIZE

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def model_evaluation(input_path, input_model_path, result_path):
    
    loan_df_merged = pd.read_csv(input_path)
    X = loan_df_merged.drop(columns=['target'])
    y = loan_df_merged['target']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=42
    )

    lgbm_model = lgb.Booster(model_file=input_model_path)
    y_pred_prob = lgbm_model.predict(X_test, num_iteration=lgbm_model.best_iteration)

    # Convert probabilities to binary output with 0.5 threshold
    y_pred = (y_pred_prob >= 0.5).astype(int)

    print("\n****** Results on the test set:************\n")
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nPrecision:", precision_score(y_test, y_pred))
    print("\nRecall:", recall_score(y_test, y_pred))
    print("\nF1 score:", f1_score(y_test, y_pred))
    print("\nAUC-ROC:", roc_auc_score(y_test, y_pred_prob))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    with open(result_path, 'w') as f:
        f.write(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")
        f.write(f"Precision: {precision_score(y_test, y_pred)}\n")
        f.write(f"Recall: {recall_score(y_test, y_pred)}\n")
        f.write(f"F1 score: {f1_score(y_test, y_pred)}\n")
        f.write(f"AUC-ROC: {roc_auc_score(y_test, y_pred_prob)}\n")
        f.write(f"Classification Report:\n {classification_report(y_test, y_pred)}\n")
        
    print(f"Model evaluation completed. Results saved at {result_path}")