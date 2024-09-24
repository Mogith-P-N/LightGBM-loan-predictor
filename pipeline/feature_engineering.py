import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

categorical_cols = ['payFrequency', 'state', 'leadType', 'fpStatus', 'isRepeatCustomer']

def feature_engineering(input_path, output_path):
    loan_df_merged = pd.read_csv(input_path, low_memory=False)
    
    # Feature Engineering
    for col in categorical_cols:
        loan_df_merged[col] = loan_df_merged[col].astype(str)
    
    le = LabelEncoder()
    for col in categorical_cols:
        loan_df_merged[col] = le.fit_transform(loan_df_merged[col])
    
    loan_df_merged = loan_df_merged.dropna()
    loan_df_merged.to_csv(output_path, index=False)