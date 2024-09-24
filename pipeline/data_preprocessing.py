import pandas as pd
import numpy as np

default_statuses = ['Paid Off Loan', 'Settlement Paid Off', 'Pending Paid Off']
non_default_statuses = ['Charged Off', 'Charged Off Paid Off', 'External Collection', 'Internal Collection', 'Settled Bankruptcy']

def data_preprocessing(input_path, output_path):
    loan_df = pd.read_csv(input_path)
    
    # Transforming the loanStatus column to binary classification
    loan_df['target'] = loan_df['loanStatus'].apply(
    lambda x: 1 if x in default_statuses else 0 if x in non_default_statuses else None )
    
    #considering all other categories irrelevant
    loan_df = loan_df[loan_df['target'].notnull()]
    loan_df = loan_df.dropna()
    
    loan_df['isRepeatCustomer'] = loan_df['nPaidOff'] > 0
    loan_df['applicationDate'] = pd.to_datetime(loan_df['applicationDate'], errors='coerce')
    loan_df['originatedDate'] = pd.to_datetime(loan_df['originatedDate'], errors='coerce')
    loan_df['loanTermDays'] = (loan_df['originatedDate'] - loan_df['applicationDate']).dt.days
    loan_df['loanTermDays'] = loan_df['loanTermDays'].fillna(0) # Fill NaN with 0 for non-originated loans.
    
    loan_df.to_csv(output_path, index=False)
    print("Data Preprocessing Completed.")   
    
def data_aggregation(input_path, cuv_path, payment_path, output_path):
    cuv = pd.read_csv(cuv_path, usecols=['underwritingid', 'clearfraudscore'], low_memory=False)
    payment_df = pd.read_csv(payment_path, low_memory=False)
    loan_df = pd.read_csv(input_path, low_memory=False)
    
    # Aggregating the payment data
    loan_df_merged = pd.merge(loan_df, cuv, left_on='clarityFraudId', right_on='underwritingid', how='left')
    loan_df_merged = loan_df_merged.drop(columns=['underwritingid'])
    
    payment_df['paymentDate'] = pd.to_datetime(payment_df['paymentDate'], errors='coerce')
    payment_agg = payment_df.groupby('loanId').agg({
    'principal': 'sum',
    'fees': 'sum',
    'paymentAmount': 'sum',
    'paymentStatus': lambda x: (x == 'Rejected').sum(),# no of failed payments
}).rename(columns={'paymentStatus': 'numFailedPayments'})
    
    loan_df_merged = loan_df_merged.merge(payment_agg, on='loanId', how='left')
    
    #Drop unwanted columns
    loan_df_merged = loan_df_merged.drop(columns=[
        'loanId', 'applicationDate', 'originatedDate', 
        'loanStatus', 'clarityFraudId','anon_ssn'
        ])
    loan_df_merged.to_csv(output_path, index=False)
    

