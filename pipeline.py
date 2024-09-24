import os

from pipeline.data_ingestion import data_ingestion
from pipeline.data_preprocessing import data_preprocessing, data_aggregation
from pipeline.feature_engineering import feature_engineering
from pipeline.model_training import model_training
from pipeline.model_evaluation import model_evaluation
from pipeline.model_deployment import model_deployment

import threading

raw_loan_data_path = 'pipeline/data/raw/loan.csv'
raw_cuv_path = 'pipeline/data/raw/clarity_underwriting_variables.csv'
raw_payment_path = 'pipeline/data/raw/payment.csv'
processed_loan_data_path = 'pipeline/data/processed/loan_data_processed.csv'
aggregated_loan_data_path = 'pipeline/data/processed/loan_data_aggregated.csv'
feature_engineered_loan_data_path = 'pipeline/data/processed/loan_data_feature_engineered.csv'
model_path = 'pipeline/models/lgbm_model.txt'
results_path = 'pipeline/results/model_evaluation_results.txt'

def main():

    # Data Ingestion
    data_ingestion(raw_loan_data_path, processed_loan_data_path)
    
    # Data Preprocessing
    data_preprocessing(processed_loan_data_path, processed_loan_data_path)
    
    # Data Aggregation
    data_aggregation(processed_loan_data_path, raw_cuv_path, raw_payment_path, aggregated_loan_data_path)
    
    # Feature Engineering
    feature_engineering(aggregated_loan_data_path, feature_engineered_loan_data_path)
    
    # Model Training
    model_training(feature_engineered_loan_data_path, model_path)
    
    # Model Evaluation
    model_evaluation(feature_engineered_loan_data_path, model_path, results_path)
    
    # Model Deployment
    model_deployment(model_path)
    
if __name__ == '__main__':
    main()
        