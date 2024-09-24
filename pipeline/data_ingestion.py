# scripts/data_ingestion.py

import os
import pandas as pd

def data_ingestion(input_path, output_path):
    df = pd.read_csv(input_path)
    df.to_csv(output_path, index=False)
    print("Initial Data Ingestion Completed.")
