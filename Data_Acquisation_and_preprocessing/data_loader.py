import pandas as pd
from pathlib import Path
import logging

def log_status(level, message):
    logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
    level = level.upper()
    
    if level == 'INFO':
        logging.info(message)
    elif level == 'WARNING':
        logging.warning(message)
    elif level == 'ERROR':
        logging.error(message)
    elif level == 'CRITICAL':
        logging.critical(message)

def load_csv_data(file_path):
    if not Path(file_path).exists():
        log_status("Warning","File does not exist")
        return None
    
    try:
        data_frame = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error in reading the file: {e}")
        return None
    
    if data_frame.empty:
        log_status("WARNING","The loaded file is Empty")
        return None
    
    return data_frame


def validate_dataframe(data_frame):
    if data_frame.columns.duplicated().any():
        raise ValueError("Duplicate values found")

    if data_frame[["gender","race","college_tier","hired"]].isnull().any().any():
        log_status("CRITICAL","Some values from critical columns are missing")
        missing_precentage = data_frame[["gender","race","college_tier","hired"]].isnull().mean() * 100
        print(f"Percentage of missing values:\n ")
        print(missing_precentage)
    else:
        log_status("INFO","No missing values in critical columns")

    valid_hiring_values = {0,1}
    if not data_frame["hired"].isin(valid_hiring_values).all():
        raise ValueError("Invalid Hiring values detected")
    else:
        log_status("INFO","All hiring values for labels are valid")
    return 

def summarize_data(data_frame):
    return data_frame.describe(), data_frame.head()

def extract_sensitive_columns(data_frame,sensitive_col):
    for col in sensitive_col:
        if col not in data_frame.columns:
            raise ValueError(f"{col} not found")
    
    sensitive_df = data_frame[sensitive_col]
    return sensitive_df

