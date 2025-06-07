from Data_Acquisation_and_preprocessing.data_loader import log_status
import pandas as pd
from sklearn.preprocessing import StandardScaler

def handle_missing_values(df,subset_of_col,strategy):
    strategy = strategy.upper()
    new_df = df.copy()
    if strategy == "DROP":
        new_df = df.dropna(subset=subset_of_col)
    elif strategy == "MEDIAN":
        for col in subset_of_col:
            if col in new_df.columns and pd.api.types.is_numeric_dtype(new_df[col]):
                new_df.loc[:, col] = new_df[col].fillna(new_df[col].median())
        
    return new_df

def eliminate_outliers(df):
    original_count = len(df)
    numeric_col = df.select_dtypes(include = 'number').columns
    mask = pd.Series([True] * len(df), index=df.index)
    for col in numeric_col:

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        IQR = q3 - q1

        lower_bound = q1 - 1.5* IQR
        upper_bound = q3 + 1.5* IQR

        mask &= (df[col] >= lower_bound) & (df[col] <= upper_bound)

    filtered_df = df[mask]
    
    log_status("INFO",f"{original_count - len(filtered_df)} rows removed as outliers")
    return filtered_df

def standardize_col(df):
    standarized_columns = [col.strip().lower().replace(" ","_") for col in df.columns]
    df.columns = standarized_columns
    log_status("INFO","Column names are standardized")
    return df

def encode_categorical_col(df, non_numeric_col):
    edu_order = { "Bachelors":0, "Masters":1, "PhD":2 }
    df = df.copy()  # Ensure you're working on a copy
    df.loc[:, "education_level"] = df["education_level"].map(edu_order)
    df = pd.get_dummies(df, columns=non_numeric_col, drop_first=True, dtype=float)
    log_status("INFO", "Non-numeric values encoded")
    return df

def scale_numerical_col(df,numeric_col):
    scaler = StandardScaler()
    scaled_val = scaler.fit_transform(df[numeric_col])
    scaled_df = df.copy()
    scaled_df[numeric_col] = scaled_val
    log_status("INFO","Numeric values scaled")
    return scaled_df
