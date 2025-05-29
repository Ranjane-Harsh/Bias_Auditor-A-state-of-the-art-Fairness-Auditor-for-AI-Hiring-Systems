from data_loader import load_csv_data, validate_dataframe, file_path
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

def handle_missing_values(df,subset_of_col,strategy):
    strategy = strategy.upper()
    if strategy == "DROP":
        new_df = df.dropna(subset=subset_of_col)
    elif strategy == "MEDIAN":
        for col in subset_of_col:
            df[col] = df[col].fillna(df[col].median())
        new_df = df
        
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
    
    print(f"{original_count - len(filtered_df)} rows removed as outliers")
    return filtered_df

def standardize_col(df):
    standarized_columns = [col.strip().lower().replace(" ","_") for col in df.columns]
    df.columns = standarized_columns
    return df

def encode_categorical_col(df,non_numeric_col):
    edu_order = { "Bachelors":0, "Masters":1, "PhD":2 }
    df["education_level"] = df["education_level"].map(edu_order)
    df = pd.get_dummies(df, columns = non_numeric_col, drop_first = True,dtype = float)
    return df

def scale_numerical_col(df,numeric_col):
    scaler = StandardScaler()
    scaled_val = scaler.fit_transform(df[numeric_col])
    scaled_df = pd.DataFrame(scaled_val, columns=numeric_col)

    return scaled_df

def main():
    data_frame = load_csv_data(file_path)
    non_numeric_col = ["gender","race","college_tier","education_level"]
    non_numeric_df = handle_missing_values(data_frame,non_numeric_col,"Drop")
    
    numeric_col = ["years_experience", "skills_score","interview_score","test_score"]
    non_null_df = handle_missing_values(non_numeric_df,numeric_col,"Median")
    print("The summary for non_empty dataframe looks like: ")
    print(validate_dataframe(non_null_df))

    outlier_free = eliminate_outliers(non_null_df)
    standardized_df = standardize_col(outlier_free)
    encoded_df= encode_categorical_col(standardized_df,non_numeric_col)
    scaled_df = scale_numerical_col(encoded_df,numeric_col)
    print(scaled_df[numeric_col].describe())
    
if __name__ == "__main__":
    main()