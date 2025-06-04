from data_loader import load_csv_data, validate_dataframe,extract_sensitive_columns,summarize_data,log_status
from data_cleaner import handle_missing_values,eliminate_outliers,standardize_col,encode_categorical_col,scale_numerical_col

file_path = r"D:\Coding\Projects\Bias_Auditor A state of the art Fairness Auditor for AI Hiring Systems\Dataset\synthetic_ai_hiring_dataset_v2.csv"
def load_and_preprocess_data(file_path):
    df = load_csv_data(file_path)
    print(validate_dataframe(df))
    
    non_numeric_col = ["gender","race","college_tier","education_level"]
    non_numeric_df = handle_missing_values(df,non_numeric_col,"Drop")

    numeric_col = ["years_experience", "skills_score","interview_score","test_score"]
    non_null_df = handle_missing_values(non_numeric_df,numeric_col,"Median")

    outlier_free = eliminate_outliers(non_null_df)
    standardized_df = standardize_col(outlier_free)
    encoded_df= encode_categorical_col(standardized_df,non_numeric_col)
    scaled_df = scale_numerical_col(encoded_df,numeric_col)

    return scaled_df

def split_labels(df):
    label_col = "hired"
    X = df.drop(columns = label_col)
    y = df[label_col]
    return X,y

def get_data(file_path):
    
    proccessed_df = load_and_preprocess_data(file_path)

    return split_labels(proccessed_df)
    
    
