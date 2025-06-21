import numpy as np
import pandas as pd

def normalize_input(y_true, y_pred, columns_values):
    y_true_arr = np.asarray(y_true).ravel()
    y_pred_arr = np.asarray(y_pred).ravel()
    col_val_arr = np.asarray(columns_values)

    return y_true_arr,y_pred_arr, col_val_arr

def confusion_matrix_by_grp(y_true_arr, y_pred_arr, col_val_arr):
    results = {}
    unique_values = np.unique(col_val_arr)
    for unique_val in unique_values:
        mask = (col_val_arr == unique_val)

        y_t_unique_val = y_true_arr[mask]
        y_p_unique_val = y_pred_arr[mask]

        TP = np.sum((y_t_unique_val == 1) & (y_p_unique_val == 1))
        FP = np.sum((y_t_unique_val == 0) & (y_p_unique_val == 1))
        TN = np.sum((y_t_unique_val == 0) & (y_p_unique_val == 0))
        FN = np.sum((y_t_unique_val == 1) & (y_p_unique_val == 0))
        results[unique_val] = {'TP': int(TP),
                      'FP': int(FP),
                      'TN':int(TN),
                      'FN':int(FN)}

    return results

def selection_rate_by_grp(y_pred_arr,col_val_arr):
    rates = {}
    unique_values = np.unique(col_val_arr)
    for unique_val in unique_values:
        mask = (col_val_arr == unique_val)
        count_val = mask.sum()
        positive_counts = y_pred_arr[mask].sum()
        rate = positive_counts/count_val
        rates[unique_val] = rate

    rates = pd.Series(rates)
    return rates

def demographic_parity_difference(rates):
    max_rate = rates.max()
    min_rate = rates.min()
    difference = max_rate - min_rate
    results = {'Selection_rate':rates,'Max_rate':max_rate,'Min_rate':min_rate,'Difference':difference}
    print(results)
    return results

def compute_all_metrices(y_true, y_pred, sensitive_df):
    if len(y_true) == len(y_pred) == len(sensitive_df):
        print("Shape of all parameters match")
    else:
        print("Shape mismatch error")

    for column in sensitive_df.columns:
        columns_values = sensitive_df[column]
        y_true_arr, y_pred_arr, col_val_arr = normalize_input(y_true,y_pred,columns_values)
        confusion_matrix_by_grp(y_true_arr,y_pred_arr,col_val_arr) 
        rates = selection_rate_by_grp(y_pred_arr,col_val_arr)
        demographic_parity_difference(rates)

    return 