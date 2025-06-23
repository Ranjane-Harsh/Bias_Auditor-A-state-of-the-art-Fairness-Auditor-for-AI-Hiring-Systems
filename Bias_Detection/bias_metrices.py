from Data_Acquisation_and_preprocessing.data_loader import log_status
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
    results = {'Selection_rate': rates,'Max_rate':max_rate,'Min_rate':min_rate,'Difference':difference}
    return results

def disparate_impact_ratios(y_pred_arr, rates):
    overall_rate = y_pred_arr.mean()
    ratios = rates / overall_rate
    return ratios

def tpr_fpr_by_grp(Confusion_matrix):
    tpr_dict = {}
    fpr_dict = {}

    for val,counts in Confusion_matrix.items():
        TP = counts['TP']
        FN = counts['FN']
        FP = counts['FP']
        TN = counts['TN']

        if (TP + FN) > 0:
            tpr = TP / (TP + FN)
        else:
            tpr = np.nan

        if (FP + TN) > 0:
            fpr = FP / (FP + TN)
        else:
            fpr = np.nan

        tpr_dict[val] = tpr
        fpr_dict[val] = fpr

    tpr_series = pd.Series(tpr_dict, name = "TPR")
    fpr_series = pd.Series(fpr_dict, name = "FPR")

    return tpr_series, fpr_series

def equalized_odds_difference(tpr_series, fpr_series):
    tpr_diff = tpr_series.max() - tpr_series.min()
    fpr_diff = fpr_series.max() - fpr_series.min()
    results = { 'TPR_difference':tpr_diff, 'FPR_difference': fpr_diff }

    return results

def bias_summary_for_attribute(y_true_arr, y_pred_arr,col_val_arr):
    summary_data = {}
    confusion_matrix = confusion_matrix_by_grp(y_true_arr,y_pred_arr,col_val_arr)
    rates = selection_rate_by_grp(y_pred_arr,col_val_arr)
    dp_results = demographic_parity_difference(rates)
    di_ratio = disparate_impact_ratios(y_pred_arr,rates)
    tpr_series, fpr_series = tpr_fpr_by_grp(confusion_matrix)
    eo_results = equalized_odds_difference(tpr_series,fpr_series)

    summary_data['Selection_rate'] = rates.to_dict()
    summary_data['Demographic_Parity_Difference'] = {**{g: np.nan for g in rates.index}, 'Difference': dp_results['Difference']}
    summary_data['Disparate_impact_ratio'] = di_ratio.to_dict()
    summary_data['True_postive_rate'] = tpr_series.to_dict()
    summary_data['True_positive_diff'] = {**{g: np.nan for g in tpr_series.index}, 'Difference': eo_results['TPR_difference']}
    summary_data['False_postive_rate'] = fpr_series.to_dict()
    summary_data['False_positive_diff'] = {**{g: np.nan for g in fpr_series.index}, 'Difference': eo_results['FPR_difference']}

    summary_df = pd.DataFrame.from_dict(summary_data, orient= 'index')
    return summary_df

def compute_all_metrices(y_true, y_pred, sensitive_df):
    if len(y_true) == len(y_pred) == len(sensitive_df):
        log_status("INFO","Shape of all parameters match")
    else:
        log_status("ERROR","Shape mismatch error")

    for column in sensitive_df.columns:
        columns_values = sensitive_df[column]
        y_true_arr, y_pred_arr, col_val_arr = normalize_input(y_true,y_pred,columns_values)

        print(f"This is the Metrices Dataframe for : {column}")
        summary_df = bias_summary_for_attribute(y_true_arr,y_pred_arr,col_val_arr)
        print(summary_df)
        print("\n")
    return 