from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, demographic_parity_difference,equalized_odds_difference
from Data_Acquisation_and_preprocessing.data_loader import log_status

def evaluate_performance(y_true,y_pred):
    y_true_arr = np.asarray(y_true).ravel()
    y_pred_arr = np.asarray(y_pred).ravel()

    if y_true_arr.shape != y_pred_arr.shape:
        log_status("ERROR",f"Shape_mismatch - y_true_shape = {y_true_arr.shape()} | y_pred_shape = {y_pred_arr.shape()}")

    metrices = {
        "Accuracy" : float(accuracy_score(y_true_arr,y_pred_arr)),
        "Precision" : float(precision_score(y_true_arr,y_pred_arr, zero_division=0)),
        "Recall" : float(recall_score(y_true_arr,y_pred_arr,zero_division=0)),
        "f1_score" : float(f1_score(y_true_arr,y_pred_arr,zero_division=0))
    }
    return metrices

def evaluate_fairness(y_true, y_pred,sensitive_df):
    y_true_arr = np.asarray(y_true).ravel()
    y_pred_arr = np.asarray(y_pred).ravel()

    fairness_data = []

    for feature_name in sensitive_df.columns:
        feature_values = sensitive_df[feature_name]

        group_metrices = MetricFrame(metrics={"Accuracy": accuracy_score, "Recall": recall_score},
                                     y_true=y_true_arr,
                                     y_pred=y_pred_arr,sensitive_features=feature_values)
        
        dp_diff = demographic_parity_difference(y_true=y_true_arr,y_pred=y_pred_arr,sensitive_features=feature_values)
        eo_diff = equalized_odds_difference(y_true=y_true_arr,y_pred=y_pred_arr,sensitive_features=feature_values)

        fairness_data.append({
            "Sensitive Feature": feature_name,
            "Demographic Parity Difference": float(dp_diff),
            "Equalized Odds Difference": float(eo_diff)
        })
    
    
    fairness_df = pd.DataFrame(fairness_data)
    return fairness_df