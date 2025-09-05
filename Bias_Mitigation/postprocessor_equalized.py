import numpy as np
import pandas as pd

from Data_Acquisation_and_preprocessing.data_loader import log_status
from Bias_Detection.bias_metrices import confusion_matrix_by_grp, tpr_fpr_by_grp, compute_all_metrices
from Bias_Mitigation.preprocessor_reweighting import load_mitigation_config

def extract_positive_scores(y_proba):
    
    if hasattr(y_proba, "values"):
        y_proba= y_proba.values

    y_proba = np.asarray(y_proba)

    scores = y_proba.astype(float).ravel()

    scores = np.clip(scores, 0.0 , 1.0)
    print(f"These is the scores array : {scores}")
    print(f"Length of Scores {len(scores)}")

    return scores

def compute_global_tpr_fpr(scores, labels, threshold):
    labels = np.asarray(labels).ravel()
    scores = np.asarray(scores).ravel()

    if scores.shape[0] != labels.shape[0]:
        print("error")
    else:
        print("No error")

    labels = labels.astype(int)

    preds = (scores >= float(threshold)).astype(int)

    P = int((labels == 1).sum())
    N = int((labels == 0).sum())

    TP = int(((preds == 1) & (labels == 1)).sum())
    FP = int(((preds == 1) & (labels == 0)).sum())

    TPR_all = (TP / P)
    FPR_all = (FP / N)

    print(f"This is TPR : {TPR_all} and this is FPR : {FPR_all} inside the threshold")

    return float(TPR_all), float(FPR_all)