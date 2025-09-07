import numpy as np
import pandas as pd
import warnings

from Bias_Detection.bias_metrices import compute_all_metrices
from Bias_Mitigation.preprocessor_reweighting import load_mitigation_config

def extract_positive_scores(y_proba):
    
    if hasattr(y_proba, "values"):
        y_proba= y_proba.values

    y_proba = np.asarray(y_proba)

    scores = y_proba.astype(float).ravel()

    scores = np.clip(scores, 0.0 , 1.0)

    return scores

def compute_global_tpr_fpr(scores, labels, threshold):
    labels = np.asarray(labels).ravel()
    scores = np.asarray(scores).ravel()

    labels = labels.astype(int)

    preds = (scores >= float(threshold)).astype(int)

    P = int((labels == 1).sum())
    N = int((labels == 0).sum())

    TP = int(((preds == 1) & (labels == 1)).sum())
    FP = int(((preds == 1) & (labels == 0)).sum())

    TPR_all = (TP / P)
    FPR_all = (FP / N)

    return float(TPR_all), float(FPR_all)

def compute_equalized_thresholds_for_attribute(scores, labels, group_values, epsilon, min_group_size, reference_threshold):
    scores = np.asarray(scores).ravel()
    labels = np.asarray(labels).ravel()
    groups = np.asarray(group_values).ravel()

    if not (scores.size == labels.size == groups.size):
        raise ValueError("Size of Scores, Labels and Group values does not Match")
    
    def tpr_fpr_for_arrays(s_arr: np.ndarray, y_arr: np.ndarray, thr: float):
        preds = (s_arr >= thr).astype(int)
        P = int((y_arr == 1).sum())
        N = int((y_arr == 0).sum())
        TP = int(((preds == 1) & (y_arr == 1)).sum())
        FP = int(((preds == 1) & (y_arr == 0)).sum())
        tpr = TP / P if P > 0 else 0.0
        fpr = FP / N if N > 0 else 0.0
        return float(tpr), float(fpr)
    
    TPR_all, FPR_all = tpr_fpr_for_arrays(scores, labels, reference_threshold)

    candidates = np.unique(scores)
    if reference_threshold not in candidates:
        
        candidates = np.sort(np.concatenate((candidates, np.array([reference_threshold]))))

    unique_groups_in_order = list(dict.fromkeys(groups.tolist()))

    thresholds = {}

    for g in unique_groups_in_order:
        idx = np.where(groups == g)[0]
        n_g = idx.size

        if n_g < min_group_size:
            thresholds[g] = float(reference_threshold)
            continue

        s_g = scores[idx]
        y_g = labels[idx]

        P_g = int((y_g == 1).sum())
        N_g = int((y_g == 0).sum())

        best_thr = float(reference_threshold)
        best_violation = float("inf")
        best_acc = -1.0

        if P_g == 0 or N_g == 0:
        
            for thr in candidates:
                tpr_g, fpr_g = tpr_fpr_for_arrays(s_g, y_g, thr)
                if P_g == 0:
                    violation = abs(fpr_g - FPR_all)
                else:
                    violation = abs(tpr_g - TPR_all)

                preds = (s_g >= thr).astype(int)
                acc = float((preds == y_g).mean()) if s_g.size > 0 else 0.0

                if (violation < best_violation) or (np.isclose(violation, best_violation) and acc > best_acc):
                    best_violation = violation
                    best_thr = float(thr)
                    best_acc = acc

            thresholds[g] = best_thr
            continue

        for thr in candidates:
            tpr_g, fpr_g = tpr_fpr_for_arrays(s_g, y_g, thr)
            violation = max(abs(tpr_g - TPR_all), abs(fpr_g - FPR_all))

            preds = (s_g >= thr).astype(int)
            acc = float((preds == y_g).mean()) if s_g.size > 0 else 0.0

            if (violation < best_violation) or (np.isclose(violation, best_violation) and acc > best_acc):
                best_violation = violation
                best_thr = float(thr)
                best_acc = acc
                if np.isclose(best_violation, 0.0):
                    break

        thresholds[g] = best_thr

    return thresholds

def apply_equalized_odds_for_attribute(scores, thresholds, group_values, default_threshold = 0.5, return_series = None):

    is_scores_series = hasattr(scores, "index") and not isinstance(scores, np.ndarray)
    is_groups_series = hasattr(group_values, "index") and not isinstance(group_values, np.ndarray)

    orig_index = None
    if is_groups_series:
        orig_index = group_values.index
    elif is_scores_series:
        orig_index = scores.index

    if hasattr(scores, "values"):
        s_arr = np.asarray(scores.values).ravel()
    else:
        s_arr = np.asarray(scores).ravel()

    if hasattr(group_values, "values"):
        g_arr = np.asarray(group_values.values).ravel()
    else:
        g_arr = np.asarray(group_values).ravel()

    
    if s_arr.ndim != 1:
        if s_arr.ndim == 2 and s_arr.shape[1] == 2:
            s_arr = s_arr[:, 1].ravel()
        else:
            raise ValueError(f"`scores` must be 1-D (or shape (N,2) from predict_proba); got shape {s_arr.shape}")

    if s_arr.shape[0] != g_arr.shape[0]:
        raise ValueError(f"`scores` and `group_values` must have same length: {s_arr.shape[0]} vs {g_arr.shape[0]}")

    y_post = np.zeros_like(s_arr, dtype=np.int64)

    unique_groups = np.unique(g_arr)

    thr_map = dict(thresholds)  
    
    lower_map = {str(k).lower(): v for k, v in thr_map.items()}

    for g in unique_groups:
        mask = (g_arr == g)
        if not np.any(mask):
            continue  

        thr = thr_map.get(g, None)
        if thr is None:
            try:
                thr = thr_map.get(str(g), None)
            except Exception:
                thr = None
        if thr is None:
            
            thr = lower_map.get(str(g).lower(), None)

        if thr is None:
            
            warnings.warn(
                f"No threshold found for group '{g}'. Using default_threshold={default_threshold}. "
                "Consider adding an explicit threshold for this group.",
                UserWarning,
            )
            thr = float(default_threshold)
        else:
            thr = float(thr)

        y_post[mask] = (s_arr[mask] >= thr).astype(np.int64)

    if return_series is None:
        return_series = is_scores_series or is_groups_series

    if return_series:
        if orig_index is None:
            orig_index = pd.RangeIndex(start=0, stop=len(y_post))
        return pd.Series(y_post, index=orig_index, name="y_post").astype(int)

    return y_post.astype(int)

def map_config_sensitive_names_to_df_columns(sensitive_attributes, sensitive_df):
    
    if not isinstance(sensitive_attributes, (list, tuple)):
        raise ValueError("sensitive_attributes must be a list of names")

    df_cols = list(sensitive_df.columns)
    norm_to_col = {}
    for c in df_cols:
        key = str(c).strip().lower().replace(" ", "_")
        norm_to_col[key] = c

    mapping = {}
    for cfg_name in sensitive_attributes:
        if cfg_name is None:
            continue
        key = str(cfg_name).strip().lower().replace(" ", "_")
        if key in norm_to_col:
            mapping[cfg_name] = norm_to_col[key]
            continue
            
        alt = key.replace("-", "_").rstrip("s")
        if alt in norm_to_col:
            mapping[cfg_name] = norm_to_col[alt]
            continue
            
        matches = [c for c in df_cols if str(c).strip().lower() == str(cfg_name).strip().lower()]
        if len(matches) == 1:
            mapping[cfg_name] = matches[0]
            continue

    return mapping

def run_equalized_odds_postprocessing(y_true, y_proba, mitigation_cfg, reference_threshold, sensitive_df):

    stages = mitigation_cfg.get("stages", [])
    sensitive_attributes_cfg = mitigation_cfg.get("sensitive_attributes", [])
    eq_cfg = mitigation_cfg.get("equalized_odds", [])
    epsilon = float(eq_cfg.get("epsilon", 0.02))
    min_group_size = int(eq_cfg.get("min_group_size", 50))

    scores = extract_positive_scores(y_proba)

    scores = np.asarray(scores).ravel()
    y_true = np.asarray(y_true).ravel()

    mapping = map_config_sensitive_names_to_df_columns(sensitive_attributes_cfg, sensitive_df)

    print("\n=== Equalized Odds Postprocessing Results ===")

    results = { "config": {"epsilon": epsilon, "min_group_size": min_group_size, "reference_threshold": reference_threshold}, "attributes": {} }

    y_before_global = (scores >= reference_threshold).astype(int)

    thresholds_by_attribute = {}
    results_data = []

    for cfg_name, df_col in mapping.items():
        print(f"\n--- Analysis for {df_col} ---")
        group_values = sensitive_df[df_col].to_numpy()

        thresholds = compute_equalized_thresholds_for_attribute(scores, y_true, group_values, epsilon, min_group_size, reference_threshold)
        y_post = apply_equalized_odds_for_attribute(scores, thresholds, group_values, reference_threshold, return_series=True)
        group_df_single = sensitive_df[[df_col]].copy()
        metrics_before = compute_all_metrices(y_true, y_before_global, group_df_single)
        y_post_arr = y_post.values if isinstance(y_post, pd.Series) else np.asarray(y_post)
        metrics_after = compute_all_metrices(y_true, y_post_arr, group_df_single)
        
        for group, threshold in thresholds.items():
            row_data = {
        'Attribute': df_col,
        'Group': group,
        'Threshold': threshold,
        'Selection_Rate_Before': metrics_before[df_col].loc['Selection_rate', group],
        'Selection_Rate_After': metrics_after[df_col].loc['Selection_rate', group],
        'DP_Diff_before': metrics_before[df_col].loc['Demographic_Parity_Difference', 'Difference'],
        'DP_Diff_after': metrics_after[df_col].loc['Demographic_Parity_Difference', 'Difference'],
        'TPR_Before': metrics_before[df_col].loc['True_postive_rate', group],
        'TPR_After': metrics_after[df_col].loc['True_postive_rate', group],
        'FPR_Before': metrics_before[df_col].loc['False_postive_rate', group],
        'FPR_After': metrics_after[df_col].loc['False_postive_rate', group],
        'TPR_Diff_Before': metrics_before[df_col].loc['True_positive_diff', 'Difference'],
        'TPR_Diff_After': metrics_after[df_col].loc['True_positive_diff', 'Difference'],
        'FPR_Diff_Before': metrics_before[df_col].loc['False_positive_diff', 'Difference'],
        'FPR_Diff_After': metrics_after[df_col].loc['False_positive_diff', 'Difference']
            }
            results_data.append(row_data)

    results_df = pd.DataFrame(results_data)

    print("\n=== Configuration ===")
    print(f"Epsilon: {epsilon}")
    print(f"Min Group Size: {min_group_size}")
    print(f"Reference Threshold: {reference_threshold}")

    print("\n=== Results ===")
    # Format float values to 4 decimal places
    float_cols = results_df.select_dtypes(include=['float64']).columns
    results_df[float_cols] = results_df[float_cols].round(4)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(results_df.to_string(index=False))

    thresholds_by_attribute[df_col] = thresholds

    results["thresholds_by_attribute"] = thresholds_by_attribute

    return results_df