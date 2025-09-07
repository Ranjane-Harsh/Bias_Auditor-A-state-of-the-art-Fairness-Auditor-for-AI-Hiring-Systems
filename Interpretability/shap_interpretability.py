
import numpy as np
import pandas as pd
import shap
from collections import Counter


def _choose_explainer(model, X_background):
    try:
        return shap.TreeExplainer(model)
    except Exception:
        pass
    try:
        return shap.LinearExplainer(model, X_background)
    except Exception:
        pass
    background = shap.kmeans(X_background, min(50, len(X_background)))
    return shap.KernelExplainer(lambda x: model.predict_proba(x)[:, 1], background)


def _extract_positive_class_shap(shap_values):
    if isinstance(shap_values, (list, tuple)):
        if len(shap_values) == 2:  # binary
            return np.array(shap_values[1])
        else:  # multiclass → pick last class
            return np.array(shap_values[-1])

    shap_values = np.array(shap_values)

    # Case 2: 3D array (n, features, classes) → pick positive class
    if shap_values.ndim == 3:
        # Assume binary classification: take class 1
        return shap_values[:, :, 1]

    # Case 3: already 2D
    if shap_values.ndim == 2:
        return shap_values


def compute_shap_values(model, X):
    explainer = _choose_explainer(model, X)
    shap_vals_raw = explainer.shap_values(X)
    shap_vals = _extract_positive_class_shap(shap_vals_raw)
    return pd.DataFrame(shap_vals, columns=X.columns, index=X.index)


def compute_local_topk_for_hired(shap_df, y_pred, y_proba, threshold=0.5, top_k=3):
    if y_pred is None:
        hired_mask = np.asarray(y_proba) >= threshold
    else:
        hired_mask = np.asarray(y_pred) == 1

    hired_idx = np.where(hired_mask)[0]
    feature_counter = Counter()

    for i in hired_idx:
        row = shap_df.iloc[i]
        pos = row[row > 0]
        top = pos.sort_values(ascending=False).head(top_k)
        feature_counter.update(top.index.tolist())

    most_common_feats = feature_counter.most_common(top_k)
    return most_common_feats


def compute_group_shap_stats(shap_df, sensitive_df):
    results = {}
    for col in sensitive_df.columns:
        groups = sensitive_df[col].astype(str)
        mean_signed = shap_df.join(groups.rename('__group')).groupby('__group').mean()
        mean_abs = shap_df.abs().join(groups.rename('__group')).groupby('__group').mean()
        disparity_abs = (mean_abs.max(axis=0) - mean_abs.min(axis=0)).sort_values(ascending=False)
        disparity_signed = (mean_signed.max(axis=0) - mean_signed.min(axis=0)).sort_values(ascending=False)
        results[col] = {
            'mean_signed': mean_signed,
            'mean_abs': mean_abs,
            'disparity_abs': disparity_abs,
            'disparity_signed': disparity_signed
        }
    return results


def run_interpretability_pipeline(model, X, y_pred, y_proba, sensitive_df, top_k=3, threshold=0.5):
    print("\n=== Running SHAP Interpretability Pipeline ===")
    shap_df = compute_shap_values(model, X)

    # Approach 1: aggregated local explanations for hired
    print("\n--- Aggregated Local Explanations (Hired candidates) ---")
    common_feats = compute_local_topk_for_hired(shap_df, y_pred, y_proba, threshold=threshold, top_k=top_k)
    if not common_feats:
        print("No hired candidates found.")
    else:
        print(f"Most common top-{top_k} features across hired candidates:")
        for feat, count in common_feats:
            print(f"  {feat}: occurred {count} times")

    # Approach 2: group-level SHAP stats
    print("\n--- Group-level SHAP Statistics ---")
    group_results = compute_group_shap_stats(shap_df, sensitive_df)
    for attr, stats in group_results.items():
        print(f"\nAttribute: {attr}")
        print("Mean signed SHAP values (per group, first 5 features):")
        print(stats['mean_signed'].iloc[:, :5])
        print("\nMean absolute SHAP values (per group, first 5 features):")
        print(stats['mean_abs'].iloc[:, :5])
        print("\nTop 5 disparity features (by mean abs SHAP):")
        print(stats['disparity_abs'].head(5))

    print("\n=== SHAP Interpretability Completed ===")
    return shap_df, common_feats, group_results
