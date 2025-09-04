import yaml
import pandas as pd
import numpy as np

def load_mitigation_config(file_path):
    try:
        with open(file_path,"r") as f:
            cfg_raw = yaml.safe_load(f)
    except FileNotFoundError:
        raise ValueError("File not Found")

    mitigation_cfg = {
        'stages' : cfg_raw['stages'],
        'sensitive_attributes' : cfg_raw['sensitive_attributes'],
        'reweighing':{},
        'adversarial':{
            'hidden_layers': cfg_raw['adversarial'].get('hidden_layers',[64,32]),
            'weight_lambda': float(cfg_raw['adversarial'].get('weight_lambda',0.5)),
            'n_epochs': int(cfg_raw['adversarial'].get('n_epochs',10)),
            'batch_size' : int(cfg_raw['adversarial'].get('batch_size',128)),
            'lr_adv': float(cfg_raw['adversarial'].get('lr_adv',1e-4))
        },
        'equalized_odds': {
            'epsilon': float(cfg_raw['equalized_odds'].get('epsilon',0.02)),
            'min_group_size': int(cfg_raw['equalized_odds'].get('min_group_size',50))
        }
    }

    return mitigation_cfg

def compute_reweighing_weights(sensitive_df):
    N = len(sensitive_df)
    
    reweighing_weights = {}
    sensitive_columns = ["gender","race","college_tier","education_level"]
    for col in sensitive_columns:
        
        counts_attr = sensitive_df[col].value_counts()
        counts_label = pd.Series(sensitive_df["hired"]).value_counts()

        joint_df = sensitive_df.copy()
        dataframe = joint_df.groupby([col, "hired"]).size().reset_index(name = 'count')

        dataframe['ideal'] = (dataframe[col].map(counts_attr) * dataframe["hired"].map(counts_label)) / N

        dataframe['weight'] = dataframe['ideal'] / dataframe['count']

        for _, row in dataframe.iterrows():
            key = (col , row[col], row["hired"])
            reweighing_weights[key] = row['weight']

    return reweighing_weights

def apply_reweighting(df,sensitive_attributes, reweighing_weights, scaling_range):
    
    weights_df = pd.DataFrame(index=df.index)
    for attr in sensitive_attributes:
        per_attr_map = {}
        for (a, val, lab), w in reweighing_weights.items():
            if a == attr:
                per_attr_map[(val, lab)] = float(w)

        pairs = list(zip(df[attr].values, df["hired"].values))
        pairs_series = pd.Series(pairs, index = df.index)

        weights_df_col = pairs_series.map(per_attr_map)

        col_name = f"w_{attr}"
        weights_df[col_name] = weights_df_col

    combined = weights_df.prod(axis=1)

    lower_range, upper_range = scaling_range
    combined_scaled = combined.clip(lower= lower_range, upper = upper_range)

    mean_val = combined_scaled.mean()

    final_weights = (combined_scaled/mean_val).astype(float)
    final_weights.name = 'sample_weights'

    return final_weights

def reweighing(sensitive_df, sensitive_columns):
    reweighing_weights = compute_reweighing_weights(sensitive_df)
    sample_weights = apply_reweighting(sensitive_df, sensitive_columns, reweighing_weights,(0.1,10))

    return sample_weights