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

def compute_reweighing_weights(sensitive_df,label_col):
    N = len(sensitive_df)
    print(f"Number of rows in Dataframe are : {N}")
    reweighing_weights = {}
    sensitive_columns = ["gender","race","college_tier","education_level"]
    for col in sensitive_columns:
        '''print(f"\n{col}:\n{sensitive_df[col].value_counts()}")

        print(f"\n{col} vs labels")
        print(pd.crosstab(sensitive_df[col], sensitive_df["hired"]))'''

        unique_attr_val = sensitive_df[col].unique()
        unique_lables = np.unique(sensitive_df["hired"])

        counts_attr = sensitive_df[col].value_counts()
        counts_label = pd.Series(sensitive_df["hired"]).value_counts()

        joint_df = sensitive_df.copy()
        dataframe = joint_df.groupby([col, "hired"]).size().reset_index(name = 'count')

        dataframe['ideal'] = (dataframe[col].map(counts_attr) * dataframe["hired"].map(counts_label)) / N

        dataframe['weight'] = dataframe['ideal'] / dataframe['count']

        for _, row in dataframe.iterrows():
            key = (col , row[col], row["hired"])
            reweighing_weights[key] = row['weight']

    for k, v in reweighing_weights.items():
        print(f"{k} : {v}")

    print(reweighing_weights)
    return reweighing_weights

        

        

    

    '''for attr in sensitive_columns:
        counts_attr = sensitive_df[attr].value_counts()
        counts_label = sensitive_df["hired"].value_counts()
        joint_df = (sensitive_df.groupby([attr,"hired"]).size().reset_index(name = 'count'))
        joint_df['ideal'] = (joint_df[attr].map(counts_attr) * sensitive_df["hired"].map(counts_label)) / N        

        joint_df['weight'] = joint_df['ideal'] / joint_df['count']

        for _, row in joint_df.iterrows():        
            key = (attr, row[attr], row["hired"])   
            reweighing_weights[key] = row['weight']        
    
    for k,v in reweighing_weights.items():
        print(f"{k}: {v}")

    return reweighing_weights'''