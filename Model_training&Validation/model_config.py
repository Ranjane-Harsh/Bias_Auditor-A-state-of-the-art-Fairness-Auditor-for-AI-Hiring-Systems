import json
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def load_model_config(config_path):
    if config_path.endswith("json"):
        with open(config_path,'r') as p:
            config_dict = json.load(p)
        
    elif config_path.endswith((".yaml", ".yml")):
        with open(config_path,'r') as p:
            config_dict = yaml.safe_load(p)  

    else:
        raise ValueError("Unexpected config file format")
    
    if not all(k in config_dict for k in ['model_name', 'params', 'fairness']):
        raise ValueError("Missing critical values in the file")

    return config_dict

config_dict = load_model_config(r"D:\Coding\Projects\Bias_Auditor A state of the art Fairness Auditor for AI Hiring Systems\Configs\random_forest.yaml")
print(config_dict)

def initialize_model(config_dict):
    model_name = config_dict.get("model_name")
    model_params = config_dict.get("params")

    MODEL_REGISTRY = {"logistic_regression" : LogisticRegression, "random_forest" : RandomForestClassifier, "xgboost" : XGBClassifier}

    model_class = MODEL_REGISTRY.get(model_name)

    if model_class is None:
        raise ValueError(f"Model: {model_name} is not supported in the registry")
    
    model_instance = model_class(**model_params)

    return model_instance

model_instance = initialize_model(config_dict)
print(type(model_instance))