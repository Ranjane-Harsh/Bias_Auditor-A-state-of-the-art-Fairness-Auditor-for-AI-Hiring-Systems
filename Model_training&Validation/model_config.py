import json
import yaml

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