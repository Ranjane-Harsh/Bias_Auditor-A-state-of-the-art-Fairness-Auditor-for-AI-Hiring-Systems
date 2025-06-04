import time
import sys
sys.path.append(r'D:\Coding\Projects\Bias_Auditor A state of the art Fairness Auditor for AI Hiring Systems\Data_Acquisation&pre-processing')
from data_preprocessor import load_and_preprocess_data

def train_model(model, X_train, y_train, verbose= True):
    return