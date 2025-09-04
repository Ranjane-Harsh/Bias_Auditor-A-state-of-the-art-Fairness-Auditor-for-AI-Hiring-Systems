import time
import os
import joblib
from datetime import datetime
from Data_Acquisation_and_preprocessing.data_loader import log_status
import numpy as np

def train_model(model, X_train, y_train, sample_weight, verbose= True):
    start_time = time.time()
    fit_kwargs = {}
    if sample_weight is None:
        trained_model = model.fit(X_train, y_train)
    else:
        sw = np.asarray(sample_weight, dtype=float)
        fit_kwargs['sample_weight'] = sw
        trained_model = model.fit(X_train,y_train, **fit_kwargs)
        end_time = time.time()
        if verbose:
            print(f"Model trained in {end_time-start_time:.2f} seconds")

    return trained_model

def generate_predictions(model, X_test):
    if hasattr(model,"predict"):
        y_pred = model.predict(X_test)
    else:
        log_status("CRITICAL","This model does not support this command")
    
    if hasattr(model,"predict_proba"):
        y_proba = model.predict_proba(X_test)
    else:
        log_status("CRITICAL","This model does not support this command")

    return y_pred,y_proba

def save_model(model, model_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    filename = f"{model_name}_{timestamp}.joblib"
    model_path = os.path.join(save_dir,filename)
    joblib.dump(model,model_path)

    return model_path