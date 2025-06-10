import time
from Data_Acquisation_and_preprocessing.data_loader import log_status
def train_model(model, X_train, y_train, verbose= True):
    start_time = time.time()
    trained_model = model.fit(X_train,y_train)
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
