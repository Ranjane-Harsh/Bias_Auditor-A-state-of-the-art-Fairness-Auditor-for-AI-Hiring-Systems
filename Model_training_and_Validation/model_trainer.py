import sys
import os
import time

def train_model(model, X_train, y_train, verbose= True):
    start_time = time.time()
    trained_model = model.fit(X_train,y_train)
    end_time = time.time()
    if verbose:
        print(f"Model trained in {end_time-start_time:.2f} seconds")
    return trained_model




