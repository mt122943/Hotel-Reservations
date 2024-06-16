import os
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from dotenv import load_dotenv, find_dotenv
import time
import psutil
import numpy as np
from load_config import load_config

# Zmienna globalna do przechowywania wyników selekcji forward
forward_selection_results = None

# Load environment variables from .env file
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

# Load configuration from config.yaml
config = load_config()

# Load training and testing sets
X_train = pd.read_csv(config['paths']['x_train_path'])
X_test = pd.read_csv(config['paths']['x_test_path'])
y_train = pd.read_csv(config['paths']['y_train_path']).values.ravel()
y_test = pd.read_csv(config['paths']['y_test_path']).values.ravel()

# Function to calculate evaluation metrics
def calculate_metrics(model, X, y):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1_score": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_proba)
    }

def log_resource_usage():
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent
    }

def forward_selection():
    """
    Perform forward feature selection.
    """
    # Initialize the logistic regression model
    log_reg = LogisticRegression(
        max_iter=config['logistic_regression_params']['max_iter'],
        random_state=config['logistic_regression_params']['random_state'],
        solver=config['logistic_regression_params']['solver']
    )
    
    # Initialize the feature selector with logistic regression and forward selection parameters
    sfs = SequentialFeatureSelector(
        log_reg,
        n_features_to_select=config['forward_selection_params']['k_features'],
        direction='backward',
        scoring=config['forward_selection_params']['scoring'],
        cv=config['forward_selection_params']['cv'],
        n_jobs=config['forward_selection_params']['n_jobs']
    )
    
    # Fit the feature selector to the training data
    sfs = sfs.fit(X_train, y_train)
    
    # Get the selected features
    selected_features = X_train.columns[sfs.get_support()]
    print(f"Selected features (forward): {selected_features}")
    
    return sfs, selected_features

def main():
    global forward_selection_results
    
    start_time = time.time()
    
    # Perform forward feature selection
    sfs, selected_features = forward_selection()
    
    # Fit the logistic regression model with selected features
    log_reg = LogisticRegression(
        max_iter=config['logistic_regression_params']['max_iter'],
        random_state=config['logistic_regression_params']['random_state'],
        solver=config['logistic_regression_params']['solver']
    )
    log_reg.fit(X_train[selected_features], y_train)
    
    # Calculate evaluation metrics
    metrics = calculate_metrics(log_reg, X_test[selected_features], y_test)
    
    # Log resource usage
    resource_usage = log_resource_usage()
    
    # Record the elapsed time
    elapsed_time = time.time() - start_time
    
    # Prepare results
    forward_selection_results = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "selected_features": list(selected_features),
        "metrics": metrics,
        "resource_usage": resource_usage,
        "elapsed_time": elapsed_time,
        "selection_type": "forward"
    }
    
    # Load existing results if available
    results_path = config['paths']['results_path']
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    if os.path.exists(results_path):
        existing_results = pd.read_csv(results_path)
    else:
        existing_results = pd.DataFrame()
    
    # Append new results
    new_results = pd.DataFrame([forward_selection_results])
    all_results = pd.concat([existing_results, new_results], ignore_index=True)
    
    # Save results to CSV
    all_results.to_csv(results_path, index=False)
    
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

def get_forward_selection_results():
    global forward_selection_results
    return forward_selection_results

if __name__ == "__main__":
    main()
