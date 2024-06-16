import os
import time
import psutil
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from load_config import load_config
from typing import Dict, List, Tuple

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

def calculate_metrics(model, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
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

def log_resource_usage() -> Dict[str, float]:
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent
    }

def forward_selection() -> Tuple[SequentialFeatureSelector, List[str]]:
    """
    Perform forward feature selection.
    """
    log_reg = LogisticRegression(
        max_iter=config['logistic_regression_params']['max_iter'],
        random_state=config['logistic_regression_params']['random_state'],
        solver=config['logistic_regression_params']['solver']
    )
    
    sfs = SequentialFeatureSelector(
        log_reg,
        n_features_to_select=config['forward_selection_params']['k_features'],
        direction='forward',
        scoring=config['forward_selection_params']['scoring'],
        cv=config['forward_selection_params']['cv'],
        n_jobs=config['forward_selection_params']['n_jobs']
    )
    
    sfs = sfs.fit(X_train, y_train)
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
    results_path = config['paths']['features_selection_results_path']
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    if os.path.exists(results_path) and os.path.getsize(results_path) > 0:
        existing_results = pd.read_csv(results_path)
    else:
        existing_results = pd.DataFrame()
    
    # Append new results
    new_results = pd.DataFrame([forward_selection_results])
    all_results = pd.concat([existing_results, new_results], ignore_index=True)
    
    # Save results to CSV
    all_results.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")

    # Save selected features to a CSV file
    selected_features_path = config['paths']['features_selected_path']
    pd.Series(selected_features).to_csv(selected_features_path, index=False, header=False)
    print(f"Selected features saved to {selected_features_path}")

if __name__ == "__main__":
    main()
