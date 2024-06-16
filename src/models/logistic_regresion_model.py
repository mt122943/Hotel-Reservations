import os
import time
import psutil
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    log_loss, average_precision_score
)
from joblib import dump
from load_config import load_config
from typing import Dict, Any

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

# Load selected features from CSV file
selected_features_path = config['paths']['features_selected_path']
selected_features = pd.read_csv(selected_features_path, header=None).iloc[:, 0].tolist()

X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Function to log resource usage
def log_resource_usage() -> Dict[str, float]:
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent
    }

# Function to perform grid search
def perform_grid_search(X_train: pd.DataFrame, y_train: pd.Series, param_grid: Dict[str, Any]) -> GridSearchCV:
    grid_search = GridSearchCV(
        estimator=LogisticRegression(
            max_iter=config['logistic_regression_params']['max_iter'],  # Zwiększenie liczby iteracji
            random_state=config['logistic_regression_params']['random_state']
        ),
        param_grid=param_grid,
        cv=config['grid_search_params']['cv'],
        scoring=config['grid_search_params']['scoring']
    )
    grid_search.fit(X_train, y_train)
    return grid_search

# Function to evaluate the model
def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_prob),
        "log_loss": log_loss(y_test, y_pred_prob),
        "avg_precision": average_precision_score(y_test, y_pred_prob)
    }

def main():
    start_time = time.time()
    
    # Perform grid search
    grid_search = perform_grid_search(X_train_selected, y_train, config['grid_search_params']['param_grid'])
    
    # Best model from grid search
    best_model = grid_search.best_estimator_
    
    # Evaluate the best model
    metrics = evaluate_model(best_model, X_test_selected, y_test)
    
    # Log resource usage
    resource_usage = log_resource_usage()
    
    # Record the elapsed time
    elapsed_time = time.time() - start_time
    
    # Prepare results
    results = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "best_params": str(grid_search.best_params_),
        "metrics": str(metrics),
        "resource_usage": str(resource_usage),
        "elapsed_time": elapsed_time,
        "selection_type": "logistic_regression_final"
    }
    
    # Save general results to CSV
    model_results_path = config['paths']['model_results_path']
    results_df = pd.DataFrame([results])
    os.makedirs(os.path.dirname(model_results_path), exist_ok=True)
    
    if os.path.exists(model_results_path) and os.path.getsize(model_results_path) > 0:
        existing_results = pd.read_csv(model_results_path)
        all_results = pd.concat([existing_results, results_df], ignore_index=True)
    else:
        all_results = results_df
    
    all_results.to_csv(model_results_path, index=False)
    print(f"Model results saved to {model_results_path}")
    
    # Save the best model to models/ directory
    model_path = config['paths']['logistic_regression_model_path']
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    dump(best_model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
