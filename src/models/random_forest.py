import os
import time
import psutil
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, classification_report, roc_curve, auc
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

# Function to evaluate the model
def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_prob),
        "mean_squared_error": mean_squared_error(y_test, y_pred),
        "mean_absolute_error": mean_absolute_error(y_test, y_pred),
        "r2_score": r2_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

def main():
    start_time = time.time()
    
    # Train Random Forest model
    rf = RandomForestClassifier(random_state=42)
    
    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'n_estimators': config['random_forest_params']['n_estimators'],
        'max_depth': config['random_forest_params']['max_depth'],
        'min_samples_split': config['random_forest_params']['min_samples_split'],
        'min_samples_leaf': config['random_forest_params']['min_samples_leaf']
    }
    
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_selected, y_train)
    
    # Best parameters and result
    print(f'Best Parameters: {grid_search.best_params_}')
    best_rf = grid_search.best_estimator_
    y_pred_best = best_rf.predict(X_test_selected)
    y_pred_best_prob = best_rf.predict_proba(X_test_selected)[:, 1]
    print(f'Best Model Accuracy: {accuracy_score(y_test, y_pred_best)}')
    
    # Evaluate the best model
    metrics = evaluate_model(best_rf, X_test_selected, y_test)
    
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
        "selection_type": "random_forest"
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
    model_path = config['paths']['random_forest_model_path']
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    dump(best_rf, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()

