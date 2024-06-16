import os
import time
import psutil
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, classification_report, confusion_matrix, roc_curve
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
def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
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
    rf.fit(X_train_selected, y_train)
    
    # Evaluate the model
    metrics = evaluate_model(rf, X_test_selected, y_test)
    
    # Log resource usage
    resource_usage = log_resource_usage()
    
    # Record the elapsed time
    elapsed_time = time.time() - start_time
    
    # Prepare results
    results = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "metrics": str(metrics),
        "resource_usage": str(resource_usage),
        "elapsed_time": elapsed_time,
        "selection_type": "random_forest_base"
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
    model_path = config['paths']['random_forest_base_model_path']
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    dump(rf, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
