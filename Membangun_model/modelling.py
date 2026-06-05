"""
modelling.py
============
Basic Model Training with MLflow Autolog.
Trains a RandomForestClassifier on the preprocessed Wine Quality dataset
using MLflow's autolog feature for experiment tracking.

Kriteria 2 - Basic (2 pts):
- Melatih model ML (Scikit-Learn) menggunakan MLflow Tracking UI lokal
- Menggunakan autolog dari MLflow
"""

import os

# Fix: Must set before importing mlflow to allow file-based tracking
os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'winequality_preprocessing')
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')

EXPERIMENT_NAME = "Wine_Quality_Classification"
RANDOM_STATE = 42

MLRUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mlruns')
mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR.replace(os.sep, '/')}")



def load_data():
    """Load preprocessed train and test datasets."""
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    target_col = 'quality_encoded'
    
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    print(f"Train set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """Train a RandomForestClassifier with default parameters."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set."""
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['high', 'low', 'medium'])
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:\n{report}")
    
    return accuracy, y_pred


def main():
    """Main function: Load data, train model, evaluate, log with MLflow autolog."""
    print("=" * 60)
    print("WINE QUALITY - MODEL TRAINING (AUTOLOG)")
    print("=" * 60)
    
    # Set MLflow experiment
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Enable autolog
    mlflow.sklearn.autolog()
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Start MLflow run
    with mlflow.start_run(run_name="RandomForest_Autolog"):
        # Train
        print("\nTraining RandomForestClassifier...")
        model = train_model(X_train, y_train)
        
        # Evaluate
        accuracy, y_pred = evaluate_model(model, X_test, y_test)
        
        print(f"\nMLflow Run ID: {mlflow.active_run().info.run_id}")
        print(f"MLflow Experiment: {EXPERIMENT_NAME}")
        print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("\nTo view results, run: mlflow ui")
    print("Then open: http://localhost:5000")


if __name__ == "__main__":
    main()
