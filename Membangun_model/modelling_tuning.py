"""
modelling_tuning.py
====================
Advanced Model Training with Hyperparameter Tuning and Manual Logging.

Kriteria 2 - Skilled (3 pts):
- Hyperparameter tuning with RandomizedSearchCV
- Manual logging (not autolog) with same metrics as autolog

Kriteria 2 - Advance (4 pts):
- Manual logging with autolog metrics + minimal 2 additional artifacts
  - Artifact 1: Classification Report (text file)
  - Artifact 2: Feature Importance Plot (image)
  - Artifact 3: Confusion Matrix Heatmap (image)
"""

import os

# Fix: Must set before importing mlflow to allow file-based tracking
os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'

import json
import tempfile
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, log_loss
)
import mlflow
import mlflow.sklearn
from scipy.stats import randint

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bankmarketing_preprocessing')
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')

EXPERIMENT_NAME = "Bank_Marketing_Classification_Tuning"
RANDOM_STATE = 42
TARGET_NAMES = ['no', 'yes']

MLRUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mlruns')
mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR.replace(os.sep, '/')}")


def load_data():
    """Load preprocessed train and test datasets."""
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    target_col = 'y'
    
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    print(f"Train set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    return X_train, X_test, y_train, y_test


def hyperparameter_tuning(X_train, y_train):
    """
    Perform hyperparameter tuning using RandomizedSearchCV.
    """
    print("\n--- Hyperparameter Tuning (RandomizedSearchCV) ---")
    
    param_distributions = {
        'n_estimators': randint(50, 300),
        'max_depth': [5, 10, 15, 20, 25, None],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy']
    }
    
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=50,
        cv=5,
        scoring='accuracy',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"\nBest Parameters: {random_search.best_params_}")
    print(f"Best CV Score: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_, random_search.best_params_, random_search.cv_results_


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return all metrics."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'log_loss': log_loss(y_test, y_pred_proba),
    }
    
    print(f"\n--- Evaluation Results ---")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    return metrics, y_pred, y_pred_proba


def create_classification_report_artifact(y_test, y_pred, artifact_dir):
    """Create classification report as text and JSON artifact."""
    report_text = classification_report(y_test, y_pred, target_names=TARGET_NAMES)
    report_dict = classification_report(y_test, y_pred, target_names=TARGET_NAMES, output_dict=True)
    
    report_path = os.path.join(artifact_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("Bank Marketing Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(report_text)
    
    report_json_path = os.path.join(artifact_dir, 'classification_report.json')
    with open(report_json_path, 'w') as f:
        json.dump(report_dict, f, indent=2, default=str)
    
    print(f"\n  [OK] Classification Report saved")
    return report_path, report_json_path


def create_feature_importance_artifact(model, feature_names, artifact_dir):
    """Create feature importance plot as image artifact."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_names)))
    
    bars = ax.barh(
        range(len(feature_names)),
        importances[indices],
        color=colors,
        edgecolor='white',
        linewidth=0.5
    )
    
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=10)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title('Random Forest - Feature Importance\n(Bank Marketing Classification)', 
                 fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    for i, (bar, imp) in enumerate(zip(bars, importances[indices])):
        ax.text(imp + 0.002, i, f'{imp:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    plot_path = os.path.join(artifact_dir, 'feature_importance.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  [OK] Feature Importance Plot saved")
    return plot_path


def create_confusion_matrix_artifact(y_test, y_pred, artifact_dir):
    """Create confusion matrix heatmap as image artifact."""
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=TARGET_NAMES,
        yticklabels=TARGET_NAMES,
        ax=ax,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': 'Count'}
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix\n(Bank Marketing Classification)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    plot_path = os.path.join(artifact_dir, 'confusion_matrix.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  [OK] Confusion Matrix Heatmap saved")
    return plot_path


def main():
    """Main function: Full pipeline with manual logging."""
    print("=" * 60)
    print("BANK MARKETING - MODEL TRAINING WITH TUNING")
    print("(Manual Logging)")
    print("=" * 60)
    
    # Set experiment
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    feature_names = list(X_train.columns)
    
    # Hyperparameter tuning
    best_model, best_params, cv_results = hyperparameter_tuning(X_train, y_train)
    
    # Evaluate
    metrics, y_pred, y_pred_proba = evaluate_model(best_model, X_test, y_test)
    
    # Start MLflow run with MANUAL LOGGING
    with mlflow.start_run(run_name="RandomForest_Tuned_Manual"):
        
        print("\n--- Logging to MLflow (Manual) ---")
        
        # Log all hyperparameters
        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)
            print(f"  [PARAM] {param_name} = {param_value}")
        
        # Log additional training params
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("n_iter_search", 50)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("train_samples", X_train.shape[0])
        mlflow.log_param("test_samples", X_test.shape[0])
        mlflow.log_param("n_features", X_train.shape[1])
        
        # Log all metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            print(f"  [METRIC] {metric_name} = {metric_value:.4f}")
        
        # Log best CV score
        mlflow.log_metric("best_cv_score", float(cv_results['mean_test_score'].max()))
        
        # Log model
        mlflow.sklearn.log_model(
            best_model,
            artifact_path="model",
            registered_model_name="bank-marketing-rf-tuned"
        )
        print("  [MODEL] Model logged")
        
        # Additional Artifacts
        print("\n--- Creating Additional Artifacts ---")
        
        with tempfile.TemporaryDirectory() as artifact_dir:
            # Artifact 1: Classification Report
            report_path, report_json_path = create_classification_report_artifact(
                y_test, y_pred, artifact_dir
            )
            mlflow.log_artifact(report_path, "reports")
            mlflow.log_artifact(report_json_path, "reports")
            
            # Artifact 2: Feature Importance Plot
            fi_path = create_feature_importance_artifact(
                best_model, feature_names, artifact_dir
            )
            mlflow.log_artifact(fi_path, "plots")
            
            # Artifact 3: Confusion Matrix Heatmap
            cm_path = create_confusion_matrix_artifact(
                y_test, y_pred, artifact_dir
            )
            mlflow.log_artifact(cm_path, "plots")
        
        # Log tags
        mlflow.set_tag("stage", "tuning")
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("dataset", "bank-marketing")
        
        run_id = mlflow.active_run().info.run_id
        print(f"\n  [RUN_ID] {run_id}")
        print(f"  [EXPERIMENT] {EXPERIMENT_NAME}")
        print(f"  [TRACKING] {mlflow.get_tracking_uri()}")
    
    print("\n" + "=" * 60)
    print("TRAINING WITH TUNING COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("\nTo view results locally, run: mlflow ui")
    print("Then open: http://localhost:5000")
    
    return run_id


if __name__ == "__main__":
    main()
