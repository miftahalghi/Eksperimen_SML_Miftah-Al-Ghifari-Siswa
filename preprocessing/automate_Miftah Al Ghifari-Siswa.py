"""
automate_Miftah Al Ghifari-Siswa.py
====================================
Automated preprocessing pipeline for Bank Marketing Dataset (UCI).
Converts the manual experiment notebook into a reusable, automated script.

Dataset: Bank Marketing (UCI ML Repository)
Task: Binary classification - predict if client will subscribe a term deposit (yes/no)

Usage:
    python "automate_Miftah Al Ghifari-Siswa.py"
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
import logging

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# 1. DATA LOADING
# ============================================================
def load_data(path: str) -> pd.DataFrame:
    """
    Load raw Bank Marketing dataset from CSV.
    
    Args:
        path: Path to the raw CSV file (semicolon-separated)
    
    Returns:
        DataFrame with loaded data
    """
    logger.info(f"Loading dataset from: {path}")
    
    df = pd.read_csv(path, sep=';')
    
    logger.info(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"Columns: {list(df.columns)}")
    
    return df


# ============================================================
# 2. HANDLING MISSING VALUES
# ============================================================
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check and handle missing values in the dataset.
    Bank Marketing dataset uses 'unknown' as missing value marker.
    Strategy: Replace 'unknown' with mode for categorical columns.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with missing values handled
    """
    logger.info("Handling missing values...")
    
    # Check for actual NaN
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        logger.info(f"Found {missing_count} NaN values")
        df = df.dropna()
    
    # Check for 'unknown' values in categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    unknown_counts = {}
    
    for col in cat_cols:
        unknown_count = (df[col] == 'unknown').sum()
        if unknown_count > 0:
            unknown_counts[col] = unknown_count
            # Replace 'unknown' with mode
            mode_val = df[df[col] != 'unknown'][col].mode()[0]
            df[col] = df[col].replace('unknown', mode_val)
            logger.info(f"  '{col}': replaced {unknown_count} 'unknown' values with mode='{mode_val}'")
    
    if not unknown_counts:
        logger.info("No 'unknown' values found.")
    
    return df


# ============================================================
# 3. REMOVING DUPLICATES
# ============================================================
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the dataset.
    """
    logger.info("Checking for duplicate rows...")
    
    duplicates = df.duplicated().sum()
    
    if duplicates > 0:
        df = df.drop_duplicates()
        logger.info(f"Removed {duplicates} duplicate rows. Remaining: {df.shape[0]} rows")
    else:
        logger.info("No duplicate rows found.")
    
    return df


# ============================================================
# 4. OUTLIER HANDLING (IQR Method)
# ============================================================
def handle_outliers(df: pd.DataFrame, columns: list = None, threshold: float = 1.5) -> pd.DataFrame:
    """
    Detect and handle outliers using IQR method on numeric columns.
    Strategy: Cap outliers at Q1 - threshold*IQR and Q3 + threshold*IQR.
    """
    logger.info(f"Handling outliers using IQR method (threshold={threshold})...")
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    total_outliers = 0
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        
        if outliers > 0:
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            total_outliers += outliers
            logger.info(f"  '{col}': {outliers} outliers capped to [{lower_bound:.4f}, {upper_bound:.4f}]")
    
    logger.info(f"Total outliers handled: {total_outliers}")
    
    return df


# ============================================================
# 5. ENCODING CATEGORICAL FEATURES
# ============================================================
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features and target variable.
    - Target 'y': yes=1, no=0
    - Categorical features: LabelEncoder
    """
    logger.info("Encoding categorical features...")
    
    # Encode target variable
    df['y'] = df['y'].map({'yes': 1, 'no': 0})
    logger.info(f"Target 'y' encoded: no=0, yes=1")
    logger.info(f"Target distribution: {dict(df['y'].value_counts())}")
    
    # Encode categorical features using LabelEncoder
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        logger.info(f"  Encoded '{col}': {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    return df


# ============================================================
# 6. FEATURE SCALING
# ============================================================
def scale_features(df: pd.DataFrame, target_col: str = 'y') -> pd.DataFrame:
    """
    Apply StandardScaler to all feature columns (except target).
    """
    logger.info("Scaling features using StandardScaler...")
    
    feature_cols = [col for col in df.columns if col != target_col]
    
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    logger.info(f"Scaled {len(feature_cols)} feature columns")
    
    return df


# ============================================================
# 7. TRAIN-TEST SPLIT
# ============================================================
def split_data(df: pd.DataFrame, target_col: str = 'y',
               test_size: float = 0.2, random_state: int = 42) -> dict:
    """
    Split data into training and testing sets.
    """
    logger.info(f"Splitting data (test_size={test_size}, random_state={random_state})...")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


# ============================================================
# 8. SAVE PREPROCESSED DATA
# ============================================================
def save_preprocessed(data: dict, output_dir: str):
    """
    Save preprocessed train and test datasets to CSV files.
    """
    logger.info(f"Saving preprocessed data to: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine X and y for saving
    train_df = pd.concat([data['X_train'].reset_index(drop=True),
                          data['y_train'].reset_index(drop=True)], axis=1)
    test_df = pd.concat([data['X_test'].reset_index(drop=True),
                         data['y_test'].reset_index(drop=True)], axis=1)
    full_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    
    # Save files
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    full_df.to_csv(os.path.join(output_dir, 'bankmarketing_preprocessed.csv'), index=False)
    
    logger.info(f"Saved train.csv ({train_df.shape[0]} rows, {train_df.shape[1]} cols)")
    logger.info(f"Saved test.csv ({test_df.shape[0]} rows, {test_df.shape[1]} cols)")
    logger.info(f"Saved bankmarketing_preprocessed.csv ({full_df.shape[0]} rows)")


# ============================================================
# MAIN PIPELINE
# ============================================================
def preprocess_pipeline(input_path: str, output_dir: str) -> dict:
    """
    Execute the full preprocessing pipeline.
    """
    logger.info("=" * 60)
    logger.info("STARTING PREPROCESSING PIPELINE - BANK MARKETING")
    logger.info("=" * 60)
    
    # Step 1: Load data
    df = load_data(input_path)
    
    # Step 2: Handle missing values (including 'unknown')
    df = handle_missing_values(df)
    
    # Step 3: Remove duplicates
    df = remove_duplicates(df)
    
    # Step 4: Handle outliers (numeric columns only)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude target 'y' if already numeric at this point
    numeric_cols = [c for c in numeric_cols if c != 'y']
    df = handle_outliers(df, columns=numeric_cols)
    
    # Step 5: Encode categorical features + target
    df = encode_features(df)
    
    # Step 6: Scale features
    df = scale_features(df)
    
    # Step 7: Split data
    data = split_data(df)
    
    # Step 8: Save preprocessed data
    save_preprocessed(data, output_dir)
    
    logger.info("=" * 60)
    logger.info("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    
    return data


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    # Determine paths relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    input_path = os.path.join(project_dir, 'bankmarketing_raw', 'bank-additional-full.csv')
    output_dir = os.path.join(script_dir, 'bankmarketing_preprocessing')
    
    # Run pipeline
    preprocess_pipeline(input_path, output_dir)
