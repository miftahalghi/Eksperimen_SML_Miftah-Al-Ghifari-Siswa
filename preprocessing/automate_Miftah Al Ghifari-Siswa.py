"""
automate_Miftah Al Ghifari-Siswa.py
====================================
Automated preprocessing pipeline for Wine Quality Dataset.
Converts the manual experiment notebook into a reusable, automated script.

Usage:
    python "automate_Miftah Al Ghifari-Siswa.py"
    
Or import as module:
    from automate_Miftah_Al_Ghifari_Siswa import preprocess_pipeline
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
    Load raw Wine Quality dataset from CSV.
    
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
    Strategy: Drop rows with missing values if < 5% of data,
              otherwise fill with median for numerical columns.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with missing values handled
    """
    logger.info("Handling missing values...")
    
    missing_count = df.isnull().sum().sum()
    missing_pct = (missing_count / (df.shape[0] * df.shape[1])) * 100
    
    if missing_count == 0:
        logger.info("No missing values found.")
        return df
    
    logger.info(f"Found {missing_count} missing values ({missing_pct:.2f}%)")
    
    if missing_pct < 5:
        df = df.dropna()
        logger.info(f"Dropped rows with missing values. Remaining: {df.shape[0]} rows")
    else:
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.info(f"  Filled '{col}' missing values with median: {median_val}")
    
    return df


# ============================================================
# 3. REMOVING DUPLICATES
# ============================================================
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the dataset.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with duplicates removed
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
    Detect and handle outliers using IQR (Interquartile Range) method.
    Strategy: Cap outliers at Q1 - threshold*IQR and Q3 + threshold*IQR.
    
    Args:
        df: Input DataFrame
        columns: List of columns to check (default: all numeric columns except 'quality')
        threshold: IQR multiplier for outlier detection (default: 1.5)
    
    Returns:
        DataFrame with outliers handled
    """
    logger.info(f"Handling outliers using IQR method (threshold={threshold})...")
    
    if columns is None:
        columns = [col for col in df.select_dtypes(include=[np.number]).columns if col != 'quality']
    
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
# 5. TARGET ENCODING (Quality → Categories)
# ============================================================
def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode the 'quality' column into categorical labels.
    Binning: quality 3-4 → 'low', 5-6 → 'medium', 7-9 → 'high'
    Then encode using LabelEncoder: high=0, low=1, medium=2
    
    Args:
        df: Input DataFrame with 'quality' column
    
    Returns:
        DataFrame with encoded 'quality_label' column
    """
    logger.info("Encoding target variable (quality → categories)...")
    
    # Create categorical bins
    def quality_category(q):
        if q <= 4:
            return 'low'
        elif q <= 6:
            return 'medium'
        else:
            return 'high'
    
    df['quality_label'] = df['quality'].apply(quality_category)
    
    # Log distribution
    dist = df['quality_label'].value_counts()
    logger.info(f"Quality distribution after encoding:")
    for label, count in dist.items():
        logger.info(f"  {label}: {count} ({count/len(df)*100:.1f}%)")
    
    # Label encode
    le = LabelEncoder()
    df['quality_encoded'] = le.fit_transform(df['quality_label'])
    
    logger.info(f"Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # Drop original quality and quality_label
    df = df.drop(columns=['quality', 'quality_label'])
    
    return df


# ============================================================
# 6. FEATURE SCALING
# ============================================================
def scale_features(df: pd.DataFrame, target_col: str = 'quality_encoded') -> pd.DataFrame:
    """
    Apply StandardScaler to all feature columns (except target).
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column to exclude from scaling
    
    Returns:
        DataFrame with scaled features
    """
    logger.info("Scaling features using StandardScaler...")
    
    feature_cols = [col for col in df.columns if col != target_col]
    
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    logger.info(f"Scaled {len(feature_cols)} feature columns: {feature_cols}")
    
    return df


# ============================================================
# 7. TRAIN-TEST SPLIT
# ============================================================
def split_data(df: pd.DataFrame, target_col: str = 'quality_encoded',
               test_size: float = 0.2, random_state: int = 42) -> dict:
    """
    Split data into training and testing sets.
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
        test_size: Proportion of test set (default: 0.2)
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary with X_train, X_test, y_train, y_test
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
    
    Args:
        data: Dictionary with X_train, X_test, y_train, y_test
        output_dir: Directory to save the preprocessed files
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
    full_df.to_csv(os.path.join(output_dir, 'winequality_preprocessed.csv'), index=False)
    
    logger.info(f"Saved train.csv ({train_df.shape[0]} rows)")
    logger.info(f"Saved test.csv ({test_df.shape[0]} rows)")
    logger.info(f"Saved winequality_preprocessed.csv ({full_df.shape[0]} rows)")


# ============================================================
# MAIN PIPELINE
# ============================================================
def preprocess_pipeline(input_path: str, output_dir: str) -> dict:
    """
    Execute the full preprocessing pipeline.
    
    Args:
        input_path: Path to raw dataset CSV
        output_dir: Directory to save preprocessed data
    
    Returns:
        Dictionary with X_train, X_test, y_train, y_test
    """
    logger.info("=" * 60)
    logger.info("STARTING PREPROCESSING PIPELINE")
    logger.info("=" * 60)
    
    # Step 1: Load data
    df = load_data(input_path)
    
    # Step 2: Handle missing values
    df = handle_missing_values(df)
    
    # Step 3: Remove duplicates
    df = remove_duplicates(df)
    
    # Step 4: Handle outliers
    df = handle_outliers(df)
    
    # Step 5: Encode target
    df = encode_target(df)
    
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
    
    input_path = os.path.join(project_dir, 'winequality_raw', 'winequality-red.csv')
    output_dir = os.path.join(script_dir, 'winequality_preprocessing')
    
    # Run pipeline
    preprocess_pipeline(input_path, output_dir)
