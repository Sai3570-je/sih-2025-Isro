"""
Training module for enhanced Kalman filter with proper validation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Tuple, Dict, List
import pickle

from src.enhanced_kalman import EnhancedKalmanFilter, forecast_day8_enhanced

logger = logging.getLogger(__name__)


def train_enhanced_filter(
    df_train: pd.DataFrame,
    dt: float = 900.0,
    validation_split: float = 0.2
) -> Tuple[EnhancedKalmanFilter, Dict]:
    """
    Train enhanced Kalman filter with validation.
    
    Args:
        df_train: Training data with columns [timestamp, X_Error, Y_Error, Z_Error, Sat_Clock_Error]
        dt: Time step in seconds
        validation_split: Fraction of data for validation
        
    Returns:
        kf: Trained filter
        metrics: Training metrics
    """
    # Split train/validation
    split_idx = int(len(df_train) * (1 - validation_split))
    df_fit = df_train.iloc[:split_idx].copy()
    df_val = df_train.iloc[split_idx:].copy()
    
    logger.info(f"Training on {len(df_fit)} samples, validating on {len(df_val)} samples")
    
    # Grid search for optimal Q and R scales
    best_mae = np.inf
    best_params = {'Q_scale': 1.0, 'R_scale': 1.0}
    best_kf = None
    
    Q_scales = [0.01, 0.1, 1.0, 10.0, 100.0]
    R_scales = [0.1, 1.0, 10.0]
    
    for Q_scale in Q_scales:
        for R_scale in R_scales:
            # Train filter
            kf = EnhancedKalmanFilter(dt=dt, Q_scale=Q_scale, R_scale=R_scale)
            
            # Fit on training data
            for _, row in df_fit.iterrows():
                if pd.notna(row['X_Error']):
                    meas = np.array([
                        row['X_Error'],
                        row['Y_Error'],
                        row['Z_Error'],
                        row['Clock_Error']
                    ])
                    
                    if not kf.initialized:
                        kf.initialize(meas, timestamp=row['timestamp'])
                    else:
                        kf.predict()
                        kf.update(meas)
            
            # Validate
            if len(df_val) > 0:
                mae = validate_filter(kf, df_val, dt)
                
                if mae < best_mae:
                    best_mae = mae
                    best_params = {'Q_scale': Q_scale, 'R_scale': R_scale}
                    best_kf = kf
                    logger.info(f"New best: Q={Q_scale:.2f}, R={R_scale:.2f}, MAE={mae:.3f}m")
    
    if best_kf is None:
        # Fallback: use default params
        logger.warning("Grid search failed, using default parameters")
        best_kf = EnhancedKalmanFilter(dt=dt, Q_scale=1.0, R_scale=1.0)
        
        for _, row in df_train.iterrows():
            if pd.notna(row['X_Error']):
                meas = np.array([
                    row['X_Error'],
                    row['Y_Error'],
                    row['Z_Error'],
                    row['Clock_Error']
                ])
                
                if not best_kf.initialized:
                    best_kf.initialize(meas, timestamp=row['timestamp'])
                else:
                    best_kf.predict()
                    best_kf.update(meas)
    
    metrics = {
        'best_Q_scale': best_params['Q_scale'],
        'best_R_scale': best_params['R_scale'],
        'validation_mae': best_mae,
        'training_samples': len(df_fit),
        'validation_samples': len(df_val)
    }
    
    return best_kf, metrics


def validate_filter(
    kf: EnhancedKalmanFilter,
    df_val: pd.DataFrame,
    dt: float
) -> float:
    """
    Compute validation MAE for filter.
    
    Args:
        kf: Kalman filter (will be copied for validation)
        df_val: Validation data
        dt: Time step
        
    Returns:
        mae: Mean absolute error in meters
    """
    # Copy filter state to avoid modifying original
    import copy
    kf_copy = copy.deepcopy(kf)
    
    errors = []
    
    for _, row in df_val.iterrows():
        if pd.notna(row['X_Error']):
            # Predict
            kf_copy.predict()
            pred = kf_copy.get_position()
            
            # Compare to ground truth
            truth = np.array([
                row['X_Error'],
                row['Y_Error'],
                row['Z_Error'],
                row['Clock_Error']
            ])
            
            error = np.abs(pred - truth)
            errors.append(error)
            
            # Update with measurement (for next prediction)
            kf_copy.update(truth)
    
    if len(errors) == 0:
        return np.inf
    
    errors = np.array(errors)
    mae = np.mean(errors[:, :3])  # Average over X, Y, Z only
    
    return mae


def train_and_predict_enhanced(
    satellite_type: str,
    data_folder: Path,
    output_folder: Path,
    dt: float = 900.0
) -> Dict:
    """
    Complete training and prediction pipeline for enhanced filter.
    
    Args:
        satellite_type: 'geo' or 'meo'
        data_folder: Path to data files
        output_folder: Path for outputs
        dt: Time step in seconds
        
    Returns:
        results: Dictionary with metrics and file paths
    """
    # Load preprocessed data
    train_file = output_folder / f"{satellite_type.upper()}_01_timeseries.parquet"
    
    if not train_file.exists():
        raise FileNotFoundError(f"Training data not found: {train_file}")
    
    df_train = pd.read_parquet(train_file)
    df_train = df_train[df_train['timestamp'] <= '2025-09-07 23:59:59'].copy()
    
    # Filter to real measurements only
    df_real = df_train[df_train['X_Error'].notna()].copy()
    logger.info(f"{satellite_type.upper()}: {len(df_real)} real measurements out of {len(df_train)}")
    
    if len(df_real) < 10:
        logger.error(f"Insufficient data for {satellite_type.upper()}: only {len(df_real)} measurements")
        return {
            'status': 'failed',
            'reason': 'insufficient_data',
            'measurements': len(df_real)
        }
    
    # Train filter
    logger.info(f"Training enhanced filter for {satellite_type.upper()}...")
    kf, metrics = train_enhanced_filter(df_real, dt=dt, validation_split=0.2)
    
    logger.info(f"Best parameters: Q={metrics['best_Q_scale']:.2f}, "
                f"R={metrics['best_R_scale']:.2f}, MAE={metrics['validation_mae']:.3f}m")
    
    # Forecast Day 8
    logger.info(f"Forecasting Day 8 for {satellite_type.upper()}...")
    predictions, uncertainties = forecast_day8_enhanced(kf, num_steps=96)
    
    # Save predictions
    timestamps = pd.date_range(
        start='2025-09-08 00:00:00',
        periods=96,
        freq='15min'
    )
    
    df_pred = pd.DataFrame({
        'timestamp': timestamps,
        'X_Error_pred': predictions[:, 0],
        'Y_Error_pred': predictions[:, 1],
        'Z_Error_pred': predictions[:, 2],
        'Sat_Clock_Error_pred': predictions[:, 3],
        'X_Error_std': uncertainties[:, 0],
        'Y_Error_std': uncertainties[:, 1],
        'Z_Error_std': uncertainties[:, 2],
        'Sat_Clock_Error_std': uncertainties[:, 3]
    })
    
    pred_file = output_folder / f"predictions_day8_{satellite_type}_enhanced.csv"
    df_pred.to_csv(pred_file, index=False)
    logger.info(f"Predictions saved to {pred_file}")
    
    # Save model
    model_file = output_folder / f"kf_model_{satellite_type}_enhanced.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(kf, f)
    logger.info(f"Model saved to {model_file}")
    
    # Compute prediction statistics
    pred_stats = {
        'X_mean': predictions[:, 0].mean(),
        'X_std': predictions[:, 0].std(),
        'X_range': [predictions[:, 0].min(), predictions[:, 0].max()],
        'Y_mean': predictions[:, 1].mean(),
        'Y_std': predictions[:, 1].std(),
        'Y_range': [predictions[:, 1].min(), predictions[:, 1].max()],
        'Z_mean': predictions[:, 2].mean(),
        'Z_std': predictions[:, 2].std(),
        'Z_range': [predictions[:, 2].min(), predictions[:, 2].max()],
    }
    
    return {
        'status': 'success',
        'satellite': satellite_type.upper(),
        'training_metrics': metrics,
        'prediction_stats': pred_stats,
        'prediction_file': str(pred_file),
        'model_file': str(model_file)
    }
