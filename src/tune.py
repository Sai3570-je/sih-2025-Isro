"""
Tuning Q and R parameters for Kalman filter
"""
import numpy as np
import logging
from .kalman_filter import run_kalman_forward, build_state_matrices, initialize_covariance_matrices

logger = logging.getLogger(__name__)

def compute_validation_mae(df_train, df_val, Q, R, F, H, x0, P0):
    """
    Compute MAE on validation set
    
    Args:
        df_train: Training data for initialization
        df_val: Validation data
        Q, R: Covariance matrices
        F, H: State matrices
        x0, P0: Initial state and covariance
    
    Returns:
        float: Mean absolute error
    """
    # Run filter on validation data
    result = run_kalman_forward(df_val, Q, R, x0, P0, F, H)
    
    # Compute MAE between predictions and actual
    target_cols = ['X_Error', 'Y_Error', 'Z_Error', 'Clock_Error']
    
    actual_vals = []
    for idx, row in df_val.iterrows():
        if all(col in row and not pd.isna(row[col]) for col in target_cols):
            actual_vals.append([row[col] for col in target_cols])
    
    if len(actual_vals) == 0:
        return float('inf')
    
    actual_vals = np.array(actual_vals)
    pred_vals = result['predicted_obs'][:len(actual_vals)]
    
    mae = np.mean(np.abs(actual_vals - pred_vals))
    
    return mae

def tune_q_r_gridsearch(df_sat, validation_fraction=0.15):
    """
    Tune Q and R using grid search
    
    Args:
        df_sat: Satellite training DataFrame
        validation_fraction: Fraction of data to use for validation
    
    Returns:
        dict: Best parameters {Q, R, x0, P0}
    """
    logger.info("Starting Q/R tuning via grid search...")
    
    # Split into train/validation
    split_idx = int(len(df_sat) * (1 - validation_fraction))
    df_train_sub = df_sat.iloc[:split_idx].copy()
    df_val = df_sat.iloc[split_idx:].copy()
    
    logger.info(f"  Train subset: {len(df_train_sub)}, Validation: {len(df_val)}")
    
    # Build state matrices
    F, H = build_state_matrices(dt=900.0)
    
    # Estimate initial R from data
    from .kalman_filter import estimate_measurement_noise
    R_base = estimate_measurement_noise(df_train_sub)
    
    # Initialize state from first observation
    target_cols = ['X_Error', 'Y_Error', 'Z_Error', 'Clock_Error']
    first_obs = df_train_sub[target_cols].dropna().iloc[0] if len(df_train_sub[target_cols].dropna()) > 0 else None
    
    if first_obs is None:
        logger.warning("  No valid observations for tuning, using defaults")
        Q, R, P0 = initialize_covariance_matrices()
        x0 = np.zeros((8, 1))
        return {'Q': Q, 'R': R, 'x0': x0, 'P0': P0, 'mae': float('inf')}
    
    x0 = np.array([
        [first_obs['X_Error']], [0],
        [first_obs['Y_Error']], [0],
        [first_obs['Z_Error']], [0],
        [first_obs['Clock_Error']], [0]
    ])
    
    # Grid search over Q scale factors
    q_scales = [0.01, 0.1, 1.0, 10.0, 100.0]
    
    best_mae = float('inf')
    best_params = None
    
    for q_scale in q_scales:
        # Scale Q
        Q_base, _, P0 = initialize_covariance_matrices()
        Q = Q_base * q_scale
        R = R_base.copy()
        
        try:
            mae = compute_validation_mae(df_train_sub, df_val, Q, R, F, H, x0, P0)
            logger.info(f"  Q_scale={q_scale:6.2f}, MAE={mae:.6f}")
            
            if mae < best_mae:
                best_mae = mae
                best_params = {'Q': Q, 'R': R, 'x0': x0, 'P0': P0, 'mae': mae}
        
        except Exception as e:
            logger.warning(f"  Q_scale={q_scale} failed: {e}")
            continue
    
    if best_params is None:
        logger.warning("  Grid search failed, using defaults")
        Q, R, P0 = initialize_covariance_matrices()
        best_params = {'Q': Q, 'R': R_base, 'x0': x0, 'P0': P0, 'mae': float('inf')}
    
    logger.info(f"[OK] Best Q scale found, Validation MAE: {best_params['mae']:.6f}")
    
    return best_params

import pandas as pd  # Add this import
