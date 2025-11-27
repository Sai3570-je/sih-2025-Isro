"""
Complete training pipeline with improved Kalman filter.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import pickle
from typing import Tuple, Dict

from src.improved_kalman import ImprovedKalmanFilter, forecast_improved

logger = logging.getLogger(__name__)


def train_improved(
    df_train: pd.DataFrame,
    dt: float = 900.0
) -> Tuple[ImprovedKalmanFilter, Dict]:
    """Train improved Kalman filter."""
    
    # Split train/validation (80/20)
    split_idx = int(len(df_train) * 0.8)
    df_fit = df_train.iloc[:split_idx].copy()
    df_val = df_train.iloc[split_idx:].copy()
    
    logger.info(f"Training: {len(df_fit)} samples, Validation: {len(df_val)} samples")
    
    # Grid search
    best_mae = np.inf
    best_params = None
    best_kf = None
    
    Q_scales = [0.1, 1.0, 10.0, 100.0]
    R_scales = [0.1, 1.0, 10.0]
    
    for Q in Q_scales:
        for R in R_scales:
            kf = ImprovedKalmanFilter(dt=dt, Q_scale=Q, R_scale=R)
            
            # Fit
            for _, row in df_fit.iterrows():
                meas = np.array([row['X_Error'], row['Y_Error'], row['Z_Error'], row['Clock_Error']])
                if not kf.initialized:
                    kf.initialize(meas)
                else:
                    kf.predict()
                    kf.update(meas)
            
            # Validate
            if len(df_val) >= 5:
                import copy
                kf_val = copy.deepcopy(kf)
                errors = []
                
                for _, row in df_val.iterrows():
                    kf_val.predict()
                    pred = kf_val.get_position()
                    truth = np.array([row['X_Error'], row['Y_Error'], row['Z_Error'], row['Clock_Error']])
                    errors.append(np.abs(pred[:3] - truth[:3]))
                    kf_val.update(truth)
                
                mae = np.mean(errors)
                
                if mae < best_mae and not np.isnan(mae) and not np.isinf(mae):
                    best_mae = mae
                    best_params = {'Q': Q, 'R': R}
                    best_kf = copy.deepcopy(kf)
                    logger.info(f"New best: Q={Q:.1f}, R={R:.1f}, MAE={mae:.3f}m")
    
    if best_kf is None:
        logger.warning("Grid search failed, using defaults")
        best_kf = ImprovedKalmanFilter(dt=dt, Q_scale=1.0, R_scale=1.0)
        best_params = {'Q': 1.0, 'R': 1.0}
        
        for _, row in df_train.iterrows():
            meas = np.array([row['X_Error'], row['Y_Error'], row['Z_Error'], row['Clock_Error']])
            if not best_kf.initialized:
                best_kf.initialize(meas)
            else:
                best_kf.predict()
                best_kf.update(meas)
    
    metrics = {
        'Q_scale': best_params['Q'],
        'R_scale': best_params['R'],
        'validation_mae': best_mae,
        'train_samples': len(df_fit),
        'val_samples': len(df_val)
    }
    
    return best_kf, metrics


def run_full_pipeline(satellite: str, output_folder: Path) -> Dict:
    """Run complete improved pipeline."""
    
    # Load data
    train_file = Path('temp') / f"{satellite.upper()}_01_timeseries.parquet"
    if not train_file.exists():
        train_file = output_folder / f"{satellite.upper()}_01_timeseries.parquet"
    
    if not train_file.exists():
        raise FileNotFoundError(f"No training data: {train_file}")
    
    df = pd.read_parquet(train_file)
    df = df[df['timestamp'] <= '2025-09-07 23:59:59']
    df_real = df[df['X_Error'].notna()].copy()
    
    logger.info(f"{satellite.upper()}: {len(df_real)} real measurements")
    
    if len(df_real) < 10:
        return {'status': 'failed', 'reason': 'insufficient_data'}
    
    # Train
    logger.info(f"Training {satellite.upper()}...")
    kf, metrics = train_improved(df_real, dt=900.0)
    
    logger.info(f"Best: Q={metrics['Q_scale']:.1f}, R={metrics['R_scale']:.1f}, MAE={metrics['validation_mae']:.3f}m")
    
    # Forecast
    logger.info(f"Forecasting Day 8...")
    preds, uncs = forecast_improved(kf, num_steps=96)
    
    # Save
    timestamps = pd.date_range('2025-09-08 00:00:00', periods=96, freq='15min')
    df_pred = pd.DataFrame({
        'timestamp': timestamps,
        'X_Error_pred': preds[:, 0],
        'Y_Error_pred': preds[:, 1],
        'Z_Error_pred': preds[:, 2],
        'Sat_Clock_Error_pred': preds[:, 3],
        'X_std': uncs[:, 0],
        'Y_std': uncs[:, 1],
        'Z_std': uncs[:, 2],
        'Clock_std': uncs[:, 3]
    })
    
    pred_file = output_folder / f"predictions_day8_{satellite}_improved.csv"
    df_pred.to_csv(pred_file, index=False)
    
    model_file = output_folder / f"kf_{satellite}_improved.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(kf, f)
    
    stats = {
        'X_mean': preds[:, 0].mean(),
        'X_std': preds[:, 0].std(),
        'X_range': [preds[:, 0].min(), preds[:, 0].max()],
        'Y_mean': preds[:, 1].mean(),
        'Y_std': preds[:, 1].std(),
        'Y_range': [preds[:, 1].min(), preds[:, 1].max()],
        'Z_mean': preds[:, 2].mean(),
        'Z_std': preds[:, 2].std(),
        'Z_range': [preds[:, 2].min(), preds[:, 2].max()],
    }
    
    return {
        'status': 'success',
        'metrics': metrics,
        'stats': stats,
        'pred_file': str(pred_file),
        'model_file': str(model_file)
    }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    output = Path('outputs')
    output.mkdir(exist_ok=True)
    
    for sat in ['geo', 'meo']:
        print(f"\n{'='*60}")
        print(f"Processing {sat.upper()}")
        print(f"{'='*60}")
        
        try:
            result = run_full_pipeline(sat, output)
            
            if result['status'] == 'success':
                print(f"✓ Success!")
                print(f"  MAE: {result['metrics']['validation_mae']:.3f}m")
                print(f"  Q={result['metrics']['Q_scale']:.1f}, R={result['metrics']['R_scale']:.1f}")
                print(f"\n  Predictions:")
                s = result['stats']
                print(f"    X: {s['X_mean']:.2f} ± {s['X_std']:.2f}m, range=[{s['X_range'][0]:.2f}, {s['X_range'][1]:.2f}]")
                print(f"    Y: {s['Y_mean']:.2f} ± {s['Y_std']:.2f}m, range=[{s['Y_range'][0]:.2f}, {s['Y_range'][1]:.2f}]")
                print(f"    Z: {s['Z_mean']:.2f} ± {s['Z_std']:.2f}m, range=[{s['Z_range'][0]:.2f}, {s['Z_range'][1]:.2f}]")
            else:
                print(f"✗ Failed: {result.get('reason', 'unknown')}")
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
