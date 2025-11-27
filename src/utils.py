"""
Utility functions for GNSS error prediction pipeline
"""
import pandas as pd
import numpy as np
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime

def setup_logging(log_file='outputs/pipeline.log'):
    """Setup logging configuration"""
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_output_dirs():
    """Create necessary output directories"""
    dirs = ['outputs', 'models', 'results', 'results/figures', 'temp']
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def save_model_params(sat_id, params, output_dir='models'):
    """Save Kalman filter parameters for a satellite"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(output_dir) / f'kalman_params_{sat_id}.pkl'
    with open(filepath, 'wb') as f:
        pickle.dump(params, f)
    logging.info(f"Saved model parameters for {sat_id} to {filepath}")

def load_model_params(sat_id, output_dir='models'):
    """Load Kalman filter parameters for a satellite"""
    filepath = Path(output_dir) / f'kalman_params_{sat_id}.pkl'
    with open(filepath, 'rb') as f:
        params = pickle.load(f)
    return params

def save_metrics(metrics, filepath='results/metrics_summary.json'):
    """Save evaluation metrics to JSON"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"Saved metrics to {filepath}")

def compute_3d_error(dx, dy, dz):
    """Compute 3D Euclidean position error"""
    return np.sqrt(dx**2 + dy**2 + dz**2)

def compute_metrics(y_true, y_pred, metric_name=''):
    """Compute MAE, RMSE, and R² metrics"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    # R² calculation
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        f'{metric_name}_MAE': float(mae),
        f'{metric_name}_RMSE': float(rmse),
        f'{metric_name}_R2': float(r2)
    }

def log_environment_info():
    """Log Python version and key package versions"""
    import sys
    import pandas
    import numpy
    
    info = {
        'timestamp': datetime.now().isoformat(),
        'python_version': sys.version,
        'pandas_version': pandas.__version__,
        'numpy_version': numpy.__version__
    }
    
    logging.info(f"Environment: {json.dumps(info, indent=2)}")
    return info
