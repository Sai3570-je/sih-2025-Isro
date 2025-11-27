"""
Prediction and forecasting module
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from .kalman_filter import build_state_matrices, run_kalman_forward, forecast_day8

logger = logging.getLogger(__name__)

def predict_day8_satellite(sat_id, df_train, params, output_dir='outputs'):
    """
    Generate Day 8 predictions for a single satellite
    
    Args:
        sat_id: Satellite identifier
        df_train: Training DataFrame
        params: Kalman filter parameters {Q, R, x0, P0}
        output_dir: Output directory
    
    Returns:
        DataFrame: Day 8 predictions
    """
    logger.info(f"Generating Day 8 forecast for {sat_id}...")
    
    # Build state matrices
    F, H = build_state_matrices(dt=900.0)
    
    # Run forward filter on training data to get final state
    result = run_kalman_forward(
        df_train, 
        params['Q'], 
        params['R'], 
        params['x0'], 
        params['P0'],
        F, 
        H
    )
    
    last_state = result['final_state']
    last_P = result['final_cov']
    
    # Generate Day 8 timestamps (96 intervals of 15 minutes)
    last_timestamp = df_train['timestamp'].max()
    day8_start = (last_timestamp + pd.Timedelta(days=1)).normalize()
    
    day8_timestamps = [day8_start + pd.Timedelta(minutes=15*i) for i in range(96)]
    
    # Forecast Day 8
    predictions = forecast_day8(last_state, last_P, steps=96, F=F, Q=params['Q'], H=H)
    
    # Create DataFrame
    pred_df = pd.DataFrame({
        'timestamp': day8_timestamps,
        'satellite_id': sat_id,
        'X_Error_pred': predictions[:, 0],
        'Y_Error_pred': predictions[:, 1],
        'Z_Error_pred': predictions[:, 2],
        'Clock_Error_pred': predictions[:, 3]
    })
    
    logger.info(f"  Generated {len(pred_df)} predictions")
    
    return pred_df

def generate_all_predictions(satellites_data, models_dir='models', output_dir='outputs'):
    """
    Generate Day 8 predictions for all satellites
    
    Args:
        satellites_data: Dict of {sat_id: {'train': df, 'test': df}}
        models_dir: Directory containing model parameters
        output_dir: Output directory
    
    Returns:
        dict: {sat_type: DataFrame} of predictions
    """
    logger.info("\n" + "="*60)
    logger.info("GENERATING DAY 8 PREDICTIONS")
    logger.info("="*60)
    
    predictions = {'GEO': [], 'MEO': []}
    
    for sat_id, data in satellites_data.items():
        try:
            # Load model parameters
            from .utils import load_model_params
            params = load_model_params(sat_id, models_dir)
            
            # Generate predictions
            pred_df = predict_day8_satellite(
                sat_id, 
                data['train'], 
                params, 
                output_dir
            )
            
            # Categorize by satellite type
            if 'GEO' in sat_id:
                predictions['GEO'].append(pred_df)
            else:
                predictions['MEO'].append(pred_df)
        
        except Exception as e:
            logger.error(f"Failed to predict for {sat_id}: {e}")
            continue
    
    # Combine predictions by type
    results = {}
    
    if predictions['GEO']:
        df_geo = pd.concat(predictions['GEO'], ignore_index=True)
        geo_file = Path(output_dir) / 'predictions_day8_geo.csv'
        df_geo.to_csv(geo_file, index=False)
        logger.info(f"[OK] Saved GEO predictions to {geo_file}")
        results['GEO'] = df_geo
    
    if predictions['MEO']:
        df_meo = pd.concat(predictions['MEO'], ignore_index=True)
        meo_file = Path(output_dir) / 'predictions_day8_meo.csv'
        df_meo.to_csv(meo_file, index=False)
        logger.info(f"[OK] Saved MEO predictions to {meo_file}")
        results['MEO'] = df_meo
    
    return results
