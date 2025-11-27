"""
Evaluation and metrics module
"""
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from .utils import compute_metrics, compute_3d_error, save_metrics

logger = logging.getLogger(__name__)

def evaluate_satellite(sat_id, df_pred, df_actual):
    """
    Evaluate predictions for a single satellite
    
    Args:
        sat_id: Satellite identifier
        df_pred: Predictions DataFrame
        df_actual: Actual values DataFrame
    
    Returns:
        dict: Evaluation metrics
    """
    logger.info(f"Evaluating {sat_id}...")
    
    # Merge on timestamp
    df_merged = pd.merge(df_pred, df_actual, on='timestamp', how='inner', suffixes=('_pred', '_actual'))
    
    if len(df_merged) == 0:
        logger.warning(f"  No matching timestamps for evaluation")
        return None
    
    metrics = {'satellite_id': sat_id}
    
    # Compute metrics for each component
    components = ['X_Error', 'Y_Error', 'Z_Error', 'Clock_Error']
    
    for comp in components:
        pred_col = f'{comp}_pred'
        actual_col = comp if comp in df_merged.columns else f'{comp}_actual'
        
        if pred_col in df_merged.columns and actual_col in df_merged.columns:
            comp_metrics = compute_metrics(
                df_merged[actual_col], 
                df_merged[pred_col], 
                metric_name=comp
            )
            metrics.update(comp_metrics)
    
    # Compute 3D position error
    if all(col in df_merged.columns for col in ['X_Error', 'Y_Error', 'Z_Error']):
        dx = df_merged['X_Error'] - df_merged['X_Error_pred']
        dy = df_merged['Y_Error'] - df_merged['Y_Error_pred']
        dz = df_merged['Z_Error'] - df_merged['Z_Error_pred']
        
        error_3d = compute_3d_error(dx, dy, dz)
        
        metrics['3D_Error_Mean'] = float(np.mean(error_3d))
        metrics['3D_Error_Median'] = float(np.median(error_3d))
        metrics['3D_Error_95th'] = float(np.percentile(error_3d, 95))
    
    logger.info(f"  Metrics computed: {len(metrics)} values")
    
    return metrics

def evaluate_all_satellites(predictions, satellites_data, output_file='results/metrics_summary.json'):
    """
    Evaluate predictions for all satellites
    
    Args:
        predictions: Dict of prediction DataFrames
        satellites_data: Dict of actual data
        output_file: Output file for metrics
    
    Returns:
        dict: All metrics
    """
    logger.info("\n" + "="*60)
    logger.info("EVALUATION")
    logger.info("="*60)
    
    all_metrics = []
    
    for sat_id, data in satellites_data.items():
        try:
            # Find corresponding predictions
            sat_type = 'GEO' if 'GEO' in sat_id else 'MEO'
            
            if sat_type not in predictions:
                logger.warning(f"No predictions found for {sat_id}")
                continue
            
            df_pred = predictions[sat_type]
            df_pred_sat = df_pred[df_pred['satellite_id'] == sat_id]
            
            if len(df_pred_sat) == 0:
                logger.warning(f"No predictions for {sat_id} in {sat_type} file")
                continue
            
            # Get actual test data
            df_actual = data.get('test')
            
            if df_actual is None or len(df_actual) == 0:
                logger.warning(f"No test data for {sat_id}")
                continue
            
            # Evaluate
            metrics = evaluate_satellite(sat_id, df_pred_sat, df_actual)
            
            if metrics:
                all_metrics.append(metrics)
        
        except Exception as e:
            logger.error(f"Evaluation failed for {sat_id}: {e}")
            continue
    
    # Compute global aggregates
    if all_metrics:
        global_metrics = compute_global_metrics(all_metrics)
        
        summary = {
            'per_satellite': all_metrics,
            'global': global_metrics
        }
        
        save_metrics(summary, output_file)
        
        logger.info(f"\n📊 GLOBAL METRICS:")
        for key, value in global_metrics.items():
            logger.info(f"  {key}: {value:.6f}")
    
    return all_metrics

def compute_global_metrics(all_metrics):
    """Compute aggregated global metrics"""
    global_metrics = {}
    
    # Aggregate across all satellites
    metric_keys = [k for k in all_metrics[0].keys() if k != 'satellite_id']
    
    for key in metric_keys:
        values = [m[key] for m in all_metrics if key in m]
        if values:
            global_metrics[f'{key}_mean'] = float(np.mean(values))
            global_metrics[f'{key}_std'] = float(np.std(values))
    
    return global_metrics

def plot_predictions_vs_actual(sat_id, df_pred, df_actual, output_dir='results/figures'):
    """
    Plot predicted vs actual time series
    
    Args:
        sat_id: Satellite identifier
        df_pred: Predictions DataFrame
        df_actual: Actual DataFrame
        output_dir: Output directory for plots
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Predictions vs Actual - {sat_id}', fontsize=16, fontweight='bold')
    
    components = ['X_Error', 'Y_Error', 'Z_Error', 'Clock_Error']
    titles = ['X Position Error', 'Y Position Error', 'Z Position Error', 'Clock Error']
    
    for idx, (comp, title, ax) in enumerate(zip(components, titles, axes.flat)):
        # Plot actual
        if comp in df_actual.columns:
            ax.plot(df_actual['timestamp'], df_actual[comp], 
                   'b-', label='Actual', linewidth=2, alpha=0.7)
        
        # Plot predicted
        pred_col = f'{comp}_pred'
        if pred_col in df_pred.columns:
            ax.plot(df_pred['timestamp'], df_pred[pred_col], 
                   'r--', label='Predicted', linewidth=2, alpha=0.7)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Error (m)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    output_file = Path(output_dir) / f'{sat_id}_predictions.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Saved plot to {output_file}")

def generate_all_plots(predictions, satellites_data, output_dir='results/figures'):
    """Generate plots for all satellites"""
    logger.info("\nGenerating plots...")
    
    for sat_id, data in satellites_data.items():
        try:
            sat_type = 'GEO' if 'GEO' in sat_id else 'MEO'
            
            if sat_type not in predictions:
                continue
            
            df_pred = predictions[sat_type]
            df_pred_sat = df_pred[df_pred['satellite_id'] == sat_id]
            
            df_actual = data.get('test')
            
            if df_actual is not None and len(df_pred_sat) > 0:
                plot_predictions_vs_actual(sat_id, df_pred_sat, df_actual, output_dir)
        
        except Exception as e:
            logger.error(f"Plot generation failed for {sat_id}: {e}")
            continue
