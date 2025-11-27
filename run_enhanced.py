"""
Main script to run enhanced Kalman filter training and prediction.
"""

import argparse
import logging
from pathlib import Path
import sys

from src.preprocess import preprocess_pipeline
from src.train_enhanced import train_and_predict_enhanced

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('enhanced_pipeline.log')
    ]
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Enhanced Kalman Filter for Satellite Prediction')
    parser.add_argument('--data-folder', type=str, default='data',
                       help='Folder containing training data')
    parser.add_argument('--output-folder', type=str, default='outputs',
                       help='Folder for outputs')
    parser.add_argument('--satellite', type=str, choices=['geo', 'meo', 'both'], default='both',
                       help='Which satellite to process')
    parser.add_argument('--dt', type=float, default=900.0,
                       help='Time step in seconds (default: 900 = 15 min)')
    
    args = parser.parse_args()
    
    data_folder = Path(args.data_folder)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("ENHANCED KALMAN FILTER PIPELINE")
    logger.info("=" * 80)
    
    # Determine which satellites to process
    satellites = []
    if args.satellite in ['geo', 'both']:
        satellites.append('geo')
    if args.satellite in ['meo', 'both']:
        satellites.append('meo')
    
    results = {}
    
    for sat_type in satellites:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Processing {sat_type.upper()} Satellite")
        logger.info(f"{'=' * 80}")
        
        try:
            # Step 1: Preprocess data (if not already done)
            logger.info(f"Step 1: Ensuring preprocessed data exists...")
            temp_folder = Path('temp')
            train_file = temp_folder / f"{sat_type.upper()}_01_timeseries.parquet"
            
            if not train_file.exists():
                logger.info(f"Preprocessing data...")
                preprocess_pipeline(data_folder=str(data_folder), save_temp=True)
            else:
                logger.info(f"Using existing preprocessed data: {train_file}")
            
            # Copy to outputs folder if needed
            output_file = output_folder / f"{sat_type.upper()}_01_timeseries.parquet"
            if not output_file.exists() and train_file.exists():
                import shutil
                shutil.copy(train_file, output_file)
            
            # Step 2: Train and predict
            logger.info(f"Step 2: Training enhanced filter for {sat_type.upper()}...")
            result = train_and_predict_enhanced(
                satellite_type=sat_type,
                data_folder=data_folder,
                output_folder=output_folder,
                dt=args.dt
            )
            
            results[sat_type] = result
            
            # Print results
            if result['status'] == 'success':
                logger.info(f"\n{sat_type.upper()} Results:")
                logger.info(f"  Training MAE: {result['training_metrics']['validation_mae']:.3f}m")
                logger.info(f"  Best Q scale: {result['training_metrics']['best_Q_scale']:.2f}")
                logger.info(f"  Best R scale: {result['training_metrics']['best_R_scale']:.2f}")
                logger.info(f"\n  Prediction Statistics:")
                stats = result['prediction_stats']
                logger.info(f"    X: {stats['X_mean']:.2f} ± {stats['X_std']:.2f}m, "
                          f"range=[{stats['X_range'][0]:.2f}, {stats['X_range'][1]:.2f}]")
                logger.info(f"    Y: {stats['Y_mean']:.2f} ± {stats['Y_std']:.2f}m, "
                          f"range=[{stats['Y_range'][0]:.2f}, {stats['Y_range'][1]:.2f}]")
                logger.info(f"    Z: {stats['Z_mean']:.2f} ± {stats['Z_std']:.2f}m, "
                          f"range=[{stats['Z_range'][0]:.2f}, {stats['Z_range'][1]:.2f}]")
                logger.info(f"\n  Files saved:")
                logger.info(f"    Predictions: {result['prediction_file']}")
                logger.info(f"    Model: {result['model_file']}")
            else:
                logger.error(f"{sat_type.upper()} failed: {result.get('reason', 'unknown')}")
                
        except Exception as e:
            logger.error(f"Error processing {sat_type.upper()}: {str(e)}", exc_info=True)
            results[sat_type] = {'status': 'error', 'error': str(e)}
    
    # Final summary
    logger.info(f"\n{'=' * 80}")
    logger.info("PIPELINE COMPLETE")
    logger.info(f"{'=' * 80}")
    
    for sat_type, result in results.items():
        if result['status'] == 'success':
            logger.info(f"✓ {sat_type.upper()}: Success")
        else:
            logger.info(f"✗ {sat_type.upper()}: Failed - {result.get('reason', result.get('error', 'unknown'))}")
    
    return results


if __name__ == '__main__':
    main()
