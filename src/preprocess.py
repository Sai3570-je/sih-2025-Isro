"""
Data preprocessing for GNSS error datasets
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_gnss_data(data_folder='data'):
    """
    Load GNSS datasets from SIH CSV files
    
    Returns:
        dict: {'GEO': df_geo, 'MEO': df_meo_combined}
    """
    data_path = Path(data_folder)
    
    datasets = {}
    
    # Load GEO data
    try:
        geo_file = data_path / 'DATA_GEO_Train.csv'
        df_geo = pd.read_csv(geo_file)
        df_geo['satellite_type'] = 'GEO'
        df_geo['satellite_id'] = 'GEO_01'  # Assume single GEO satellite
        logger.info(f"Loaded GEO data: {len(df_geo)} records from {geo_file}")
        datasets['GEO'] = df_geo
    except Exception as e:
        logger.warning(f"Failed to load GEO data: {e}")
    
    # Load MEO data (combine MEO_Train and MEO_Train2)
    try:
        meo_files = [
            data_path / 'DATA_MEO_Train.csv',
            data_path / 'DATA_MEO_Train2.csv'
        ]
        
        meo_dfs = []
        for meo_file in meo_files:
            if meo_file.exists():
                df = pd.read_csv(meo_file)
                df['satellite_type'] = 'MEO'
                meo_dfs.append(df)
                logger.info(f"Loaded MEO data: {len(df)} records from {meo_file}")
        
        if meo_dfs:
            df_meo = pd.concat(meo_dfs, ignore_index=True)
            df_meo['satellite_id'] = 'MEO_01'  # Assume single MEO satellite
            datasets['MEO'] = df_meo
            logger.info(f"Combined MEO data: {len(df_meo)} total records")
    except Exception as e:
        logger.warning(f"Failed to load MEO data: {e}")
    
    return datasets

def standardize_columns(df):
    """Standardize column names across datasets"""
    # Map various column name formats to standard names
    column_mapping = {
        'utc_time': 'timestamp',
        'x_error (m)': 'X_Error',
        'y_error (m)': 'Y_Error',
        'y_error  (m)': 'Y_Error',  # Handle double space
        'z_error (m)': 'Z_Error',
        'satclockerror (m)': 'Clock_Error'
    }
    
    # Rename columns - this may create duplicates
    df_renamed = df.rename(columns=column_mapping)
    
    # Remove duplicate columns (keep first occurrence)
    df_renamed = df_renamed.loc[:, ~df_renamed.columns.duplicated()]
    
    # Parse timestamp
    if 'timestamp' in df_renamed.columns:
        df_renamed['timestamp'] = pd.to_datetime(df_renamed['timestamp'])
    
    # Remove duplicate timestamps by averaging values
    if 'timestamp' in df_renamed.columns and df_renamed.duplicated(subset=['timestamp']).any():
        logger = logging.getLogger(__name__)
        n_dupes = df_renamed.duplicated(subset=['timestamp']).sum()
        logger.warning(f"Found {n_dupes} duplicate timestamps, averaging values")
        # Preserve satellite_id and only average numeric columns
        sat_id = df_renamed['satellite_id'].iloc[0] if 'satellite_id' in df_renamed.columns else None
        df_renamed = df_renamed.groupby('timestamp').mean(numeric_only=True).reset_index()
        if sat_id is not None:
            df_renamed['satellite_id'] = sat_id
    
    return df_renamed

def resample_to_15min(df_sat, max_gap_minutes=45):
    """
    Resample satellite data to exact 15-minute intervals
    
    Args:
        df_sat: DataFrame for single satellite
        max_gap_minutes: Maximum gap to interpolate (minutes)
    
    Returns:
        DataFrame with 15-minute intervals
    """
    df = df_sat.copy()
    df = df.set_index('timestamp').sort_index()
    
    # Create full 15-minute range from first to last ACTUAL data point
    start_time = df.index.min().floor('15min')
    end_time = df.index.max().ceil('15min')
    full_range = pd.date_range(start=start_time, end=end_time, freq='15min')
    
    # Reindex to 15-minute intervals
    df_resampled = df.reindex(full_range)
    
    # Interpolate short gaps only (both forward and backward fill for limits)
    target_cols = ['X_Error', 'Y_Error', 'Z_Error', 'Clock_Error']
    max_limit = max_gap_minutes // 15  # Convert to number of 15-min intervals
    
    for col in target_cols:
        if col in df_resampled.columns:
            # Interpolate with strict limit - only fill gaps up to max_limit intervals
            df_resampled[col] = df_resampled[col].interpolate(
                method='linear', 
                limit=max_limit,
                limit_direction='both'
            )
    
    df_resampled.index.name = 'timestamp'
    df_resampled = df_resampled.reset_index()
    
    # Restore satellite metadata
    for col in ['satellite_id', 'satellite_type']:
        if col in df_sat.columns:
            df_resampled[col] = df_sat[col].iloc[0]
    
    logger.info(f"Resampled to {len(df_resampled)} 15-minute intervals")
    
    return df_resampled

def split_train_test(df, train_days=7):
    """
    Split data into training (days 1-7) and test (day 8)
    For forecasting, we use all available data for training since Day 8 has no observations
    
    Args:
        df: DataFrame with timestamp column
        train_days: Number of days for training
    
    Returns:
        tuple: (df_train, df_test)
    """
    df = df.sort_values('timestamp')
    
    # Find the last timestamp with actual data (not NaN)
    target_cols = ['X_Error', 'Y_Error', 'Z_Error', 'Clock_Error']
    has_data = df[target_cols].notna().any(axis=1)
    last_real_data = df[has_data]['timestamp'].max()
    
    # Use all data up to last real observation for training
    df_train = df[df['timestamp'] <= last_real_data].copy()
    
    # Test set is for Day 8 forecasting (will be empty/NaN in this dataset)
    start_date = df['timestamp'].min().normalize()
    split_date = start_date + pd.Timedelta(days=train_days)
    df_test = df[df['timestamp'] >= split_date].copy()
    
    logger.info(f"Train: {len(df_train)} records (until {last_real_data})")
    logger.info(f"Test: {len(df_test)} records (from {split_date})")
    logger.info(f"Train data with measurements: {has_data.sum()} of {len(df_train)}")
    
    return df_train, df_test

def preprocess_pipeline(data_folder='data', save_temp=True):
    """
    Complete preprocessing pipeline
    
    Returns:
        dict: {satellite_id: {'train': df_train, 'test': df_test}}
    """
    logger.info("="*60)
    logger.info("PREPROCESSING PIPELINE")
    logger.info("="*60)
    
    # Load data
    datasets = load_gnss_data(data_folder)
    
    processed_satellites = {}
    
    for sat_type, df in datasets.items():
        logger.info(f"\nProcessing {sat_type} data...")
        
        # Standardize columns
        df = standardize_columns(df)
        
        # Get unique satellites
        if 'satellite_id' in df.columns:
            satellite_ids = df['satellite_id'].unique()
        else:
            satellite_ids = [f'{sat_type}_01']
            df['satellite_id'] = satellite_ids[0]
        
        for sat_id in satellite_ids:
            try:
                logger.info(f"\n  Satellite: {sat_id}")
                
                df_sat = df[df['satellite_id'] == sat_id].copy()
                
                # Check data sufficiency
                if len(df_sat) < 10:
                    logger.warning(f"  Insufficient data for {sat_id}, skipping")
                    continue
                
                # Resample to 15-minute intervals
                df_resampled = resample_to_15min(df_sat)
                
                # Split train/test
                df_train, df_test = split_train_test(df_resampled, train_days=7)
                
                # Save to temp if requested
                if save_temp:
                    temp_file = Path('temp') / f'{sat_id}_timeseries.parquet'
                    df_resampled.to_parquet(temp_file)
                    logger.info(f"  Saved to {temp_file}")
                
                processed_satellites[sat_id] = {
                    'train': df_train,
                    'test': df_test,
                    'full': df_resampled
                }
                
            except Exception as e:
                logger.error(f"  Error processing {sat_id}: {e}")
                continue
    
    logger.info(f"\n[OK] Preprocessed {len(processed_satellites)} satellites")
    
    return processed_satellites
