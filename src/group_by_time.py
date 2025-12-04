import os
import pandas as pd
from pathlib import Path

"""
Group satellite data by identical time-of-day across all days and datasets.
Creates one CSV per unique time found (e.g., 06:00.csv, 06:01.csv, etc.)
Each file contains all occurrences of that time across Days 1-9 from all satellites.
"""

def group_by_time_of_day(
    input_files: list,
    satellite_names: list,
    output_dir: str = 'outputs/time_series_groups',
):
    """
    Group all datasets by time-of-day and create one CSV per unique time.
    
    Args:
        input_files: List of CSV file paths
        satellite_names: List of satellite identifiers
        output_dir: Output directory for grouped CSVs
    """
    print("="*80)
    print("GROUPING DATA BY TIME-OF-DAY")
    print("="*80)
    
    all_data = []
    
    # Load all datasets
    for path, sat_name in zip(input_files, satellite_names):
        if not os.path.isfile(path):
            print(f"Warning: File not found, skipping: {path}")
            continue
        
        df = pd.read_csv(path)
        
        # Find time column
        time_col = None
        for col in df.columns:
            norm = str(col).strip().lower().replace(' ', '').replace('-', '').replace('_', '')
            if norm in {'timestamp', 'time', 'utctime', 'utc'}:
                time_col = col
                break
        
        if time_col is None:
            print(f"Warning: No time column found in {path}, skipping")
            continue
        
        # Parse timestamps
        df['timestamp'] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        
        # Extract date and time components
        df['date'] = df['timestamp'].dt.date
        df['time_of_day'] = df['timestamp'].dt.strftime('%H:%M')
        df['satellite'] = sat_name
        
        # Normalize column names
        rename_map = {}
        for col in df.columns:
            if col in ['timestamp', 'date', 'time_of_day', 'satellite']:
                continue
            norm = str(col).lower().strip().replace('(m)', '').replace('  ', ' ').strip().replace(' ', '_')
            if 'x_error' in norm or 'xerror' in norm:
                rename_map[col] = 'x_error'
            elif 'y_error' in norm or 'yerror' in norm:
                rename_map[col] = 'y_error'
            elif 'z_error' in norm or 'zerror' in norm:
                rename_map[col] = 'z_error'
            elif 'clock' in norm or 'satclock' in norm:
                rename_map[col] = 'clock_error'
        
        df = df.rename(columns=rename_map)
        
        # Keep relevant columns
        keep_cols = ['timestamp', 'date', 'time_of_day', 'satellite', 
                     'x_error', 'y_error', 'z_error', 'clock_error']
        available = [c for c in keep_cols if c in df.columns]
        df = df[available]
        
        all_data.append(df)
        print(f"✓ Loaded {len(df)} rows from {sat_name}")
    
    if not all_data:
        raise ValueError("No valid data loaded")
    
    # Combine all data
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values(['time_of_day', 'date', 'satellite']).reset_index(drop=True)
    
    print(f"\n✓ Total combined: {len(combined)} rows")
    print(f"  Date range: {combined['date'].min()} to {combined['date'].max()}")
    print(f"  Satellites: {sorted(combined['satellite'].unique())}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by time_of_day and save each group
    unique_times = sorted(combined['time_of_day'].unique())
    print(f"\n✓ Found {len(unique_times)} unique times")
    print(f"  Creating one CSV file per time...")
    
    for time_val in unique_times:
        group = combined[combined['time_of_day'] == time_val].copy()
        group = group.sort_values(['date', 'satellite']).reset_index(drop=True)
        
        # Create filename (replace : with -)
        filename = f"{time_val.replace(':', '-')}.csv"
        filepath = os.path.join(output_dir, filename)
        
        group.to_csv(filepath, index=False)
    
    # Also save a combined reference file
    combined_path = os.path.join(output_dir, '_all_combined.csv')
    combined.to_csv(combined_path, index=False)
    
    print(f"\n{'='*80}")
    print(f"✓ GROUPING COMPLETE!")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    print(f"Files created: {len(unique_times)} time-specific CSVs + 1 combined CSV")
    print(f"\nExample files:")
    for i, time_val in enumerate(sorted(unique_times)[:5]):
        count = len(combined[combined['time_of_day'] == time_val])
        print(f"  {time_val.replace(':', '-')}.csv -> {count} rows (same time across all days/satellites)")
    if len(unique_times) > 5:
        print(f"  ... and {len(unique_times) - 5} more")
    print(f"\nNow you can train on each time group separately!")
    print(f"{'='*80}")
    
    return combined


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Group satellite data by time-of-day (creates one CSV per unique time)'
    )
    parser.add_argument('--inputs', nargs='+', required=True, 
                       help='Input CSV files (e.g., DATA_MEO_Train.csv DATA_MEO_Train2.csv DATA_GEO_Train.csv)')
    parser.add_argument('--names', nargs='+', required=True,
                       help='Satellite names (e.g., MEO1 MEO2 GEO)')
    parser.add_argument('--out', default='outputs/time_series_groups',
                       help='Output directory (default: outputs/time_series_groups)')
    
    args = parser.parse_args()
    
    if len(args.inputs) != len(args.names):
        raise ValueError("Number of input files must match number of satellite names")
    
    group_by_time_of_day(args.inputs, args.names, args.out)
