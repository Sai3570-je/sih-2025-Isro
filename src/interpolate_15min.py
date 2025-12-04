"""
Interpolate time-series predictions to 15-minute intervals
Takes hourly/irregular predictions and creates smooth 15-min grid (96 points for Day 8)
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def interpolate_to_15min_grid(
    input_csv='outputs/day8_forecast_timeslots.csv',
    output_csv='outputs/day8_forecast_15min.csv'
):
    """
    Convert irregular time predictions to standard 15-minute grid
    
    Args:
        input_csv: Path to time-slot predictions
        output_csv: Path to save 15-min interpolated predictions
    """
    print("="*80)
    print("INTERPOLATING TO 15-MINUTE INTERVALS")
    print("="*80)
    
    # Load predictions
    df = pd.read_csv(input_csv)
    print(f"\n✓ Loaded {len(df)} time-slot predictions")
    
    # Parse time_of_day to minutes since midnight
    def time_to_minutes(time_str):
        """Convert HH:MM to minutes since midnight"""
        h, m = map(int, time_str.split(':'))
        return h * 60 + m
    
    df['minutes'] = df['time_of_day'].apply(time_to_minutes)
    df = df.sort_values('minutes').reset_index(drop=True)
    
    print(f"  Time range: {df['time_of_day'].min()} to {df['time_of_day'].max()}")
    
    # Generate 15-minute grid (96 points for 24 hours)
    grid_times = []
    for h in range(24):
        for m in [0, 15, 30, 45]:
            grid_times.append(f"{h:02d}:{m:02d}")
    
    grid_minutes = [time_to_minutes(t) for t in grid_times]
    
    print(f"\n✓ Generating 15-minute grid: 96 intervals")
    
    # Interpolate each error column
    error_cols = ['x_error_day8', 'y_error_day8', 'z_error_day8', 'clock_error_day8']
    
    interpolated_data = []
    
    for i, (grid_time, grid_min) in enumerate(zip(grid_times, grid_minutes)):
        row = {'time_of_day': grid_time, 'minutes': grid_min}
        
        for col in error_cols:
            if col not in df.columns:
                row[col] = np.nan
                continue
            
            # Check if we have exact match
            exact_match = df[df['minutes'] == grid_min]
            if len(exact_match) > 0:
                # Use exact prediction
                row[col] = exact_match.iloc[0][col]
            else:
                # Interpolate between nearest times
                before = df[df['minutes'] < grid_min]
                after = df[df['minutes'] > grid_min]
                
                if len(before) > 0 and len(after) > 0:
                    # Linear interpolation
                    t1 = before.iloc[-1]['minutes']
                    t2 = after.iloc[0]['minutes']
                    v1 = before.iloc[-1][col]
                    v2 = after.iloc[0][col]
                    
                    # Interpolation factor
                    alpha = (grid_min - t1) / (t2 - t1)
                    row[col] = v1 + alpha * (v2 - v1)
                    
                elif len(before) > 0:
                    # Use last known value
                    row[col] = before.iloc[-1][col]
                elif len(after) > 0:
                    # Use next known value
                    row[col] = after.iloc[0][col]
                else:
                    # No data available
                    row[col] = np.nan
        
        interpolated_data.append(row)
    
    # Create output dataframe
    result = pd.DataFrame(interpolated_data)
    result = result[['time_of_day'] + error_cols]
    
    # Save
    result.to_csv(output_csv, index=False)
    
    print(f"\n{'='*80}")
    print("INTERPOLATION COMPLETE!")
    print(f"{'='*80}")
    print(f"Output saved to: {output_csv}")
    print(f"Total predictions: {len(result)} (96 time slots at 15-min intervals)")
    
    # Show statistics
    print(f"\n{'='*80}")
    print("DAY 8 FORECAST - 15-MINUTE INTERVALS")
    print(f"{'='*80}")
    
    for col in error_cols:
        if col in result.columns:
            values = result[col].dropna()
            if len(values) > 0:
                print(f"\n{col}:")
                print(f"  Mean:   {values.mean():8.4f} m")
                print(f"  Std:    {values.std():8.4f} m")
                print(f"  Min:    {values.min():8.4f} m")
                print(f"  Max:    {values.max():8.4f} m")
                print(f"  Points: {len(values)}/96")
    
    # Show sample predictions
    print(f"\n{'='*80}")
    print("SAMPLE PREDICTIONS (Every Hour)")
    print(f"{'='*80}")
    print("\nTime    | X_Error  | Y_Error  | Z_Error  | Clock_Error")
    print("-" * 60)
    
    for i in range(0, len(result), 4):  # Every hour (4 x 15min = 1 hour)
        row = result.iloc[i]
        print(f"{row['time_of_day']} | {row['x_error_day8']:8.4f} | {row['y_error_day8']:8.4f} | {row['z_error_day8']:8.4f} | {row['clock_error_day8']:8.4f}")
    
    print(f"\n{'='*80}")
    print("✓ READY FOR ISRO SUBMISSION!")
    print(f"{'='*80}")
    print("Your Day 8 forecast now has:")
    print("  - 96 predictions (complete 24-hour coverage)")
    print("  - 15-minute intervals (00:00, 00:15, ..., 23:45)")
    print("  - Time-specific learned patterns (each time is different!)")
    print("  - Smooth transitions between trained time slots")
    print(f"{'='*80}\n")
    
    return result


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Interpolate time-slot predictions to 15-minute grid'
    )
    parser.add_argument('--input', default='outputs/day8_forecast_timeslots.csv',
                       help='Input CSV with time-slot predictions')
    parser.add_argument('--output', default='outputs/day8_forecast_15min.csv',
                       help='Output CSV with 15-min interpolated predictions')
    
    args = parser.parse_args()
    
    interpolate_to_15min_grid(args.input, args.output)
