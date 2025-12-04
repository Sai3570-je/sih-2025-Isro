"""
Train Wavelet+Kalman on time-series grouped data and forecast Day 8
Uses the same proven approach but applies it per time slot for better pattern learning
"""
import os
import glob
import pandas as pd
import numpy as np
import pywt
from pathlib import Path
from datetime import datetime, timedelta

class WaveletKalmanTimeSeriesPredictor:
    def __init__(self, Q=0.01, R=0.1, wavelet='coif2', level=2):
        self.Q = Q  # Process noise
        self.R = R  # Measurement noise
        self.wavelet = wavelet
        self.level = level
        
    def wavelet_denoise(self, data):
        """Denoise using wavelet transform"""
        # Remove NaN values first
        clean_data = data[~np.isnan(data)]
        
        if len(clean_data) < 4:
            return clean_data  # Not enough points for wavelet
        
        # If data range is too small, skip denoising
        if np.abs(clean_data).max() < 1e-6:
            return clean_data
        
        try:
            # Use level=1 for very short sequences (4-7 points)
            actual_level = 1 if len(clean_data) < 8 else self.level
            coeffs = pywt.wavedec(clean_data, self.wavelet, level=actual_level, mode='smooth')
            
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            if sigma < 1e-10:  # Avoid division by very small numbers
                return clean_data
            
            threshold = sigma * np.sqrt(2 * np.log(len(clean_data)))
            coeffs_thresh = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
            denoised = pywt.waverec(coeffs_thresh, self.wavelet, mode='smooth')
            return denoised[:len(clean_data)]
        except:
            return clean_data  # Fallback to original if wavelet fails
    
    def kalman_predict(self, measurements):
        """
        1D Kalman filter for time series prediction with uncertainty estimation
        
        Returns:
            tuple: (prediction, uncertainty_std)
        """
        if len(measurements) == 0:
            return np.nan, np.nan
        
        # Remove NaN values
        clean_measurements = measurements[~np.isnan(measurements)]
        
        if len(clean_measurements) == 0:
            return np.nan, np.nan
        
        # If all values very close to zero, use mean instead of Kalman
        if np.abs(clean_measurements).max() < 1e-6:
            mean_val = float(np.mean(clean_measurements))
            # Uncertainty = std of data (or small value if constant)
            uncertainty = float(np.std(clean_measurements)) if len(clean_measurements) > 1 else 0.1
            return mean_val, uncertainty
        
        # Initialize state with first measurement
        x = clean_measurements[0]
        P = 1.0  # Estimate uncertainty
        
        # Process measurements
        for z in clean_measurements:
            # Predict
            x_prior = x
            P_prior = P + self.Q
            
            # Update
            K = P_prior / (P_prior + self.R)
            x = x_prior + K * (z - x_prior)
            P = (1 - K) * P_prior
        
        # One-step-ahead prediction with trend estimation
        # Use last two points to estimate trend if available
        if len(clean_measurements) >= 2:
            trend = clean_measurements[-1] - clean_measurements[-2]
            x_forecast = x + trend * 0.5  # Add 50% of last trend
            
            # Increase uncertainty when extrapolating with trend
            # This prevents overfitting to recent changes
            P_forecast = P + self.Q * 2  # Double process noise for forecast step
        else:
            x_forecast = x
            P_forecast = P + self.Q
        
        # Return prediction and uncertainty (standard deviation)
        uncertainty = np.sqrt(P_forecast)
        
        return x_forecast, uncertainty
    
    def train_and_predict(self, time_series_data):
        """
        Train on days 1-7 and predict day 8 with uncertainty
        
        Args:
            time_series_data: List of values [day1, day2, ..., day7]
        
        Returns:
            tuple: (day8_prediction, uncertainty_std, confidence_lower, confidence_upper)
        """
        # Convert to numpy array and remove NaN
        data_array = np.array(time_series_data)
        clean_data = data_array[~np.isnan(data_array)]
        
        if len(clean_data) < 2:
            # Not enough data, return mean if available
            if len(clean_data) == 1:
                return clean_data[0], np.nan, np.nan, np.nan
            return np.nan, np.nan, np.nan, np.nan
        
        # Overfitting prevention: Reject if variance is too small
        # (indicates model might memorize noise instead of learning pattern)
        data_std = np.std(clean_data)
        if data_std < 1e-8 and len(clean_data) < 5:
            # Too few samples + no variation = high overfitting risk
            # Return mean with wide uncertainty
            mean_val = np.mean(clean_data)
            uncertainty = 1.0  # Wide uncertainty
            return mean_val, uncertainty, mean_val - 1.96*uncertainty, mean_val + 1.96*uncertainty
        
        # Step 1: Wavelet denoise
        denoised = self.wavelet_denoise(clean_data)
        
        # Step 2: Kalman filter and predict with uncertainty
        prediction, kalman_uncertainty = self.kalman_predict(denoised)
        
        # Step 3: Adjust uncertainty based on data variability
        # Prevents overfitting by ensuring uncertainty reflects actual data spread
        data_variability = np.std(clean_data)
        
        # Use maximum of Kalman uncertainty and data-based uncertainty
        # This prevents unrealistically confident predictions
        if not np.isnan(kalman_uncertainty) and not np.isnan(data_variability):
            # Combine both sources of uncertainty
            # Kalman gives us short-term uncertainty, data std gives long-term
            uncertainty = np.sqrt(kalman_uncertainty**2 + (data_variability * 0.5)**2)
            
            # Additional penalty for small sample sizes (overfitting prevention)
            if len(clean_data) < 5:
                uncertainty *= 1.5  # Increase uncertainty by 50% for <5 samples
            elif len(clean_data) < 7:
                uncertainty *= 1.2  # Increase uncertainty by 20% for <7 samples
        else:
            uncertainty = kalman_uncertainty if not np.isnan(kalman_uncertainty) else data_variability
        
        # Step 4: Calculate 95% confidence intervals
        # Using 1.96 std deviations (95% for Gaussian)
        if not np.isnan(uncertainty):
            confidence_lower = prediction - 1.96 * uncertainty
            confidence_upper = prediction + 1.96 * uncertainty
        else:
            confidence_lower = np.nan
            confidence_upper = np.nan
        
        return prediction, uncertainty, confidence_lower, confidence_upper


def process_time_slot_file(filepath, predictor):
    """Process a single time slot file and predict Day 8"""
    df = pd.read_csv(filepath)
    
    if len(df) == 0:
        return None
    
    # Extract time of day from filename
    filename = os.path.basename(filepath)
    time_of_day = filename.replace('.csv', '').replace('-', ':')
    
    # Group by date and average if multiple satellites at same time
    df['date'] = pd.to_datetime(df['date'])
    daily_avg = df.groupby('date').agg({
        'x_error': 'mean',
        'y_error': 'mean',
        'z_error': 'mean',
        'clock_error': 'mean'
    }).reset_index()
    
    # Sort by date
    daily_avg = daily_avg.sort_values('date')
    
    # Extract days 1-7 (filter out day 8 and 9 if present)
    min_date = daily_avg['date'].min()
    max_train_date = min_date + timedelta(days=6)
    train_data = daily_avg[daily_avg['date'] <= max_train_date]
    
    if len(train_data) < 2:
        return None  # Not enough training data
    
    # Predict Day 8 for each error type
    predictions = {
        'time_of_day': time_of_day,
        'num_training_days': len(train_data)
    }
    
    for col in ['x_error', 'y_error', 'z_error', 'clock_error']:
        if col in train_data.columns:
            series = train_data[col].values
            pred, uncertainty, conf_lower, conf_upper = predictor.train_and_predict(series)
            predictions[f'{col}_day8'] = pred
            predictions[f'{col}_uncertainty'] = uncertainty
            predictions[f'{col}_conf_lower'] = conf_lower
            predictions[f'{col}_conf_upper'] = conf_upper
        else:
            predictions[f'{col}_day8'] = np.nan
            predictions[f'{col}_uncertainty'] = np.nan
            predictions[f'{col}_conf_lower'] = np.nan
            predictions[f'{col}_conf_upper'] = np.nan
    
    return predictions


def train_all_timeslots_and_forecast(
    input_dir='outputs/time_series_groups',
    output_csv='outputs/day8_forecast_timeslots.csv',
    Q=0.01,
    R=0.1
):
    """
    Train on all time slot files and generate Day 8 forecast
    
    Args:
        input_dir: Directory containing time slot CSV files
        output_csv: Output file for Day 8 predictions
        Q: Kalman process noise
        R: Kalman measurement noise
    """
    print("="*80)
    print("TIME-SERIES WAVELET+KALMAN TRAINING AND FORECASTING")
    print("="*80)
    
    # Initialize predictor
    predictor = WaveletKalmanTimeSeriesPredictor(Q=Q, R=R, wavelet='coif2', level=2)
    print(f"\n✓ Initialized predictor:")
    print(f"  Wavelet: coif2, Level: 2")
    print(f"  Kalman Q: {Q}, R: {R}")
    
    # Find all time slot CSV files
    pattern = os.path.join(input_dir, '*.csv')
    files = [f for f in glob.glob(pattern) if not f.endswith('_all_combined.csv')]
    
    print(f"\n✓ Found {len(files)} time slot files")
    
    if len(files) == 0:
        raise ValueError(f"No CSV files found in {input_dir}")
    
    # Process each time slot
    results = []
    success_count = 0
    skip_count = 0
    
    print(f"\n{'='*80}")
    print("TRAINING EACH TIME SLOT...")
    print(f"{'='*80}\n")
    
    for i, filepath in enumerate(sorted(files), 1):
        filename = os.path.basename(filepath)
        
        try:
            result = process_time_slot_file(filepath, predictor)
            
            if result is not None:
                results.append(result)
                success_count += 1
                if i % 20 == 0:
                    print(f"  Processed {i}/{len(files)} time slots... (✓ {success_count} predictions)")
            else:
                skip_count += 1
                
        except Exception as e:
            print(f"  ⚠ Error processing {filename}: {e}")
            skip_count += 1
    
    print(f"\n✓ Training complete!")
    print(f"  Successful predictions: {success_count}")
    print(f"  Skipped (insufficient data): {skip_count}")
    
    # Create output dataframe
    forecast_df = pd.DataFrame(results)
    
    # Sort by time of day
    forecast_df = forecast_df.sort_values('time_of_day').reset_index(drop=True)
    
    # Save results
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    forecast_df.to_csv(output_csv, index=False)
    
    print(f"\n{'='*80}")
    print("DAY 8 FORECAST COMPLETE!")
    print(f"{'='*80}")
    print(f"Output saved to: {output_csv}")
    print(f"Total predictions: {len(forecast_df)} time slots")
    
    # Show summary statistics
    print(f"\n{'='*80}")
    print("FORECAST SUMMARY (Day 8 Predictions)")
    print(f"{'='*80}")
    
    for col in ['x_error_day8', 'y_error_day8', 'z_error_day8', 'clock_error_day8']:
        if col in forecast_df.columns:
            values = forecast_df[col].dropna()
            if len(values) > 0:
                print(f"\n{col}:")
                print(f"  Mean:   {values.mean():8.4f} m")
                print(f"  Std:    {values.std():8.4f} m")
                print(f"  Min:    {values.min():8.4f} m")
                print(f"  Max:    {values.max():8.4f} m")
    
    print(f"\n{'='*80}")
    print("NEXT STEPS:")
    print(f"{'='*80}")
    print("1. Review forecast file for predicted Day 8 values per time slot")
    print("2. When Day 8 ground truth available, compute residuals")
    print("3. Test Gaussianity per time slot (Shapiro-Wilk p > 0.05)")
    print("4. Validate MAE/RMSE per time slot")
    print(f"{'='*80}\n")
    
    return forecast_df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train Wavelet+Kalman on time-series groups and forecast Day 8'
    )
    parser.add_argument('--input', default='outputs/time_series_groups',
                       help='Directory with time slot CSV files')
    parser.add_argument('--output', default='outputs/day8_forecast_timeslots.csv',
                       help='Output CSV for Day 8 predictions')
    parser.add_argument('--Q', type=float, default=0.01,
                       help='Kalman process noise (default: 0.01)')
    parser.add_argument('--R', type=float, default=0.1,
                       help='Kalman measurement noise (default: 0.1)')
    
    args = parser.parse_args()
    
    train_all_timeslots_and_forecast(
        input_dir=args.input,
        output_csv=args.output,
        Q=args.Q,
        R=args.R
    )
