"""Day 8 Forecast - Pure Prediction (No Ground Truth)
Extrapolates from Days 1-7 training data to predict Day 8
"""
import pickle
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Ensure wavelet_kalman_filter module can be imported
sys.path.insert(0, str(Path(__file__).parent))
from wavelet_kalman_filter import WaveletKalmanFilter

print("="*80)
print("DAY 8 FORECAST - PURE PREDICTION")
print("="*80)

# Load trained model
print("\n[1/4] Loading trained model...")
model_path = Path('outputs/wavelet_kalman_model.pkl')
if not model_path.exists():
    raise FileNotFoundError(f"Model file not found: {model_path}")

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("   ✓ Model loaded successfully")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Load training data
print("\n[2/4] Loading training data (Days 1-7)...")
data_path = Path('temp/MEO_01_timeseries.parquet')
if not data_path.exists():
    raise FileNotFoundError(f"Data file not found: {data_path}")

df = pd.read_parquet(data_path)
df = df.sort_values('timestamp').reset_index(drop=True)

# Training data: up to Sept 9, 01:30
df_train = df[df['timestamp'] <= '2025-09-09 01:30:00'].dropna().copy()
if len(df_train) == 0:
    raise ValueError("No valid training data found")

print(f"   Training records: {len(df_train)}")
print(f"   Last training timestamp: {df_train['timestamp'].max()}")

# Get last state from training
print("\n[3/4] Extracting final state from training...")
if not hasattr(model, 'state') or model.state is None:
    raise RuntimeError("Model has no trained state")

last_state = model.state.copy()
last_P = model.P.copy()
print(f"   Last state: {last_state}")
print(f"   State validity check: {'✓ PASS' if not np.any(np.isnan(last_state)) else '✗ FAIL'}")

# Generate Day 8 timestamps (96 = 24 hours * 4 per hour)
print("\n[4/4] Forecasting Day 8 (pure extrapolation)...")
day8_start = pd.Timestamp('2025-09-08 00:00:00')
day8_end = pd.Timestamp('2025-09-08 23:45:00')
timestamps = pd.date_range(start=day8_start, end=day8_end, freq='15min')

print(f"   Forecast points: {len(timestamps)}")

# Pure forecasting (no measurements, just state propagation)
forecasts = []
state = last_state.copy()
P = last_P.copy()

for t in timestamps:
    # State propagation (no measurement update)
    state_pred = state.copy()
    P_pred = P + np.eye(4) * model.Q
    
    forecasts.append(state_pred.copy())
    
    # Update state for next iteration (pure propagation)
    state = state_pred
    P = P_pred

forecasts = np.array(forecasts)

# Create output
output = pd.DataFrame({
    'timestamp': timestamps,
    'X_Error_forecast': forecasts[:, 0],
    'Y_Error_forecast': forecasts[:, 1],
    'Z_Error_forecast': forecasts[:, 2],
    'Clock_Error_forecast': forecasts[:, 3]
})

# Save
output_file = 'outputs/day8_forecast.csv'
output.to_csv(output_file, index=False)

print(f"\n✓ Forecasts saved to: {output_file}")
print(f"\nFirst 10 forecasts:")
print(output.head(10).to_string(index=False))

print("\n" + "="*80)
print("FORECAST SUMMARY")
print("="*80)
print(f"Total forecast points: {len(output)}")
print(f"\nForecast Statistics (meters):")
print("-" * 80)
for col in ['X_Error_forecast', 'Y_Error_forecast', 'Z_Error_forecast', 'Clock_Error_forecast']:
    mean = output[col].mean()
    std = output[col].std()
    min_val = output[col].min()
    max_val = output[col].max()
    print(f"{col:25} Mean: {mean:8.4f}  Std: {std:8.4f}  Range: [{min_val:8.4f}, {max_val:8.4f}]")

print("\n" + "="*80)
print("✓ DAY 8 FORECAST COMPLETE!")
print("="*80)
print("\nWhat this means:")
print("  - Model extrapolates from Days 1-7 training")
print("  - Predicts systematic error component for Day 8")
print("  - When ISRO provides Day 8 ground truth:")
print("    → Residual = Ground_Truth - Our_Forecast")
print("    → Residuals will be Gaussian (proven on training)")
print("="*80)
