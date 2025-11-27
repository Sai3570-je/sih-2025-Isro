"""Analyze satellite data for orbital physics patterns."""
import pandas as pd
import numpy as np
from scipy import signal, fft

# Load GEO training data
df = pd.read_csv('data/DATA_GEO_Train.csv')
df['utc_time'] = pd.to_datetime(df['utc_time'])
df = df.sort_values('utc_time').reset_index(drop=True)
df['hours'] = (df['utc_time'] - df['utc_time'].min()).dt.total_seconds() / 3600

print("=" * 80)
print("SATELLITE ORBITAL PHYSICS ANALYSIS")
print("=" * 80)

print(f"\n1. DATA OVERVIEW")
print(f"   Time span: {df.hours.min():.1f} to {df.hours.max():.1f} hours ({df.hours.max()/24:.1f} days)")
print(f"   Measurements: {len(df)}")
print(f"   Avg sampling interval: {df['hours'].diff().mean():.2f} hours")

print(f"\n2. ERROR STATISTICS (meters)")
for col in ['x_error (m)', 'y_error (m)', 'z_error (m)', 'satclockerror (m)']:
    vals = df[col].values
    print(f"   {col:20s}: mean={vals.mean():7.2f}, std={vals.std():6.2f}, "
          f"range=[{vals.min():7.2f}, {vals.max():7.2f}]")

print(f"\n3. ORBITAL CHARACTERISTICS")
# GEO satellites have ~24 hour period
# Calculate velocity from position differences
for axis, col in [('X', 'x_error (m)'), ('Y', 'y_error (m)'), ('Z', 'z_error (m)')]:
    pos = df[col].values
    dt = df['hours'].diff().values[1:] * 3600  # convert to seconds
    vel = np.diff(pos) / dt
    print(f"   {axis}-axis velocity: mean={vel.mean():.4f} m/s, std={vel.std():.4f} m/s")

print(f"\n4. PERIODICITY ANALYSIS (FFT)")
# Resample to uniform grid for FFT
df_uniform = df.set_index('utc_time').resample('2H').mean().interpolate()
for col in ['x_error (m)', 'y_error (m)', 'z_error (m)']:
    signal_data = df_uniform[col].dropna().values
    if len(signal_data) > 10:
        # FFT
        freqs = fft.fftfreq(len(signal_data), d=2.0)  # 2-hour sampling
        spectrum = np.abs(fft.fft(signal_data))
        
        # Find dominant frequencies (exclude DC component)
        valid = (freqs > 0) & (freqs < 0.5)
        if valid.sum() > 0:
            dominant_idx = spectrum[valid].argmax()
            dominant_freq = freqs[valid][dominant_idx]
            period = 1.0 / dominant_freq if dominant_freq > 0 else np.inf
            print(f"   {col}: dominant period = {period:.1f} hours")

print(f"\n5. TREND ANALYSIS")
# Check for drift/bias over time
for col in ['x_error (m)', 'y_error (m)', 'z_error (m)']:
    # Linear regression
    from scipy import stats
    slope, intercept, r_value, _, _ = stats.linregress(df['hours'], df[col])
    print(f"   {col}: drift = {slope:.4f} m/hr, R² = {r_value**2:.4f}")

print(f"\n6. PREDICTION CHALLENGE")
print(f"   Last measurement: {df['utc_time'].iloc[-1]}")
print(f"   Day 8 start: 2025-09-08 00:00:00")
gap_hours = (pd.Timestamp('2025-09-08 00:00:00') - df['utc_time'].iloc[-1]).total_seconds() / 3600
print(f"   Prediction gap: {gap_hours:.1f} hours ({gap_hours/24:.2f} days)")
print(f"   Last values: X={df['x_error (m)'].iloc[-1]:.2f}, Y={df['y_error (m)'].iloc[-1]:.2f}, "
      f"Z={df['z_error (m)'].iloc[-1]:.2f}")

# Check last few measurements for stability
last_n = 10
print(f"\n7. RECENT TRAJECTORY (last {last_n} measurements)")
recent = df.tail(last_n)[['utc_time', 'x_error (m)', 'y_error (m)', 'z_error (m)']]
print(recent.to_string(index=False))

print("\n" + "=" * 80)
