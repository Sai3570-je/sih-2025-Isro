"""
ENHANCED WAVELET OPTIMIZATION
==============================
Try multiple wavelet families and preprocessing strategies to maximize Gaussian components.
"""

import numpy as np
import pandas as pd
import pywt
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path

print("="*90)
print("WAVELET OPTIMIZATION - FIND BEST CONFIGURATION")
print("="*90)

# Load data
df = pd.read_parquet('temp/MEO_01_timeseries.parquet')
df_train = df[df['timestamp'] <= '2025-09-07 18:45:00'].copy()
valid_mask = (df_train['X_Error'].notna() & df_train['Y_Error'].notna() & 
              df_train['Z_Error'].notna() & df_train['Clock_Error'].notna())
df_valid = df_train[valid_mask].copy()

components = ['X_Error', 'Y_Error', 'Z_Error', 'Clock_Error']

# Test different wavelet configurations
wavelets = ['db4', 'db6', 'db8', 'sym4', 'sym6', 'coif2', 'coif3']
levels = [3, 4, 5]
threshold_modes = ['soft', 'hard']

results = []

print(f"\nTesting {len(wavelets)} wavelets × {len(levels)} levels × {len(threshold_modes)} modes = {len(wavelets)*len(levels)*len(threshold_modes)} configurations...\n")

for wavelet in wavelets:
    for level in levels:
        for threshold_mode in threshold_modes:
            
            config_results = {
                'wavelet': wavelet,
                'level': level,
                'mode': threshold_mode
            }
            
            n_gaussian = 0
            total_p = 0
            total_kurt = 0
            
            for comp in components:
                signal = df_valid[comp].values
                
                try:
                    # Wavelet decomposition
                    coeffs = pywt.wavedec(signal, wavelet, level=level)
                    
                    # Estimate noise and threshold
                    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
                    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
                    
                    # Apply thresholding
                    coeffs_thresh = [coeffs[0]]
                    for c in coeffs[1:]:
                        coeffs_thresh.append(pywt.threshold(c, threshold, mode=threshold_mode))
                    
                    # Reconstruct
                    denoised = pywt.waverec(coeffs_thresh, wavelet)[:len(signal)]
                    
                    # Extract noise
                    noise = signal - denoised
                    
                    # Test Gaussianity
                    shapiro_stat, shapiro_p = stats.shapiro(noise)
                    kurt = stats.kurtosis(noise)
                    
                    config_results[f'{comp}_p'] = shapiro_p
                    config_results[f'{comp}_kurt'] = kurt
                    config_results[f'{comp}_pass'] = shapiro_p > 0.05
                    
                    if shapiro_p > 0.05:
                        n_gaussian += 1
                    
                    total_p += shapiro_p
                    total_kurt += abs(kurt)
                    
                except Exception as e:
                    config_results[f'{comp}_p'] = 0
                    config_results[f'{comp}_kurt'] = 999
                    config_results[f'{comp}_pass'] = False
            
            config_results['n_gaussian'] = n_gaussian
            config_results['avg_p'] = total_p / 4
            config_results['avg_kurt'] = total_kurt / 4
            config_results['score'] = n_gaussian + (total_p / 4) - (total_kurt / 40)
            
            results.append(config_results)

# Convert to DataFrame and sort
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('score', ascending=False)

# Display top 10
print("TOP 10 CONFIGURATIONS:")
print("-" * 120)
print(f"{'Rank':<6} {'Wavelet':<8} {'Lvl':<5} {'Mode':<6} {'Gauss':<7} {'X_p':<8} {'Y_p':<8} {'Z_p':<8} {'Clk_p':<8} {'Avg_Kurt':<10} {'Score':<8}")
print("-" * 120)

for idx, (i, row) in enumerate(results_df.head(10).iterrows(), 1):
    print(f"{idx:<6} {row['wavelet']:<8} {row['level']:<5} {row['mode']:<6} "
          f"{row['n_gaussian']}/4    {row['X_Error_p']:<8.4f} {row['Y_Error_p']:<8.4f} "
          f"{row['Z_Error_p']:<8.4f} {row['Clock_Error_p']:<8.4f} {row['avg_kurt']:<10.3f} {row['score']:<8.3f}")

# Get best configuration
best = results_df.iloc[0]

print(f"\n{'='*120}")
print(f"BEST CONFIGURATION:")
print(f"  Wavelet: {best['wavelet']}")
print(f"  Level: {best['level']}")
print(f"  Threshold mode: {best['mode']}")
print(f"  Gaussian components: {best['n_gaussian']}/4")
print(f"  Components passing:")
for comp in components:
    status = "✓ PASS" if best[f'{comp}_pass'] else "✗ FAIL"
    print(f"    {comp}: p={best[f'{comp}_p']:.6f}, kurtosis={best[f'{comp}_kurt']:.3f} {status}")

# Save results
output_dir = Path('outputs')
results_df.to_csv(output_dir / 'wavelet_optimization_results.csv', index=False)
print(f"\n[OK] Saved full results to: outputs/wavelet_optimization_results.csv")

# Visualize best configuration
print(f"\n[VISUAL] Creating visualization for best configuration...")

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for i, comp in enumerate(components):
    signal = df_valid[comp].values
    
    # Apply best wavelet
    coeffs = pywt.wavedec(signal, best['wavelet'], level=int(best['level']))
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs_thresh = [coeffs[0]] + [pywt.threshold(c, threshold, mode=best['mode']) for c in coeffs[1:]]
    denoised = pywt.waverec(coeffs_thresh, best['wavelet'])[:len(signal)]
    noise = signal - denoised
    
    # Histogram
    axes[0, i].hist(noise, bins=30, alpha=0.7, edgecolor='black', density=True)
    mu, sigma_plot = np.mean(noise), np.std(noise)
    x = np.linspace(noise.min(), noise.max(), 100)
    axes[0, i].plot(x, stats.norm.pdf(x, mu, sigma_plot), 'r-', linewidth=2)
    axes[0, i].set_title(f'{comp}\np={best[f"{comp}_p"]:.4f}, kurt={best[f"{comp}_kurt"]:.2f}')
    axes[0, i].set_xlabel('Noise (m)')
    axes[0, i].set_ylabel('Density')
    axes[0, i].grid(alpha=0.3)
    
    # Q-Q plot
    stats.probplot(noise, dist="norm", plot=axes[1, i])
    axes[1, i].set_title(f'{comp} - Q-Q Plot')
    axes[1, i].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'best_wavelet_configuration.png', dpi=150)
print(f"[OK] Saved visualization: outputs/best_wavelet_configuration.png")

print(f"\n{'='*120}")
