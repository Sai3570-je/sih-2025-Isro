"""
Visualize Predictions with Confidence Intervals
Shows prediction uncertainty and helps identify overfitting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

sns.set_style("whitegrid")

def plot_predictions_with_confidence(predictions_csv, output_dir='outputs/plots'):
    """
    Generate visualization of predictions with 95% confidence intervals
    
    Args:
        predictions_csv: Path to predictions with confidence intervals
        output_dir: Directory to save plots
    """
    print("\n" + "="*70)
    print("VISUALIZING PREDICTIONS WITH CONFIDENCE INTERVALS")
    print("="*70)
    
    # Load predictions
    df = pd.read_csv(predictions_csv)
    print(f"\n✓ Loaded {len(df)} predictions from {predictions_csv}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert time_of_day to minutes for plotting
    def time_to_minutes(time_str):
        if ':' in str(time_str):
            h, m = map(int, str(time_str).split(':'))
        else:
            h, m = map(int, str(time_str).split('-'))
        return h * 60 + m
    
    df['minutes'] = df['time_of_day'].apply(time_to_minutes)
    df = df.sort_values('minutes')
    
    # Plot 1: All errors with confidence bands
    print("\n📊 Generating comprehensive confidence interval plot...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    error_types = [
        ('x_error_day8', 'x_error_uncertainty', 'x_error_conf_lower', 'x_error_conf_upper', 'X Position Error'),
        ('y_error_day8', 'y_error_uncertainty', 'y_error_conf_lower', 'y_error_conf_upper', 'Y Position Error'),
        ('z_error_day8', 'z_error_uncertainty', 'z_error_conf_lower', 'z_error_conf_upper', 'Z Position Error'),
        ('clock_error_day8', 'clock_error_uncertainty', 'clock_error_conf_lower', 'clock_error_conf_upper', 'Clock Bias Error')
    ]
    
    for idx, (pred_col, unc_col, lower_col, upper_col, title) in enumerate(error_types):
        ax = axes[idx // 2, idx % 2]
        
        # Get data
        x = df['minutes'].values
        y = df[pred_col].values
        lower = df[lower_col].values
        upper = df[upper_col].values
        
        # Plot prediction line
        ax.plot(x, y, 'b-', linewidth=2, label='Prediction', alpha=0.8)
        
        # Plot confidence band
        ax.fill_between(x, lower, upper, alpha=0.3, color='blue', label='95% Confidence Interval')
        
        # Add scatter points
        ax.scatter(x, y, c='darkblue', s=20, alpha=0.6, zorder=5)
        
        ax.set_xlabel('Time of Day (minutes since midnight)', fontweight='bold')
        ax.set_ylabel('Predicted Error (m)', fontweight='bold')
        ax.set_title(f'{title}\nPredictions with 95% Confidence Bands', fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add time labels
        hours = [0, 6, 12, 18, 24]
        ax.set_xticks([h*60 for h in hours])
        ax.set_xticklabels([f'{h:02d}:00' for h in hours])
    
    plt.tight_layout()
    plt.savefig(output_path / 'predictions_with_confidence.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {output_path / 'predictions_with_confidence.png'}")
    
    # Plot 2: Uncertainty distribution
    print("📊 Generating uncertainty distribution analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, (pred_col, unc_col, lower_col, upper_col, title) in enumerate(error_types):
        ax = axes[idx // 2, idx % 2]
        
        uncertainty = df[unc_col].dropna()
        
        # Histogram
        ax.hist(uncertainty, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(uncertainty.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {uncertainty.mean():.3f}m')
        ax.axvline(uncertainty.median(), color='green', linestyle='--', linewidth=2,
                   label=f'Median: {uncertainty.median():.3f}m')
        
        ax.set_xlabel('Uncertainty (Standard Deviation, m)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title(f'{title}\nUncertainty Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path / 'uncertainty_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {output_path / 'uncertainty_distribution.png'}")
    
    # Plot 3: Confidence interval width over time
    print("📊 Generating confidence interval width analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    for idx, (pred_col, unc_col, lower_col, upper_col, title) in enumerate(error_types):
        ax = axes[idx // 2, idx % 2]
        
        # Calculate interval width
        interval_width = df[upper_col] - df[lower_col]
        
        # Plot
        ax.plot(df['minutes'], interval_width, 'o-', color='purple', linewidth=2, markersize=4)
        ax.axhline(interval_width.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean Width: {interval_width.mean():.3f}m')
        
        ax.set_xlabel('Time of Day (minutes since midnight)', fontweight='bold')
        ax.set_ylabel('95% CI Width (m)', fontweight='bold')
        ax.set_title(f'{title}\nConfidence Interval Width Over Time', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add time labels
        hours = [0, 6, 12, 18, 24]
        ax.set_xticks([h*60 for h in hours])
        ax.set_xticklabels([f'{h:02d}:00' for h in hours])
    
    plt.tight_layout()
    plt.savefig(output_path / 'confidence_width_over_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {output_path / 'confidence_width_over_time.png'}")
    
    # Generate summary statistics
    print("\n" + "="*70)
    print("UNCERTAINTY SUMMARY STATISTICS")
    print("="*70)
    
    for pred_col, unc_col, lower_col, upper_col, title in error_types:
        uncertainty = df[unc_col].dropna()
        interval_width = (df[upper_col] - df[lower_col]).dropna()
        
        print(f"\n{title}:")
        print(f"  Mean Uncertainty:     {uncertainty.mean():.4f} m")
        print(f"  Median Uncertainty:   {uncertainty.median():.4f} m")
        print(f"  Std Uncertainty:      {uncertainty.std():.4f} m")
        print(f"  Mean CI Width (95%):  {interval_width.mean():.4f} m")
        print(f"  Max CI Width:         {interval_width.max():.4f} m")
        print(f"  Min CI Width:         {interval_width.min():.4f} m")
    
    # Overfitting check
    print("\n" + "="*70)
    print("OVERFITTING CHECK")
    print("="*70)
    
    for pred_col, unc_col, lower_col, upper_col, title in error_types:
        uncertainty = df[unc_col].dropna()
        predictions = df[pred_col].dropna()
        
        # Check 1: Unrealistically low uncertainty
        low_unc_count = (uncertainty < 0.01).sum()
        low_unc_pct = low_unc_count / len(uncertainty) * 100
        
        # Check 2: Uncertainty much smaller than prediction variation
        pred_std = predictions.std()
        avg_uncertainty = uncertainty.mean()
        ratio = pred_std / avg_uncertainty if avg_uncertainty > 0 else np.inf
        
        print(f"\n{title}:")
        print(f"  Low uncertainty slots (<0.01m): {low_unc_count}/{len(uncertainty)} ({low_unc_pct:.1f}%)")
        print(f"  Prediction std / Avg uncertainty: {ratio:.2f}")
        
        if low_unc_pct > 50:
            print(f"  ⚠️  WARNING: High % of low uncertainty - possible overfitting!")
        elif ratio > 5:
            print(f"  ⚠️  WARNING: Uncertainty too low vs prediction variation!")
        else:
            print(f"  ✅ Uncertainty levels look healthy")
    
    print("\n" + "="*70)
    print("✅ ALL CONFIDENCE INTERVAL PLOTS GENERATED!")
    print("="*70)
    print(f"\n📁 Saved to: {output_path}")
    print("\nGenerated plots:")
    print("  1. predictions_with_confidence.png")
    print("  2. uncertainty_distribution.png")
    print("  3. confidence_width_over_time.png")
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize predictions with confidence intervals'
    )
    parser.add_argument('--predictions', default='outputs/day8_forecast_timeslots.csv',
                       help='CSV file with predictions and confidence intervals')
    parser.add_argument('--output_dir', default='outputs/plots',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    plot_predictions_with_confidence(args.predictions, args.output_dir)


if __name__ == '__main__':
    main()
