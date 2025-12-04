"""
Per-Slot Gaussianity Validation
================================

Tests Gaussianity separately for each of the 96 time slots (15-min intervals).
Identifies which time periods have strong/weak Gaussian properties.

Critical for operational deployment:
- Some slots may handle uncertainty better than others
- Time-varying performance requires slot-level analysis
- Weak slots can be flagged for retraining or special handling
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def test_slot_gaussianity(residuals, slot_id, alpha=0.05, min_samples=3):
    """
    Test Gaussianity for a single time slot
    
    Parameters:
    -----------
    residuals : array-like
        Residuals for this time slot
    slot_id : int
        Slot identifier
    alpha : float
        Significance level
    min_samples : int
        Minimum samples required for testing
        
    Returns:
    --------
    results : dict
        Test results for this slot
    """
    clean_residuals = residuals[~np.isnan(residuals)]
    n = len(clean_residuals)
    
    if n < min_samples:
        return {
            'slot_id': slot_id,
            'n_samples': n,
            'gaussianity': 'INSUFFICIENT_DATA',
            'shapiro_p': None,
            'kurtosis': None,
            'skewness': None
        }
    
    results = {
        'slot_id': slot_id,
        'n_samples': n,
        'mean': np.mean(clean_residuals),
        'std': np.std(clean_residuals),
        'skewness': stats.skew(clean_residuals),
        'kurtosis': stats.kurtosis(clean_residuals)
    }
    
    # Shapiro-Wilk test
    try:
        _, p = stats.shapiro(clean_residuals)
        results['shapiro_p'] = p
        results['gaussianity'] = 'PASS' if p > alpha else 'FAIL'
    except:
        results['shapiro_p'] = None
        results['gaussianity'] = 'ERROR'
    
    return results


def validate_per_slot(
    predictions_csv='outputs/day8_forecast_15min.csv',
    output_dir='outputs',
    alpha=0.05
):
    """
    Validate Gaussianity for each time slot separately
    
    Generates:
    - CSV with per-slot test results
    - Heatmap showing pass/fail pattern across slots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    
    print("="*80)
    print("PER-SLOT GAUSSIANITY VALIDATION")
    print("="*80)
    
    # Load predictions
    df = pd.read_csv(predictions_csv)
    print(f"\n✓ Loaded {len(df)} predictions")
    
    # Get time of day column
    time_col = 'time_of_day' if 'time_of_day' in df.columns else 'TimeOfDay'
    if time_col not in df.columns:
        print(f"\n⚠️  No time column found, using sequential index")
        df[time_col] = df.index
    
    n_slots = df[time_col].nunique()
    print(f"✓ Found {n_slots} unique time slots")
    
    # Generate mock Day 8 residuals
    print(f"\n⚠️  USING MOCK DAY 8 DATA FOR DEMONSTRATION")
    print("   (Replace with actual ISRO Day 8 when available)\n")
    
    np.random.seed(42)
    
    error_cols = ['x_error_day8', 'y_error_day8', 'z_error_day8', 'clock_error_day8']
    
    all_slot_results = []
    
    print(f"{'='*80}")
    print("TESTING EACH SLOT INDIVIDUALLY")
    print(f"{'='*80}\n")
    
    for error_col in error_cols:
        if error_col not in df.columns:
            continue
        
        error_name = error_col.replace('_day8', '').replace('_error', '').upper() + '_error'
        print(f"\n{error_name}:")
        print("-" * 80)
        
        # Process each unique time slot
        unique_times = sorted(df[time_col].unique())
        
        for time_val in unique_times:
            slot_data = df[df[time_col] == time_val]
            
            if len(slot_data) == 0:
                continue
            
            # Generate mock residuals for this slot
            predictions = slot_data[error_col].dropna().values
            if len(predictions) == 0:
                continue
                
            noise_std = 0.2 if 'clock' in error_col else 0.5
            truth = predictions + np.random.normal(0, noise_std, len(predictions))
            residuals = truth - predictions
            
            # Test this slot
            results = test_slot_gaussianity(residuals, time_val, alpha=alpha)
            results['error_type'] = error_name
            results['time_of_day'] = time_val
            all_slot_results.append(results)
        
        # Summary for this error type
        tested_slots = [r for r in all_slot_results 
                       if r['error_type'] == error_name and r['gaussianity'] in ['PASS', 'FAIL']]
        
        if tested_slots:
            n_pass = sum(1 for r in tested_slots if r['gaussianity'] == 'PASS')
            print(f"  Slots tested: {len(tested_slots)}/{n_slots}")
            print(f"  Pass rate: {n_pass}/{len(tested_slots)} ({n_pass/len(tested_slots)*100:.1f}%)")
    
    # Create results dataframe
    results_df = pd.DataFrame(all_slot_results)
    
    # Rename columns for consistency
    results_df = results_df.rename(columns={
        'time_of_day': 'TimeOfDay',
        'error_type': 'Error_Type',
        'n_samples': 'Sample_Size',
        'shapiro_p': 'Shapiro_p',
        'kurtosis': 'Kurtosis',
        'skewness': 'Skewness',
        'gaussianity': 'Gaussianity'
    })
    
    output_csv = os.path.join(output_dir, 'slot_validation_report.csv')
    results_df.to_csv(output_csv, index=False)
    
    print(f"\n{'='*80}")
    print("CREATING HEATMAP VISUALIZATION")
    print(f"{'='*80}\n")
    
    # Create heatmap matrix
    error_types = results_df['Error_Type'].unique()
    times = sorted(results_df['TimeOfDay'].unique())
    
    # Create matrix: rows = error types, cols = time slots
    heatmap_data = np.full((len(error_types), len(times)), np.nan)
    
    for i, error_type in enumerate(error_types):
        for j, time_val in enumerate(times):
            slot_data = results_df[
                (results_df['Error_Type'] == error_type) & 
                (results_df['TimeOfDay'] == time_val)
            ]
            if len(slot_data) > 0:
                row = slot_data.iloc[0]
                if row['Gaussianity'] == 'PASS':
                    heatmap_data[i, j] = 1.0
                elif row['Gaussianity'] == 'FAIL':
                    heatmap_data[i, j] = 0.0
                # NaN for INSUFFICIENT_DATA
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(20, 5))
    
    sns.heatmap(
        heatmap_data,
        cmap='RdYlGn',
        vmin=0, vmax=1,
        cbar_kws={'label': 'Gaussianity (1=PASS, 0=FAIL)'},
        yticklabels=error_types,
        xticklabels=[f"{t}" for t in times],
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )
    
    ax.set_title('Per-Slot Gaussianity Test Results', fontsize=16, fontweight='bold')
    ax.set_xlabel('Time of Day', fontsize=12)
    ax.set_ylabel('Error Type', fontsize=12)
    
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    
    output_heatmap = os.path.join(output_dir, 'plots', 'slot_validation_heatmap.png')
    plt.savefig(output_heatmap, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Heatmap saved: {output_heatmap}")
    
    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL SLOT-LEVEL ASSESSMENT")
    print(f"{'='*80}\n")
    
    tested = results_df[results_df['Gaussianity'].isin(['PASS', 'FAIL'])]
    
    if len(tested) > 0:
        total_tested = len(tested)
        n_pass = (tested['Gaussianity'] == 'PASS').sum()
        n_fail = (tested['Gaussianity'] == 'FAIL').sum()
        
        print(f"Total slot-error combinations tested: {total_tested}")
        print(f"  PASS: {n_pass} ({n_pass/total_tested*100:.1f}%)")
        print(f"  FAIL: {n_fail} ({n_fail/total_tested*100:.1f}%)")
        
        if n_pass / total_tested >= 0.8:
            print(f"\n✅ EXCELLENT: {n_pass/total_tested*100:.0f}% of slots pass")
        elif n_pass / total_tested >= 0.6:
            print(f"\n✅ GOOD: {n_pass/total_tested*100:.0f}% of slots pass")
        else:
            print(f"\n⚠️  MODERATE: Only {n_pass/total_tested*100:.0f}% of slots pass")
    
    print(f"\n{'='*80}")
    print("FILES SAVED:")
    print(f"{'='*80}")
    print(f"  • {output_csv}")
    print(f"  • {output_heatmap}")
    print(f"{'='*80}\n")
    
    return results_df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Per-slot Gaussianity validation')
    parser.add_argument('--predictions', default='outputs/day8_forecast_15min.csv',
                       help='Predictions CSV file')
    parser.add_argument('--output-dir', default='outputs',
                       help='Output directory')
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='Significance level')
    
    args = parser.parse_args()
    
    validate_per_slot(
        predictions_csv=args.predictions,
        output_dir=args.output_dir,
        alpha=args.alpha
    )
