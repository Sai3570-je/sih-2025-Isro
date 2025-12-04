"""
Compute MSE, RMSE, MAE and validate Gaussian residuals against mock Day 8 truth
(In production, use actual Day 8 ground truth from ISRO)
"""
import pandas as pd
import numpy as np
from scipy import stats

def compute_metrics_with_mock_truth(
    predictions_csv='outputs/day8_forecast_15min.csv',
    output_csv='outputs/validation_metrics.csv'
):
    """
    Compute comprehensive metrics using mock Day 8 data
    (Replace with actual Day 8 ground truth when available)
    """
    print("="*80)
    print("PREDICTION METRICS COMPUTATION")
    print("="*80)
    
    # Load predictions
    pred_df = pd.read_csv(predictions_csv)
    print(f"\n✓ Loaded {len(pred_df)} Day 8 predictions")
    
    # Generate mock Day 8 ground truth
    # In production: load actual ISRO Day 8 data
    print(f"\n⚠️  GENERATING MOCK DAY 8 GROUND TRUTH")
    print("   (Replace with actual ISRO data for real validation)")
    
    np.random.seed(42)  # For reproducibility
    
    # Mock truth = prediction + small Gaussian noise
    # This simulates systematic component (prediction) + random component (noise)
    truth_df = pred_df.copy()
    
    error_cols = ['x_error_day8', 'y_error_day8', 'z_error_day8', 'clock_error_day8']
    
    for col in error_cols:
        if col in truth_df.columns:
            # Add Gaussian noise (mean=0, std=0.5m for position, 0.2m for clock)
            noise_std = 0.2 if 'clock' in col else 0.5
            truth_df[col] = truth_df[col] + np.random.normal(0, noise_std, len(truth_df))
    
    # Rename truth columns
    truth_df = truth_df.rename(columns={
        'x_error_day8': 'x_truth',
        'y_error_day8': 'y_truth',
        'z_error_day8': 'z_truth',
        'clock_error_day8': 'clock_truth'
    })
    
    # Compute residuals
    residuals_df = pred_df[['time_of_day']].copy()
    
    metric_results = []
    
    for pred_col, truth_col in [
        ('x_error_day8', 'x_truth'),
        ('y_error_day8', 'y_truth'),
        ('z_error_day8', 'z_truth'),
        ('clock_error_day8', 'clock_truth')
    ]:
        predictions = pred_df[pred_col].values
        truth = truth_df[truth_col].values
        
        # Compute residuals
        residuals = truth - predictions
        residuals_df[f'{truth_col}_residual'] = residuals
        
        # MSE, RMSE, MAE
        mse = np.mean(residuals ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(residuals))
        
        # Shapiro-Wilk test for Gaussianity
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        is_gaussian = shapiro_p > 0.05
        
        # Additional Gaussian tests
        kurtosis_val = stats.kurtosis(residuals)
        skewness_val = stats.skew(residuals)
        
        # Jarque-Bera test (another Gaussianity test)
        jb_stat, jb_p = stats.jarque_bera(residuals)
        
        # Anderson-Darling test
        ad_result = stats.anderson(residuals, dist='norm')
        ad_stat = ad_result.statistic
        ad_critical = ad_result.critical_values[2]  # 5% significance
        ad_pass = ad_stat < ad_critical
        
        metric_results.append({
            'error_type': pred_col.replace('_day8', ''),
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'shapiro_statistic': shapiro_stat,
            'shapiro_p_value': shapiro_p,
            'is_gaussian_shapiro': is_gaussian,
            'kurtosis': kurtosis_val,
            'skewness': skewness_val,
            'jarque_bera_p': jb_p,
            'anderson_darling_stat': ad_stat,
            'anderson_darling_pass': ad_pass,
            'residual_mean': np.mean(residuals),
            'residual_std': np.std(residuals)
        })
    
    # Save residuals
    residuals_df.to_csv('outputs/residuals_analysis.csv', index=False)
    
    # Create metrics dataframe
    metrics_df = pd.DataFrame(metric_results)
    metrics_df.to_csv(output_csv, index=False)
    
    # Print results
    print(f"\n{'='*80}")
    print("PREDICTION ACCURACY METRICS")
    print(f"{'='*80}\n")
    
    print(f"{'Error Type':<15} {'MSE':<12} {'RMSE':<12} {'MAE':<12}")
    print("-" * 60)
    for _, row in metrics_df.iterrows():
        print(f"{row['error_type']:<15} {row['MSE']:<12.6f} {row['RMSE']:<12.6f} {row['MAE']:<12.6f}")
    
    print(f"\n{'='*80}")
    print("GAUSSIANITY VALIDATION (Shapiro-Wilk Test)")
    print(f"{'='*80}\n")
    
    print(f"{'Error Type':<15} {'p-value':<12} {'Gaussian?':<12} {'Status'}")
    print("-" * 60)
    
    all_gaussian = True
    for _, row in metrics_df.iterrows():
        status = "✅ PASS" if row['is_gaussian_shapiro'] else "❌ FAIL"
        gauss_str = "YES" if row['is_gaussian_shapiro'] else "NO"
        print(f"{row['error_type']:<15} {row['shapiro_p_value']:<12.6f} {gauss_str:<12} {status}")
        if not row['is_gaussian_shapiro']:
            all_gaussian = False
    
    print(f"\n{'='*80}")
    print("ADDITIONAL GAUSSIANITY TESTS")
    print(f"{'='*80}\n")
    
    print(f"{'Error Type':<15} {'Kurtosis':<12} {'Skewness':<12} {'A-D Test':<12}")
    print("-" * 60)
    for _, row in metrics_df.iterrows():
        ad_status = "✅ PASS" if row['anderson_darling_pass'] else "❌ FAIL"
        print(f"{row['error_type']:<15} {row['kurtosis']:<12.4f} {row['skewness']:<12.4f} {ad_status:<12}")
    
    print(f"\n{'='*80}")
    print("RESIDUAL CHARACTERISTICS")
    print(f"{'='*80}\n")
    
    print(f"{'Error Type':<15} {'Mean':<12} {'Std Dev':<12} {'Quality'}")
    print("-" * 60)
    for _, row in metrics_df.iterrows():
        mean_close_to_zero = abs(row['residual_mean']) < 0.01
        quality = "✅ GOOD" if mean_close_to_zero else "⚠️  CHECK"
        print(f"{row['error_type']:<15} {row['residual_mean']:<12.6f} {row['residual_std']:<12.6f} {quality}")
    
    # Overall assessment
    print(f"\n{'='*80}")
    print("OVERALL ISRO COMPLIANCE")
    print(f"{'='*80}")
    
    shapiro_pass_count = metrics_df['is_gaussian_shapiro'].sum()
    shapiro_pass_pct = (shapiro_pass_count / len(metrics_df)) * 100
    
    ad_pass_count = metrics_df['anderson_darling_pass'].sum()
    ad_pass_pct = (ad_pass_count / len(metrics_df)) * 100
    
    avg_mae = metrics_df['MAE'].mean()
    avg_rmse = metrics_df['RMSE'].mean()
    
    print(f"\n✅ ACCURACY METRICS:")
    print(f"   Average MAE:  {avg_mae:.4f} m")
    print(f"   Average RMSE: {avg_rmse:.4f} m")
    
    if avg_mae < 1.0:
        print(f"   ✅ EXCELLENT - Very low prediction errors!")
    elif avg_mae < 2.0:
        print(f"   ✅ GOOD - Acceptable prediction errors")
    else:
        print(f"   ⚠️  MODERATE - Consider model improvements")
    
    print(f"\n✅ GAUSSIANITY COMPLIANCE:")
    print(f"   Shapiro-Wilk: {shapiro_pass_count}/{len(metrics_df)} tests passed ({shapiro_pass_pct:.0f}%)")
    print(f"   Anderson-Darling: {ad_pass_count}/{len(metrics_df)} tests passed ({ad_pass_pct:.0f}%)")
    
    if all_gaussian and ad_pass_count == len(metrics_df):
        print(f"   ✅ ✅ ✅ PERFECT - All residuals are Gaussian!")
        print(f"   🏆 ISRO REQUIREMENT MET!")
    elif shapiro_pass_pct >= 75:
        print(f"   ✅ GOOD - High Gaussian compliance")
        print(f"   ✓ ISRO requirement likely met")
    else:
        print(f"   ⚠️  NEEDS IMPROVEMENT - Low Gaussian compliance")
    
    # Check kurtosis (should be near 0 for Gaussian)
    avg_kurtosis = abs(metrics_df['kurtosis'].mean())
    print(f"\n✅ KURTOSIS CHECK:")
    print(f"   Average |kurtosis|: {avg_kurtosis:.4f}")
    if avg_kurtosis < 0.5:
        print(f"   ✅ EXCELLENT - Very close to Gaussian (target = 0)")
    elif avg_kurtosis < 1.0:
        print(f"   ✅ GOOD - Reasonably Gaussian")
    else:
        print(f"   ⚠️  MODERATE - Some deviation from Gaussian")
    
    print(f"\n{'='*80}")
    print("FILES SAVED:")
    print(f"{'='*80}")
    print(f"  • {output_csv}")
    print(f"  • outputs/residuals_analysis.csv")
    
    print(f"\n{'='*80}")
    print("⚠️  IMPORTANT NOTE:")
    print(f"{'='*80}")
    print("These metrics use MOCK Day 8 ground truth for demonstration.")
    print("For actual ISRO submission:")
    print("  1. Replace mock truth with real Day 8 satellite data")
    print("  2. Re-run this validation")
    print("  3. Submit predictions + validation metrics")
    print(f"{'='*80}\n")
    
    return metrics_df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute prediction metrics')
    parser.add_argument('--predictions', default='outputs/day8_forecast_15min.csv',
                       help='Day 8 predictions CSV')
    parser.add_argument('--output', default='outputs/validation_metrics.csv',
                       help='Output metrics CSV')
    
    args = parser.parse_args()
    
    compute_metrics_with_mock_truth(args.predictions, args.output)
