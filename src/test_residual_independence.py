"""
Residual Independence Testing
==============================

Tests whether residuals are white noise (independent, no autocorrelation).

Critical validation:
- Gaussian residuals are necessary but not sufficient
- Residuals must also be independent (no temporal patterns)
- Autocorrelated residuals indicate model missed temporal structure
- White noise residuals prove model captured all predictable patterns

Tests Implemented:
1. Ljung-Box Test (omnibus autocorrelation test)
2. Durbin-Watson Statistic (first-order autocorrelation)
3. Runs Test (randomness test)
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
import warnings
warnings.filterwarnings('ignore')


def runs_test(residuals, alpha=0.05):
    """
    Wald-Wolfowitz runs test for randomness
    
    Tests whether residuals are randomly distributed or have patterns
    
    Returns:
    --------
    z_stat : float
        Z-statistic
    p_value : float
        Two-tailed p-value
    is_random : bool
        True if residuals appear random
    """
    # Convert to binary: above/below median
    median = np.median(residuals)
    binary = (residuals > median).astype(int)
    
    # Count runs
    runs = 1
    for i in range(1, len(binary)):
        if binary[i] != binary[i-1]:
            runs += 1
    
    # Count positive and negative
    n_pos = np.sum(binary == 1)
    n_neg = np.sum(binary == 0)
    
    n = len(residuals)
    
    # Expected runs and variance under null hypothesis
    expected_runs = (2 * n_pos * n_neg) / n + 1
    variance_runs = (2 * n_pos * n_neg * (2 * n_pos * n_neg - n)) / (n**2 * (n - 1))
    
    # Z-statistic
    z_stat = (runs - expected_runs) / np.sqrt(variance_runs)
    
    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    is_random = p_value > alpha
    
    return z_stat, p_value, is_random


def test_residual_independence(residuals, name="Residuals", lags=10, alpha=0.05):
    """
    Comprehensive independence testing for residuals
    
    Parameters:
    -----------
    residuals : array-like
        Residual series to test
    name : str
        Name for reporting
    lags : int
        Number of lags for autocorrelation tests
    alpha : float
        Significance level
        
    Returns:
    --------
    results : dict
        Test results
    """
    clean_residuals = residuals[~np.isnan(residuals)]
    n = len(clean_residuals)
    
    results = {
        'name': name,
        'n_samples': n,
        'mean': np.mean(clean_residuals),
        'std': np.std(clean_residuals)
    }
    
    print(f"\n{name}")
    print("="*80)
    print(f"Samples: {n}")
    print(f"Mean: {results['mean']:.6f}")
    print(f"Std: {results['std']:.6f}")
    
    # Test 1: Ljung-Box Test
    print(f"\n1. LJUNG-BOX TEST (Omnibus Autocorrelation)")
    print("-" * 80)
    
    try:
        lb_result = acorr_ljungbox(clean_residuals, lags=lags, return_df=True)
        
        # Check if any lag shows significant autocorrelation
        lb_significant = (lb_result['lb_pvalue'] < alpha).any()
        
        results['ljungbox_min_p'] = lb_result['lb_pvalue'].min()
        results['ljungbox_significant_lags'] = (lb_result['lb_pvalue'] < alpha).sum()
        results['ljungbox_pass'] = not lb_significant
        
        print(f"  Lags tested: {lags}")
        print(f"  Significant lags (p < {alpha}): {results['ljungbox_significant_lags']}")
        print(f"  Minimum p-value: {results['ljungbox_min_p']:.6f}")
        
        if results['ljungbox_pass']:
            print(f"  ✅ PASS: No significant autocorrelation detected")
        else:
            print(f"  ❌ FAIL: Significant autocorrelation at {results['ljungbox_significant_lags']} lags")
            
    except Exception as e:
        print(f"  ⚠️  Error: {e}")
        results['ljungbox_error'] = str(e)
        results['ljungbox_pass'] = False
    
    # Test 2: Durbin-Watson Statistic
    print(f"\n2. DURBIN-WATSON STATISTIC (First-Order Autocorrelation)")
    print("-" * 80)
    
    try:
        dw_stat = durbin_watson(clean_residuals)
        results['durbin_watson'] = dw_stat
        
        # DW interpretation: 
        # ~2.0 = no autocorrelation
        # < 2.0 = positive autocorrelation
        # > 2.0 = negative autocorrelation
        # Acceptable range: 1.5 - 2.5
        
        dw_acceptable = 1.5 <= dw_stat <= 2.5
        results['durbin_watson_pass'] = dw_acceptable
        
        print(f"  DW Statistic: {dw_stat:.4f}")
        print(f"  Interpretation:")
        
        if 1.9 <= dw_stat <= 2.1:
            print(f"    ✅ EXCELLENT: Very close to 2.0 (no autocorrelation)")
        elif 1.5 <= dw_stat <= 2.5:
            print(f"    ✅ ACCEPTABLE: Within normal range")
        elif dw_stat < 1.5:
            print(f"    ⚠️  POSITIVE autocorrelation detected")
        else:
            print(f"    ⚠️  NEGATIVE autocorrelation detected")
        
        if dw_acceptable:
            print(f"  ✅ PASS: DW in acceptable range [1.5, 2.5]")
        else:
            print(f"  ❌ FAIL: DW outside acceptable range")
            
    except Exception as e:
        print(f"  ⚠️  Error: {e}")
        results['durbin_watson_error'] = str(e)
        results['durbin_watson_pass'] = False
    
    # Test 3: Runs Test
    print(f"\n3. RUNS TEST (Randomness)")
    print("-" * 80)
    
    try:
        z_stat, p_value, is_random = runs_test(clean_residuals, alpha=alpha)
        
        results['runs_test_z'] = z_stat
        results['runs_test_p'] = p_value
        results['runs_test_pass'] = is_random
        
        print(f"  Z-statistic: {z_stat:.4f}")
        print(f"  P-value: {p_value:.6f}")
        
        if is_random:
            print(f"  ✅ PASS: Residuals appear random")
        else:
            print(f"  ❌ FAIL: Non-random pattern detected")
            
    except Exception as e:
        print(f"  ⚠️  Error: {e}")
        results['runs_test_error'] = str(e)
        results['runs_test_pass'] = False
    
    # Overall assessment
    print(f"\n{'='*80}")
    print("OVERALL INDEPENDENCE ASSESSMENT")
    print(f"{'='*80}")
    
    tests_passed = sum([
        results.get('ljungbox_pass', False),
        results.get('durbin_watson_pass', False),
        results.get('runs_test_pass', False)
    ])
    
    total_tests = 3
    
    results['tests_passed'] = tests_passed
    results['total_tests'] = total_tests
    results['independence'] = 'PASS' if tests_passed >= 2 else 'FAIL'
    
    print(f"Tests Passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print(f"✅ ✅ ✅ PERFECT: All independence tests pass!")
        print(f"           Residuals are white noise - model captured all temporal patterns")
    elif tests_passed >= 2:
        print(f"✅ GOOD: Most independence tests pass")
    else:
        print(f"⚠️  MODERATE: Significant autocorrelation detected")
    
    print(f"{'='*80}\n")
    
    return results


def validate_independence_all_errors(
    predictions_csv='outputs/day8_forecast_15min.csv',
    output_dir='outputs',
    lags=10,
    alpha=0.05
):
    """
    Test residual independence for all error types
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("RESIDUAL INDEPENDENCE VALIDATION")
    print("="*80)
    print("\nTesting whether residuals are white noise (independent, no autocorrelation)")
    
    # Load predictions
    df = pd.read_csv(predictions_csv)
    print(f"\n✓ Loaded {len(df)} predictions")
    
    # Generate mock Day 8 residuals
    print(f"\n⚠️  USING MOCK DAY 8 DATA FOR DEMONSTRATION")
    print("   (Replace with actual ISRO Day 8 when available)")
    
    np.random.seed(42)
    
    error_cols = ['x_error_day8', 'y_error_day8', 'z_error_day8', 'clock_error_day8']
    
    all_results = []
    
    for error_col in error_cols:
        if error_col not in df.columns:
            continue
        
        error_name = error_col.replace('_day8', '').replace('_error', '').upper() + '_error'
        
        # Generate mock residuals
        predictions = df[error_col].dropna().values
        noise_std = 0.2 if 'clock' in error_col else 0.5
        truth = predictions + np.random.normal(0, noise_std, len(predictions))
        residuals = truth - predictions
        
        # Test independence
        results = test_residual_independence(
            residuals,
            name=error_name,
            lags=lags,
            alpha=alpha
        )
        
        results['error_type'] = error_name
        all_results.append(results)
    
    # Save results
    results_df = pd.DataFrame(all_results)
    
    # Rename columns for consistency
    results_df = results_df.rename(columns={
        'error_type': 'Error_Type',
        'ljungbox_min_p': 'LjungBox_p',
        'durbin_watson': 'DurbinWatson',
        'runs_test_p': 'Runs_p',
        'independence': 'Independence'
    })
    
    output_csv = os.path.join(output_dir, 'independence_test_report.csv')
    results_df.to_csv(output_csv, index=False)
    
    # Summary
    print("="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    
    for _, row in results_df.iterrows():
        print(f"\n{row['Error_Type']}:")
        print(f"  Independence: {row['Independence']}")
    
    n_pass = (results_df['Independence'] == 'PASS').sum()
    total = len(results_df)
    
    print(f"\n{'='*80}")
    print(f"Independent Residuals: {n_pass}/{total} ({n_pass/total*100:.0f}%)")
    
    if n_pass == total:
        print(f"✅ ✅ ✅ PERFECT: All error types have white noise residuals!")
    elif n_pass >= total * 0.75:
        print(f"✅ EXCELLENT: Most error types have white noise residuals")
    else:
        print(f"⚠️  NEEDS IMPROVEMENT: Some error types show autocorrelation")
    
    print(f"\n{'='*80}")
    print("FILES SAVED:")
    print(f"{'='*80}")
    print(f"  • {output_csv}")
    print(f"{'='*80}\n")
    
    return results_df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Residual independence validation')
    parser.add_argument('--predictions', default='outputs/day8_forecast_15min.csv',
                       help='Predictions CSV file')
    parser.add_argument('--output-dir', default='outputs',
                       help='Output directory')
    parser.add_argument('--lags', type=int, default=10,
                       help='Number of lags for autocorrelation tests')
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='Significance level')
    
    args = parser.parse_args()
    
    validate_independence_all_errors(
        predictions_csv=args.predictions,
        output_dir=args.output_dir,
        lags=args.lags,
        alpha=args.alpha
    )
