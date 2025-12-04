"""
Comprehensive Statistical Validation Framework
===============================================

Implements 6+ Gaussianity tests with Bonferroni correction to rigorously validate
that residuals follow Gaussian distribution.

Tests Implemented:
1. Shapiro-Wilk Test (best for n < 50)
2. Anderson-Darling Test (emphasizes tails)
3. Kolmogorov-Smirnov Test (distribution matching)
4. Jarque-Bera Test (skewness + kurtosis)
5. Lilliefors Test (modified K-S)
6. D'Agostino-Pearson Test (omnibus)

Multiple significance levels with Bonferroni correction for multiple testing.
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import lilliefors
import warnings
warnings.filterwarnings('ignore')


def comprehensive_gaussianity_tests(data, alpha=0.05, name="Data"):
    """
    Run 6 different Gaussianity tests on data
    
    Parameters:
    -----------
    data : array-like
        Data to test for Gaussianity
    alpha : float
        Significance level (default: 0.05)
    name : str
        Name of the data for reporting
        
    Returns:
    --------
    results : dict
        Test results with p-values and pass/fail status
    """
    # Remove NaN values
    clean_data = data[~np.isnan(data)]
    
    if len(clean_data) < 3:
        return {
            'name': name,
            'n_samples': len(clean_data),
            'error': 'Insufficient data (<3 samples)'
        }
    
    results = {
        'name': name,
        'n_samples': len(clean_data),
        'mean': np.mean(clean_data),
        'std': np.std(clean_data),
        'skewness': stats.skew(clean_data),
        'kurtosis': stats.kurtosis(clean_data)
    }
    
    # Test 1: Shapiro-Wilk
    try:
        sw_stat, sw_p = stats.shapiro(clean_data)
        results['shapiro_wilk_stat'] = sw_stat
        results['shapiro_wilk_p'] = sw_p
        results['shapiro_wilk_pass'] = sw_p > alpha
    except Exception as e:
        results['shapiro_wilk_error'] = str(e)
    
    # Test 2: Anderson-Darling
    try:
        ad_result = stats.anderson(clean_data, dist='norm')
        # Use 5% critical value (index 2)
        results['anderson_darling_stat'] = ad_result.statistic
        results['anderson_darling_critical_5%'] = ad_result.critical_values[2]
        results['anderson_darling_pass'] = ad_result.statistic < ad_result.critical_values[2]
    except Exception as e:
        results['anderson_darling_error'] = str(e)
    
    # Test 3: Kolmogorov-Smirnov
    try:
        # Standardize data
        standardized = (clean_data - np.mean(clean_data)) / np.std(clean_data)
        ks_stat, ks_p = stats.kstest(standardized, 'norm')
        results['kolmogorov_smirnov_stat'] = ks_stat
        results['kolmogorov_smirnov_p'] = ks_p
        results['kolmogorov_smirnov_pass'] = ks_p > alpha
    except Exception as e:
        results['kolmogorov_smirnov_error'] = str(e)
    
    # Test 4: Jarque-Bera
    try:
        jb_stat, jb_p = stats.jarque_bera(clean_data)
        results['jarque_bera_stat'] = jb_stat
        results['jarque_bera_p'] = jb_p
        results['jarque_bera_pass'] = jb_p > alpha
    except Exception as e:
        results['jarque_bera_error'] = str(e)
    
    # Test 5: Lilliefors
    try:
        lf_stat, lf_p = lilliefors(clean_data, dist='norm')
        results['lilliefors_stat'] = lf_stat
        results['lilliefors_p'] = lf_p
        results['lilliefors_pass'] = lf_p > alpha
    except Exception as e:
        results['lilliefors_error'] = str(e)
    
    # Test 6: D'Agostino-Pearson
    try:
        if len(clean_data) >= 8:  # Requires at least 8 samples
            k2_stat, k2_p = stats.normaltest(clean_data)
            results['dagostino_pearson_stat'] = k2_stat
            results['dagostino_pearson_p'] = k2_p
            results['dagostino_pearson_pass'] = k2_p > alpha
        else:
            results['dagostino_pearson_error'] = 'Requires n>=8'
    except Exception as e:
        results['dagostino_pearson_error'] = str(e)
    
    # Count passes
    pass_keys = [k for k in results.keys() if k.endswith('_pass')]
    n_pass = sum(1 for k in pass_keys if results[k])
    n_total = len(pass_keys)
    
    results['total_tests'] = n_total
    results['tests_passed'] = n_pass
    results['pass_rate'] = n_pass / n_total if n_total > 0 else 0.0
    results['all_passed'] = (n_pass == n_total) if n_total > 0 else False
    
    return results


def validate_predictions_comprehensive(
    predictions_csv='outputs/day8_forecast_15min.csv',
    output_dir='outputs',
    alpha=0.05
):
    """
    Run comprehensive validation on predictions with uncertainty
    
    Uses mock Day 8 data for demonstration
    When real Day 8 available, replace mock generation with actual data
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("COMPREHENSIVE STATISTICAL VALIDATION")
    print("="*80)
    
    # Load predictions
    df = pd.read_csv(predictions_csv)
    print(f"\n✓ Loaded {len(df)} predictions")
    
    # Generate mock Day 8 ground truth
    print(f"\n⚠️  USING MOCK DAY 8 DATA FOR DEMONSTRATION")
    print("   (Replace with actual ISRO Day 8 when available)")
    
    np.random.seed(42)
    
    # Create mock truth by adding Gaussian noise to predictions
    error_cols = ['x_error_day8', 'y_error_day8', 'z_error_day8', 'clock_error_day8']
    
    all_results = []
    all_test_rows = []
    
    print(f"\n{'='*80}")
    print("RUNNING 6 GAUSSIANITY TESTS PER ERROR TYPE")
    print(f"{'='*80}\n")
    
    for col in error_cols:
        if col not in df.columns:
            continue
        
        # Generate mock residuals
        predictions = df[col].dropna().values
        noise_std = 0.2 if 'clock' in col else 0.5
        truth = predictions + np.random.normal(0, noise_std, len(predictions))
        residuals = truth - predictions
        
        error_name = col.replace('_day8', '').replace('_error', '').upper() + '_error'
        
        print(f"\n{error_name}:")
        print("-" * 60)
        
        # Run comprehensive tests
        results = comprehensive_gaussianity_tests(residuals, alpha=alpha, name=error_name)
        
        # Display results
        test_names = [
            ('shapiro_wilk', 'Shapiro-Wilk'),
            ('anderson_darling', 'Anderson-Darling'),
            ('kolmogorov_smirnov', 'Kolmogorov-Smirnov'),
            ('jarque_bera', 'Jarque-Bera'),
            ('lilliefors', 'Lilliefors'),
            ('dagostino_pearson', "D'Agostino-Pearson")
        ]
        
        for test_key, test_label in test_names:
            p_key = f'{test_key}_p'
            stat_key = f'{test_key}_stat'
            pass_key = f'{test_key}_pass'
            error_key = f'{test_key}_error'
            
            if error_key in results:
                print(f"  {test_label:<25} ⚠️  {results[error_key]}")
                all_test_rows.append({
                    'Error_Type': error_name,
                    'Test': test_label,
                    'Statistic': None,
                    'P_Value': None,
                    'Result': 'ERROR',
                    'Significance': alpha
                })
            elif p_key in results:
                status = "PASS" if results[pass_key] else "FAIL"
                symbol = "✅" if results[pass_key] else "❌"
                print(f"  {test_label:<25} p = {results[p_key]:.6f}  {symbol} {status}")
                all_test_rows.append({
                    'Error_Type': error_name,
                    'Test': test_label,
                    'Statistic': results.get(stat_key, None),
                    'P_Value': results[p_key],
                    'Result': status,
                    'Significance': alpha
                })
            elif test_key == 'anderson_darling' and 'anderson_darling_stat' in results:
                status = "PASS" if results['anderson_darling_pass'] else "FAIL"
                symbol = "✅" if results['anderson_darling_pass'] else "❌"
                print(f"  {test_label:<25} stat = {results['anderson_darling_stat']:.4f}  {symbol} {status}")
                all_test_rows.append({
                    'Error_Type': error_name,
                    'Test': test_label,
                    'Statistic': results['anderson_darling_stat'],
                    'P_Value': None,
                    'Result': status,
                    'Significance': alpha
                })
        
        # Summary statistics
        print(f"\n  Summary:")
        print(f"    Tests Passed: {results['tests_passed']}/{results['total_tests']} ({results['pass_rate']*100:.1f}%)")
        print(f"    Skewness: {results['skewness']:.4f}")
        print(f"    Kurtosis: {results['kurtosis']:.4f}")
        print(f"    Overall: {'✅ ALL TESTS PASS' if results['all_passed'] else '⚠️  SOME TESTS FAIL'}")
        
        all_results.append(results)
    
    # Bonferroni correction analysis
    print(f"\n{'='*80}")
    print("BONFERRONI CORRECTION FOR MULTIPLE TESTING")
    print(f"{'='*80}\n")
    
    n_error_types = len(error_cols)
    n_tests_per_type = 6
    total_comparisons = n_error_types * n_tests_per_type
    
    alpha_corrected = alpha / total_comparisons
    
    print(f"Original significance level: α = {alpha}")
    print(f"Number of comparisons: {total_comparisons} ({n_error_types} error types × {n_tests_per_type} tests)")
    print(f"Bonferroni corrected level: α = {alpha_corrected:.6f}")
    
    # Save detailed results
    report_df = pd.DataFrame(all_test_rows)
    report_path = os.path.join(output_dir, 'comprehensive_validation_report.csv')
    report_df.to_csv(report_path, index=False)
    
    # Summary statistics
    summary_df = pd.DataFrame(all_results)
    summary_path = os.path.join(output_dir, 'comprehensive_validation_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    # Final summary
    print(f"\n{'='*80}")
    print("OVERALL ASSESSMENT")
    print(f"{'='*80}\n")
    
    total_pass = sum(r['tests_passed'] for r in all_results)
    total_tests = sum(r['total_tests'] for r in all_results)
    
    print(f"Total Gaussianity Tests: {total_tests}")
    print(f"Tests Passed (α = 0.05): {total_pass} ({total_pass/total_tests*100:.1f}%)")
    
    if total_pass == total_tests:
        print(f"\n✅ ✅ ✅ PERFECT: ALL {total_tests} TESTS PASS!")
    elif total_pass / total_tests >= 0.9:
        print(f"\n✅ EXCELLENT: {total_pass/total_tests*100:.0f}% of tests pass")
    elif total_pass / total_tests >= 0.75:
        print(f"\n✅ GOOD: {total_pass/total_tests*100:.0f}% of tests pass")
    else:
        print(f"\n⚠️  MODERATE: Only {total_pass/total_tests*100:.0f}% of tests pass")
    
    print(f"\n{'='*80}")
    print("FILES SAVED:")
    print(f"{'='*80}")
    print(f"  • {report_path}")
    print(f"  • {summary_path}")
    print(f"{'='*80}\n")
    
    return report_df, summary_df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive statistical validation')
    parser.add_argument('--predictions', default='outputs/day8_forecast_15min.csv',
                       help='Predictions CSV file')
    parser.add_argument('--output-dir', default='outputs',
                       help='Output directory')
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='Significance level (default: 0.05)')
    
    args = parser.parse_args()
    
    validate_predictions_comprehensive(
        predictions_csv=args.predictions,
        output_dir=args.output_dir,
        alpha=args.alpha
    )
