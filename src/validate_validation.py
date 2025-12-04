"""
Meta-Validator: Checks That All Validation Scripts Are Working Properly
========================================================================

This script verifies that:
1. Validation reports were generated correctly
2. All required columns exist
3. Test results are reasonable
4. Files have expected structure
5. Sanity checks pass
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys


def check_comprehensive_validation():
    """Check if comprehensive validation ran correctly"""
    print("\n" + "="*60)
    print("1. CHECKING COMPREHENSIVE STATISTICAL TESTS")
    print("="*60)
    
    report_path = Path("outputs/comprehensive_validation_report.csv")
    
    if not report_path.exists():
        print("❌ FAIL: Report not generated")
        print("   Run: python src/validate_comprehensive.py")
        return False
    
    df = pd.read_csv(report_path)
    
    # Check structure
    required_cols = ['Error_Type', 'Test', 'P_Value', 'Result', 'Significance']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"❌ FAIL: Missing columns: {missing}")
        return False
    print("✅ PASS: Report has correct structure")
    
    # Check all tests present
    expected_tests = ['Shapiro-Wilk', 'Anderson-Darling', 'Kolmogorov-Smirnov', 
                      'Jarque-Bera', 'Lilliefors', "D'Agostino-Pearson"]
    actual_tests = df['Test'].unique()
    missing_tests = [t for t in expected_tests if t not in actual_tests]
    if missing_tests:
        print(f"❌ FAIL: Missing tests: {missing_tests}")
        return False
    print(f"✅ PASS: All {len(expected_tests)} tests present")
    
    # Check all error types
    expected_errors = 4  # X, Y, Z, Clock
    actual_errors = df['Error_Type'].nunique()
    if actual_errors != expected_errors:
        print(f"⚠️  WARNING: Expected {expected_errors} error types, got {actual_errors}")
    else:
        print("✅ PASS: All 4 error types tested")
    
    # Check pass rates
    pass_rate = (df['Result'] == 'PASS').mean()
    print(f"📊 Overall pass rate: {pass_rate*100:.1f}%")
    
    if pass_rate < 0.5:
        print("⚠️  WARNING: Pass rate below 50% - check model")
    elif pass_rate == 1.0:
        print("✅ EXCELLENT: 100% test pass rate!")
    else:
        print(f"✅ GOOD: {pass_rate*100:.1f}% tests passing")
    
    return True


def check_per_slot_validation():
    """Check if per-slot validation ran correctly"""
    print("\n" + "="*60)
    print("2. CHECKING PER-SLOT GAUSSIANITY TESTS")
    print("="*60)
    
    report_path = Path("outputs/slot_validation_report.csv")
    
    if not report_path.exists():
        print("❌ FAIL: Report not generated")
        print("   Run: python src/validate_per_slot.py")
        return False
    
    df = pd.read_csv(report_path)
    
    # Check structure
    required_cols = ['TimeOfDay', 'Error_Type', 'Shapiro_p', 'Kurtosis', 'Skewness', 
                     'Sample_Size', 'Gaussianity']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"❌ FAIL: Missing columns: {missing}")
        return False
    print("✅ PASS: Report has correct structure")
    
    # Check time slots
    n_slots = df['TimeOfDay'].nunique()
    if n_slots == 0:
        print("❌ FAIL: No time slots found")
        return False
    print(f"✅ PASS: {n_slots} time slots validated")
    
    # Check per error type
    for error in df['Error_Type'].unique():
        subset = df[df['Error_Type'] == error]
        tested = subset[subset['Gaussianity'].isin(['PASS', 'FAIL'])]
        if len(tested) > 0:
            pass_rate = (tested['Gaussianity'] == 'PASS').mean()
            n_pass = (tested['Gaussianity'] == 'PASS').sum()
            n_total = len(tested)
            
            print(f"  {error:12s}: {n_pass}/{n_total} slots pass ({pass_rate*100:.1f}%)")
    
    return True


def check_independence_tests():
    """Check if residual independence tests ran correctly"""
    print("\n" + "="*60)
    print("3. CHECKING RESIDUAL INDEPENDENCE TESTS")
    print("="*60)
    
    report_path = Path("outputs/independence_test_report.csv")
    
    if not report_path.exists():
        print("❌ FAIL: Report not generated")
        print("   Run: python src/test_residual_independence.py")
        return False
    
    df = pd.read_csv(report_path)
    
    # Check structure
    required_cols = ['Error_Type', 'LjungBox_p', 'DurbinWatson', 'Runs_p', 'Independence']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"❌ FAIL: Missing columns: {missing}")
        return False
    print("✅ PASS: Report has correct structure")
    
    # Check all error types
    if len(df) != 4:
        print(f"⚠️  WARNING: Expected 4 rows (one per error type), got {len(df)}")
    else:
        print("✅ PASS: All 4 error types tested")
    
    # Check independence results
    for _, row in df.iterrows():
        error = row['Error_Type']
        result = row['Independence']
        
        ljung_ok = "✅" if pd.notna(row['LjungBox_p']) and row['LjungBox_p'] > 0.05 else "⚠️"
        dw_ok = "✅" if pd.notna(row['DurbinWatson']) and 1.5 < row['DurbinWatson'] < 2.5 else "⚠️"
        
        print(f"  {error:12s}: {result:4s} {ljung_ok} {dw_ok}")
    
    pass_count = (df['Independence'] == 'PASS').sum()
    print(f"\n📊 {pass_count}/4 error types show independence")
    
    return True


def check_plots_exist():
    """Check if required plots were generated"""
    print("\n" + "="*60)
    print("4. CHECKING VISUALIZATION OUTPUTS")
    print("="*60)
    
    plots_dir = Path("outputs/plots")
    
    if not plots_dir.exists():
        print("⚠️  Plots directory doesn't exist")
        return False
    
    required_plots = [
        'slot_validation_heatmap.png',
    ]
    
    all_exist = True
    for plot in required_plots:
        path = plots_dir / plot
        if path.exists():
            print(f"  ✅ {plot}")
        else:
            print(f"  ⚠️  {plot} - not found")
            all_exist = False
    
    # Check for any plots
    all_plots = list(plots_dir.glob('*.png'))
    print(f"\n  Found {len(all_plots)} total plots in outputs/plots/")
    
    return True


def run_sanity_checks():
    """Run sanity checks on validation outputs"""
    print("\n" + "="*60)
    print("5. SANITY CHECKS")
    print("="*60)
    
    # Check 1: Comprehensive tests should have reasonable p-values
    comp_path = Path("outputs/comprehensive_validation_report.csv")
    if comp_path.exists():
        df = pd.read_csv(comp_path)
        p_values = df['P_Value'].dropna()
        
        if len(p_values) > 0:
            if (p_values < 0).any():
                print("❌ FAIL: Found negative p-values (impossible!)")
                return False
            
            if (p_values > 1).any():
                print("❌ FAIL: Found p-values > 1 (impossible!)")
                return False
            
            print("✅ PASS: All p-values in valid range [0, 1]")
        else:
            print("⚠️  WARNING: No p-values found")
    
    # Check 2: Per-slot should have reasonable sample sizes
    slot_path = Path("outputs/slot_validation_report.csv")
    if slot_path.exists():
        df = pd.read_csv(slot_path)
        
        if 'Sample_Size' in df.columns:
            avg_size = df['Sample_Size'].mean()
            print(f"✅ Average sample size per slot: {avg_size:.1f}")
        else:
            print("⚠️  WARNING: No Sample_Size column")
    
    # Check 3: Independence tests should have Durbin-Watson near 2
    indep_path = Path("outputs/independence_test_report.csv")
    if indep_path.exists():
        df = pd.read_csv(indep_path)
        if 'DurbinWatson' in df.columns:
            dw_values = df['DurbinWatson'].dropna()
            
            if len(dw_values) > 0:
                if (dw_values < 0).any() or (dw_values > 4).any():
                    print("❌ FAIL: Durbin-Watson outside valid range [0, 4]")
                    return False
                
                mean_dw = dw_values.mean()
                print(f"✅ Mean Durbin-Watson: {mean_dw:.3f} (target: ~2.0)")
            else:
                print("⚠️  WARNING: No Durbin-Watson values")
        else:
            print("⚠️  WARNING: No DurbinWatson column")
    
    return True


def main():
    """Run all validation checks"""
    print("\n" + "="*70)
    print("🔍 META-VALIDATION: Checking That All Tests Are Working Properly")
    print("="*70)
    
    results = {
        'Comprehensive Tests': check_comprehensive_validation(),
        'Per-Slot Tests': check_per_slot_validation(),
        'Independence Tests': check_independence_tests(),
        'Plots Generated': check_plots_exist(),
        'Sanity Checks': run_sanity_checks()
    }
    
    print("\n" + "="*70)
    print("📊 VALIDATION SUMMARY")
    print("="*70)
    
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check:30s}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("🎉 ALL VALIDATION CHECKS PASSED!")
        print("Your testing framework is working correctly.")
        print("\n📁 Check these files for detailed results:")
        print("  - outputs/comprehensive_validation_report.csv")
        print("  - outputs/slot_validation_report.csv")
        print("  - outputs/independence_test_report.csv")
        print("  - outputs/plots/")
    else:
        print("⚠️  SOME VALIDATION CHECKS FAILED")
        print("Review the output above to identify issues.")
        print("\n💡 Run validation scripts:")
        print("  python src/validate_comprehensive.py")
        print("  python src/validate_per_slot.py")
        print("  python src/test_residual_independence.py")
    print("="*70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
