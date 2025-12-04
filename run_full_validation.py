"""
Master Validation Pipeline
===========================

Runs all validation scripts in correct order and generates final report.

Usage:
    python run_full_validation.py
"""

import subprocess
import sys
from pathlib import Path
import time


def run_script(script_path, description):
    """Run a Python script and report results"""
    print(f"\n{'='*70}")
    print(f"🚀 Running: {description}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path.cwd()
        )
        
        elapsed = time.time() - start_time
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        print(f"✅ Completed in {elapsed:.2f}s")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        
        print(f"❌ FAILED after {elapsed:.2f}s")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ ERROR after {elapsed:.2f}s: {e}")
        return False


def main():
    """Run complete validation pipeline"""
    
    print("\n" + "="*70)
    print("🔬 FULL VALIDATION PIPELINE")
    print("="*70)
    print("\nThis will:")
    print("  1. Run comprehensive statistical tests (6 tests × 4 errors)")
    print("  2. Validate each time slot individually")
    print("  3. Test residual independence (autocorrelation)")
    print("  4. Verify everything is working correctly")
    print("\nEstimated time: 1-2 minutes\n")
    
    # Check if predictions file exists
    predictions_file = Path("outputs/day8_forecast_15min.csv")
    if not predictions_file.exists():
        print(f"⚠️  WARNING: {predictions_file} not found!")
        print("   Make sure you have run the forecasting scripts first.")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return 1
    
    # Pipeline steps
    steps = [
        ("src/validate_comprehensive.py", "Comprehensive Statistical Tests"),
        ("src/validate_per_slot.py", "Per-Slot Gaussianity Validation"),
        ("src/test_residual_independence.py", "Residual Independence Tests"),
        ("src/validate_validation.py", "Meta-Validation (Check Everything Works)")
    ]
    
    results = {}
    
    for script, description in steps:
        success = run_script(script, description)
        results[description] = success
        
        if not success:
            print(f"\n⚠️  {description} failed - continuing anyway...")
            time.sleep(1)
    
    # Final summary
    print("\n" + "="*70)
    print("📊 VALIDATION PIPELINE RESULTS")
    print("="*70)
    
    for step, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {step:50s}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("🎉 ALL VALIDATION STEPS COMPLETED SUCCESSFULLY!")
        print("\n📁 Generated outputs:")
        print("  ✅ outputs/comprehensive_validation_report.csv")
        print("  ✅ outputs/comprehensive_validation_summary.csv")
        print("  ✅ outputs/slot_validation_report.csv")
        print("  ✅ outputs/independence_test_report.csv")
        print("  ✅ outputs/plots/slot_validation_heatmap.png")
        print("\n🎯 Next steps:")
        print("  1. Review outputs/comprehensive_validation_report.csv")
        print("  2. Check pass rates in slot_validation_report.csv")
        print("  3. Verify independence in independence_test_report.csv")
        print("  4. Present results to ISRO with confidence!")
    else:
        print("⚠️  SOME STEPS FAILED")
        print("\n💡 Troubleshooting:")
        print("  - Check that outputs/day8_forecast_15min.csv exists")
        print("  - Ensure all dependencies installed (scipy, statsmodels)")
        print("  - Review error messages above")
        print("  - Re-run failed scripts individually")
    
    print("="*70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
