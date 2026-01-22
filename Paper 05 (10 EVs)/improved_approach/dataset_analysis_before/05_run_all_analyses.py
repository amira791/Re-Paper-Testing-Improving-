# file: 05_run_all_analyses.py
#!/usr/bin/env python3
"""
Master script to run all dataset analyses
"""

import subprocess
import sys
from pathlib import Path
import time

def run_script(script_name, description):
    """Run a Python script and capture output"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            check=True
        )
        
        print(result.stdout)
        
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
        
        elapsed = time.time() - start_time
        print(f"\n✓ Completed in {elapsed:.1f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running {script_name}:")
        print(e.stdout)
        print(e.stderr)
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def main():
    """Run all analysis scripts"""
    print("=" * 70)
    print("COMPREHENSIVE DATASET ANALYSIS FOR 10 EVs")
    print("=" * 70)
    
    scripts = [
        ("01_data_summary.py", "Basic Dataset Summary and Statistics"),
        ("02_charging_analysis.py", "Charging Event Detection and Analysis"),
        ("03_soh_estimation.py", "SOH Estimation using Paper's Method"),
        ("04_data_quality_report.py", "Data Quality Assessment and Anomaly Detection")
    ]
    
    all_successful = True
    
    for script_file, description in scripts:
        success = run_script(script_file, description)
        all_successful = all_successful and success
        
        # Small pause between scripts
        time.sleep(1)
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    
    if all_successful:
        print("✓ All analyses completed successfully!")
        print("\nGenerated output files:")
        
        output_dirs = [
            "./analysis_output",
            "./analysis_output/charging_analysis",
            "./analysis_output/soh_estimation",
            "./analysis_output/data_quality"
        ]
        
        for output_dir in output_dirs:
            path = Path(output_dir)
            if path.exists():
                files = list(path.glob("*"))
                if files:
                    print(f"\n{output_dir}/")
                    for file in sorted(files):
                        if file.is_file():
                            size = file.stat().st_size / 1024  # KB
                            print(f"  {file.name} ({size:.1f} KB)")
    else:
        print("✗ Some analyses failed. Check the output above for errors.")
    
    print(f"\nNext steps:")
    print("1. Review the generated CSV files and plots")
    print("2. Check data quality reports for cleaning recommendations")
    print("3. Use SOH estimates as labels for ML model training")
    print("4. Consider feature engineering based on the analysis")

if __name__ == "__main__":
    main()