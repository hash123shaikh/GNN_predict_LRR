#!/usr/bin/env python3
"""
Automated Setup Script for RadGraph Implementation
Checks environment, data structure, and runs initial tests

Usage:
    python setup.py
"""

import sys
import os
from pathlib import Path
import subprocess

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def check_python_version():
    """Check Python version"""
    print_header("Checking Python Version")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    
    print("✅ Python version OK")
    return True

def check_packages():
    """Check required packages"""
    print_header("Checking Required Packages")
    
    required_packages = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('torch', 'torch'),
        ('sklearn', 'scikit-learn'),
        ('SimpleITK', 'SimpleITK'),
    ]
    
    missing = []
    
    for display_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"✅ {display_name}")
        except ImportError:
            print(f"❌ {display_name} - NOT FOUND")
            missing.append(display_name)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def check_directory_structure():
    """Check directory structure"""
    print_header("Checking Directory Structure")
    
    base_dir = Path(__file__).parent
    
    required_dirs = [
        'data',
        'outputs',
        'models',
        'logs'
    ]
    
    all_exist = True
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            print(f"✅ {dir_name}/")
        else:
            print(f"❌ {dir_name}/ - CREATING...")
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"   Created {dir_path}")
    
    return True

def check_config_file():
    """Check if config.py exists and is valid"""
    print_header("Checking Configuration")
    
    try:
        import config
        print("✅ config.py loaded successfully")
        
        # Check critical paths
        paths_to_check = [
            ('CT_SCANS_DIR', config.CT_SCANS_DIR),
            ('RTSTRUCT_DIR', config.RTSTRUCT_DIR),
            ('CLINICAL_DATA_FILE', config.CLINICAL_DATA_FILE),
        ]
        
        for name, path in paths_to_check:
            print(f"\n{name}: {path}")
            if path.exists():
                print(f"  ✅ Path exists")
            else:
                print(f"  ⚠️  Path does not exist - please update config.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading config.py: {e}")
        return False

def check_data_files():
    """Check if data files exist"""
    print_header("Checking Data Files")
    
    try:
        import config
        import pandas as pd
        
        # Check clinical data
        if config.CLINICAL_DATA_FILE.exists():
            print(f"✅ Clinical data: {config.CLINICAL_DATA_FILE}")
            
            df = pd.read_csv(config.CLINICAL_DATA_FILE)
            print(f"   Patients: {len(df)}")
            print(f"   Columns: {len(df.columns)}")
            
            # Check required columns
            required_cols = [config.PATIENT_ID_COL] + config.CLINICAL_FEATURES
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"   ⚠️  Missing columns: {missing_cols}")
            else:
                print(f"   ✅ All required columns present")
        else:
            print(f"❌ Clinical data not found: {config.CLINICAL_DATA_FILE}")
        
        # Check radiomics features
        if hasattr(config, 'RADIOMICS_FEATURES_FILE') and config.RADIOMICS_FEATURES_FILE.exists():
            print(f"✅ Radiomics features: {config.RADIOMICS_FEATURES_FILE}")
            
            df = pd.read_csv(config.RADIOMICS_FEATURES_FILE)
            print(f"   Patients: {len(df)}")
            print(f"   Features: {len(df.columns) - 1}")  # Exclude patient_id
        else:
            print(f"⚠️  Radiomics features not found")
            print(f"   You may need to extract features first")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking data files: {e}")
        return False

def run_quick_test():
    """Run quick test of core functionality"""
    print_header("Running Quick Functionality Test")
    
    try:
        # Test imports
        print("Testing imports...")
        from utils import set_seed, get_device
        from data_loader import HNSCCDataLoader
        
        print("✅ Core modules imported successfully")
        
        # Test seed setting
        print("\nTesting seed setting...")
        set_seed(42)
        print("✅ Seed set successfully")
        
        # Test device detection
        print("\nTesting device detection...")
        device = get_device(use_cuda=False)  # Use CPU for testing
        print(f"✅ Device: {device}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def print_summary():
    """Print setup summary"""
    print_header("Setup Summary")
    
    print("""
✅ Your environment is ready!

Next steps:

1. Verify your data paths in config.py:
   - Update CT_SCANS_DIR
   - Update RTSTRUCT_DIR  
   - Update CLINICAL_DATA_FILE
   - Update RADIOMICS_FEATURES_FILE

2. Run the baseline model:
   python main_simple.py --task LR --split_data --train --evaluate

3. Check results in:
   - outputs/test_results_LR.csv
   - outputs/roc_curve_LR.png

For detailed instructions, see README.md

Good luck! 🚀
""")

def main():
    """Main setup function"""
    print("""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║           RadGraph Implementation Setup                    ║
║         CMC Vellore - Radiation Oncology                  ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
""")
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_packages),
        ("Directory Structure", check_directory_structure),
        ("Configuration", check_config_file),
        ("Data Files", check_data_files),
        ("Functionality", run_quick_test),
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"\n❌ {check_name} check failed with error: {e}")
            results.append((check_name, False))
    
    # Summary
    print_header("Setup Check Results")
    
    all_passed = True
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10} - {check_name}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All checks passed!")
        print_summary()
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  1. Install missing packages: pip install -r requirements.txt")
        print("  2. Update paths in config.py")
        print("  3. Ensure data files are in correct format")

if __name__ == '__main__':
    main()
