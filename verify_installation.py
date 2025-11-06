#!/usr/bin/env python3
"""
Installation verification script for LiDAR Converter.

This script checks that all components are properly installed and working.
Run this after installation to verify everything is set up correctly.
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Check Python version compatibility."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} (compatible)")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (requires Python 3.8+)")
        return False

def check_dependencies():
    """Check required dependencies."""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        ("numpy", "Core numerical computing"),
        ("scipy", "Scientific computing"),
        ("laspy", "LAS file handling"),
        ("dpkt", "PCAP parsing"),
        ("click", "CLI framework"),
        ("rich", "Rich terminal output"),
        ("tqdm", "Progress bars")
    ]
    
    optional_packages = [
        ("ouster", "Ouster SDK (for Ouster sensors)"),
        ("pandas", "Data analysis"),
        ("scikit-learn", "Machine learning utilities")
    ]
    
    all_good = True
    
    for package, description in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package} - {description}")
        except ImportError:
            print(f"   ❌ {package} - {description} (MISSING)")
            all_good = False
    
    print("\n   Optional packages:")
    for package, description in optional_packages:
        try:
            __import__(package)
            print(f"   ✅ {package} - {description}")
        except ImportError:
            print(f"   ⚠️  {package} - {description} (optional, not installed)")
    
    return all_good

def check_lidar_converter():
    """Check LiDAR Converter modules."""
    print("\n🔧 Checking LiDAR Converter modules...")
    
    # Add current directory to path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    modules_to_check = [
        ("Lidar_Converter.detector", "Vendor detection"),
        ("Lidar_Converter.converter", "Main converter"),
        ("Lidar_Converter.cli", "Command-line interface"),
        ("Lidar_Converter.Wrappers", "Vendor wrappers"),
        ("Lidar_Converter.Wrappers.base_wrapper", "Base wrapper class"),
        ("Lidar_Converter.Wrappers.ouster_wrapper", "Ouster wrapper"),
        ("Lidar_Converter.Wrappers.velodyne_wrapper", "Velodyne wrapper")
    ]
    
    all_good = True
    
    for module, description in modules_to_check:
        try:
            __import__(module)
            print(f"   ✅ {module} - {description}")
        except ImportError as e:
            print(f"   ❌ {module} - {description} (ERROR: {e})")
            all_good = False
    
    return all_good

def check_cli():
    """Check CLI functionality."""
    print("\n💻 Checking CLI functionality...")
    
    try:
        # Try to import and run health check
        from Lidar_Converter.converter import LiDARConverter
        
        converter = LiDARConverter()
        health = converter.health_check()
        
        print(f"   ✅ CLI health check: {health.get('status', 'unknown')}")
        
        # Check vendor support
        vendors = health.get('vendors', {})
        for vendor, info in vendors.items():
            status = "✅" if info.get('available') else "⚠️"
            print(f"   {status} {vendor}: {info.get('status', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ CLI check failed: {e}")
        return False

def main():
    """Run all verification checks."""
    print("🚀 LiDAR Converter Installation Verification")
    print("=" * 50)
    
    checks = [
        check_python_version(),
        check_dependencies(),
        check_lidar_converter(),
        check_cli()
    ]
    
    print("\n" + "=" * 50)
    
    if all(checks):
        print("🎉 All checks passed! LiDAR Converter is ready to use.")
        print("\nNext steps:")
        print("  1. Run: python Lidar_Converter/cli.py health")
        print("  2. Try: python Lidar_Converter/cli.py detect <your_file.pcap>")
        print("  3. Convert: python Lidar_Converter/cli.py convert <your_file.pcap> -o output.las")
        print("\nFor more information, see README.md and Lidar_Converter/CLI_README.md")
        return 0
    else:
        print("❌ Some checks failed. Please install missing dependencies:")
        print("  pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())