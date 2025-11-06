#!/usr/bin/env python3
"""
Example usage of LiDAR Converter library.

This script demonstrates how to use the LiDAR Converter programmatically
for both single file conversion and batch processing.
"""

import sys
import os
from pathlib import Path

# Add Lidar_Converter to path
sys.path.insert(0, str(Path(__file__).parent / "Lidar_Converter"))

from converter import LiDARConverter
from detector import VendorDetector
from Wrappers import OusterWrapper, VelodyneWrapper

def example_vendor_detection():
    """Example: Detect vendor of a LiDAR file."""
    print("🔍 Example: Vendor Detection")
    print("-" * 30)
    
    # Initialize detector
    detector = VendorDetector()
    
    # Example file path (replace with your actual file)
    file_path = "sample_data.pcap"
    
    if not Path(file_path).exists():
        print(f"   ⚠️  Sample file not found: {file_path}")
        print("   Create a sample file or update the path in this script")
        return
    
    # Detect vendor
    result = detector.detect_vendor(file_path)
    
    if result["success"]:
        print(f"   ✅ Detected vendor: {result['vendor_name']}")
        print(f"   📊 Confidence: {result['confidence']:.2%}")
        print(f"   🔖 File signature: {result['file_signature']}")
        
        # Show detection metadata
        metadata = result.get("metadata", {})
        methods = metadata.get("detection_methods", [])
        if methods:
            print(f"   🔬 Detection methods: {', '.join(methods)}")
    else:
        print(f"   ❌ Detection failed: {result['error']}")

def example_automatic_conversion():
    """Example: Automatic vendor detection and conversion."""
    print("\n🔄 Example: Automatic Conversion")
    print("-" * 35)
    
    # Initialize converter
    converter = LiDARConverter()
    
    # Example file paths (replace with your actual files)
    input_file = "sample_data.pcap"
    output_file = "output_example.las"
    
    if not Path(input_file).exists():
        print(f"   ⚠️  Sample file not found: {input_file}")
        print("   Create a sample file or update the path in this script")
        return
    
    # Convert with automatic vendor detection
    result = converter.convert(
        input_path=input_file,
        output_path=output_file,
        max_scans=100,  # Limit scans for faster processing
        preserve_intensity=True,
        validate_output=True
    )
    
    if result["success"]:
        print(f"   ✅ Conversion successful!")
        print(f"   📁 Input: {result['input_file']}")
        print(f"   📁 Output: {result['output_file']}")
        print(f"   🏭 Vendor: {result['vendor']}")
        print(f"   📊 Points converted: {result['points_converted']:,}")
        print(f"   ⏱️  Conversion time: {result['conversion_time']:.2f}s")
        print(f"   🎯 Detection confidence: {result['detection_confidence']:.2%}")
    else:
        print(f"   ❌ Conversion failed: {result['message']}")
        if result.get('errors'):
            for error in result['errors']:
                print(f"      • {error}")

def example_manual_wrapper_usage():
    """Example: Manual wrapper usage for specific vendors."""
    print("\n🔧 Example: Manual Wrapper Usage")
    print("-" * 35)
    
    # Example 1: Ouster wrapper
    print("   Ouster Wrapper:")
    ouster_wrapper = OusterWrapper()
    
    if ouster_wrapper.sdk_available:
        print(f"   ✅ Ouster SDK available (version: {ouster_wrapper.sdk_version})")
        
        # Get vendor info
        info = ouster_wrapper.get_vendor_info()
        print(f"   📋 Supported models: {', '.join(info['supported_sensor_models'][:3])}...")
        print(f"   📄 Supported formats: {', '.join(info['supported_output_formats'])}")
    else:
        print(f"   ❌ Ouster SDK not available")
    
    # Example 2: Velodyne wrapper
    print("\n   Velodyne Wrapper:")
    velodyne_wrapper = VelodyneWrapper()
    
    if velodyne_wrapper.sdk_available:
        print(f"   ✅ Velodyne processing available (method: {velodyne_wrapper.sdk_version})")
        
        # Get vendor info
        info = velodyne_wrapper.get_vendor_info()
        print(f"   📋 Supported models: {', '.join(info['supported_sensor_models'][:3])}...")
        print(f"   📄 Supported formats: {', '.join(info['supported_output_formats'])}")
    else:
        print(f"   ❌ Velodyne processing not available")

def example_batch_processing():
    """Example: Batch processing multiple files."""
    print("\n📦 Example: Batch Processing")
    print("-" * 30)
    
    # Initialize converter
    converter = LiDARConverter()
    
    # Example directory (replace with your actual directory)
    input_dir = "sample_data_dir"
    output_dir = "batch_output"
    
    if not Path(input_dir).exists():
        print(f"   ⚠️  Sample directory not found: {input_dir}")
        print("   Create a sample directory with PCAP files or update the path")
        return
    
    # Find PCAP files
    pcap_files = list(Path(input_dir).glob("*.pcap"))
    
    if not pcap_files:
        print(f"   ⚠️  No PCAP files found in: {input_dir}")
        return
    
    print(f"   📁 Found {len(pcap_files)} PCAP files")
    
    # Batch convert
    results = converter.convert_batch(
        file_paths=[str(f) for f in pcap_files],
        output_dir=output_dir,
        output_format="las",
        max_scans=50  # Limit for faster processing
    )
    
    # Summary
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    
    if successful > 0:
        total_points = sum(r.get("points_converted", 0) for r in results if r["success"])
        print(f"   📊 Total points converted: {total_points:,}")

def example_health_check():
    """Example: System health check."""
    print("\n🏥 Example: Health Check")
    print("-" * 25)
    
    # Initialize converter
    converter = LiDARConverter()
    
    # Get health status
    health = converter.health_check()
    
    print(f"   🎯 Overall status: {health['status'].upper()}")
    
    # Show vendor status
    vendors = health.get('vendors', {})
    for vendor, info in vendors.items():
        status_icon = "✅" if info.get('available') else "❌"
        print(f"   {status_icon} {vendor.capitalize()}: {info.get('status', 'unknown')}")
        if info.get('sdk_version'):
            print(f"      Version: {info['sdk_version']}")

def main():
    """Run all examples."""
    print("🚀 LiDAR Converter Usage Examples")
    print("=" * 50)
    
    # Run examples
    example_health_check()
    example_vendor_detection()
    example_automatic_conversion()
    example_manual_wrapper_usage()
    example_batch_processing()
    
    print("\n" + "=" * 50)
    print("📚 For more examples and documentation:")
    print("   • README.md - Project overview")
    print("   • Lidar_Converter/CLI_README.md - CLI usage guide")
    print("   • Lidar_Converter/TESTING_GUIDE.md - Testing instructions")
    print("   • Lidar_Converter/Wrappers/README.md - Wrapper documentation")

if __name__ == "__main__":
    main()