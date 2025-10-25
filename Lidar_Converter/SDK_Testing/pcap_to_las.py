#!/usr/bin/env python3
"""
Fast Ouster PCAP to LAS Converter
Optimized version for quick testing and conversion
"""

import os
import sys
import numpy as np
import laspy
from pathlib import Path
import argparse
from typing import Optional, Tuple

def find_data_files(data_dir: str = "sample_data") -> Tuple[Optional[str], Optional[str]]:
    """Find PCAP and JSON files in the data directory."""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ Data directory '{data_dir}' not found!")
        return None, None
    
    # Find PCAP file
    pcap_files = list(data_path.glob("*.pcap"))
    if not pcap_files:
        print(f"❌ No PCAP files found in '{data_dir}'")
        return None, None
    
    # Find JSON file
    json_files = list(data_path.glob("*.json"))
    if not json_files:
        print(f"❌ No JSON metadata files found in '{data_dir}'")
        return None, None
    
    pcap_file = pcap_files[0]
    json_file = json_files[0]
    
    print(f"✓ Found PCAP file: {pcap_file.name}")
    print(f"✓ Found JSON file: {json_file.name}")
    
    return str(pcap_file), str(json_file)

def convert_pcap_to_las_fast(pcap_path: str, json_path: str, output_path: str = "output.las", max_scans: int = 10):
    """Fast conversion of Ouster PCAP data to LAS format."""
    
    print(f"\n=== Fast PCAP to LAS Conversion ===")
    print(f"Input PCAP: {pcap_path}")
    print(f"Input JSON: {json_path}")
    print(f"Output LAS: {output_path}")
    print(f"Max scans to process: {max_scans}")
    
    try:
        # Import Ouster SDK
        from ouster.sdk import open_source
        from ouster.sdk.core import SensorInfo, XYZLut, ChanField
        
        print("✓ Ouster SDK imported successfully")
        
        # Read sensor metadata
        print("📂 Reading sensor metadata...")
        with open(json_path, 'r') as f:
            metadata = SensorInfo(f.read())
        
        print("✓ Metadata loaded successfully")
        
        # Open the data source
        print("📂 Opening data source...")
        source = open_source(pcap_path, meta=[json_path])
        print("✓ Data source opened successfully")
        
        # Precompute the XYZ lookup table
        print("🔧 Computing XYZ lookup table...")
        xyzlut = XYZLut(metadata)
        print("✓ XYZ lookup table computed")
        
        # Collect point cloud data
        all_points = []
        scan_count = 0
        
        print("🔄 Processing scans...")
        
        # Iterate over scans in the PCAP
        for scans in source:
            for scan in scans:
                if scan is None:
                    continue
                    
                scan_count += 1
                print(f"  Processing scan {scan_count}/{max_scans}...", end="\r")
                
                # Get range data
                range_data = scan.field(ChanField.RANGE)
                
                # Compute XYZ coordinates using the lookup table
                xyz = xyzlut(range_data)
                
                # Get intensity (reflectivity)
                intensity = scan.field(ChanField.REFLECTIVITY)
                
                # Filter out invalid points FIRST (range = 0) - this is much faster
                valid_mask = range_data > 0
                
                if not np.any(valid_mask):
                    continue  # Skip if no valid points
                
                # Only process valid points
                valid_xyz = xyz[valid_mask]
                valid_intensity = intensity[valid_mask]
                valid_range = range_data[valid_mask]
                
                # Create points array directly
                points = np.column_stack([
                    valid_xyz[:, 0],  # X
                    valid_xyz[:, 1],  # Y
                    valid_xyz[:, 2],  # Z
                    valid_intensity,  # Intensity
                    valid_range,      # Range
                    np.full(len(valid_intensity), scan_count, dtype=np.uint16)  # Scan ID
                ])
                
                all_points.append(points)
                
                # Limit number of scans to process
                if scan_count >= max_scans:
                    print(f"\n✓ Processed {scan_count} scans (limited by max_scans)")
                    break
            
            if scan_count >= max_scans:
                break
        
        if not all_points:
            print("❌ No valid points found in the data!")
            return False
        
        # Combine all points
        print(f"\n📊 Combining {len(all_points)} scan point clouds...")
        combined_points = np.vstack(all_points)
        
        print(f"✓ Total points: {len(combined_points):,}")
        print(f"  X range: {combined_points[:, 0].min():.2f} to {combined_points[:, 0].max():.2f}")
        print(f"  Y range: {combined_points[:, 1].min():.2f} to {combined_points[:, 1].max():.2f}")
        print(f"  Z range: {combined_points[:, 2].min():.2f} to {combined_points[:, 2].max():.2f}")
        
        # Create LAS file
        print("💾 Creating LAS file...")
        
        # Create LAS header
        header = laspy.LasHeader(point_format=1, version="1.2")
        header.x_scale = 0.001  # 1mm precision
        header.y_scale = 0.001
        header.z_scale = 0.001
        header.x_offset = combined_points[:, 0].mean()
        header.y_offset = combined_points[:, 1].mean()
        header.z_offset = combined_points[:, 2].mean()
        
        # Create LAS data
        las = laspy.LasData(header)
        
        # Set point data
        las.x = combined_points[:, 0]
        las.y = combined_points[:, 1]
        las.z = combined_points[:, 2]
        las.intensity = combined_points[:, 3].astype(np.uint16)
        
        # Write LAS file
        las.write(output_path)
        
        print(f"✅ LAS file created successfully: {output_path}")
        print(f"   Points: {len(combined_points):,}")
        print(f"   File size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Fast convert Ouster PCAP to LAS format")
    parser.add_argument("--data-dir", default="sample_data", help="Directory containing PCAP and JSON files")
    parser.add_argument("--output", default="fast_output.las", help="Output LAS file path")
    parser.add_argument("--max-scans", type=int, default=5, help="Maximum number of scans to process")
    
    args = parser.parse_args()
    
    print("🚀 Fast Ouster PCAP to LAS Converter")
    print("=" * 50)
    
    # Find data files
    pcap_path, json_path = find_data_files(args.data_dir)
    
    if not pcap_path or not json_path:
        print("\n❌ Required data files not found!")
        print("Please ensure you have:")
        print("1. A PCAP file in the data directory")
        print("2. A corresponding JSON metadata file")
        return 1
    
    # Convert to LAS
    success = convert_pcap_to_las_fast(pcap_path, json_path, args.output, args.max_scans)
    
    if success:
        print(f"\n🎉 Conversion completed successfully!")
        print(f"📁 Output file: {args.output}")
        print(f"🔍 You can now open this file in CloudCompare or other point cloud viewers")
        return 0
    else:
        print(f"\n❌ Conversion failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
