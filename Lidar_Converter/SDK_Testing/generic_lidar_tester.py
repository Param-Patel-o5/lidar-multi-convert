#!/usr/bin/env python3
"""
Generic LiDAR SDK Testing Framework
This script provides a unified interface for testing any LiDAR vendor's SDK
without needing vendor-specific code.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LidarSDKInterface(ABC):
    """Abstract base class for all LiDAR SDK implementations."""
    
    @abstractmethod
    def detect_vendor(self, file_path: str) -> bool:
        """Detect if this SDK can handle the given file."""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Return list of supported file formats."""
        pass
    
    @abstractmethod
    def convert_to_las(self, input_path: str, output_path: str, **kwargs) -> bool:
        """Convert input file to LAS format."""
        pass
    
    @abstractmethod
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get information about the file (points, bounds, etc.)."""
        pass
    
    @abstractmethod
    def validate_conversion(self, input_path: str, output_path: str) -> bool:
        """Validate that the conversion was successful."""
        pass

class OusterSDK(LidarSDKInterface):
    """Ouster LiDAR SDK implementation."""
    
    def __init__(self):
        self.vendor_name = "Ouster"
        self.supported_formats = [".pcap"]
        
    def detect_vendor(self, file_path: str) -> bool:
        """Detect Ouster files by extension and content."""
        try:
            file_path = Path(file_path)
            
            # Check file extension
            if file_path.suffix.lower() not in self.supported_formats:
                return False
            
            # Check if corresponding JSON metadata exists
            json_path = file_path.with_suffix('.json')
            if not json_path.exists():
                logger.warning(f"No JSON metadata found for {file_path}")
                return False
            
            # Try to read and parse the JSON metadata
            with open(json_path, 'r') as f:
                metadata = json.load(f)
                
            # Check for Ouster-specific fields
            if 'ouster' in str(metadata).lower() or 'lidar_mode' in metadata:
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error detecting Ouster vendor: {e}")
            return False
    
    def get_supported_formats(self) -> List[str]:
        return self.supported_formats
    
    def convert_to_las(self, input_path: str, output_path: str, **kwargs) -> bool:
        """Convert Ouster PCAP to LAS format."""
        try:
            from ouster.sdk import open_source
            from ouster.sdk.core import SensorInfo, XYZLut, ChanField
            import numpy as np
            import laspy
            
            logger.info(f"Converting Ouster file: {input_path}")
            
            # Find corresponding JSON file
            input_path = Path(input_path)
            json_path = input_path.with_suffix('.json')
            
            # Try different JSON file patterns
            if not json_path.exists():
                # Try with different naming pattern
                json_path = input_path.parent / (input_path.stem + '.json')
                if not json_path.exists():
                    # Try with the exact filename from the directory
                    json_files = list(input_path.parent.glob('*.json'))
                    if json_files:
                        json_path = json_files[0]
                    else:
                        logger.error(f"JSON metadata file not found: {json_path}")
                        return False
            
            # Read sensor metadata
            with open(json_path, 'r') as f:
                metadata = SensorInfo(f.read())
            
            # Open data source
            source = open_source(str(input_path), meta=[str(json_path)])
            
            # Compute XYZ lookup table
            xyzlut = XYZLut(metadata)
            
            # Process scans
            all_points = []
            max_scans = kwargs.get('max_scans', 10)
            scan_count = 0
            
            for scans in source:
                for scan in scans:
                    if scan is None:
                        continue
                    
                    scan_count += 1
                    logger.info(f"Processing scan {scan_count}/{max_scans}")
                    
                    # Get range data
                    range_data = scan.field(ChanField.RANGE)
                    xyz = xyzlut(range_data)
                    intensity = scan.field(ChanField.REFLECTIVITY)
                    
                    # Filter valid points
                    valid_mask = range_data > 0
                    if not np.any(valid_mask):
                        continue
                    
                    valid_xyz = xyz[valid_mask]
                    valid_intensity = intensity[valid_mask]
                    
                    # Create points array
                    points = np.column_stack([
                        valid_xyz[:, 0],  # X
                        valid_xyz[:, 1],  # Y
                        valid_xyz[:, 2],  # Z
                        valid_intensity,  # Intensity
                        range_data[valid_mask],  # Range
                        np.full(len(valid_intensity), scan_count, dtype=np.uint16)  # Scan ID
                    ])
                    
                    all_points.append(points)
                    
                    if scan_count >= max_scans:
                        break
                
                if scan_count >= max_scans:
                    break
            
            if not all_points:
                logger.error("No valid points found")
                return False
            
            # Combine all points
            combined_points = np.vstack(all_points)
            logger.info(f"Total points: {len(combined_points):,}")
            
            # Create LAS file
            header = laspy.LasHeader(point_format=1, version="1.2")
            header.x_scale = 0.001
            header.y_scale = 0.001
            header.z_scale = 0.001
            header.x_offset = combined_points[:, 0].mean()
            header.y_offset = combined_points[:, 1].mean()
            header.z_offset = combined_points[:, 2].mean()
            
            las = laspy.LasData(header)
            las.x = combined_points[:, 0]
            las.y = combined_points[:, 1]
            las.z = combined_points[:, 2]
            las.intensity = combined_points[:, 3].astype(np.uint16)
            
            # Write LAS file
            las.write(output_path)
            logger.info(f"LAS file created: {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Ouster conversion failed: {e}")
            return False
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get information about the Ouster file."""
        try:
            file_path = Path(file_path)
            json_path = file_path.with_suffix('.json')
            
            if not json_path.exists():
                return {"error": "No JSON metadata found"}
            
            with open(json_path, 'r') as f:
                metadata = json.load(f)
            
            return {
                "vendor": "Ouster",
                "file_type": file_path.suffix,
                "metadata": metadata,
                "file_size_mb": file_path.stat().st_size / (1024 * 1024)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def validate_conversion(self, input_path: str, output_path: str) -> bool:
        """Validate the conversion by checking if LAS file can be read."""
        try:
            import laspy
            las = laspy.read(output_path)
            return len(las.points) > 0
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False

class HesaiSDK(LidarSDKInterface):
    """Hesai LiDAR SDK implementation."""
    
    def __init__(self):
        self.vendor_name = "Hesai"
        self.supported_formats = [".pcap", ".bin"]
        
    def detect_vendor(self, file_path: str) -> bool:
        """Detect Hesai files."""
        file_path = Path(file_path)
        return file_path.suffix.lower() in self.supported_formats
    
    def get_supported_formats(self) -> List[str]:
        return self.supported_formats
    
    def convert_to_las(self, input_path: str, output_path: str, **kwargs) -> bool:
        """Convert Hesai file to LAS format."""
        # TODO: Implement Hesai conversion using hesai-sdk or pandar-sdk
        logger.warning("Hesai conversion not yet implemented")
        return False
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get information about the Hesai file."""
        return {
            "vendor": "Hesai",
            "file_type": Path(file_path).suffix,
            "status": "Not implemented yet"
        }
    
    def validate_conversion(self, input_path: str, output_path: str) -> bool:
        """Validate Hesai conversion."""
        return False

class GenericLidarTester:
    """Generic tester that works with any LiDAR vendor."""
    
    def __init__(self):
        self.sdks = {
            "ouster": OusterSDK(),
            "hesai": HesaiSDK(),
        }
        self.detected_vendor = None
    
    def detect_vendor(self, file_path: str) -> Optional[str]:
        """Automatically detect the vendor of a LiDAR file."""
        logger.info(f"Detecting vendor for: {file_path}")
        
        # Use the centralized detector
        try:
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent.parent / "Lidar_Converter"))
            from detector import detect_lidar_vendor
            vendor, info = detect_lidar_vendor(file_path)
            if vendor:
                logger.info(f"Detected vendor: {vendor}")
                self.detected_vendor = vendor
                return vendor
        except Exception as e:
            logger.warning(f"Centralized detection failed: {e}")
        
        # Fallback to SDK-based detection
        for vendor_name, sdk in self.sdks.items():
            if sdk.detect_vendor(file_path):
                logger.info(f"Detected vendor: {vendor_name}")
                self.detected_vendor = vendor_name
                return vendor_name
        
        logger.warning("No vendor detected")
        return None
    
    def test_conversion(self, input_path: str, output_path: str, vendor: Optional[str] = None, **kwargs) -> bool:
        """Test conversion for any vendor."""
        if vendor is None:
            vendor = self.detect_vendor(input_path)
            if vendor is None:
                logger.error("Could not detect vendor and none specified")
                return False
        
        if vendor not in self.sdks:
            logger.error(f"Unknown vendor: {vendor}")
            return False
        
        sdk = self.sdks[vendor]
        logger.info(f"Testing {vendor} conversion...")
        
        # Get file info
        file_info = sdk.get_file_info(input_path)
        logger.info(f"File info: {file_info}")
        
        # Convert to LAS
        success = sdk.convert_to_las(input_path, output_path, **kwargs)
        
        if success:
            # Validate conversion
            validation = sdk.validate_conversion(input_path, output_path)
            logger.info(f"Conversion validation: {'PASSED' if validation else 'FAILED'}")
            return validation
        
        return False
    
    def list_supported_vendors(self) -> List[str]:
        """List all supported vendors."""
        return list(self.sdks.keys())
    
    def get_vendor_info(self, vendor: str) -> Dict[str, Any]:
        """Get information about a specific vendor."""
        if vendor not in self.sdks:
            return {"error": f"Unknown vendor: {vendor}"}
        
        sdk = self.sdks[vendor]
        return {
            "vendor": vendor,
            "supported_formats": sdk.get_supported_formats(),
            "status": "Available" if vendor == "ouster" else "In development"
        }

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Generic LiDAR SDK Tester")
    parser.add_argument("--input", "-i", help="Input LiDAR file path")
    parser.add_argument("--output", "-o", default="output.las", help="Output LAS file path")
    parser.add_argument("--vendor", "-v", help="Specify vendor (auto-detect if not provided)")
    parser.add_argument("--max-scans", type=int, default=10, help="Maximum scans to process")
    parser.add_argument("--list-vendors", action="store_true", help="List supported vendors")
    parser.add_argument("--info", action="store_true", help="Show file information only")
    
    args = parser.parse_args()
    
    tester = GenericLidarTester()
    
    if args.list_vendors:
        print("Supported vendors:")
        for vendor in tester.list_supported_vendors():
            info = tester.get_vendor_info(vendor)
            print(f"  {vendor}: {info['supported_formats']} ({info['status']})")
        return 0
    
    if args.input and not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return 1
    
    if args.info:
        vendor = args.vendor or tester.detect_vendor(args.input)
        if vendor:
            sdk = tester.sdks[vendor]
            info = sdk.get_file_info(args.input)
            print(json.dumps(info, indent=2))
        return 0
    
    # Test conversion
    success = tester.test_conversion(
        args.input, 
        args.output, 
        vendor=args.vendor,
        max_scans=args.max_scans
    )
    
    if success:
        logger.info("✅ Conversion test PASSED")
        return 0
    else:
        logger.error("❌ Conversion test FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
