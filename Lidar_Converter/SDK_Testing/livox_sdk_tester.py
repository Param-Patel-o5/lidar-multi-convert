#!/usr/bin/env python3
"""
Livox LiDAR SDK Testing Script
Converts Livox data files (.lvx, .bag, .csv) to LAS/LAZ format

Livox sensors (Avia, Horizon, Tele-15, Mid-40/70) use custom data formats.
This script provides manual parsing capabilities for Livox data.

Supported formats:
- .lvx (Livox custom format)
- .csv (exported point cloud data)
- .bag (ROS bag files - requires rosbag)
"""

import os
import sys
import struct
import numpy as np
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import laspy
    LASPY_AVAILABLE = True
except ImportError:
    LASPY_AVAILABLE = False
    logger.warning("laspy not available. Install with: pip install laspy")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("pandas not available. Install with: pip install pandas")


class LivoxLVXParser:
    """
    Parser for Livox .lvx files (Livox custom format).
    
    LVX file structure:
    - Header (24 bytes): Magic bytes, version, frame duration
    - Device info blocks
    - Frame data blocks with point cloud data
    """
    
    # LVX file constants
    LVX_MAGIC = b'livox_tech'
    LVX_HEADER_SIZE = 24
    
    def __init__(self):
        self.points = []
        self.device_info = {}
    
    def parse_lvx_header(self, data: bytes) -> Dict[str, Any]:
        """Parse LVX file header"""
        if len(data) < self.LVX_HEADER_SIZE:
            raise ValueError("Invalid LVX header size")
        
        # Check magic bytes
        magic = data[:10]
        if magic != self.LVX_MAGIC:
            raise ValueError(f"Invalid LVX magic bytes: {magic}")
        
        # Parse header fields
        version_major = data[10]
        version_minor = data[11]
        magic_code = struct.unpack('<I', data[12:16])[0]
        
        header = {
            "version": f"{version_major}.{version_minor}",
            "magic_code": magic_code,
            "valid": True
        }
        
        logger.info(f"LVX Header: Version {header['version']}, Magic Code: {magic_code}")
        return header
    
    def parse_lvx_file(self, lvx_file: str, max_frames: int = 1000) -> bool:
        """Parse Livox .lvx file and extract point cloud data"""
        logger.info(f"Parsing Livox LVX file: {lvx_file}")
        
        try:
            with open(lvx_file, 'rb') as f:
                # Read and parse header
                header_data = f.read(self.LVX_HEADER_SIZE)
                header = self.parse_lvx_header(header_data)
                
                if not header["valid"]:
                    logger.error("Invalid LVX header")
                    return False
                
                # Read device info section
                device_count = struct.unpack('<B', f.read(1))[0]
                logger.info(f"Number of devices: {device_count}")
                
                # Skip device info blocks (variable size)
                # For simplicity, we'll read the rest of the file as frame data
                
                frame_count = 0
                while frame_count < max_frames:
                    # Read frame header (24 bytes)
                    frame_header = f.read(24)
                    if len(frame_header) < 24:
                        break  # End of file
                    
                    # Parse frame header
                    current_offset = struct.unpack('<Q', frame_header[0:8])[0]
                    next_offset = struct.unpack('<Q', frame_header[8:16])[0]
                    frame_index = struct.unpack('<Q', frame_header[16:24])[0]
                    
                    # Calculate frame data size
                    frame_size = next_offset - current_offset - 24
                    
                    if frame_size <= 0 or frame_size > 10000000:  # Sanity check
                        break
                    
                    # Read frame data
                    frame_data = f.read(frame_size)
                    if len(frame_data) < frame_size:
                        break
                    
                    # Parse points from frame data
                    # Livox point format: x, y, z (float32), intensity (uint8), tag (uint8)
                    point_size = 14  # 3*4 + 1 + 1 bytes
                    num_points = len(frame_data) // point_size
                    
                    for i in range(num_points):
                        offset = i * point_size
                        if offset + point_size > len(frame_data):
                            break
                        
                        # Extract point data
                        x = struct.unpack('<f', frame_data[offset:offset+4])[0]
                        y = struct.unpack('<f', frame_data[offset+4:offset+8])[0]
                        z = struct.unpack('<f', frame_data[offset+8:offset+12])[0]
                        intensity = frame_data[offset+12]
                        tag = frame_data[offset+13]
                        
                        # Filter valid points
                        if abs(x) < 200 and abs(y) < 200 and abs(z) < 200:
                            self.points.append((x, y, z, intensity, tag))
                    
                    frame_count += 1
                    
                    if frame_count % 100 == 0:
                        logger.info(f"Processed {frame_count} frames, {len(self.points)} points")
                
                logger.info(f"Total frames processed: {frame_count}")
                logger.info(f"Total points extracted: {len(self.points)}")
                
                return len(self.points) > 0
                
        except Exception as e:
            logger.error(f"Error parsing LVX file: {e}")
            import traceback
            traceback.print_exc()
            return False


class LivoxCSVParser:
    """Parser for Livox CSV point cloud files"""
    
    def __init__(self):
        self.points = []
    
    def parse_csv_file(self, csv_file: str, max_points: int = 1000000) -> bool:
        """Parse Livox CSV file"""
        logger.info(f"Parsing Livox CSV file: {csv_file}")
        
        if not PANDAS_AVAILABLE:
            logger.error("pandas not available. Install with: pip install pandas")
            return False
        
        try:
            # Read CSV file
            # Expected columns: x, y, z, intensity, (optional: reflectivity, tag, etc.)
            df = pd.read_csv(csv_file, nrows=max_points)
            
            logger.info(f"CSV columns: {list(df.columns)}")
            logger.info(f"Loaded {len(df)} points from CSV")
            
            # Extract point data
            if 'x' in df.columns and 'y' in df.columns and 'z' in df.columns:
                x = df['x'].values
                y = df['y'].values
                z = df['z'].values
                
                # Get intensity if available
                if 'intensity' in df.columns:
                    intensity = df['intensity'].values
                elif 'reflectivity' in df.columns:
                    intensity = df['reflectivity'].values
                else:
                    intensity = np.ones(len(df)) * 128  # Default intensity
                
                # Get tag if available
                if 'tag' in df.columns:
                    tag = df['tag'].values
                else:
                    tag = np.zeros(len(df))
                
                # Combine into points list
                for i in range(len(df)):
                    self.points.append((x[i], y[i], z[i], int(intensity[i]), int(tag[i])))
                
                logger.info(f"Extracted {len(self.points)} points from CSV")
                return True
            else:
                logger.error("CSV file missing required columns (x, y, z)")
                return False
                
        except Exception as e:
            logger.error(f"Error parsing CSV file: {e}")
            import traceback
            traceback.print_exc()
            return False


class LivoxLASConverter:
    """Convert Livox point cloud to LAS format"""
    
    def __init__(self, points: List[Tuple[float, float, float, int, int]]):
        self.points = np.array(points)
    
    def convert_to_las(self, output_file: str) -> bool:
        """Convert points to LAS format"""
        if not LASPY_AVAILABLE:
            logger.error("laspy not available. Install with: pip install laspy")
            return False
        
        try:
            logger.info(f"Converting {len(self.points)} points to LAS format...")
            
            # Create LAS header
            header = laspy.LasHeader(point_format=1, version="1.2")
            header.x_scale = 0.001  # 1mm precision
            header.y_scale = 0.001
            header.z_scale = 0.001
            
            # Set offsets to center the point cloud
            header.x_offset = float(self.points[:, 0].mean())
            header.y_offset = float(self.points[:, 1].mean())
            header.z_offset = float(self.points[:, 2].mean())
            
            # Create LAS data
            las = laspy.LasData(header)
            
            # Set point data
            las.x = self.points[:, 0]
            las.y = self.points[:, 1]
            las.z = self.points[:, 2]
            las.intensity = self.points[:, 3].astype(np.uint16)
            
            # Write LAS file
            las.write(output_file)
            
            logger.info(f"LAS file created: {output_file}")
            logger.info(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating LAS file: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_point_cloud_stats(self) -> Dict[str, Any]:
        """Get statistics about the point cloud"""
        if len(self.points) == 0:
            return {"error": "No points available"}
        
        return {
            "total_points": len(self.points),
            "x_range": [float(self.points[:, 0].min()), float(self.points[:, 0].max())],
            "y_range": [float(self.points[:, 1].min()), float(self.points[:, 1].max())],
            "z_range": [float(self.points[:, 2].min()), float(self.points[:, 2].max())],
            "intensity_range": [int(self.points[:, 3].min()), int(self.points[:, 3].max())],
            "tag_range": [int(self.points[:, 4].min()), int(self.points[:, 4].max())]
        }


def find_livox_files(data_dir: str = "sample_data/livox") -> Dict[str, List[str]]:
    """Find Livox data files in the data directory"""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.error(f"Data directory '{data_dir}' not found!")
        return {}
    
    lvx_files = list(data_path.glob("*.lvx")) + list(data_path.glob("*.lvx2"))
    csv_files = list(data_path.glob("*.csv"))
    bag_files = list(data_path.glob("*.bag"))
    pcap_files = list(data_path.glob("*.pcap"))
    
    files = {
        "lvx": [str(f) for f in lvx_files],
        "csv": [str(f) for f in csv_files],
        "bag": [str(f) for f in bag_files],
        "pcap": [str(f) for f in pcap_files]
    }
    
    logger.info(f"Found {len(lvx_files)} LVX files, {len(csv_files)} CSV files, {len(bag_files)} BAG files, {len(pcap_files)} PCAP files")
    
    return files


def test_livox_conversion(input_file: str, output_file: str, file_type: str = "auto", max_data: int = 1000) -> bool:
    """Test Livox data to LAS conversion"""
    logger.info(f"Testing Livox conversion: {input_file} -> {output_file}")
    
    # Auto-detect file type
    if file_type == "auto":
        ext = Path(input_file).suffix.lower()
        if ext in [".lvx", ".lvx2"]:
            file_type = "lvx"
        elif ext == ".csv":
            file_type = "csv"
        elif ext == ".bag":
            file_type = "bag"
        elif ext == ".pcap":
            file_type = "pcap"
        else:
            logger.error(f"Unknown file type: {ext}")
            return False
    
    # Parse file based on type
    points = []
    
    if file_type == "lvx":
        parser = LivoxLVXParser()
        if not parser.parse_lvx_file(input_file, max_frames=max_data):
            logger.error("Failed to parse LVX file")
            return False
        points = parser.points
        
    elif file_type == "csv":
        parser = LivoxCSVParser()
        if not parser.parse_csv_file(input_file, max_points=max_data):
            logger.error("Failed to parse CSV file")
            return False
        points = parser.points
        
    elif file_type == "bag":
        logger.error("ROS bag file support not yet implemented")
        logger.info("Please convert BAG files to CSV or LVX format first")
        return False
    
    elif file_type == "pcap":
        logger.error("Livox PCAP file support not yet implemented")
        logger.info("Livox PCAP files require special parsing - use Livox Viewer to convert to LVX or CSV first")
        return False
    
    if not points:
        logger.error("No points extracted from file")
        return False
    
    # Convert to LAS
    converter = LivoxLASConverter(points)
    
    # Get statistics
    stats = converter.get_point_cloud_stats()
    logger.info(f"Point cloud statistics: {stats}")
    
    # Convert to LAS
    if not converter.convert_to_las(output_file):
        logger.error("Failed to convert to LAS")
        return False
    
    # Validate conversion
    try:
        las = laspy.read(output_file)
        logger.info(f"Validation: LAS file contains {len(las.points)} points")
        return True
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Livox LiDAR SDK Tester")
    parser.add_argument("--data-dir", default="sample_data/livox", help="Directory containing Livox data files")
    parser.add_argument("--input", help="Specific input file to process")
    parser.add_argument("--output", default="livox_output.las", help="Output LAS file path")
    parser.add_argument("--type", default="auto", choices=["auto", "lvx", "csv", "bag", "pcap"], help="Input file type")
    parser.add_argument("--max-data", type=int, default=1000, help="Maximum frames/points to process")
    parser.add_argument("--list-files", action="store_true", help="List available data files")
    
    args = parser.parse_args()
    
    logger.info("Livox LiDAR SDK Tester")
    logger.info("=" * 50)
    
    # Find Livox files
    files = find_livox_files(args.data_dir)
    
    if args.list_files:
        logger.info("Available Livox files:")
        for file_type, file_list in files.items():
            if file_list:
                logger.info(f"\n  {file_type.upper()} files:")
                for i, file in enumerate(file_list, 1):
                    logger.info(f"    {i}. {Path(file).name}")
        return 0
    
    # Determine input file
    if args.input:
        test_file = args.input
    else:
        # Try to find any available file
        test_file = None
        for file_type in ["lvx", "csv", "pcap", "bag"]:
            if files.get(file_type):
                test_file = files[file_type][0]
                break
        
        if not test_file:
            logger.error("No Livox data files found!")
            logger.info("Please provide a file with --input or place files in the data directory")
            return 1
    
    logger.info(f"Testing with file: {Path(test_file).name}")
    
    success = test_livox_conversion(test_file, args.output, args.type, args.max_data)
    
    if success:
        logger.info("✅ Livox conversion test PASSED")
        logger.info(f"📁 Output file: {args.output}")
        logger.info("🔍 You can now open this file in CloudCompare")
        return 0
    else:
        logger.error("❌ Livox conversion test FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())