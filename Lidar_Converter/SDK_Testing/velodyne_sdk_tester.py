#!/usr/bin/env python3
"""
Velodyne LiDAR SDK Testing Script
Converts Velodyne PCAP files to LAS/LAZ format using multiple approaches
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
    from scapy.all import rdpcap
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    logger.warning("scapy not available. Install with: pip install scapy")

class VelodynePacketParser:
    """Parser for Velodyne LiDAR packets"""
    
    def __init__(self, model: str = "VLP16"):
        self.model = model
        self.setup_model_parameters()
    
    def setup_model_parameters(self):
        """Setup model-specific parameters"""
        if self.model == "VLP16":
            self.lasers_per_packet = 32
            self.blocks_per_packet = 12
            self.points_per_block = 32
            self.packet_size = 1206
            self.laser_angles = np.array([
                -15, 1, -13, 3, -11, 5, -9, 7, -7, 9, -5, 11, -3, 13, -1, 15,
                1, -15, 3, -13, 5, -11, 7, -9, 9, -7, 11, -5, 13, -3, 15, -1
            ])
        elif self.model == "HDL32E":
            self.lasers_per_packet = 32
            self.blocks_per_packet = 12
            self.points_per_block = 32
            self.packet_size = 1206
            self.laser_angles = np.array([
                -30.67, -9.33, -29.33, -8.0, -28.0, -6.67, -26.67, -5.33,
                -25.33, -4.0, -24.0, -2.67, -22.67, -1.33, -21.33, 0.0,
                0.0, 1.33, -1.33, 2.67, -2.67, 4.0, -4.0, 5.33,
                -5.33, 6.67, -6.67, 8.0, -8.0, 9.33, -9.33, 10.67
            ])
        else:
            # Default to VLP16 parameters
            self.lasers_per_packet = 32
            self.blocks_per_packet = 12
            self.points_per_block = 32
            self.packet_size = 1206
            self.laser_angles = np.array([
                -15, 1, -13, 3, -11, 5, -9, 7, -7, 9, -5, 11, -3, 13, -1, 15,
                1, -15, 3, -13, 5, -11, 7, -9, 9, -7, 11, -5, 13, -3, 15, -1
            ])
    
    def parse_packet(self, packet_data: bytes) -> List[Tuple[float, float, float, int, float]]:
        """Parse a single Velodyne packet and return points as (x, y, z, intensity, timestamp)"""
        if len(packet_data) < self.packet_size:
            return []
        
        points = []
        
        # Extract timestamp (first 4 bytes)
        timestamp = struct.unpack('<I', packet_data[:4])[0]
        
        # Parse data blocks
        for block_idx in range(self.blocks_per_packet):
            block_start = 42 + block_idx * 100  # Skip header, 100 bytes per block
            
            if block_start + 100 > len(packet_data):
                break
            
            # Extract azimuth for this block
            azimuth_start = block_start + 2
            azimuth = struct.unpack('<H', packet_data[azimuth_start:azimuth_start+2])[0] / 100.0
            
            # Parse laser returns
            for laser_idx in range(self.lasers_per_packet):
                laser_start = block_start + 4 + laser_idx * 3
                
                if laser_start + 3 > len(packet_data):
                    break
                
                # Extract distance and intensity
                distance_raw = struct.unpack('<H', packet_data[laser_start:laser_start+2])[0]
                intensity = packet_data[laser_start + 2]
                
                # Convert distance (2mm resolution)
                distance = (distance_raw & 0xFFF) * 0.002
                
                # Filter valid points
                if 0.1 < distance < 100.0:  # Valid range
                    # Get laser angle
                    laser_angle = self.laser_angles[laser_idx]
                    
                    # Convert to Cartesian coordinates
                    x = distance * np.cos(np.radians(laser_angle)) * np.cos(np.radians(azimuth))
                    y = distance * np.cos(np.radians(laser_angle)) * np.sin(np.radians(azimuth))
                    z = distance * np.sin(np.radians(laser_angle))
                    
                    points.append((x, y, z, intensity, timestamp))
        
        return points

class VelodynePCAPProcessor:
    """Process Velodyne PCAP files"""
    
    def __init__(self, model: str = "VLP16"):
        self.parser = VelodynePacketParser(model)
        self.points = []
        self.timestamps = []
    
    def process_pcap_file(self, pcap_file: str, max_packets: int = 1000) -> bool:
        """Process a PCAP file and extract point cloud data"""
        logger.info(f"Processing Velodyne PCAP file: {pcap_file}")
        
        if not SCAPY_AVAILABLE:
            logger.error("scapy not available. Install with: pip install scapy")
            return False
        
        try:
            # Read PCAP file using scapy
            packets = rdpcap(pcap_file)
            logger.info(f"Loaded {len(packets)} packets from PCAP file")
            
            packet_count = 0
            for packet in packets:
                packet_count += 1
                
                # Extract raw packet data
                if hasattr(packet, 'load'):
                    packet_data = bytes(packet.load)
                else:
                    continue
                
                # Parse packet
                points = self.parser.parse_packet(packet_data)
                self.points.extend(points)
                
                if packet_count % 100 == 0:
                    logger.info(f"Processed {packet_count} packets, {len(self.points)} points")
                
                if packet_count >= max_packets:
                    logger.info(f"Reached maximum packet limit: {max_packets}")
                    break
            
            logger.info(f"Total packets processed: {packet_count}")
            logger.info(f"Total points extracted: {len(self.points)}")
            
            return len(self.points) > 0
            
        except Exception as e:
            logger.error(f"Error processing PCAP file: {e}")
            return False
    
    def get_point_cloud_stats(self) -> Dict[str, Any]:
        """Get statistics about the extracted point cloud"""
        if not self.points:
            return {"error": "No points extracted"}
        
        points_array = np.array(self.points)
        
        return {
            "total_points": len(self.points),
            "x_range": [float(points_array[:, 0].min()), float(points_array[:, 0].max())],
            "y_range": [float(points_array[:, 1].min()), float(points_array[:, 1].max())],
            "z_range": [float(points_array[:, 2].min()), float(points_array[:, 2].max())],
            "intensity_range": [int(points_array[:, 3].min()), int(points_array[:, 3].max())],
            "timestamp_range": [float(points_array[:, 4].min()), float(points_array[:, 4].max())]
        }

class VelodyneLASConverter:
    """Convert Velodyne point cloud to LAS format"""
    
    def __init__(self, points: List[Tuple[float, float, float, int, float]]):
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
            return False

def find_velodyne_files(data_dir: str = "sample_data/velodyne") -> List[str]:
    """Find Velodyne PCAP files in the data directory"""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.error(f"Data directory '{data_dir}' not found!")
        return []
    
    pcap_files = list(data_path.glob("*.pcap"))
    logger.info(f"Found {len(pcap_files)} PCAP files in {data_dir}")
    
    return [str(f) for f in pcap_files]

def test_velodyne_conversion(pcap_file: str, output_file: str, model: str = "VLP16", max_packets: int = 1000) -> bool:
    """Test Velodyne PCAP to LAS conversion"""
    logger.info(f"Testing Velodyne conversion: {pcap_file} -> {output_file}")
    
    # Process PCAP file
    processor = VelodynePCAPProcessor(model)
    if not processor.process_pcap_file(pcap_file, max_packets):
        logger.error("Failed to process PCAP file")
        return False
    
    # Get statistics
    stats = processor.get_point_cloud_stats()
    logger.info(f"Point cloud statistics: {stats}")
    
    # Convert to LAS
    converter = VelodyneLASConverter(processor.points)
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
    parser = argparse.ArgumentParser(description="Velodyne LiDAR SDK Tester")
    parser.add_argument("--data-dir", default="sample_data/velodyne", help="Directory containing Velodyne PCAP files")
    parser.add_argument("--output", default="velodyne_output.las", help="Output LAS file path")
    parser.add_argument("--model", default="VLP16", choices=["VLP16", "HDL32E"], help="Velodyne model")
    parser.add_argument("--max-packets", type=int, default=1000, help="Maximum packets to process")
    parser.add_argument("--list-files", action="store_true", help="List available PCAP files")
    
    args = parser.parse_args()
    
    logger.info("Velodyne LiDAR SDK Tester")
    logger.info("=" * 50)
    
    # Find Velodyne files
    pcap_files = find_velodyne_files(args.data_dir)
    
    if not pcap_files:
        logger.error("No Velodyne PCAP files found!")
        return 1
    
    if args.list_files:
        logger.info("Available PCAP files:")
        for i, file in enumerate(pcap_files, 1):
            logger.info(f"  {i}. {Path(file).name}")
        return 0
    
    # Test conversion with first file
    test_file = pcap_files[0]
    logger.info(f"Testing with file: {Path(test_file).name}")
    
    success = test_velodyne_conversion(test_file, args.output, args.model, args.max_packets)
    
    if success:
        logger.info("✅ Velodyne conversion test PASSED")
        logger.info(f"📁 Output file: {args.output}")
        logger.info("🔍 You can now open this file in CloudCompare")
        return 0
    else:
        logger.error("❌ Velodyne conversion test FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
