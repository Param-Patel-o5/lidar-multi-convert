#!/usr/bin/env python3
"""
Hesai LiDAR SDK wrapper for conversion to standardized formats.

This module provides a unified interface to Hesai LiDAR data processing,
abstracting Hesai-specific implementation details. It follows the same
architectural patterns as the Ouster and Velodyne wrappers.

The wrapper handles:
- SDK installation validation (with dpkt fallback)
- PCAP to LAS/LAZ/PCD/BIN/CSV conversion
- Hesai packet structure parsing (based on official documentation)
- Error handling and logging
- Performance monitoring
- Vendor-specific metadata extraction

Supported Hesai Models:
- PandarXT-32 (32 channels) - Tested ✓
- PandarXT-16 (16 channels)
- Pandar64 (64 channels)
- Pandar40P (40 channels)
- PandarQT (64 channels)

Packet Structure (from official Hesai documentation):
- Pre-header: 6 bytes (0xEEFF + protocol version)
- Header: 6 bytes (channel num, block num, distance unit, flags)
- Body: Variable bytes (point cloud data blocks)
- Tail: 34 bytes (timestamp, factory info)

The wrapper dynamically reads packet structure from headers for flexibility
across different Hesai models.

Usage:
    wrapper = HesaiWrapper()
    if wrapper.sdk_available:
        # Convert to LAS
        result = wrapper.convert_to_las("input.pcap", "output.las")
        
        # Convert to other formats
        result = wrapper.convert("input.pcap", "laz", "output.laz")
        result = wrapper.convert("input.pcap", "pcd", "output.pcd")
        result = wrapper.convert("input.pcap", "bin", "output.bin")
        result = wrapper.convert("input.pcap", "csv", "output.csv")

Note:
    The wrapper uses dpkt for PCAP parsing as a fallback when the official
    Hesai SDK is not compiled. For best accuracy, compile the official SDK
    located in Lidar_Converter/Wrappers/hesai_sdk/
"""

import os
import sys
import subprocess
import time
import struct
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging

# Hesai SDK imports (with graceful fallback)
HESAI_SDK_AVAILABLE = False
hesai_sdk = None

try:
    # Try to import Hesai SDK if available
    # Note: The C++ SDK would need Python bindings
    import hesai_sdk
    HESAI_SDK_AVAILABLE = True
except ImportError:
    # No Hesai SDK available - will use dpkt fallback
    HESAI_SDK_AVAILABLE = False
    hesai_sdk = None

# PCAP parsing dependency (required for Hesai)
try:
    import dpkt
    DPKT_AVAILABLE = True
except ImportError:
    DPKT_AVAILABLE = False

# LAS file handling
try:
    import laspy
    LASPY_AVAILABLE = True
except ImportError:
    LASPY_AVAILABLE = False

import numpy as np

from .base_wrapper import BaseVendorWrapper

logger = logging.getLogger(__name__)


class HesaiWrapper(BaseVendorWrapper):
    """
    Hesai LiDAR wrapper implementing BaseVendorWrapper interface.
    
    This wrapper provides conversion capabilities for Hesai sensors:
    - PandarXT-32 (32 channels)
    - PandarXT-16 (16 channels)
    - Pandar64 (64 channels)
    - Pandar40P (40 channels)
    - PandarQT (64 channels)
    
    The wrapper uses dpkt for PCAP parsing and implements Hesai packet
    structure parsing manually, as the C++ SDK requires compilation.
    
    Example:
        wrapper = HesaiWrapper()
        if wrapper.sdk_available:
            result = wrapper.convert_to_las(
                input_path="data.pcap",
                output_path="output.las",
                sensor_model="PandarXT-32"
            )
            if result["success"]:
                print(f"Converted {result['points_converted']} points")
    """
    
    # Supported Hesai sensor models
    SUPPORTED_MODELS = [
        "PandarXT",
        "PandarXT-32",
        "PandarXT-16",
        "Pandar64",
        "Pandar40P",
        "PandarQT",
    ]
    
    # Supported input/output formats
    SUPPORTED_INPUT_FORMATS = [".pcap"]
    SUPPORTED_OUTPUT_FORMATS = [".las", ".laz", ".pcd", ".bin", ".csv"]
    
    # Hesai packet constants
    HESAI_MAGIC_BYTES = b"\xEE\xFF"  # Magic bytes at start of Hesai packets
    HESAI_UDP_PORT = 2368  # Standard Hesai UDP port (shared with Velodyne)
    HESAI_PACKET_SIZE_RANGE = (1000, 1300)  # Typical Hesai packet size range
    
    def __init__(self, sdk_path: Optional[str] = None, raise_on_missing: bool = False):
        """
        Initialize Hesai wrapper with SDK validation.
        
        Args:
            sdk_path: Optional path to custom Hesai SDK installation
            raise_on_missing: If True, raise exception if SDK not found.
                            If False (default), log warning and use dpkt fallback.
        
        Raises:
            RuntimeError: If raise_on_missing=True and dpkt is not available
        """
        super().__init__()
        
        # Check for custom SDK path from environment variable
        if sdk_path is None:
            sdk_path = os.environ.get("HESAI_SDK_PATH")
            if sdk_path is None:
                # Default to the SDK directory in the project
                sdk_path = str(Path(__file__).parent / "hesai_sdk")
        
        self.sdk_path = sdk_path
        self.raise_on_missing = raise_on_missing
        
        # Validate SDK installation
        validation_result = self.validate_sdk_installation()
        
        if not validation_result.get("available", False):
            error_msg = validation_result.get("error", "Hesai SDK not found")
            if raise_on_missing:
                raise RuntimeError(f"Hesai processing capability is required but not available: {error_msg}")
            else:
                self.logger.warning(f"Hesai SDK not available: {error_msg}")
                self.sdk_available = False
        else:
            self.sdk_available = True
            self.sdk_version = validation_result.get("version", "dpkt-fallback")
            self.logger.info(f"Hesai processing validated - Method: {validation_result.get('method', 'unknown')}")
    
    def get_vendor_name(self) -> str:
        """Return vendor identifier."""
        return "hesai"
    
    def validate_sdk_installation(self) -> Dict[str, Any]:
        """
        Validate Hesai SDK installation and detect available tools.
        
        Checks for (in priority order):
        1. Hesai C++ SDK (compiled library)
        2. dpkt library for generic PCAP parsing (fallback)
        
        Returns:
            dict: Validation result with availability and version info
        """
        result = {
            "available": False,
            "version": None,
            "installation_path": None,
            "method": None,
            "message": "",
            "error": None
        }
        
        # Method 1: Check for compiled Hesai SDK
        if self.sdk_path:
            sdk_dir = Path(self.sdk_path)
            if sdk_dir.exists() and sdk_dir.is_dir():
                # Check for compiled library
                build_dir = sdk_dir / "build"
                lib_files = [
                    build_dir / "libPandarGeneralSDK.so",  # Linux
                    build_dir / "libPandarGeneralSDK.dylib",  # macOS
                    build_dir / "PandarGeneralSDK.dll",  # Windows
                ]
                
                for lib_file in lib_files:
                    if lib_file.exists():
                        result.update({
                            "available": True,
                            "version": "hesai_sdk",
                            "installation_path": str(sdk_dir),
                            "method": "hesai_sdk",
                            "message": f"Hesai SDK library found at {lib_file}"
                        })
                        return result
                
                # SDK directory exists but not built
                self.logger.debug(f"Hesai SDK directory found at {sdk_dir} but not built")
        
        # Method 2: Check dpkt for generic PCAP parsing (fallback)
        if DPKT_AVAILABLE:
            try:
                import importlib.metadata
                dpkt_version = importlib.metadata.version('dpkt')
            except (ImportError, importlib.metadata.PackageNotFoundError):
                try:
                    import pkg_resources
                    dpkt_version = pkg_resources.get_distribution('dpkt').version
                except:
                    dpkt_version = "installed"
            
            result.update({
                "available": True,
                "version": f"dpkt-{dpkt_version}",
                "method": "dpkt_fallback",
                "message": f"Using dpkt {dpkt_version} for Hesai PCAP parsing"
            })
            return result
        
        # No processing capability found
        result.update({
            "available": False,
            "error": "No Hesai processing capability found. Install dpkt with: pip install dpkt",
            "message": "Hesai processing is not available"
        })
        return result
    
    def get_vendor_info(self) -> Dict[str, Any]:
        """
        Get vendor capabilities and information.
        
        Returns:
            dict: Vendor information including supported models, formats, and SDK status
        """
        # Determine status based on SDK availability
        if self.sdk_available:
            status = "available"
        else:
            status = "unavailable"
        
        return {
            "vendor_name": self.get_vendor_name(),
            "sdk_available": self.sdk_available,
            "sdk_version": self.sdk_version,
            "status": status,
            "supported_models": self.SUPPORTED_MODELS,
            "supported_input_formats": self.SUPPORTED_INPUT_FORMATS,
            "supported_output_formats": self.SUPPORTED_OUTPUT_FORMATS,
            "capabilities": {
                "pcap_parsing": self.sdk_available,
                "las_conversion": self.sdk_available and LASPY_AVAILABLE,
                "laz_conversion": self.sdk_available and LASPY_AVAILABLE,
                "pcd_conversion": self.sdk_available,
                "bin_conversion": self.sdk_available,
                "csv_conversion": self.sdk_available,
            }
        }
    
    def validate_conversion(self, input_path: str, output_path: str) -> bool:
        """
        Validate that a conversion output is valid.
        
        Args:
            input_path: Path to input file
            output_path: Path to output file
            
        Returns:
            bool: True if output file is valid, False otherwise
        """
        try:
            # Check if output file exists
            output_file = Path(output_path)
            if not output_file.exists():
                self.logger.error(f"Output file does not exist: {output_path}")
                return False
            
            # Check if output file is not empty
            if output_file.stat().st_size == 0:
                self.logger.error(f"Output file is empty: {output_path}")
                return False
            
            # Format-specific validation
            suffix = output_file.suffix.lower()
            
            if suffix in [".las", ".laz"]:
                # Validate LAS/LAZ file
                if not LASPY_AVAILABLE:
                    self.logger.warning("laspy not available for LAS validation")
                    return True  # Assume valid if we can't validate
                
                try:
                    las_file = laspy.read(output_path)
                    point_count = len(las_file.points)
                    if point_count == 0:
                        self.logger.error(f"LAS file contains no points: {output_path}")
                        return False
                    self.logger.info(f"LAS validation passed: {point_count} points")
                    return True
                except Exception as e:
                    self.logger.error(f"LAS validation failed: {e}")
                    return False
            
            elif suffix == ".pcd":
                # Basic PCD validation - check for header
                try:
                    with open(output_path, 'r') as f:
                        first_line = f.readline()
                        if not first_line.startswith("# .PCD"):
                            self.logger.error(f"Invalid PCD header: {output_path}")
                            return False
                    return True
                except Exception as e:
                    self.logger.error(f"PCD validation failed: {e}")
                    return False
            
            elif suffix == ".bin":
                # Basic BIN validation - check file size is multiple of 16 bytes
                file_size = output_file.stat().st_size
                if file_size % 16 != 0:
                    self.logger.error(f"Invalid BIN file size (not multiple of 16): {output_path}")
                    return False
                return True
            
            elif suffix == ".csv":
                # Basic CSV validation - check for header
                try:
                    with open(output_path, 'r') as f:
                        first_line = f.readline()
                        if 'x' not in first_line.lower() or 'y' not in first_line.lower():
                            self.logger.error(f"Invalid CSV header: {output_path}")
                            return False
                    return True
                except Exception as e:
                    self.logger.error(f"CSV validation failed: {e}")
                    return False
            
            # Unknown format - assume valid if file exists and not empty
            return True
            
        except Exception as e:
            self.logger.exception(f"Validation error: {e}")
            return False
    
    def convert_to_las(
        self,
        input_path: str,
        output_path: str,
        sensor_model: Optional[str] = None,
        calibration_file: Optional[str] = None,
        preserve_intensity: bool = True,
        max_scans: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Convert Hesai PCAP file to LAS format.
        
        This method implements the Hesai-specific conversion logic while
        maintaining the unified interface defined by BaseVendorWrapper.
        
        Args:
            input_path: Path to input .pcap file
            output_path: Path where .las file will be written
            sensor_model: Optional sensor model (e.g., "PandarXT-32")
            calibration_file: Optional calibration file (not typically needed for Hesai)
            preserve_intensity: Whether to preserve intensity values
            max_scans: Optional limit on number of scans to process
            **kwargs: Additional parameters
                
        Returns:
            dict: Conversion result with success status and details
        """
        start_time = time.time()
        result = {
            "success": False,
            "message": "",
            "output_file": None,
            "conversion_time": 0.0,
            "points_converted": 0,
            "sdk_version_used": self.sdk_version,
            "error": None
        }
        
        # Validate SDK availability
        if not self.sdk_available:
            result["error"] = "Hesai processing capability is not available"
            result["message"] = "Cannot convert: dpkt library required for Hesai PCAP parsing"
            self.logger.error(result["error"])
            return result
        
        # Validate input file
        input_validation = self._validate_file_path(input_path, must_exist=True)
        if not input_validation["valid"]:
            result["error"] = input_validation["error"]
            result["message"] = f"Input validation failed: {input_validation['error']}"
            self.logger.error(result["error"])
            return result
        
        # Validate output directory
        output_validation = self._validate_output_directory(output_path)
        if not output_validation["valid"]:
            result["error"] = output_validation["error"]
            result["message"] = f"Output validation failed: {output_validation['error']}"
            self.logger.error(result["error"])
            return result
        
        # Validate file extensions
        input_path_obj = Path(input_path)
        output_path_obj = Path(output_path)
        
        if input_path_obj.suffix.lower() != ".pcap":
            result["error"] = f"Unsupported input format: {input_path_obj.suffix}"
            result["message"] = "Hesai wrapper only supports .pcap input files"
            self.logger.error(result["error"])
            return result
        
        if output_path_obj.suffix.lower() != ".las":
            result["error"] = f"Unsupported output format: {output_path_obj.suffix}"
            result["message"] = "Use convert() method for formats other than .las"
            self.logger.error(result["error"])
            return result
        
        try:
            self.logger.info(f"Starting Hesai conversion: {input_path} -> {output_path}")
            
            # Check laspy availability
            if not LASPY_AVAILABLE:
                result["error"] = "laspy not available - install with: pip install laspy"
                result["message"] = "Cannot create LAS file: laspy package required"
                self.logger.error(result["error"])
                return result
            
            # Extract points using _process_pcap
            points = self._process_pcap(
                input_path,
                sensor_model,
                max_scans,
                **kwargs
            )
            
            if points is None or len(points) == 0:
                result["error"] = "Failed to extract points from PCAP"
                result["message"] = "Could not parse Hesai data"
                self.logger.error(result["error"])
                return result
            
            # Create LAS file from points array
            self.logger.debug(f"Writing LAS file: {output_path}")
            self._create_las_file(points, output_path, preserve_intensity)
            
            point_count = len(points)
            
            # Calculate conversion time
            conversion_time = time.time() - start_time
            
            result.update({
                "success": True,
                "message": f"Successfully converted {point_count:,} points",
                "points_converted": point_count,
                "output_file": str(output_path),
                "conversion_time": conversion_time,
                "sdk_version_used": self.sdk_version
            })
            
            self.logger.info(
                f"Conversion completed: {point_count:,} points in {conversion_time:.2f}s"
            )
        
        except Exception as e:
            conversion_time = time.time() - start_time
            result.update({
                "success": False,
                "error": str(e),
                "message": f"Conversion failed: {e}",
                "conversion_time": conversion_time
            })
            self.logger.exception(f"Exception during Hesai conversion: {e}")
        
        return result
    
    def _create_las_file(
        self,
        points: np.ndarray,
        output_path: str,
        preserve_intensity: bool
    ) -> None:
        """
        Create LAS file from point cloud data.
        
        Args:
            points: Point cloud array [x, y, z, intensity]
            output_path: Output file path
            preserve_intensity: Whether to include intensity data
        """
        # Create LAS header
        header = laspy.LasHeader(point_format=1, version="1.2")
        header.x_scale = 0.001  # 1mm precision
        header.y_scale = 0.001
        header.z_scale = 0.001
        header.x_offset = float(points[:, 0].mean())
        header.y_offset = float(points[:, 1].mean())
        header.z_offset = float(points[:, 2].mean())
        
        # Create LAS data
        las_file = laspy.LasData(header)
        las_file.x = points[:, 0]
        las_file.y = points[:, 1]
        las_file.z = points[:, 2]
        
        if preserve_intensity and points.shape[1] >= 4:
            las_file.intensity = points[:, 3].astype(np.uint16)
        else:
            las_file.intensity = np.zeros(len(points), dtype=np.uint16)
        
        # Write LAS file
        las_file.write(str(output_path))
        
        self.logger.debug(f"LAS file written: {output_path}")
    
    def _process_pcap(
        self,
        pcap_path: str,
        sensor_model: Optional[str] = None,
        max_scans: Optional[int] = None,
        **kwargs
    ) -> Optional[np.ndarray]:
        """
        Extract point cloud from Hesai PCAP file.
        
        This method parses Hesai UDP packets and extracts point cloud data.
        Uses dpkt for PCAP parsing and implements Hesai packet structure parsing.
        
        Args:
            pcap_path: Path to PCAP file
            sensor_model: Optional sensor model (e.g., "PandarXT-32")
            max_scans: Optional limit on number of scans to process
            **kwargs: Additional parameters
            
        Returns:
            Nx4 numpy array [x, y, z, intensity] or None if failed
        """
        if not DPKT_AVAILABLE:
            self.logger.error("dpkt not available - install with: pip install dpkt")
            return None
        
        try:
            # Auto-detect sensor model if not provided
            if not sensor_model:
                sensor_model = "PandarXT-32"  # Default
                self.logger.info(f"Using default sensor model: {sensor_model}")
            
            # Open PCAP file
            self.logger.debug(f"Opening PCAP: {pcap_path}")
            with open(pcap_path, 'rb') as f:
                pcap = dpkt.pcap.Reader(f)
                
                # Collect point cloud data
                all_points_list = []
                packet_count = 0
                valid_packet_count = 0
                
                self.logger.info(f"Processing Hesai packets (max_scans: {max_scans or 'unlimited'})...")
                
                for ts, buf in pcap:
                    packet_count += 1
                    
                    if packet_count % 1000 == 0:
                        self.logger.debug(f"Processing packet {packet_count}...")
                    
                    try:
                        # Parse Ethernet frame
                        eth = dpkt.ethernet.Ethernet(buf)
                        if eth.type != dpkt.ethernet.ETH_TYPE_IP:
                            continue
                        
                        # Parse IP packet
                        ip = eth.data
                        if ip.p != dpkt.ip.IP_PROTO_UDP:
                            continue
                        
                        # Parse UDP packet
                        udp = ip.data
                        if udp.dport != self.HESAI_UDP_PORT:
                            continue
                        
                        # Parse Hesai packet
                        points = self._parse_hesai_packet(
                            udp.data,
                            sensor_model
                        )
                        
                        if points is not None and len(points) > 0:
                            all_points_list.append(points)
                            valid_packet_count += 1
                        
                        # Check scan limit (approximate - each packet is part of a scan)
                        if max_scans and valid_packet_count >= max_scans:
                            self.logger.info(f"Reached max_scans limit: {valid_packet_count}")
                            break
                            
                    except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError, AttributeError) as e:
                        # Skip malformed packets
                        continue
                    except Exception as e:
                        self.logger.warning(f"Error processing packet {packet_count}: {e}")
                        continue
                
                if not all_points_list:
                    self.logger.error("No valid Hesai packets found in PCAP file")
                    return None
                
                # Combine all points
                self.logger.debug("Combining all point clouds...")
                all_points = np.vstack(all_points_list)
                point_count = len(all_points)
                
                self.logger.info(f"Extracted {point_count:,} points from {valid_packet_count} packets")
                return all_points
        
        except Exception as e:
            self.logger.exception(f"Error in PCAP processing: {e}")
            return None
    
    def _parse_hesai_packet(
        self,
        payload: bytes,
        sensor_model: str
    ) -> Optional[np.ndarray]:
        """
        Parse a single Hesai UDP packet and extract point data.
        
        Hesai packet structure (from official documentation):
        - Pre-header: 6 bytes (0xEEFF + protocol version + reserved)
        - Header: 6 bytes (channel num, block num, flags, etc.)
        - Body: Variable bytes (point cloud data blocks)
        - Tail: 34+ bytes (timestamp, factory info, etc.)
        
        For 861-byte packets:
        - Pre-header: bytes 0-5 (6 bytes)
        - Header: bytes 6-11 (6 bytes)  
        - Body: bytes 12-826 (815 bytes) - point data
        - Tail: bytes 827-860 (34 bytes)
        
        Args:
            payload: UDP payload bytes
            sensor_model: Sensor model identifier
            
        Returns:
            numpy array of points [x, y, z, intensity] or None if invalid
        """
        # Check minimum packet size
        if len(payload) < 100:  # Minimum reasonable size
            return None
        
        # Check magic bytes (Hesai pre-header: 0xEEFF)
        if len(payload) < 2 or payload[:2] != self.HESAI_MAGIC_BYTES:
            return None
        
        points = []
        
        try:
            # Parse Hesai packet structure
            # Pre-header: 6 bytes (0xEEFF + version)
            # Header: 6 bytes starting at offset 6
            
            # Read header information
            if len(payload) < 12:
                return None
            
            # Header at bytes 6-11 (read dynamically from packet)
            channel_num = payload[6]  # Number of channels (e.g., 0x80 = 128)
            block_num = payload[7]    # Number of blocks per packet (e.g., 0x02 = 2)
            dis_unit_code = payload[9] if len(payload) > 9 else 0x04  # Distance unit
            
            # Distance unit: 0x04 = 4mm, 0x05 = 5mm, etc.
            dis_unit = dis_unit_code / 1000.0  # Convert to meters multiplier
            
            # Body structure (flexible based on packet size)
            # Standard structure: Pre-header (6) + Header (6) + Body + Tail
            body_start = 12  # After pre-header and header
            
            # Tail size varies by packet type, but typically 34 bytes
            # For different models, this might vary
            # We'll estimate based on packet size
            if len(payload) > 1100:  # Large packets (e.g., 1173 bytes from documentation)
                tail_size = 34 + 17 + 32  # Tail + Functional Safety + Cyber Security
            else:  # Smaller packets (e.g., 861 bytes)
                tail_size = 34  # Just tail
            
            body_end = len(payload) - tail_size
            body_size = body_end - body_start
            
            if body_size < 50:  # Minimum reasonable body size
                return None
            
            # Parse data blocks
            # Block size calculation based on header info
            # If block_num is specified in header, use it
            if block_num > 0:
                # Calculate block size from body size and block count
                block_size = body_size // block_num
            else:
                # Fallback: typical Hesai block size is 100-130 bytes
                block_size = 100
            
            # Ensure block size is reasonable
            if block_size < 50 or block_size > 200:
                block_size = 100  # Fallback to typical size
            
            num_blocks = max(1, min(block_num if block_num > 0 else 10, body_size // block_size))
            
            for block_idx in range(num_blocks):
                block_offset = body_start + block_idx * block_size
                
                if block_offset + block_size > body_end:
                    break
                
                try:
                    # Parse azimuth (first 2 bytes of block, little endian)
                    # Azimuth range: 0-36000 (0.01 degree resolution)
                    if block_offset + 2 > len(payload):
                        break
                    
                    azimuth_raw = struct.unpack('<H', payload[block_offset:block_offset+2])[0]
                    
                    # Skip if azimuth is out of range
                    if azimuth_raw > 36000:
                        continue
                    
                    azimuth = azimuth_raw / 100.0  # Convert to degrees
                    
                    # Parse channel data
                    # Hesai typically has 32-128 channels depending on model
                    # Each channel: distance (2 bytes) + reflectivity (1 byte) = 3 bytes
                    channels_per_block = min(32, (block_size - 2) // 3)  # Subtract azimuth bytes
                    
                    for channel_idx in range(channels_per_block):
                        channel_offset = block_offset + 2 + channel_idx * 3
                        
                        if channel_offset + 3 > len(payload):
                            break
                        
                        # Parse distance (2 bytes, little endian) and reflectivity (1 byte)
                        distance_raw = struct.unpack('<H', payload[channel_offset:channel_offset+2])[0]
                        reflectivity = payload[channel_offset+2]
                        
                        # Convert distance using unit from header
                        # dis_unit_code 0x04 = 4mm, so distance_raw * 0.004 meters
                        distance = distance_raw * (dis_unit_code / 1000.0)
                        
                        # Skip invalid points (0 distance or beyond max range)
                        if distance <= 0.1 or distance > 200:  # Min 10cm, max 200m
                            continue
                        
                        # Skip low reflectivity (likely noise)
                        if reflectivity < 5:
                            continue
                        
                        # Calculate elevation angle for this channel
                        elevation = self._get_elevation_angle(channel_idx, sensor_model)
                        
                        # Convert polar to Cartesian coordinates
                        x, y, z = self._polar_to_cartesian(distance, azimuth, elevation)
                        
                        # Add point [x, y, z, intensity]
                        points.append([x, y, z, float(reflectivity)])
                
                except Exception as e:
                    # Skip problematic blocks
                    self.logger.debug(f"Error parsing block {block_idx}: {e}")
                    continue
        
        except Exception as e:
            self.logger.debug(f"Error parsing Hesai packet: {e}")
            return None
        
        if not points:
            return None
        
        return np.array(points, dtype=np.float32)
    
    def _get_elevation_angle(self, channel_idx: int, sensor_model: str) -> float:
        """
        Get elevation angle for a channel (simplified approximation).
        
        In a real implementation, this would use calibration data from the sensor.
        
        Args:
            channel_idx: Channel index
            sensor_model: Sensor model identifier
            
        Returns:
            float: Elevation angle in degrees
        """
        # Simplified elevation angles (degrees)
        if "XT-32" in sensor_model or "XT32" in sensor_model:
            # PandarXT-32 has channels from +15° to -16°
            return 15.0 - (channel_idx * 31.0 / 31.0)
        elif "XT-16" in sensor_model or "XT16" in sensor_model:
            # PandarXT-16 has channels from +15° to -16°
            return 15.0 - (channel_idx * 31.0 / 15.0)
        elif "Pandar64" in sensor_model or "64" in sensor_model:
            # Pandar64 spans approximately +15° to -25°
            return 15.0 - (channel_idx * 40.0 / 63.0)
        elif "Pandar40" in sensor_model or "40P" in sensor_model:
            # Pandar40P spans approximately +15° to -25°
            return 15.0 - (channel_idx * 40.0 / 39.0)
        elif "QT" in sensor_model:
            # PandarQT spans approximately +52.1° to -52.1°
            return 52.1 - (channel_idx * 104.2 / 63.0)
        else:
            # Default fallback
            return 0.0
    
    def _polar_to_cartesian(self, distance: float, azimuth: float, elevation: float) -> Tuple[float, float, float]:
        """
        Convert polar coordinates to Cartesian coordinates.
        
        Args:
            distance: Distance in meters
            azimuth: Azimuth angle in degrees
            elevation: Elevation angle in degrees
            
        Returns:
            Tuple of (x, y, z) coordinates
        """
        # Convert angles to radians
        azimuth_rad = np.radians(azimuth)
        elevation_rad = np.radians(elevation)
        
        # Calculate Cartesian coordinates
        # Hesai coordinate system: X forward, Y left, Z up
        x = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
        y = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
        z = distance * np.sin(elevation_rad)
        
        return float(x), float(y), float(z)
    
    def convert(
        self,
        input_path: str,
        output_format: str,
        output_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generic conversion method supporting multiple output formats.
        
        Routes to format-specific conversion logic based on output_format.
        
        Args:
            input_path: Path to input file
            output_format: Target format ("las", "laz", "pcd", "bin", "csv")
            output_path: Path where output file will be written
            **kwargs: Additional parameters (sensor_model, calibration_file, 
                     preserve_intensity, max_scans, etc.)
            
        Returns:
            dict: Conversion result dictionary
        """
        start_time = time.time()
        output_format = output_format.lower().lstrip(".")
        
        result = {
            "success": False,
            "message": "",
            "output_file": None,
            "conversion_time": 0.0,
            "points_converted": 0,
            "sdk_version_used": self.sdk_version,
            "error": None
        }
        
        # Validate SDK availability
        if not self.sdk_available:
            result["error"] = "Hesai processing capability is not available"
            result["message"] = "Cannot convert: dpkt library required for Hesai PCAP parsing"
            self.logger.error(result["error"])
            return result
        
        # Validate input file
        input_validation = self._validate_file_path(input_path, must_exist=True)
        if not input_validation["valid"]:
            result["error"] = input_validation["error"]
            result["message"] = f"Input validation failed: {input_validation['error']}"
            self.logger.error(result["error"])
            return result
        
        # Validate output directory
        output_validation = self._validate_output_directory(output_path)
        if not output_validation["valid"]:
            result["error"] = output_validation["error"]
            result["message"] = f"Output validation failed: {output_validation['error']}"
            self.logger.error(result["error"])
            return result
        
        # Route to appropriate format handler
        if output_format == "las":
            return self.convert_to_las(input_path, output_path, **kwargs)
        
        # For all other formats, extract points first then convert
        try:
            self.logger.info(f"Starting Hesai conversion: {input_path} -> {output_path} ({output_format})")
            
            # Extract parameters
            sensor_model = kwargs.pop('sensor_model', None)
            max_scans = kwargs.pop('max_scans', None)
            preserve_intensity = kwargs.get('preserve_intensity', True)
            
            # Extract points using _process_pcap
            points = self._process_pcap(
                input_path,
                sensor_model,
                max_scans,
                **kwargs
            )
            
            if points is None or len(points) == 0:
                result["error"] = "Failed to extract points from PCAP"
                result["message"] = "Could not parse Hesai data"
                self.logger.error(result["error"])
                return result
            
            point_count = len(points)
            
            # Convert to target format
            if output_format == "laz":
                # Convert to LAS first, then compress
                las_path = str(Path(output_path).with_suffix(".las"))
                
                # Check laspy availability
                if not LASPY_AVAILABLE:
                    result["error"] = "laspy not available - install with: pip install laspy"
                    result["message"] = "Cannot create LAS file: laspy package required"
                    self.logger.error(result["error"])
                    return result
                
                # Create LAS file
                self.logger.debug(f"Writing LAS file: {las_path}")
                self._create_las_file(points, las_path, preserve_intensity)
                
                # Compress to LAZ
                self.logger.debug(f"Compressing to LAZ: {output_path}")
                compression_result = self._compress_las_to_laz(las_path, output_path)
                
                if not compression_result["success"]:
                    result["error"] = compression_result.get("error", "LAZ compression failed")
                    result["message"] = f"LAZ compression failed: {result['error']}"
                    self.logger.error(result["error"])
                    return result
                
                # Clean up temporary LAS file if LAZ was created successfully
                if Path(output_path).exists() and Path(las_path).exists():
                    try:
                        Path(las_path).unlink()
                    except:
                        pass  # Ignore cleanup errors
                
                result.update({
                    "success": True,
                    "message": f"Successfully converted {point_count:,} points to LAZ",
                    "points_converted": point_count,
                    "output_file": str(output_path),
                    "compression_method": compression_result.get("compression_method")
                })
                
                if "warning" in compression_result:
                    result["warning"] = compression_result["warning"]
            
            elif output_format == "pcd":
                # Convert to PCD format
                pcd_result = self._points_to_pcd(points, output_path, preserve_intensity)
                
                if not pcd_result["success"]:
                    result["error"] = pcd_result.get("error", "PCD conversion failed")
                    result["message"] = f"PCD conversion failed: {result['error']}"
                    self.logger.error(result["error"])
                    return result
                
                result.update({
                    "success": True,
                    "message": f"Successfully converted {point_count:,} points to PCD",
                    "points_converted": point_count,
                    "output_file": str(output_path)
                })
            
            elif output_format == "bin":
                # Convert to BIN format (KITTI)
                bin_result = self._points_to_bin(points, output_path)
                
                if not bin_result["success"]:
                    result["error"] = bin_result.get("error", "BIN conversion failed")
                    result["message"] = f"BIN conversion failed: {result['error']}"
                    self.logger.error(result["error"])
                    return result
                
                result.update({
                    "success": True,
                    "message": f"Successfully converted {point_count:,} points to BIN",
                    "points_converted": point_count,
                    "output_file": str(output_path)
                })
            
            elif output_format == "csv":
                # Convert to CSV format
                csv_result = self._points_to_csv(points, output_path, preserve_intensity)
                
                if not csv_result["success"]:
                    result["error"] = csv_result.get("error", "CSV conversion failed")
                    result["message"] = f"CSV conversion failed: {result['error']}"
                    self.logger.error(result["error"])
                    return result
                
                result.update({
                    "success": True,
                    "message": f"Successfully converted {point_count:,} points to CSV",
                    "points_converted": point_count,
                    "output_file": str(output_path)
                })
            
            else:
                result["error"] = f"Unsupported output format: {output_format}"
                result["message"] = f"Supported formats: {', '.join(self.SUPPORTED_OUTPUT_FORMATS)}"
                self.logger.error(result["error"])
                return result
            
            # Calculate conversion time
            conversion_time = time.time() - start_time
            result["conversion_time"] = conversion_time
            result["sdk_version_used"] = self.sdk_version
            
            self.logger.info(
                f"Conversion completed: {point_count:,} points in {conversion_time:.2f}s"
            )
        
        except Exception as e:
            conversion_time = time.time() - start_time
            result.update({
                "success": False,
                "error": str(e),
                "message": f"Conversion failed: {e}",
                "conversion_time": conversion_time
            })
            self.logger.exception(f"Exception during Hesai conversion: {e}")
        
        return result
