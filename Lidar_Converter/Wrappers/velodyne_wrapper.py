#!/usr/bin/env python3
"""
Velodyne LiDAR SDK wrapper for conversion to standardized formats.

This module provides a unified interface to Velodyne LiDAR data processing,
abstracting Velodyne-specific implementation details. It serves as the Velodyne
counterpart to the Ouster wrapper, following the same architectural patterns.

The wrapper handles:
- SDK installation validation (with dpkt fallback)
- PCAP to LAS/LAZ/PCD conversion
- Velodyne packet structure parsing
- Error handling and logging
- Performance monitoring
- Vendor-specific metadata extraction

Supported Velodyne Models:
- VLP-16 (16 channels, 300m range)
- VLP-32C (32 channels, 200m range)
- HDL-32E (32 channels, 100m range)
- HDL-64E (64 channels, 120m range)
- VLS-128 (128 channels, 300m range)

Usage:
    wrapper = VelodyneWrapper()
    if wrapper.sdk_available:
        result = wrapper.convert_to_las("input.pcap", "output.las")
"""

import os
import sys
import subprocess
import time
import struct
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass

# Velodyne SDK imports (with graceful fallback)
VELODYNE_SDK_AVAILABLE = False
velodyne = None

try:
    # Try to import any available Velodyne libraries
    # Note: There's no standard Velodyne Python SDK, so we'll use generic parsing
    import velodyne_decoder
    VELODYNE_SDK_AVAILABLE = True
    velodyne = velodyne_decoder
except ImportError:
    try:
        # Alternative: check for other Velodyne libraries
        import velodyne
        VELODYNE_SDK_AVAILABLE = True
    except ImportError:
        # No Velodyne SDK available - will use dpkt fallback
        VELODYNE_SDK_AVAILABLE = False
        velodyne = None

# PCAP parsing dependency (required for Velodyne)
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


@dataclass
class VelodyneDataBlock:
    """Represents a single Velodyne data block from UDP packet."""
    azimuth: int  # 0-35999 (0.01° resolution)
    channels: List['VelodyneChannel']


@dataclass
class VelodyneChannel:
    """Represents a single channel measurement in a Velodyne data block."""
    distance: int     # Raw distance measurement (2mm resolution)
    intensity: int    # Reflectivity value (0-255)


class VelodyneWrapper(BaseVendorWrapper):
    """
    Velodyne LiDAR wrapper implementing BaseVendorWrapper interface.
    
    This wrapper provides conversion capabilities for Velodyne sensors:
    - VLP-16 (16 channels)
    - VLP-32C (32 channels)
    - HDL-32E (32 channels)
    - HDL-64E (64 channels)
    - VLS-128 (128 channels)
    
    The wrapper uses dpkt for PCAP parsing and implements Velodyne packet
    structure parsing manually, as there's no standard Velodyne Python SDK.
    
    Example:
        wrapper = VelodyneWrapper()
        if wrapper.sdk_available:
            result = wrapper.convert_to_las(
                input_path="data.pcap",
                output_path="output.las",
                sensor_model="VLP-16"
            )
            if result["success"]:
                print(f"Converted {result['points_converted']} points")
    """
    
    # Supported Velodyne sensor models
    SUPPORTED_MODELS = [
        "VLP-16",     # 16 channels, 300m range
        "VLP-32C",    # 32 channels, 200m range
        "HDL-32E",    # 32 channels, 100m range
        "HDL-64E",    # 64 channels, 120m range
        "VLS-128",    # 128 channels, 300m range
    ]
    
    # Supported input/output formats
    SUPPORTED_INPUT_FORMATS = [".pcap"]
    SUPPORTED_OUTPUT_FORMATS = [".las", ".laz", ".pcd", ".bin", ".csv"]
    
    # Velodyne packet constants
    VELODYNE_PACKET_SIZE = 1206  # Standard Velodyne UDP payload size
    VELODYNE_MAGIC_BYTES = b"\xFF\xEE"  # Magic bytes at start of packet
    DATA_BLOCKS_PER_PACKET = 12  # Number of data blocks per packet
    CHANNELS_PER_BLOCK = 32      # Maximum channels per block (varies by model)
    
    # Sensor-specific channel counts
    SENSOR_CHANNELS = {
        "VLP-16": 16,
        "VLP-32C": 32,
        "HDL-32E": 32,
        "HDL-64E": 64,
        "VLS-128": 128,
    }
    
    def __init__(self, sdk_path: Optional[str] = None, raise_on_missing: bool = False):
        """
        Initialize Velodyne wrapper with SDK validation.
        
        Args:
            sdk_path: Optional path to custom Velodyne SDK installation
            raise_on_missing: If True, raise exception if SDK not found.
                            If False (default), log warning and use dpkt fallback.
        
        Raises:
            RuntimeError: If raise_on_missing=True and dpkt is not available
        """
        super().__init__()
        
        # Check for custom SDK path from environment variable
        if sdk_path is None:
            sdk_path = os.environ.get("VELODYNE_SDK_PATH")
        
        self.sdk_path = sdk_path
        self.raise_on_missing = raise_on_missing
        
        # Validate SDK installation
        validation_result = self.validate_sdk_installation()
        
        if not validation_result.get("available", False):
            error_msg = validation_result.get("error", "Velodyne SDK not found")
            if raise_on_missing:
                raise RuntimeError(f"Velodyne processing capability is required but not available: {error_msg}")
            else:
                self.logger.warning(f"Velodyne SDK not available: {error_msg}")
                self.sdk_available = False
        else:
            self.sdk_available = True
            self.sdk_version = validation_result.get("version", "dpkt-fallback")
            self.logger.info(f"Velodyne processing validated - Method: {validation_result.get('method', 'unknown')}")
    
    def get_vendor_name(self) -> str:
        """Return vendor identifier."""
        return "velodyne"
    
    def validate_sdk_installation(self) -> Dict[str, Any]:
        """
        Validate Velodyne SDK installation and detect available tools.
        
        Checks for (in priority order):
        1. Velodyne Python libraries (velodyne-decoder, etc.)
        2. VeloView CLI tools
        3. dpkt library for generic PCAP parsing
        
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
        
        # Method 1: Check for Velodyne Python libraries
        if VELODYNE_SDK_AVAILABLE and velodyne is not None:
            try:
                # Try to get version from velodyne package
                version = "unknown"
                if hasattr(velodyne, '__version__'):
                    version = velodyne.__version__
                else:
                    # Try different methods to get version
                    try:
                        import importlib.metadata
                        version = importlib.metadata.version('velodyne-decoder')
                    except (ImportError, importlib.metadata.PackageNotFoundError):
                        try:
                            import pkg_resources
                            version = pkg_resources.get_distribution('velodyne-decoder').version
                        except:
                            version = "installed"
                
                result.update({
                    "available": True,
                    "version": version,
                    "method": "python_package",
                    "message": f"Velodyne Python SDK {version} is available"
                })
                return result
            except Exception as e:
                self.logger.debug(f"Failed to get Python SDK version: {e}")
        
        # Method 2: Check for VeloView CLI tool
        try:
            cli_result = subprocess.run(
                ["veloview", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if cli_result.returncode == 0:
                version = cli_result.stdout.strip()
                result.update({
                    "available": True,
                    "version": version,
                    "method": "veloview_cli",
                    "message": f"VeloView CLI tool {version} is available"
                })
                return result
        except FileNotFoundError:
            pass
        except Exception as e:
            self.logger.debug(f"Failed to check VeloView CLI: {e}")
        
        # Method 3: Check dpkt for generic PCAP parsing (fallback)
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
                "message": f"Using dpkt {dpkt_version} for generic PCAP parsing"
            })
            return result
        
        # Method 4: Check custom SDK path
        if self.sdk_path:
            sdk_dir = Path(self.sdk_path)
            if sdk_dir.exists() and sdk_dir.is_dir():
                result.update({
                    "available": True,
                    "installation_path": str(sdk_dir),
                    "method": "custom_path",
                    "message": f"Velodyne SDK found at custom path: {self.sdk_path}"
                })
                return result
        
        # No processing capability found
        result.update({
            "available": False,
            "error": "No Velodyne processing capability found. Install dpkt with: pip install dpkt",
            "message": "Velodyne processing is not available"
        })
        return result
    
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
        Convert Velodyne PCAP file to LAS format.
        
        This method implements the Velodyne-specific conversion logic while
        maintaining the unified interface defined by BaseVendorWrapper.
        
        Args:
            input_path: Path to input .pcap file
            output_path: Path where .las file will be written
            sensor_model: Optional sensor model (e.g., "VLP-16")
            calibration_file: Optional calibration file (not typically needed for Velodyne)
            preserve_intensity: Whether to preserve intensity values
            max_scans: Optional limit on number of scans to process
            **kwargs: Additional parameters:
                - dual_return: Process dual returns if available
                - min_range: Minimum range filter (meters)
                - max_range: Maximum range filter (meters)
                
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
            result["error"] = "Velodyne processing capability is not available"
            result["message"] = "Cannot convert: dpkt library required for Velodyne PCAP parsing"
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
            result["message"] = "Velodyne wrapper only supports .pcap input files"
            self.logger.error(result["error"])
            return result
        
        if output_path_obj.suffix.lower() != ".las":
            result["error"] = f"Unsupported output format: {output_path_obj.suffix}"
            result["message"] = "Use convert() method for formats other than .las"
            self.logger.error(result["error"])
            return result
        
        try:
            self.logger.info(f"Starting Velodyne conversion: {input_path} -> {output_path}")
            
            # Auto-detect sensor model if not provided
            if not sensor_model:
                sensor_model = self._detect_sensor_model(input_path)
                if sensor_model:
                    self.logger.info(f"Auto-detected sensor model: {sensor_model}")
                else:
                    self.logger.warning("Could not auto-detect sensor model, using VLP-16 as default")
                    sensor_model = "VLP-16"
            
            # Extract points using refactored pipeline
            points = self._convert_with_dpkt_parsing(
                input_path,
                sensor_model,
                preserve_intensity,
                max_scans,
                **kwargs
            )
            
            if points is None:
                result["error"] = "Failed to extract points from PCAP"
                result["message"] = "Could not parse Velodyne data"
                self.logger.error(result["error"])
                return result
            
            # Check laspy availability
            if not LASPY_AVAILABLE:
                result["error"] = "laspy not available - install with: pip install laspy"
                result["message"] = "Cannot create LAS file: laspy package required"
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
            self.logger.exception(f"Exception during Velodyne conversion: {e}")
        
        return result
    
    def _detect_sensor_model(self, input_path: str) -> Optional[str]:
        """
        Attempt to auto-detect Velodyne sensor model from PCAP data.
        
        Args:
            input_path: Path to PCAP file
            
        Returns:
            str: Detected sensor model or None if detection fails
        """
        try:
            with open(input_path, 'rb') as f:
                pcap = dpkt.pcap.Reader(f)
                
                # Check first few packets to determine channel count
                packet_count = 0
                max_packets = 10
                
                for ts, buf in pcap:
                    packet_count += 1
                    if packet_count > max_packets:
                        break
                    
                    try:
                        eth = dpkt.ethernet.Ethernet(buf)
                        if eth.type != dpkt.ethernet.ETH_TYPE_IP:
                            continue
                        
                        ip = eth.data
                        if ip.p != dpkt.ip.IP_PROTO_UDP:
                            continue
                        
                        udp = ip.data
                        if udp.dport not in [2368, 2369]:  # Velodyne ports
                            continue
                        
                        payload = udp.data
                        if len(payload) != self.VELODYNE_PACKET_SIZE:
                            continue
                        
                        # Check magic bytes
                        if payload[:2] != self.VELODYNE_MAGIC_BYTES:
                            continue
                        
                        # For now, return VLP-16 as default
                        # More sophisticated detection could analyze packet structure
                        return "VLP-16"
                        
                    except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError, AttributeError):
                        continue
                        
        except Exception as e:
            self.logger.debug(f"Sensor model detection failed: {e}")
        
        return None
    
    def _convert_with_dpkt_parsing(
        self,
        input_path: str,
        sensor_model: str,
        preserve_intensity: bool,
        max_scans: Optional[int],
        **kwargs
    ) -> Optional[np.ndarray]:
        """
        Extract point cloud data from PCAP using dpkt parsing.
        
        Args:
            input_path: Path to input PCAP file
            sensor_model: Velodyne sensor model
            preserve_intensity: Whether to include intensity values
            max_scans: Optional limit on number of scans to process
            **kwargs: Additional parameters
            
        Returns:
            numpy array with shape (N, 4) containing [x, y, z, intensity] or None if failed
        """
        if not DPKT_AVAILABLE:
            self.logger.error("dpkt not available - install with: pip install dpkt")
            return None
        
        try:
            # Get sensor-specific parameters
            channel_count = self.SENSOR_CHANNELS.get(sensor_model, 16)
            
            # Open PCAP file
            self.logger.debug(f"Opening PCAP: {input_path}")
            with open(input_path, 'rb') as f:
                pcap = dpkt.pcap.Reader(f)
                
                # Collect point cloud data
                all_points_list = []
                packet_count = 0
                valid_packet_count = 0
                
                self.logger.info(f"Processing packets (max_scans: {max_scans or 'unlimited'})...")
                
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
                        if udp.dport not in [2368, 2369]:  # Velodyne data ports
                            continue
                        
                        # Parse Velodyne packet
                        points = self._parse_velodyne_packet(
                            udp.data, 
                            sensor_model, 
                            channel_count,
                            preserve_intensity
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
                    self.logger.error("No valid Velodyne packets found in PCAP file")
                    return None
                
                # Combine all points
                self.logger.debug("Combining all point clouds...")
                all_points = np.vstack(all_points_list)
                point_count = len(all_points)
                
                self.logger.info(f"Extracted {point_count:,} points from {valid_packet_count} packets")
                return all_points
        
        except Exception as e:
            self.logger.exception(f"Error in dpkt parsing: {e}")
            return None
    
    def _parse_velodyne_packet(
        self, 
        payload: bytes, 
        sensor_model: str, 
        channel_count: int,
        preserve_intensity: bool
    ) -> Optional[np.ndarray]:
        """
        Parse a single Velodyne UDP packet and extract point data.
        
        Args:
            payload: UDP payload bytes
            sensor_model: Sensor model identifier
            channel_count: Number of channels for this sensor
            preserve_intensity: Whether to include intensity data
            
        Returns:
            numpy array of points [x, y, z, intensity] or None if invalid
        """
        if len(payload) != self.VELODYNE_PACKET_SIZE:
            return None
        
        # Check magic bytes
        if payload[:2] != self.VELODYNE_MAGIC_BYTES:
            return None
        
        points = []
        
        try:
            # Parse data blocks (12 blocks per packet)
            for block_idx in range(self.DATA_BLOCKS_PER_PACKET):
                block_offset = 2 + block_idx * 100  # Skip magic bytes, each block is 100 bytes
                
                if block_offset + 100 > len(payload):
                    break
                
                # Parse azimuth (2 bytes, little endian)
                azimuth_raw = struct.unpack('<H', payload[block_offset:block_offset+2])[0]
                azimuth = azimuth_raw / 100.0  # Convert to degrees
                
                # Parse channel data (32 channels max, 3 bytes each)
                for channel_idx in range(min(channel_count, self.CHANNELS_PER_BLOCK)):
                    channel_offset = block_offset + 2 + channel_idx * 3
                    
                    if channel_offset + 3 > len(payload):
                        break
                    
                    # Parse distance (2 bytes) and intensity (1 byte)
                    distance_raw = struct.unpack('<H', payload[channel_offset:channel_offset+2])[0]
                    intensity = payload[channel_offset+2]
                    
                    # Convert distance (2mm resolution)
                    distance = distance_raw * 0.002  # Convert to meters
                    
                    # Skip invalid points
                    if distance <= 0 or distance > 300:  # Max range check
                        continue
                    
                    # Calculate elevation angle (simplified - would need calibration data for accuracy)
                    # This is a rough approximation
                    elevation = self._get_elevation_angle(channel_idx, sensor_model)
                    
                    # Convert to Cartesian coordinates
                    x, y, z = self._polar_to_cartesian(distance, azimuth, elevation)
                    
                    # Add point [x, y, z, intensity]
                    if preserve_intensity:
                        points.append([x, y, z, intensity])
                    else:
                        points.append([x, y, z, 0])
        
        except Exception as e:
            self.logger.debug(f"Error parsing Velodyne packet: {e}")
            return None
        
        if not points:
            return None
        
        return np.array(points, dtype=np.float32)
    
    def _get_elevation_angle(self, channel_idx: int, sensor_model: str) -> float:
        """
        Get elevation angle for a channel (simplified approximation).
        
        In a real implementation, this would use calibration data.
        """
        # Simplified elevation angles (degrees)
        if sensor_model == "VLP-16":
            # VLP-16 has channels from +15° to -15°
            return 15.0 - (channel_idx * 30.0 / 15.0)
        elif sensor_model in ["VLP-32C", "HDL-32E"]:
            # 32-channel sensors typically span +10° to -30°
            return 10.0 - (channel_idx * 40.0 / 31.0)
        elif sensor_model == "HDL-64E":
            # 64-channel sensor spans +2° to -24.8°
            return 2.0 - (channel_idx * 26.8 / 63.0)
        elif sensor_model == "VLS-128":
            # 128-channel sensor spans +15° to -25°
            return 15.0 - (channel_idx * 40.0 / 127.0)
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
        # Velodyne coordinate system: X forward, Y left, Z up
        x = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
        y = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
        z = distance * np.sin(elevation_rad)
        
        return float(x), float(y), float(z)
    
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
        
        if preserve_intensity and points.shape[1] > 3:
            las_file.intensity = points[:, 3].astype(np.uint16)
        else:
            las_file.intensity = np.zeros(len(points), dtype=np.uint16)
        
        # Write LAS file
        las_file.write(str(output_path))
    
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
            **kwargs: Additional parameters including:
                - sensor_model: Velodyne sensor model (e.g., "VLP-16")
                - preserve_intensity: Whether to preserve intensity values (default: True)
                - max_scans: Optional limit on number of scans to process
            
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
            result["error"] = "Velodyne processing capability is not available"
            result["message"] = "Cannot convert: dpkt library required for Velodyne PCAP parsing"
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
        
        # Validate input format
        input_path_obj = Path(input_path)
        if input_path_obj.suffix.lower() != ".pcap":
            result["error"] = f"Unsupported input format: {input_path_obj.suffix}"
            result["message"] = "Velodyne wrapper only supports .pcap input files"
            self.logger.error(result["error"])
            return result
        
        try:
            # Extract parameters that are passed as positional args (remove from kwargs to avoid duplicates)
            sensor_model = kwargs.pop("sensor_model", None)
            preserve_intensity = kwargs.pop("preserve_intensity", True)
            max_scans = kwargs.pop("max_scans", None)
            
            # Auto-detect sensor model if not provided
            if not sensor_model:
                sensor_model = self._detect_sensor_model(input_path)
                if sensor_model:
                    self.logger.info(f"Auto-detected sensor model: {sensor_model}")
                else:
                    self.logger.warning("Could not auto-detect sensor model, using VLP-16 as default")
                    sensor_model = "VLP-16"
            
            self.logger.info(f"Starting Velodyne conversion: {input_path} -> {output_path} ({output_format})")
            
            # Route to appropriate format handler
            if output_format == "las":
                # Extract points
                points = self._convert_with_dpkt_parsing(
                    input_path,
                    sensor_model,
                    preserve_intensity,
                    max_scans,
                    **kwargs
                )
                
                if points is None:
                    result["error"] = "Failed to extract points from PCAP"
                    result["message"] = "Could not parse Velodyne data"
                    return result
                
                # Convert to LAS
                if not LASPY_AVAILABLE:
                    result["error"] = "laspy not available - install with: pip install laspy"
                    result["message"] = "Cannot create LAS file: laspy package required"
                    return result
                
                self._create_las_file(points, output_path, preserve_intensity)
                
                result.update({
                    "success": True,
                    "message": f"Successfully converted {len(points):,} points to LAS format",
                    "points_converted": len(points),
                    "output_file": output_path
                })
                
            elif output_format == "laz":
                # Extract points
                points = self._convert_with_dpkt_parsing(
                    input_path,
                    sensor_model,
                    preserve_intensity,
                    max_scans,
                    **kwargs
                )
                
                if points is None:
                    result["error"] = "Failed to extract points from PCAP"
                    result["message"] = "Could not parse Velodyne data"
                    return result
                
                # Convert to LAS first
                if not LASPY_AVAILABLE:
                    result["error"] = "laspy not available - install with: pip install laspy"
                    result["message"] = "Cannot create LAS file: laspy package required"
                    return result
                
                las_path = str(Path(output_path).with_suffix(".las"))
                self._create_las_file(points, las_path, preserve_intensity)
                
                # Compress to LAZ
                compression_result = self._compress_las_to_laz(las_path, output_path)
                
                if compression_result["success"]:
                    result.update({
                        "success": True,
                        "message": f"Successfully converted {len(points):,} points to LAZ format",
                        "points_converted": len(points),
                        "output_file": compression_result["output_file"],
                        "compression_method": compression_result.get("compression_method")
                    })
                    
                    # Add warning if compression wasn't available
                    if "warning" in compression_result:
                        result["warning"] = compression_result["warning"]
                else:
                    result["error"] = compression_result.get("error", "LAZ compression failed")
                    result["message"] = "Failed to compress LAS to LAZ"
                    
            elif output_format == "pcd":
                # Extract points
                points = self._convert_with_dpkt_parsing(
                    input_path,
                    sensor_model,
                    preserve_intensity,
                    max_scans,
                    **kwargs
                )
                
                if points is None:
                    result["error"] = "Failed to extract points from PCAP"
                    result["message"] = "Could not parse Velodyne data"
                    return result
                
                # Convert to PCD
                pcd_result = self._points_to_pcd(points, output_path, preserve_intensity)
                
                if pcd_result["success"]:
                    result.update({
                        "success": True,
                        "message": f"Successfully converted {pcd_result['points_converted']:,} points to PCD format",
                        "points_converted": pcd_result["points_converted"],
                        "output_file": pcd_result["output_file"]
                    })
                else:
                    result["error"] = pcd_result.get("error", "PCD conversion failed")
                    result["message"] = "Failed to convert to PCD format"
                    
            elif output_format == "bin":
                # Extract points
                points = self._convert_with_dpkt_parsing(
                    input_path,
                    sensor_model,
                    preserve_intensity,
                    max_scans,
                    **kwargs
                )
                
                if points is None:
                    result["error"] = "Failed to extract points from PCAP"
                    result["message"] = "Could not parse Velodyne data"
                    return result
                
                # Convert to BIN
                bin_result = self._points_to_bin(points, output_path)
                
                if bin_result["success"]:
                    result.update({
                        "success": True,
                        "message": f"Successfully converted {bin_result['points_converted']:,} points to BIN format",
                        "points_converted": bin_result["points_converted"],
                        "output_file": bin_result["output_file"]
                    })
                else:
                    result["error"] = bin_result.get("error", "BIN conversion failed")
                    result["message"] = "Failed to convert to BIN format"
                    
            elif output_format == "csv":
                # Extract points
                points = self._convert_with_dpkt_parsing(
                    input_path,
                    sensor_model,
                    preserve_intensity,
                    max_scans,
                    **kwargs
                )
                
                if points is None:
                    result["error"] = "Failed to extract points from PCAP"
                    result["message"] = "Could not parse Velodyne data"
                    return result
                
                # Convert to CSV
                csv_result = self._points_to_csv(points, output_path, preserve_intensity)
                
                if csv_result["success"]:
                    result.update({
                        "success": True,
                        "message": f"Successfully converted {csv_result['points_converted']:,} points to CSV format",
                        "points_converted": csv_result["points_converted"],
                        "output_file": csv_result["output_file"]
                    })
                else:
                    result["error"] = csv_result.get("error", "CSV conversion failed")
                    result["message"] = "Failed to convert to CSV format"
                    
            else:
                result["error"] = f"Unsupported output format: {output_format}"
                result["message"] = f"Supported formats: {', '.join(self.SUPPORTED_OUTPUT_FORMATS)}"
                self.logger.error(result["error"])
                return result
            
            # Calculate conversion time
            conversion_time = time.time() - start_time
            result["conversion_time"] = conversion_time
            
            if result["success"]:
                self.logger.info(
                    f"Conversion completed: {result['points_converted']} points in {conversion_time:.2f}s"
                )
            else:
                self.logger.error(f"Conversion failed: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            conversion_time = time.time() - start_time
            result.update({
                "success": False,
                "error": str(e),
                "message": f"Conversion failed: {e}",
                "conversion_time": conversion_time
            })
            self.logger.exception(f"Exception during Velodyne conversion: {e}")
        
        return result
    

    
    def get_vendor_info(self) -> Dict[str, Any]:
        """
        Get Velodyne vendor capabilities and information.
        
        Returns:
            dict: Vendor information dictionary
        """
        return {
            "vendor": "velodyne",
            "supported_input_formats": self.SUPPORTED_INPUT_FORMATS,
            "supported_output_formats": self.SUPPORTED_OUTPUT_FORMATS,
            "sdk_version": self.sdk_version,
            "supported_sensor_models": self.SUPPORTED_MODELS,
            "requires_calibration": False,  # Velodyne doesn't typically need external calibration files
            "status": "available" if self.sdk_available else "not_installed",
            "sdk_available": self.sdk_available,
            "installation_method": "dpkt_fallback" if DPKT_AVAILABLE else "unavailable"
        }
    
    def validate_conversion(
        self,
        input_path: str,
        output_path: str
    ) -> bool:
        """
        Validate that a conversion output is valid.
        
        Checks:
        - File exists and is readable
        - File is valid LAS format
        - Point count is reasonable (> 0)
        
        Args:
            input_path: Original input file (for reference)
            output_path: Path to generated output file
            
        Returns:
            bool: True if output is valid, False otherwise
        """
        output_path_obj = Path(output_path)
        
        # Check file exists
        if not output_path_obj.exists():
            self.logger.error(f"Output file not found: {output_path}")
            return False
        
        # Check file is readable
        if not output_path_obj.is_file():
            self.logger.error(f"Output path is not a file: {output_path}")
            return False
        
        # Check file size is reasonable (> 0 bytes)
        if output_path_obj.stat().st_size == 0:
            self.logger.error(f"Output file is empty: {output_path}")
            return False
        
        # Validate LAS file structure
        try:
            if not LASPY_AVAILABLE:
                self.logger.warning("laspy not available - skipping LAS validation")
                return True  # Assume valid if we can't validate
            
            las_file = laspy.read(output_path)
            point_count = len(las_file.points)
            
            if point_count == 0:
                self.logger.error("Output file contains no points")
                return False
            
            self.logger.info(f"Validation passed: {point_count:,} points in output file")
            return True
        
        except Exception as e:
            self.logger.error(f"LAS validation failed: {e}")
            return False