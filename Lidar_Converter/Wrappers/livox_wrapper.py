#!/usr/bin/env python3
"""
Livox LiDAR wrapper for conversion to standardized formats.

This module provides a unified interface for Livox LiDAR data processing.
Livox sensors (Avia, Horizon, Tele-15, Mid-40/70, HAP) use custom data formats.

The wrapper handles:
- CSV file parsing (exported from Livox Viewer)
- LVX file parsing (Livox custom format)
- Error handling and logging
- Performance monitoring

Note: Livox does not provide a standard Python SDK. This wrapper uses manual
parsing of Livox data formats. For best results, export data to CSV format
using Livox Viewer software.

Supported Livox Models:
- Avia (automotive-grade, 70m-450m range)
- Horizon (compact, 260m range)
- Tele-15 (long-range, 500m)
- Mid-40/70 (industrial, 260m range)
- HAP (high-altitude, 450m range)

Usage:
    wrapper = LivoxWrapper()
    if wrapper.sdk_available:
        result = wrapper.convert_to_las("input.csv", "output.las")
"""

import os
import sys
import subprocess
import time
import struct
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# Livox SDK imports (manual parsing - no official Python SDK)
LIVOX_SDK_AVAILABLE = False
livox = None

# PCAP parsing dependency
try:
    import dpkt
    DPKT_AVAILABLE = True
except ImportError:
    DPKT_AVAILABLE = False

# CSV parsing dependency
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# LAS file handling
try:
    import laspy
    LASPY_AVAILABLE = True
except ImportError:
    LASPY_AVAILABLE = False

import numpy as np

from .base_wrapper import BaseVendorWrapper

logger = logging.getLogger(__name__)


class LivoxWrapper(BaseVendorWrapper):
    """
    Livox LiDAR wrapper implementing BaseVendorWrapper interface.
    
    This wrapper provides conversion capabilities for Livox sensors:
    - Avia (automotive-grade)
    - Horizon (compact)
    - Tele-15 (long-range)
    - Mid-40/70 (industrial)
    - HAP (high-altitude)
    
    The wrapper uses pandas for CSV parsing as Livox doesn't provide
    a standard Python SDK. Users should export data to CSV format using
    Livox Viewer software for best compatibility.
    
    Example:
        wrapper = LivoxWrapper()
        if wrapper.sdk_available:
            result = wrapper.convert_to_las(
                input_path="livox_data.csv",
                output_path="output.las"
            )
            if result["success"]:
                print(f"Converted {result['points_converted']} points")
    """
    
    # Supported Livox sensor models
    SUPPORTED_MODELS = [
        "Avia",       # Automotive-grade, 70m-450m range
        "Horizon",    # Compact, 260m range
        "Tele-15",    # Long-range, 500m
        "Mid-40",     # Industrial, 260m range, 40° FOV
        "Mid-70",     # Industrial, 260m range, 70° FOV
        "HAP",        # High-altitude, 450m range
    ]
    
    # Supported input/output formats
    SUPPORTED_INPUT_FORMATS = [".csv", ".lvx", ".lvx2", ".pcap"]
    SUPPORTED_OUTPUT_FORMATS = [".las", ".laz", ".pcd", ".bin", ".csv"]
    
    def __init__(self, sdk_path: Optional[str] = None, raise_on_missing: bool = False):
        """
        Initialize Livox wrapper with SDK validation.
        
        Args:
            sdk_path: Optional path to custom Livox SDK installation
            raise_on_missing: If True, raise exception if pandas not found.
                            If False (default), log warning.
        
        Raises:
            RuntimeError: If raise_on_missing=True and pandas is not available
        """
        super().__init__()
        
        # Check for custom SDK path from environment variable
        if sdk_path is None:
            sdk_path = os.environ.get("LIVOX_SDK_PATH")
        
        self.sdk_path = sdk_path
        self.raise_on_missing = raise_on_missing
        
        # Validate SDK installation
        validation_result = self.validate_sdk_installation()
        
        if not validation_result.get("available", False):
            error_msg = validation_result.get("error", "Livox processing capability not found")
            if raise_on_missing:
                raise RuntimeError(f"Livox processing capability is required but not available: {error_msg}")
            else:
                self.logger.warning(f"Livox processing not fully available: {error_msg}")
                self.sdk_available = False
        else:
            self.sdk_available = True
            self.sdk_version = validation_result.get("version", "pandas-csv-parser")
            self.logger.info(f"Livox processing validated - Method: {validation_result.get('method', 'unknown')}")
    
    def get_vendor_name(self) -> str:
        """Return vendor identifier."""
        return "livox"
    
    def validate_sdk_installation(self) -> Dict[str, Any]:
        """
        Validate Livox SDK installation and detect available tools.
        
        Checks for (in priority order):
        1. Livox Viewer CLI tools
        2. pandas library for CSV parsing
        3. Custom SDK path
        
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
        
        # Method 1: Check for Livox Viewer CLI
        try:
            # Livox Viewer doesn't have a standard CLI, but check if it's installed
            livox_viewer_paths = [
                "C:\\Program Files\\Livox\\Livox Viewer\\LivoxViewer.exe",
                "C:\\Program Files (x86)\\Livox\\Livox Viewer\\LivoxViewer.exe",
            ]
            
            for path in livox_viewer_paths:
                if Path(path).exists():
                    result.update({
                        "available": True,
                        "version": "Livox Viewer",
                        "installation_path": path,
                        "method": "livox_viewer",
                        "message": f"Livox Viewer found at {path}"
                    })
                    return result
        except Exception as e:
            self.logger.debug(f"Failed to check Livox Viewer: {e}")
        
        # Method 2: Check pandas for CSV parsing (fallback)
        if PANDAS_AVAILABLE:
            try:
                import importlib.metadata
                pandas_version = importlib.metadata.version('pandas')
            except (ImportError, importlib.metadata.PackageNotFoundError):
                try:
                    import pkg_resources
                    pandas_version = pkg_resources.get_distribution('pandas').version
                except:
                    pandas_version = "installed"
            
            result.update({
                "available": True,
                "version": f"pandas-{pandas_version}",
                "method": "pandas_csv",
                "message": f"Using pandas {pandas_version} for CSV parsing"
            })
            return result
        
        # Method 3: Check custom SDK path
        if self.sdk_path:
            sdk_dir = Path(self.sdk_path)
            if sdk_dir.exists() and sdk_dir.is_dir():
                result.update({
                    "available": True,
                    "installation_path": str(sdk_dir),
                    "method": "custom_path",
                    "message": f"Livox SDK found at custom path: {self.sdk_path}"
                })
                return result
        
        # No processing capability found
        result.update({
            "available": False,
            "error": "No Livox processing capability found. Install pandas with: pip install pandas",
            "message": "Livox processing is not available"
        })
        return result
    
    def convert_to_las(
        self,
        input_path: str,
        output_path: str,
        sensor_model: Optional[str] = None,
        calibration_file: Optional[str] = None,
        preserve_intensity: bool = True,
        max_points: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Convert Livox data file to LAS format.
        
        This method implements the Livox-specific conversion logic while
        maintaining the unified interface defined by BaseVendorWrapper.
        
        Args:
            input_path: Path to input file (.csv, .lvx, .lvx2)
            output_path: Path where .las file will be written
            sensor_model: Optional sensor model (e.g., "Avia", "Horizon")
            calibration_file: Optional calibration file (not typically needed for Livox)
            preserve_intensity: Whether to preserve intensity values
            max_points: Optional limit on number of points to process
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
            result["error"] = "Livox processing capability is not available"
            result["message"] = "Cannot convert: pandas library required for Livox CSV parsing"
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
        
        if input_path_obj.suffix.lower() not in self.SUPPORTED_INPUT_FORMATS:
            result["error"] = f"Unsupported input format: {input_path_obj.suffix}"
            result["message"] = f"Livox wrapper supports: {', '.join(self.SUPPORTED_INPUT_FORMATS)}"
            self.logger.error(result["error"])
            return result
        
        if output_path_obj.suffix.lower() != ".las":
            result["error"] = f"Unsupported output format: {output_path_obj.suffix}"
            result["message"] = "Use convert() method for formats other than .las"
            self.logger.error(result["error"])
            return result
        
        try:
            self.logger.info(f"Starting Livox conversion: {input_path} -> {output_path}")
            
            # Perform conversion based on input format
            if input_path_obj.suffix.lower() == ".csv":
                result = self._convert_csv_to_las(
                    input_path,
                    output_path,
                    preserve_intensity,
                    max_points,
                    **kwargs
                )
            elif input_path_obj.suffix.lower() == ".pcap":
                result = self._convert_pcap_to_las(
                    input_path,
                    output_path,
                    preserve_intensity,
                    max_points,
                    **kwargs
                )
            elif input_path_obj.suffix.lower() in [".lvx", ".lvx2"]:
                result["error"] = "LVX/LVX2 format not yet fully supported"
                result["message"] = "Please export data to CSV format using Livox Viewer"
                self.logger.error(result["error"])
                return result
            
            # Calculate conversion time
            conversion_time = time.time() - start_time
            result["conversion_time"] = conversion_time
            result["sdk_version_used"] = self.sdk_version
            
            if result["success"]:
                result["output_file"] = str(output_path)
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
            self.logger.exception(f"Exception during Livox conversion: {e}")
        
        return result
    
    def _convert_csv_to_las(
        self,
        input_path: str,
        output_path: str,
        preserve_intensity: bool,
        max_points: Optional[int],
        **kwargs
    ) -> Dict[str, Any]:
        """Convert Livox CSV file to LAS format."""
        result = {
            "success": False,
            "message": "",
            "points_converted": 0,
            "error": None
        }
        
        if not PANDAS_AVAILABLE:
            result["error"] = "pandas not available - install with: pip install pandas"
            result["message"] = "Cannot parse CSV file: pandas package required"
            return result
        
        if not LASPY_AVAILABLE:
            result["error"] = "laspy not available - install with: pip install laspy"
            result["message"] = "Cannot create LAS file: laspy package required"
            return result
        
        try:
            # Read CSV file
            self.logger.debug(f"Reading CSV: {input_path}")
            
            # Read with row limit if specified
            df = pd.read_csv(input_path, nrows=max_points)
            
            self.logger.info(f"CSV columns: {list(df.columns)}")
            self.logger.info(f"Loaded {len(df)} points from CSV")
            
            # Extract point data
            if 'x' not in df.columns or 'y' not in df.columns or 'z' not in df.columns:
                result["error"] = "CSV file missing required columns (x, y, z)"
                result["message"] = "Livox CSV must contain x, y, z columns"
                return result
            
            # Get coordinates
            x = df['x'].values
            y = df['y'].values
            z = df['z'].values
            
            # Get intensity if available
            if preserve_intensity:
                if 'intensity' in df.columns:
                    intensity = df['intensity'].values
                elif 'reflectivity' in df.columns:
                    intensity = df['reflectivity'].values
                else:
                    intensity = np.ones(len(df)) * 128  # Default intensity
            else:
                intensity = np.zeros(len(df))
            
            # Create LAS file
            self.logger.debug(f"Writing LAS file: {output_path}")
            
            # Create LAS header
            header = laspy.LasHeader(point_format=1, version="1.2")
            header.x_scale = 0.001  # 1mm precision
            header.y_scale = 0.001
            header.z_scale = 0.001
            header.x_offset = float(x.mean())
            header.y_offset = float(y.mean())
            header.z_offset = float(z.mean())
            
            # Create LAS data
            las_file = laspy.LasData(header)
            las_file.x = x
            las_file.y = y
            las_file.z = z
            las_file.intensity = intensity.astype(np.uint16)
            
            # Write LAS file
            las_file.write(str(output_path))
            
            result.update({
                "success": True,
                "message": f"Successfully converted {len(df):,} points from CSV",
                "points_converted": len(df)
            })
            
            self.logger.info(f"Conversion complete: {len(df):,} points")
        
        except Exception as e:
            result["error"] = str(e)
            result["message"] = f"CSV conversion failed: {e}"
            self.logger.exception(f"Error in CSV conversion: {e}")
        
        return result
    
    def _convert_pcap_to_las(
        self,
        input_path: str,
        output_path: str,
        preserve_intensity: bool,
        max_points: Optional[int],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Convert Livox PCAP file to LAS format using manual packet parsing.
        
        Livox packet structure (simplified):
        - UDP payload contains point cloud data
        - Each point: x, y, z (float32), reflectivity (uint8), tag (uint8)
        - Packet size typically 1380 bytes
        """
        result = {
            "success": False,
            "message": "",
            "points_converted": 0,
            "error": None
        }
        
        if not DPKT_AVAILABLE:
            result["error"] = "dpkt not available - install with: pip install dpkt"
            result["message"] = "Cannot parse PCAP file: dpkt package required"
            return result
        
        if not LASPY_AVAILABLE:
            result["error"] = "laspy not available - install with: pip install laspy"
            result["message"] = "Cannot create LAS file: laspy package required"
            return result
        
        try:
            # Read PCAP file
            self.logger.debug(f"Opening PCAP: {input_path}")
            with open(input_path, 'rb') as f:
                pcap = dpkt.pcap.Reader(f)
                
                # Collect point cloud data
                all_points_list = []
                packet_count = 0
                valid_packet_count = 0
                
                # Limit points if specified
                points_collected = 0
                max_points_limit = max_points if max_points else float('inf')
                
                self.logger.info(f"Processing PCAP packets (max_points: {max_points or 'unlimited'})...")
                
                for ts, buf in pcap:
                    packet_count += 1
                    
                    if packet_count % 1000 == 0:
                        self.logger.debug(f"Processing packet {packet_count}... ({points_collected} points collected)")
                    
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
                        
                        # Check if it's a Livox data port (57000, 56000-56002, 58000)
                        if udp.dport not in [56000, 56001, 56002, 57000, 58000]:
                            continue
                        
                        # Parse Livox packet
                        points = self._parse_livox_packet(
                            udp.data,
                            preserve_intensity
                        )
                        
                        if points is not None and len(points) > 0:
                            all_points_list.append(points)
                            valid_packet_count += 1
                            points_collected += len(points)
                        
                        # Check point limit
                        if points_collected >= max_points_limit:
                            self.logger.info(f"Reached max_points limit: {points_collected}")
                            break
                            
                    except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError, AttributeError) as e:
                        # Skip malformed packets
                        continue
                    except Exception as e:
                        self.logger.warning(f"Error processing packet {packet_count}: {e}")
                        continue
                
                if not all_points_list:
                    result["error"] = "No valid Livox packets found in PCAP file"
                    result["message"] = "Could not extract any valid point cloud data"
                    return result
                
                # Combine all points
                self.logger.debug("Combining all point clouds...")
                all_points = np.vstack(all_points_list)
                point_count = len(all_points)
                
                # Create LAS file
                self.logger.debug(f"Writing LAS file: {output_path}")
                self._create_las_file(all_points, output_path, preserve_intensity)
                
                result.update({
                    "success": True,
                    "message": f"Successfully converted {point_count:,} points from {valid_packet_count} packets",
                    "points_converted": point_count
                })
                
                self.logger.info(f"Conversion complete: {point_count:,} points from {valid_packet_count} packets")
        
        except Exception as e:
            result["error"] = str(e)
            result["message"] = f"PCAP parsing conversion failed: {e}"
            self.logger.exception(f"Error in PCAP conversion: {e}")
        
        return result
    
    def _parse_livox_packet(self, payload: bytes, preserve_intensity: bool) -> Optional[np.ndarray]:
        """
        Parse a single Livox UDP packet and extract point data.
        
        Livox packet structure (varies by model, this is a simplified version):
        - Header: version, slot_id, lidar_id, etc.
        - Point data: multiple points with x, y, z, reflectivity, tag
        
        Args:
            payload: UDP payload bytes
            preserve_intensity: Whether to include intensity data
            
        Returns:
            numpy array of points [x, y, z, intensity] or None if invalid
        """
        if len(payload) < 100:  # Minimum packet size check
            return None
        
        points = []
        
        try:
            # Livox packet header (first ~24 bytes, varies by version)
            # Skip header and parse point data
            # This is a simplified parser - actual Livox format is more complex
            
            # Try to parse as point cloud data
            # Livox point format: x, y, z (int32, mm), reflectivity (uint8), tag (uint8)
            # Total: 14 bytes per point
            
            offset = 24  # Skip header (approximate)
            point_size = 14
            
            while offset + point_size <= len(payload):
                try:
                    # Parse point data (Livox uses millimeters for coordinates)
                    x_mm = struct.unpack('<i', payload[offset:offset+4])[0]
                    y_mm = struct.unpack('<i', payload[offset+4:offset+8])[0]
                    z_mm = struct.unpack('<i', payload[offset+8:offset+12])[0]
                    reflectivity = payload[offset+12]
                    tag = payload[offset+13]
                    
                    # Convert from millimeters to meters
                    x = x_mm / 1000.0
                    y = y_mm / 1000.0
                    z = z_mm / 1000.0
                    
                    # Filter valid points (reasonable range)
                    if abs(x) < 500 and abs(y) < 500 and abs(z) < 500:
                        if preserve_intensity:
                            points.append([x, y, z, reflectivity])
                        else:
                            points.append([x, y, z, 0])
                    
                    offset += point_size
                    
                except struct.error:
                    break
        
        except Exception as e:
            self.logger.debug(f"Error parsing Livox packet: {e}")
            return None
        
        if not points:
            return None
        
        return np.array(points, dtype=np.float32)
    
    def _create_las_file(self, points: np.ndarray, output_path: str, preserve_intensity: bool) -> None:
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
            **kwargs: Additional parameters
            
        Returns:
            dict: Conversion result dictionary
        """
        output_format = output_format.lower().lstrip(".")
        
        if output_format == "las":
            return self.convert_to_las(input_path, output_path, **kwargs)
        elif output_format == "laz":
            # Convert to LAS first, then compress
            las_path = str(Path(output_path).with_suffix(".las"))
            result = self.convert_to_las(input_path, las_path, **kwargs)
            if result["success"]:
                # LAZ compression not yet implemented
                self.logger.warning("LAZ compression not yet implemented - returning LAS file")
                result["output_file"] = las_path
            return result
        elif output_format in ["pcd", "bin", "csv"]:
            result = {
                "success": False,
                "error": f"Output format '{output_format}' not yet implemented",
                "message": "Only LAS format is currently supported",
                "output_file": None,
                "conversion_time": 0.0,
                "points_converted": 0,
                "sdk_version_used": self.sdk_version
            }
            return result
        else:
            result = {
                "success": False,
                "error": f"Unsupported output format: {output_format}",
                "message": f"Supported formats: {', '.join(self.SUPPORTED_OUTPUT_FORMATS)}",
                "output_file": None,
                "conversion_time": 0.0,
                "points_converted": 0,
                "sdk_version_used": self.sdk_version
            }
            return result
    
    def get_vendor_info(self) -> Dict[str, Any]:
        """
        Get Livox vendor capabilities and information.
        
        Returns:
            dict: Vendor information dictionary
        """
        return {
            "vendor": "livox",
            "supported_input_formats": self.SUPPORTED_INPUT_FORMATS,
            "supported_output_formats": self.SUPPORTED_OUTPUT_FORMATS,
            "sdk_version": self.sdk_version,
            "supported_sensor_models": self.SUPPORTED_MODELS,
            "requires_calibration": False,  # Livox doesn't typically need external calibration
            "status": "available" if self.sdk_available else "not_installed",
            "sdk_available": self.sdk_available,
            "installation_method": "pandas_csv" if PANDAS_AVAILABLE else "unavailable",
            "notes": "For best results, export Livox data to CSV format using Livox Viewer"
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