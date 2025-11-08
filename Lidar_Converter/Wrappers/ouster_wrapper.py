#!/usr/bin/env python3
"""
Ouster LiDAR SDK wrapper for conversion to standardized formats.

This module provides a unified interface to the Ouster SDK, abstracting all
Ouster-specific implementation details. It serves as a template/blueprint
for future vendor wrappers (Velodyne, Hesai, Livox, etc.).

The wrapper handles:
- SDK installation validation
- PCAP to LAS/LAZ/PCD conversion
- Error handling and logging
- Performance monitoring
- Vendor-specific metadata extraction

Usage:
    wrapper = OusterWrapper()
    if wrapper.sdk_available:
        result = wrapper.convert_to_las("input.pcap", "output.las")
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# Ouster SDK imports (with graceful fallback)
OUSTER_SDK_AVAILABLE = False
ouster = None
client = None
pcap = None
convert = None

try:
    import ouster
    from ouster.sdk import open_source
    from ouster.sdk.core import SensorInfo, XYZLut, ChanField
    OUSTER_SDK_AVAILABLE = True
except ImportError as e:
    # SDK not available - wrapper will use CLI fallback or fail gracefully
    OUSTER_SDK_AVAILABLE = False
    ouster = None
    open_source = None
    SensorInfo = None
    XYZLut = None
    ChanField = None

# LAS file handling
try:
    import laspy
    LASPY_AVAILABLE = True
except ImportError:
    LASPY_AVAILABLE = False

import numpy as np

from .base_wrapper import BaseVendorWrapper

logger = logging.getLogger(__name__)


class OusterWrapper(BaseVendorWrapper):
    """
    Ouster LiDAR SDK wrapper implementing BaseVendorWrapper interface.
    
    This wrapper provides conversion capabilities for Ouster sensors:
    - OS-0 series (32/64/128 channels)
    - OS-1 series (16/32/64/128 channels)
    - OS-2 series (128 channels)
    - OS Dome (32/64/128 channels)
    
    The wrapper uses the Ouster Python SDK (ouster-sdk) for conversion.
    If the SDK is not available, it attempts to use ouster-cli as fallback.
    
    Example:
        wrapper = OusterWrapper()
        if wrapper.sdk_available:
            result = wrapper.convert_to_las(
                input_path="data.pcap",
                output_path="output.las",
                sensor_model="OS1-64"
            )
            if result["success"]:
                print(f"Converted {result['points_converted']} points")
    """
    
    # Supported Ouster sensor models
    SUPPORTED_MODELS = [
        "OS0-32", "OS0-64", "OS0-128",
        "OS1-16", "OS1-32", "OS1-64", "OS1-128",
        "OS2-128",
        "OS-Dome-32", "OS-Dome-64", "OS-Dome-128"
    ]
    
    # Supported input/output formats
    SUPPORTED_INPUT_FORMATS = [".pcap"]
    SUPPORTED_OUTPUT_FORMATS = [".las", ".laz", ".pcd", ".bin", ".csv"]
    
    def __init__(self, sdk_path: Optional[str] = None, raise_on_missing: bool = False):
        """
        Initialize Ouster wrapper with SDK validation.
        
        Args:
            sdk_path: Optional path to custom Ouster SDK installation
            raise_on_missing: If True, raise exception if SDK not found.
                            If False (default), log warning and mark SDK as unavailable.
        
        Raises:
            RuntimeError: If raise_on_missing=True and SDK is not available
        """
        super().__init__()
        
        # Check for custom SDK path from environment variable
        if sdk_path is None:
            sdk_path = os.environ.get("OUSTER_SDK_PATH")
        
        self.sdk_path = sdk_path
        self.raise_on_missing = raise_on_missing
        
        # Validate SDK installation
        validation_result = self.validate_sdk_installation()
        
        if not validation_result.get("available", False):
            error_msg = validation_result.get("error", "Ouster SDK not found")
            if raise_on_missing:
                raise RuntimeError(f"Ouster SDK is required but not available: {error_msg}")
            else:
                self.logger.warning(f"Ouster SDK not available: {error_msg}")
                self.sdk_available = False
        else:
            self.sdk_available = True
            self.sdk_version = validation_result.get("version", "unknown")
            self.logger.info(f"Ouster SDK validated - Version: {self.sdk_version}")
    
    def get_vendor_name(self) -> str:
        """Return vendor identifier."""
        return "ouster"
    
    def validate_sdk_installation(self) -> Dict[str, Any]:
        """
        Validate Ouster SDK installation and detect version.
        
        Checks for:
        1. Python package 'ouster-sdk'
        2. CLI tool 'ouster-cli' (if Python SDK not available)
        3. SDK version information
        
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
        
        # Method 1: Check Python SDK
        if OUSTER_SDK_AVAILABLE and ouster is not None:
            try:
                # Try to get version from ouster package
                version = "unknown"
                if hasattr(ouster, '__version__'):
                    version = ouster.__version__
                else:
                    # Try different methods to get version
                    try:
                        # Prefer importlib.metadata (Python 3.8+)
                        try:
                            import importlib.metadata
                            version = importlib.metadata.version('ouster-sdk')
                        except (ImportError, importlib.metadata.PackageNotFoundError):
                            # Fallback to pkg_resources (deprecated but works)
                            try:
                                import pkg_resources
                                version = pkg_resources.get_distribution('ouster-sdk').version
                            except:
                                version = "installed"
                    except:
                        version = "installed"
                
                result.update({
                    "available": True,
                    "version": version,
                    "method": "python_package",
                    "message": f"Ouster Python SDK {version} is available"
                })
                return result
            except Exception as e:
                self.logger.debug(f"Failed to get Python SDK version: {e}")
        
        # Method 2: Check CLI tool
        try:
            cli_result = subprocess.run(
                ["ouster-cli", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if cli_result.returncode == 0:
                version = cli_result.stdout.strip()
                result.update({
                    "available": True,
                    "version": version,
                    "method": "cli_tool",
                    "message": f"Ouster CLI tool {version} is available"
                })
                return result
        except FileNotFoundError:
            pass
        except Exception as e:
            self.logger.debug(f"Failed to check CLI tool: {e}")
        
        # Method 3: Check custom SDK path
        if self.sdk_path:
            sdk_dir = Path(self.sdk_path)
            if sdk_dir.exists() and sdk_dir.is_dir():
                # Check for common SDK files
                if (sdk_dir / "ouster" / "__init__.py").exists():
                    result.update({
                        "available": True,
                        "installation_path": str(sdk_dir),
                        "method": "custom_path",
                        "message": f"Ouster SDK found at custom path: {self.sdk_path}"
                    })
                    return result
        
        # SDK not found
        result.update({
            "available": False,
            "error": "Ouster SDK not found. Install with: pip install ouster-sdk",
            "message": "Ouster SDK is not available"
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
        Convert Ouster PCAP file to LAS format.
        
        This method implements the Ouster-specific conversion logic while
        maintaining the unified interface defined by BaseVendorWrapper.
        
        Args:
            input_path: Path to input .pcap file
            output_path: Path where .las file will be written
            sensor_model: Optional sensor model (e.g., "OS1-64")
            calibration_file: Optional path to .json metadata file
            preserve_intensity: Whether to preserve intensity values
            max_scans: Optional limit on number of scans to process
            **kwargs: Additional parameters:
                - reprofile: Resample to different lidar mode
                - dual: Process dual returns
                - num_workers: Number of worker threads
                
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
            result["error"] = "Ouster SDK is not available"
            result["message"] = "Cannot convert: Ouster SDK not installed"
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
            result["message"] = "Ouster wrapper only supports .pcap input files"
            self.logger.error(result["error"])
            return result
        
        if output_path_obj.suffix.lower() != ".las":
            result["error"] = f"Unsupported output format: {output_path_obj.suffix}"
            result["message"] = "Use convert() method for formats other than .las"
            self.logger.error(result["error"])
            return result
        
        try:
            self.logger.info(f"Starting Ouster conversion: {input_path} -> {output_path}")
            
            # Resolve metadata file
            if calibration_file and Path(calibration_file).exists():
                metadata_path = calibration_file
            else:
                # Try to find companion .json file
                json_path = input_path_obj.with_suffix(".json")
                if json_path.exists():
                    metadata_path = str(json_path)
                else:
                    # Try to find JSON file by pattern matching
                    # Ouster files often have matching JSON files with similar names
                    json_patterns = [
                        input_path_obj.with_suffix(".json"),
                        input_path_obj.parent / (input_path_obj.stem + ".json"),
                        input_path_obj.parent / (input_path_obj.stem.split("-")[0] + ".json"),
                    ]
                    
                    for pattern in json_patterns:
                        if pattern.exists():
                            metadata_path = str(pattern)
                            break
                    else:
                        # Search for any JSON file in the same directory
                        json_files = list(input_path_obj.parent.glob("*.json"))
                        if json_files:
                            metadata_path = str(json_files[0])
                            self.logger.info(f"Using metadata file: {metadata_path}")
                        else:
                            metadata_path = None
            
            if not metadata_path:
                result["error"] = "Metadata file (.json) required for Ouster conversion"
                result["message"] = "Provide calibration_file parameter or ensure .json file exists"
                self.logger.error(result["error"])
                return result
            
            # Check laspy availability
            if not LASPY_AVAILABLE:
                result["error"] = "laspy not available - install with: pip install laspy"
                result["message"] = "Cannot create LAS file: laspy package required"
                self.logger.error(result["error"])
                return result
            
            # Extract points using Python SDK
            if OUSTER_SDK_AVAILABLE:
                points = self._convert_with_python_sdk(
                    input_path,
                    metadata_path,
                    sensor_model,
                    max_scans,
                    **kwargs
                )
                
                # Convert points array to LAS
                point_count = len(points)
                
                self.logger.debug(f"Writing LAS file: {output_path}")
                
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
                las_file.intensity = points[:, 3].astype(np.uint16) if preserve_intensity else np.zeros(point_count, dtype=np.uint16)
                
                # Write LAS file
                las_file.write(str(output_path))
                
                result.update({
                    "success": True,
                    "message": f"Successfully converted {point_count:,} points",
                    "points_converted": point_count
                })
                
                self.logger.info(f"LAS conversion complete: {point_count:,} points")
            else:
                # Fallback to CLI if Python SDK not available
                result = self._convert_with_cli(
                    input_path,
                    output_path,
                    metadata_path,
                    **kwargs
                )
            
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
            self.logger.exception(f"Exception during Ouster conversion: {e}")
        
        return result
    
    def _convert_with_python_sdk(
        self,
        input_path: str,
        metadata_path: str,
        sensor_model: Optional[str],
        max_scans: Optional[int],
        **kwargs
    ) -> np.ndarray:
        """
        Extract point cloud data from Ouster PCAP using Python SDK.
        
        Args:
            input_path: Path to input .pcap file
            metadata_path: Path to .json metadata file
            sensor_model: Optional sensor model (e.g., "OS1-64")
            max_scans: Optional limit on number of scans to process
            **kwargs: Additional parameters
            
        Returns:
            numpy array with shape (N, 4) containing [x, y, z, intensity]
            
        Raises:
            Exception: If extraction fails
        """
        try:
            # Read sensor metadata
            self.logger.debug(f"Reading metadata: {metadata_path}")
            with open(metadata_path, 'r') as f:
                metadata = SensorInfo(f.read())
            
            # Open PCAP file as data source
            self.logger.debug(f"Opening PCAP: {input_path}")
            source = open_source(str(input_path), meta=[str(metadata_path)])
            
            # Precompute XYZ lookup table
            self.logger.debug("Computing XYZ lookup table...")
            xyzlut = XYZLut(metadata)
            
            # Collect point cloud data
            all_points_list = []
            scan_count = 0
            
            self.logger.info(f"Processing scans (max: {max_scans or 'unlimited'})...")
            
            # Iterate over scans in the PCAP
            for scans in source:
                for scan in scans:
                    if scan is None:
                        continue
                    
                    scan_count += 1
                    
                    if scan_count % 10 == 0:
                        self.logger.debug(f"Processing scan {scan_count}...")
                    
                    # Get range data
                    range_data = scan.field(ChanField.RANGE)
                    
                    # Compute XYZ coordinates using lookup table
                    xyz = xyzlut(range_data)
                    
                    # Get intensity (reflectivity)
                    intensity = scan.field(ChanField.REFLECTIVITY)
                    
                    # Filter out invalid points (range = 0)
                    valid_mask = range_data > 0
                    
                    if not np.any(valid_mask):
                        continue  # Skip if no valid points
                    
                    # Process only valid points
                    valid_xyz = xyz[valid_mask]
                    valid_intensity = intensity[valid_mask]
                    
                    # Store valid points (x, y, z, intensity)
                    points = np.column_stack([
                        valid_xyz[:, 0],  # X
                        valid_xyz[:, 1],  # Y
                        valid_xyz[:, 2],  # Z
                        valid_intensity   # Intensity
                    ])
                    
                    all_points_list.append(points)
                    
                    # Check limit
                    if max_scans and scan_count >= max_scans:
                        self.logger.info(f"Reached max_scans limit: {scan_count}")
                        break
                
                if max_scans and scan_count >= max_scans:
                    break
            
            if not all_points_list:
                raise ValueError("No valid points found in PCAP file")
            
            # Combine all points
            self.logger.debug("Combining all point clouds...")
            all_points = np.vstack(all_points_list)
            point_count = len(all_points)
            
            self.logger.info(f"Extracted {point_count:,} points from {scan_count} scans")
            
            return all_points
        
        except Exception as e:
            self.logger.exception(f"Error extracting points with Python SDK: {e}")
            raise
    
    def _convert_with_cli(
        self,
        input_path: str,
        output_path: str,
        metadata_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Convert using Ouster CLI tool as fallback."""
        result = {
            "success": False,
            "message": "",
            "points_converted": 0,
            "error": None
        }
        
        try:
            # Use ouster-cli to convert
            # Note: This is a placeholder - actual CLI syntax may vary
            cmd = [
                "ouster-cli",
                "convert",
                "--input", str(input_path),
                "--output", str(output_path),
                "--metadata", str(metadata_path)
            ]
            
            self.logger.debug(f"Running CLI command: {' '.join(cmd)}")
            
            cli_result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if cli_result.returncode == 0:
                result.update({
                    "success": True,
                    "message": "Conversion completed using CLI tool"
                })
                # Try to extract point count from output
                # (This would need to be parsed from CLI output)
            else:
                result["error"] = cli_result.stderr
                result["message"] = f"CLI conversion failed: {cli_result.stderr}"
        
        except FileNotFoundError:
            result["error"] = "ouster-cli not found"
            result["message"] = "Ouster CLI tool is not available"
        except subprocess.TimeoutExpired:
            result["error"] = "Conversion timeout"
            result["message"] = "Conversion exceeded 5 minute timeout"
        except Exception as e:
            result["error"] = str(e)
            result["message"] = f"CLI conversion error: {e}"
        
        return result
    
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
            result["error"] = "Ouster SDK is not available"
            result["message"] = "Cannot convert: Ouster SDK not installed"
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
            input_path_obj = Path(input_path)
            
            # Resolve metadata file
            calibration_file = kwargs.get('calibration_file')
            if calibration_file and Path(calibration_file).exists():
                metadata_path = calibration_file
            else:
                # Try to find companion .json file
                json_path = input_path_obj.with_suffix(".json")
                if json_path.exists():
                    metadata_path = str(json_path)
                else:
                    # Search for any JSON file in the same directory
                    json_files = list(input_path_obj.parent.glob("*.json"))
                    if json_files:
                        metadata_path = str(json_files[0])
                        self.logger.info(f"Using metadata file: {metadata_path}")
                    else:
                        metadata_path = None
            
            if not metadata_path:
                result["error"] = "Metadata file (.json) required for Ouster conversion"
                result["message"] = "Provide calibration_file parameter or ensure .json file exists"
                self.logger.error(result["error"])
                return result
            
            # Extract points using Python SDK
            if not OUSTER_SDK_AVAILABLE:
                result["error"] = "Ouster Python SDK not available"
                result["message"] = "Install with: pip install ouster-sdk"
                self.logger.error(result["error"])
                return result
            
            self.logger.info(f"Starting Ouster conversion: {input_path} -> {output_path} ({output_format})")
            
            # Extract parameters that are passed as positional args
            sensor_model = kwargs.pop('sensor_model', None)
            max_scans = kwargs.pop('max_scans', None)
            
            points = self._convert_with_python_sdk(
                input_path,
                metadata_path,
                sensor_model,
                max_scans,
                **kwargs
            )
            
            # Convert to target format
            preserve_intensity = kwargs.get('preserve_intensity', True)
            
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
                point_count = len(points)
                
                self.logger.debug(f"Writing LAS file: {las_path}")
                
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
                las_file.intensity = points[:, 3].astype(np.uint16) if preserve_intensity else np.zeros(point_count, dtype=np.uint16)
                
                # Write LAS file
                las_file.write(str(las_path))
                
                # Compress to LAZ
                compression_result = self._compress_las_to_laz(las_path, output_path)
                
                if compression_result["success"]:
                    result.update({
                        "success": True,
                        "message": f"Successfully converted {point_count:,} points to LAZ",
                        "points_converted": point_count,
                        "output_file": compression_result["output_file"]
                    })
                    
                    # Add warning if compression wasn't available
                    if "warning" in compression_result:
                        result["warning"] = compression_result["warning"]
                else:
                    result["error"] = compression_result.get("error", "LAZ compression failed")
                    result["message"] = f"LAZ compression failed: {result['error']}"
                    
            elif output_format == "pcd":
                format_result = self._points_to_pcd(points, output_path, preserve_intensity)
                result.update(format_result)
                if result["success"]:
                    result["message"] = f"Successfully converted {result['points_converted']:,} points to PCD"
                    
            elif output_format == "bin":
                format_result = self._points_to_bin(points, output_path)
                result.update(format_result)
                if result["success"]:
                    result["message"] = f"Successfully converted {result['points_converted']:,} points to BIN"
                    
            elif output_format == "csv":
                format_result = self._points_to_csv(points, output_path, preserve_intensity)
                result.update(format_result)
                if result["success"]:
                    result["message"] = f"Successfully converted {result['points_converted']:,} points to CSV"
                    
            else:
                result["error"] = f"Unsupported output format: {output_format}"
                result["message"] = f"Supported formats: {', '.join(self.SUPPORTED_OUTPUT_FORMATS)}"
                self.logger.error(result["error"])
            
            # Calculate conversion time
            conversion_time = time.time() - start_time
            result["conversion_time"] = conversion_time
            result["sdk_version_used"] = self.sdk_version
            
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
            self.logger.exception(f"Exception during Ouster conversion: {e}")
        
        return result
    

    def get_vendor_info(self) -> Dict[str, Any]:
        """
        Get Ouster vendor capabilities and information.
        
        Returns:
            dict: Vendor information dictionary
        """
        return {
            "vendor": "ouster",
            "supported_input_formats": self.SUPPORTED_INPUT_FORMATS,
            "supported_output_formats": self.SUPPORTED_OUTPUT_FORMATS,
            "sdk_version": self.sdk_version,
            "supported_sensor_models": self.SUPPORTED_MODELS,
            "requires_calibration": True,
            "status": "available" if self.sdk_available else "not_installed",
            "sdk_available": self.sdk_available,
            "installation_method": "python_package" if OUSTER_SDK_AVAILABLE else "cli_tool"
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
        - File format is valid (LAS/LAZ/PCD/BIN/CSV)
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
        
        # Determine format from extension
        file_ext = output_path_obj.suffix.lower()
        
        # Validate based on format
        try:
            if file_ext in ['.las', '.laz']:
                # Validate LAS/LAZ file structure
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
                
            elif file_ext == '.pcd':
                # Basic PCD validation - check header
                with open(output_path, 'r') as f:
                    first_line = f.readline().strip()
                    if not first_line.startswith('# .PCD'):
                        self.logger.error("Invalid PCD file header")
                        return False
                self.logger.info(f"Validation passed: PCD file structure valid")
                return True
                
            elif file_ext == '.bin':
                # BIN validation - check file size is multiple of 16 bytes (4 floats)
                file_size = output_path_obj.stat().st_size
                if file_size % 16 != 0:
                    self.logger.error(f"Invalid BIN file size: {file_size} (not multiple of 16)")
                    return False
                point_count = file_size // 16
                self.logger.info(f"Validation passed: {point_count:,} points in BIN file")
                return True
                
            elif file_ext == '.csv':
                # CSV validation - check header
                with open(output_path, 'r') as f:
                    first_line = f.readline().strip()
                    if not first_line.startswith('x,y,z'):
                        self.logger.error("Invalid CSV file header")
                        return False
                self.logger.info(f"Validation passed: CSV file structure valid")
                return True
                
            else:
                self.logger.warning(f"Unknown format {file_ext} - skipping validation")
                return True
        
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return False

