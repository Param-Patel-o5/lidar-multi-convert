#!/usr/bin/env python3
"""
LiDAR Converter Module - Orchestrates vendor detection and conversion.

This module provides the main conversion pipeline that combines:
- VendorDetector: Identifies the LiDAR vendor from file analysis
- Vendor Wrappers: Handle vendor-specific conversion logic

The converter provides a unified interface for converting LiDAR files
from any supported vendor to standardized formats (LAS, LAZ, PCD, etc.).

Workflow:
    1. User calls converter.convert("file.pcap", "output.las")
    2. Converter detects vendor using VendorDetector
    3. Converter instantiates appropriate wrapper (e.g., OusterWrapper)
    4. Converter calls wrapper's conversion method
    5. Converter validates output and returns comprehensive result

Example:
    from converter import LiDARConverter
    
    converter = LiDARConverter()
    result = converter.convert("data.pcap", "output.las")
    if result["success"]:
        print(f"Converted {result['points_converted']} points")
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import logging

from .detector import VendorDetector
from .Wrappers import OusterWrapper, VelodyneWrapper, LivoxWrapper, HesaiWrapper, BaseVendorWrapper

logger = logging.getLogger(__name__)


class LiDARConverter:
    """
    Main orchestrator for LiDAR vendor detection and conversion.
    
    This class manages the complete conversion pipeline:
    - Automatic vendor detection
    - Wrapper instantiation and management
    - Format validation and conversion
    - Error handling and logging
    - Batch processing
    
    The converter is designed to be extensible - adding new vendors
    only requires:
    1. Creating a wrapper class (inheriting from BaseVendorWrapper)
    2. Registering it in the wrapper registry
    
    Example:
        converter = LiDARConverter()
        
        # Single file conversion
        result = converter.convert("file.pcap", "output.las")
        
        # Batch conversion
        results = converter.convert_batch(
            ["file1.pcap", "file2.pcap"],
            output_dir="output/"
        )
    """
    
    # Wrapper registry: maps vendor name to wrapper class
    WRAPPER_REGISTRY = {
        "ouster": OusterWrapper,
        "velodyne": VelodyneWrapper,
        "livox": LivoxWrapper,
        "hesai": HesaiWrapper,
        # Future vendors will be added here:
        # "riegl": RieglWrapper,
        # "sick": SickWrapper,
    }
    
    def __init__(self, enable_detection_cache: bool = False):
        """
        Initialize LiDAR converter.
        
        Args:
            enable_detection_cache: If True, enable caching in VendorDetector
        """
        self.detector = VendorDetector(enable_cache=enable_detection_cache)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Track wrapper instances (for reuse if stateless)
        self._wrapper_instances = {}
        
        self.logger.info("LiDARConverter initialized")
    
    def register_wrapper(self, vendor_name: str, wrapper_class: type) -> bool:
        """
        Dynamically register a new vendor wrapper.
        
        This allows plugin-style architecture where wrappers can be
        added at runtime without modifying converter code.
        
        Args:
            vendor_name: Vendor identifier (lowercase, e.g., "velodyne")
            wrapper_class: Wrapper class (must inherit from BaseVendorWrapper)
            
        Returns:
            bool: True if registered successfully, False otherwise
        """
        if not issubclass(wrapper_class, BaseVendorWrapper):
            self.logger.error(f"Wrapper class must inherit from BaseVendorWrapper")
            return False
        
        self.WRAPPER_REGISTRY[vendor_name.lower()] = wrapper_class
        self.logger.info(f"Registered wrapper for vendor: {vendor_name}")
        return True
    
    def get_wrapper(self, vendor_name: str) -> Optional[BaseVendorWrapper]:
        """
        Get or instantiate wrapper for a vendor.
        
        Args:
            vendor_name: Vendor identifier (lowercase)
            
        Returns:
            BaseVendorWrapper instance or None if vendor not supported
        """
        vendor_name = vendor_name.lower()
        
        if vendor_name not in self.WRAPPER_REGISTRY:
            self.logger.error(f"Vendor not supported: {vendor_name}")
            return None
        
        # Return cached instance if available (wrappers should be stateless)
        if vendor_name in self._wrapper_instances:
            return self._wrapper_instances[vendor_name]
        
        # Instantiate wrapper
        try:
            wrapper_class = self.WRAPPER_REGISTRY[vendor_name]
            wrapper = wrapper_class()
            self._wrapper_instances[vendor_name] = wrapper
            self.logger.debug(f"Instantiated wrapper for: {vendor_name}")
            return wrapper
        except Exception as e:
            self.logger.error(f"Failed to instantiate wrapper for {vendor_name}: {e}")
            return None
    
    def convert(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        output_format: str = "las",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Convert LiDAR file to specified output format.
        
        This is the main conversion method that orchestrates the entire
        workflow: detection → wrapper selection → conversion → validation.
        
        Args:
            input_path: Path to input LiDAR file
            output_path: Optional output file path (auto-generated if not provided)
            output_format: Output format ("las", "laz", "pcd", etc.)
            **kwargs: Additional parameters passed to wrapper:
                - sensor_model: Sensor model identifier
                - calibration_file: Path to calibration/metadata file
                - preserve_intensity: Whether to preserve intensity values
                - max_scans: Limit on number of scans to process
                - Any vendor-specific parameters
        
        Returns:
            dict: Comprehensive conversion result containing:
                - "success": bool - Whether conversion succeeded
                - "vendor": str - Detected vendor name
                - "input_file": str - Input file path
                - "output_file": str - Output file path
                - "output_format": str - Output format used
                - "conversion_time": float - Total time in seconds
                - "detection_time": float - Time spent on detection
                - "detection_confidence": float - Vendor detection confidence
                - "points_converted": int - Number of points converted
                - "message": str - Success or error description
                - "errors": List[str] - List of errors if any
                - "warnings": List[str] - List of warnings if any
        """
        start_time = time.time()
        
        # Initialize result dict
        result = {
            "success": False,
            "vendor": None,
            "input_file": str(input_path),
            "output_file": None,
            "output_format": output_format,
            "conversion_time": 0.0,
            "detection_time": 0.0,
            "detection_confidence": 0.0,
            "points_converted": 0,
            "message": "",
            "errors": [],
            "warnings": []
        }
        
        try:
            # Step 1: Pre-conversion validation
            validation_result = self._validate_inputs(input_path, output_path, output_format)
            if not validation_result["valid"]:
                result["errors"].append(validation_result["error"])
                result["message"] = f"Validation failed: {validation_result['error']}"
                self.logger.error(result["message"])
                return result
            
            # Generate output path if not provided
            if output_path is None:
                output_path = self._generate_output_path(input_path, output_format)
            
            result["output_file"] = str(output_path)
            
            # Step 2: Detect vendor
            self.logger.info(f"Detecting vendor for: {input_path}")
            detection_start = time.time()
            detection_result = self.detector.detect_vendor(input_path)
            detection_time = time.time() - detection_start
            result["detection_time"] = detection_time
            result["detection_confidence"] = detection_result.get("confidence", 0.0)
            
            if not detection_result.get("success", False):
                error_msg = detection_result.get("error", "Vendor detection failed")
                result["errors"].append(error_msg)
                result["message"] = f"Cannot convert: {error_msg}"
                self.logger.error(result["message"])
                return result
            
            vendor_name = detection_result.get("vendor_name")
            result["vendor"] = vendor_name
            
            self.logger.info(f"Detected vendor: {vendor_name} (confidence: {result['detection_confidence']:.2%})")
            
            # Step 3: Get wrapper
            wrapper = self.get_wrapper(vendor_name)
            if wrapper is None:
                error_msg = f"Vendor '{vendor_name}' is not yet supported"
                result["errors"].append(error_msg)
                result["message"] = error_msg
                self.logger.error(result["message"])
                return result
            
            # Check if wrapper SDK is available
            if not wrapper.sdk_available:
                error_msg = f"SDK for {vendor_name} is not installed"
                result["errors"].append(error_msg)
                result["message"] = f"Cannot convert: {error_msg}"
                self.logger.error(result["message"])
                return result
            
            # Step 4: Validate output format is supported by wrapper
            vendor_info = wrapper.get_vendor_info()
            supported_formats = vendor_info.get("supported_output_formats", [])
            format_ext = f".{output_format}"
            
            if format_ext not in supported_formats:
                error_msg = f"Format '{output_format}' not supported by {vendor_name} wrapper"
                result["errors"].append(error_msg)
                result["message"] = f"Cannot convert: {error_msg}. Supported formats: {', '.join(supported_formats)}"
                self.logger.error(result["message"])
                return result
            
            # Step 5: Perform conversion
            self.logger.info(f"Starting conversion: {input_path} -> {output_path}")
            conversion_result = None
            
            if output_format == "las":
                conversion_result = wrapper.convert_to_las(
                    input_path,
                    output_path,
                    **kwargs
                )
            else:
                conversion_result = wrapper.convert(
                    input_path,
                    output_format,
                    output_path,
                    **kwargs
                )
            
            # Step 6: Process conversion result
            if conversion_result.get("success", False):
                result["success"] = True
                result["points_converted"] = conversion_result.get("points_converted", 0)
                result["message"] = f"Successfully converted {vendor_name} file to {output_format}"
                
                # Optionally validate output
                if kwargs.get("validate_output", True):
                    validation = wrapper.validate_conversion(input_path, output_path)
                    if not validation:
                        result["warnings"].append("Output validation failed - file may be corrupted")
                
                self.logger.info(
                    f"Conversion completed: {result['points_converted']} points in {time.time() - start_time:.2f}s"
                )
            else:
                error_msg = conversion_result.get("error", "Conversion failed")
                result["errors"].append(error_msg)
                result["message"] = f"Conversion failed: {error_msg}"
                self.logger.error(result["message"])
        
        except Exception as e:
            error_msg = f"Unexpected error during conversion: {e}"
            result["errors"].append(error_msg)
            result["message"] = error_msg
            self.logger.exception(f"Exception during conversion: {e}")
        
        finally:
            result["conversion_time"] = time.time() - start_time
        
        return result
    
    def convert_batch(
        self,
        file_paths: List[str],
        output_dir: Optional[str] = None,
        output_format: str = "las",
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Convert multiple files in batch.
        
        Args:
            file_paths: List of input file paths
            output_dir: Optional output directory (auto-generated if not provided)
            output_format: Output format for all files
            progress_callback: Optional callback function(current, total, file_path)
            **kwargs: Additional parameters passed to convert()
            
        Returns:
            list: List of conversion result dicts (one per file)
        """
        results = []
        total_files = len(file_paths)
        
        self.logger.info(f"Starting batch conversion: {total_files} files")
        
        # Generate output directory if not provided
        if output_dir is None:
            output_dir = Path.cwd() / "converted_output"
            output_dir.mkdir(parents=True, exist_ok=True)
        
        output_dir = Path(output_dir)
        
        for idx, input_path in enumerate(file_paths, 1):
            if progress_callback:
                progress_callback(idx, total_files, input_path)
            
            # Generate output path for this file
            input_path_obj = Path(input_path)
            output_filename = f"{input_path_obj.stem}.{output_format}"
            output_path = output_dir / output_filename
            
            # Convert file
            result = self.convert(
                input_path,
                str(output_path),
                output_format,
                **kwargs
            )
            
            results.append(result)
            
            # Log progress
            if result["success"]:
                self.logger.info(
                    f"[{idx}/{total_files}] ✓ {input_path_obj.name} -> {result['points_converted']} points"
                )
            else:
                self.logger.warning(
                    f"[{idx}/{total_files}] ✗ {input_path_obj.name} - {result['message']}"
                )
        
        # Summary
        successful = sum(1 for r in results if r["success"])
        self.logger.info(f"Batch conversion complete: {successful}/{total_files} successful")
        
        return results
    
    def test_pipeline(self, input_file: str) -> Dict[str, Any]:
        """
        End-to-end test on a single file (detects, converts, validates).
        
        Args:
            input_file: Path to test file
            
        Returns:
            dict: Test result with detection, conversion, and validation status
        """
        result = {
            "success": False,
            "detection": {},
            "conversion": {},
            "validation": False,
            "message": ""
        }
        
        # Step 1: Detection test
        self.logger.info(f"Testing detection for: {input_file}")
        detection_result = self.detector.detect_vendor(input_file)
        result["detection"] = detection_result
        
        if not detection_result.get("success", False):
            result["message"] = "Detection test failed"
            return result
        
        # Step 2: Conversion test
        vendor = detection_result.get("vendor_name")
        wrapper = self.get_wrapper(vendor)
        
        if wrapper is None:
            result["message"] = f"Wrapper not available for vendor: {vendor}"
            return result
        
        # Use wrapper's test_conversion method
        test_output = Path(input_file).with_suffix(".las")
        test_result = wrapper.test_conversion(input_file, str(test_output))
        result["conversion"] = test_result
        
        # Step 3: Overall success
        result["success"] = (
            detection_result.get("success", False) and
            test_result.get("success", False)
        )
        result["validation"] = test_result.get("validation_result", False)
        
        if result["success"]:
            result["message"] = "Pipeline test passed"
        else:
            result["message"] = "Pipeline test failed"
        
        return result
    
    def health_check(self) -> Dict[str, Any]:
        """
        Verify all wrappers are installed and working.
        
        Returns:
            dict: Health status for each registered vendor
        """
        health = {
            "status": "ok",
            "vendors": {}
        }
        
        for vendor_name, wrapper_class in self.WRAPPER_REGISTRY.items():
            try:
                wrapper = wrapper_class()
                vendor_info = wrapper.get_vendor_info()
                
                health["vendors"][vendor_name] = {
                    "available": wrapper.sdk_available,
                    "status": vendor_info.get("status", "unknown"),
                    "sdk_version": vendor_info.get("sdk_version"),
                    "supported_formats": vendor_info.get("supported_output_formats", [])
                }
                
                if not wrapper.sdk_available:
                    health["status"] = "degraded"
                    
            except Exception as e:
                health["vendors"][vendor_name] = {
                    "available": False,
                    "status": "error",
                    "error": str(e)
                }
                health["status"] = "error"
        
        return health
    
    def _validate_inputs(
        self,
        input_path: str,
        output_path: Optional[str],
        output_format: str
    ) -> Dict[str, Any]:
        """Validate input parameters before conversion."""
        # Validate input file
        input_path_obj = Path(input_path)
        if not input_path_obj.exists():
            return {"valid": False, "error": f"Input file not found: {input_path}"}
        
        if not input_path_obj.is_file():
            return {"valid": False, "error": f"Input path is not a file: {input_path}"}
        
        if input_path_obj.stat().st_size == 0:
            return {"valid": False, "error": "Input file is empty"}
        
        # Validate output format
        valid_formats = ["las", "laz", "pcd", "bin", "csv"]
        if output_format.lower() not in valid_formats:
            return {"valid": False, "error": f"Unsupported output format: {output_format}"}
        
        # Validate output path if provided
        if output_path:
            output_path_obj = Path(output_path)
            output_dir = output_path_obj.parent
            
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                return {"valid": False, "error": f"Cannot create output directory: {e}"}
            
            if not output_dir.is_dir():
                return {"valid": False, "error": f"Output directory is not valid: {output_dir}"}
        
        return {"valid": True}
    
    def _generate_output_path(self, input_path: str, output_format: str) -> Path:
        """Generate output path from input path and format."""
        input_path_obj = Path(input_path)
        output_path = input_path_obj.with_suffix(f".{output_format}")
        return output_path

