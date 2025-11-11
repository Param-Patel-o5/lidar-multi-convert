"""
LiDAR Converter - Multi-vendor LiDAR data conversion library.

This package provides automatic conversion of raw LiDAR sensor data from various
manufacturers (Ouster, Velodyne, Livox, Hesai) into standardized formats
(LAS, LAZ, PCD, BIN, CSV).

Main Features:
- Automatic vendor detection with confidence scoring
- Multi-format output support (LAS, LAZ, PCD, BIN, CSV)
- Batch processing capabilities
- CLI and Python API
- Support for 4 major LiDAR vendors

Supported Vendors:
- Ouster (OS-0, OS-1, OS-2, OS-Dome)
- Velodyne (VLP-16, VLP-32C, HDL-32E, HDL-64E, VLS-128)
- Livox (Avia, Horizon, Tele-15, Mid-40/70, HAP)
- Hesai (PandarXT-32, PandarXT-16, Pandar64, Pandar40P, PandarQT)

Basic Usage:
    >>> from lidar_converter import LiDARConverter, detect_vendor
    >>> 
    >>> # Convert a LiDAR file
    >>> converter = LiDARConverter()
    >>> result = converter.convert("input.pcap", "output.las")
    >>> print(f"Converted {result['points_converted']} points")
    >>> 
    >>> # Detect vendor before conversion
    >>> detection = detect_vendor("input.pcap")
    >>> if detection["success"]:
    ...     print(f"Detected: {detection['vendor_name']}")
    ...     print(f"Confidence: {detection['confidence']:.1%}")

Batch Processing:
    >>> converter = LiDARConverter()
    >>> files = ["file1.pcap", "file2.pcap", "file3.pcap"]
    >>> results = converter.convert_batch(files, output_dir="output/", output_format="laz")
    >>> successful = sum(1 for r in results if r["success"])
    >>> print(f"Converted {successful}/{len(files)} files")

For more information, see: https://github.com/Param-Patel-o5/lidar-converter
"""

from .converter import LiDARConverter
from .detector import VendorDetector

# Convenience function for vendor detection
def detect_vendor(file_path: str):
    """
    Detect the vendor/manufacturer of a LiDAR file.
    
    This is a convenience function that creates a VendorDetector instance
    and performs detection. For repeated detections, consider creating
    a VendorDetector instance directly for better performance.
    
    Args:
        file_path: Path to the LiDAR file (PCAP, LVX, CSV, etc.)
        
    Returns:
        dict: Detection result containing:
            - success (bool): Whether vendor was detected
            - vendor_name (str): Detected vendor name (lowercase)
            - confidence (float): Detection confidence (0.0-1.0)
            - file_signature (str): Identifying signature found
            - message (str): Human-readable detection message
            - metadata (dict): Additional detection information
            - error (str): Error message if detection failed
        
    Example:
        >>> result = detect_vendor("data.pcap")
        >>> if result["success"]:
        ...     print(f"Vendor: {result['vendor_name']}")
        ...     print(f"Confidence: {result['confidence']:.1%}")
        ...     print(f"Signature: {result['file_signature']}")
        ... else:
        ...     print(f"Detection failed: {result['error']}")
    """
    detector = VendorDetector()
    return detector.detect_vendor(file_path)


# Package metadata
__version__ = "0.2.0"
__author__ = "Param Patel"
__license__ = "MIT"

# Public API exports
__all__ = [
    # Main classes
    "LiDARConverter",
    "VendorDetector",
    
    # Convenience functions
    "detect_vendor",
    
    # Metadata
    "__version__",
    "__author__",
    "__license__",
]
