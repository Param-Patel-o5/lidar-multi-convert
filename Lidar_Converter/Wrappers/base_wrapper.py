"""
Base wrapper abstract class for all LiDAR vendor wrappers.

This module defines the unified interface that all vendor-specific wrappers
must implement. This ensures consistent behavior across all vendors and
allows converter.py to treat all vendors identically.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class BaseVendorWrapper(ABC):
    """
    Abstract base class defining the interface for all vendor wrappers.
    
    This class establishes the contract that all vendor-specific wrappers
    must follow. All methods marked as @abstractmethod must be implemented
    by subclasses.
    """
    
    def __init__(self):
        """Initialize the vendor wrapper."""
        self.vendor_name = self.get_vendor_name()
        self.sdk_version = None
        self.sdk_available = False
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def get_vendor_name(self) -> str:
        """Return the vendor name (e.g., "ouster", "velodyne")."""
        pass
    
    @abstractmethod
    def validate_sdk_installation(self) -> Dict[str, Any]:
        """Validate that the vendor's SDK is properly installed."""
        pass
    
    @abstractmethod
    def convert_to_las(self, input_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        """Convert LiDAR data file to LAS format."""
        pass
    
    @abstractmethod
    def convert(self, input_path: str, output_format: str, output_path: str, **kwargs) -> Dict[str, Any]:
        """Generic conversion method supporting multiple output formats."""
        pass
    
    @abstractmethod
    def get_vendor_info(self) -> Dict[str, Any]:
        """Get vendor capabilities and information."""
        pass
    
    @abstractmethod
    def validate_conversion(self, input_path: str, output_path: str) -> bool:
        """Validate that a conversion output is valid."""
        pass
    
    def _validate_file_path(self, file_path: str, must_exist: bool = True) -> Dict[str, Any]:
        """
        Helper method to validate file paths.
        
        Args:
            file_path: Path to validate
            must_exist: Whether file must exist
            
        Returns:
            dict: Validation result with "valid" key and optional "error" key
        """
        path = Path(file_path)
        
        if must_exist and not path.exists():
            return {
                "valid": False,
                "error": f"File not found: {file_path}"
            }
        
        if must_exist and not path.is_file():
            return {
                "valid": False,
                "error": f"Path is not a file: {file_path}"
            }
        
        if must_exist:
            try:
                # Try to open for reading
                with open(path, 'rb') as f:
                    f.read(1)
            except PermissionError:
                return {
                    "valid": False,
                    "error": f"File is not readable: {file_path}"
                }
        
        return {"valid": True}
    
    def _validate_output_directory(self, output_path: str) -> Dict[str, Any]:
        """
        Helper method to validate output directory is writable.
        
        Args:
            output_path: Output file path
            
        Returns:
            dict: Validation result with "valid" key and optional "error" key
        """
        path = Path(output_path)
        output_dir = path.parent
        
        # Create directory if it doesn't exist
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return {
                "valid": False,
                "error": f"Cannot create output directory: {e}"
            }
        
        # Check if directory is writable
        if not output_dir.is_dir():
            return {
                "valid": False,
                "error": f"Output path is not a directory: {output_dir}"
            }
        
        return {"valid": True}

