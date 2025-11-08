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
import numpy as np
import subprocess
import shutil

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
    
    def _points_to_pcd(self, points: np.ndarray, output_path: str, 
                       preserve_intensity: bool = True) -> Dict[str, Any]:
        """
        Convert numpy point array to PCD format.
        
        Args:
            points: Nx4 array [x, y, z, intensity]
            output_path: Output file path
            preserve_intensity: Whether to include intensity values
            
        Returns:
            Result dict with success status, point count, and output file path
        """
        try:
            # Validate points array
            if points is None or len(points) == 0:
                return {
                    "success": False,
                    "error": "No points to convert",
                    "points_converted": 0,
                    "output_file": None
                }
            
            # Validate output directory
            validation = self._validate_output_directory(output_path)
            if not validation["valid"]:
                return {
                    "success": False,
                    "error": validation["error"],
                    "points_converted": 0,
                    "output_file": None
                }
            
            point_count = len(points)
            
            # Build PCD header
            header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z intensity
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH {point_count}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {point_count}
DATA ascii
"""
            
            # Write file
            with open(output_path, 'w') as f:
                f.write(header)
                for point in points:
                    if preserve_intensity and len(point) >= 4:
                        f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {point[3]:.1f}\n")
                    else:
                        f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} 0.0\n")
            
            # Validate output file was created
            if not Path(output_path).exists():
                return {
                    "success": False,
                    "error": "Output file was not created",
                    "points_converted": 0,
                    "output_file": None
                }
            
            # Check file size is reasonable
            file_size = Path(output_path).stat().st_size
            if file_size == 0:
                return {
                    "success": False,
                    "error": "Output file is empty",
                    "points_converted": 0,
                    "output_file": None
                }
            
            self.logger.info(f"Successfully converted {point_count} points to PCD format")
            return {
                "success": True,
                "points_converted": point_count,
                "output_file": output_path
            }
            
        except IOError as e:
            return {
                "success": False,
                "error": f"File I/O error: {str(e)}",
                "points_converted": 0,
                "output_file": None
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error during PCD conversion: {str(e)}",
                "points_converted": 0,
                "output_file": None
            }
    
    def _points_to_bin(self, points: np.ndarray, output_path: str) -> Dict[str, Any]:
        """
        Convert numpy point array to BIN format (KITTI standard).
        
        Args:
            points: Nx4 array [x, y, z, intensity]
            output_path: Output file path
            
        Returns:
            Result dict with success status, point count, and output file path
        """
        try:
            # Validate points array
            if points is None or len(points) == 0:
                return {
                    "success": False,
                    "error": "No points to convert",
                    "points_converted": 0,
                    "output_file": None
                }
            
            # Validate output directory
            validation = self._validate_output_directory(output_path)
            if not validation["valid"]:
                return {
                    "success": False,
                    "error": validation["error"],
                    "points_converted": 0,
                    "output_file": None
                }
            
            # Ensure float32 type for KITTI format
            points_f32 = points.astype(np.float32)
            
            # Write binary file in little-endian format
            with open(output_path, 'wb') as f:
                points_f32.tofile(f)
            
            # Validate output file was created
            if not Path(output_path).exists():
                return {
                    "success": False,
                    "error": "Output file was not created",
                    "points_converted": 0,
                    "output_file": None
                }
            
            # Check file size is reasonable (16 bytes per point)
            expected_size = len(points) * 16  # 4 floats * 4 bytes each
            actual_size = Path(output_path).stat().st_size
            if actual_size == 0:
                return {
                    "success": False,
                    "error": "Output file is empty",
                    "points_converted": 0,
                    "output_file": None
                }
            
            point_count = len(points)
            self.logger.info(f"Successfully converted {point_count} points to BIN format")
            return {
                "success": True,
                "points_converted": point_count,
                "output_file": output_path
            }
            
        except IOError as e:
            return {
                "success": False,
                "error": f"File I/O error: {str(e)}",
                "points_converted": 0,
                "output_file": None
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error during BIN conversion: {str(e)}",
                "points_converted": 0,
                "output_file": None
            }
    
    def _points_to_csv(self, points: np.ndarray, output_path: str,
                       preserve_intensity: bool = True) -> Dict[str, Any]:
        """
        Convert numpy point array to CSV format.
        
        Args:
            points: Nx4 array [x, y, z, intensity]
            output_path: Output file path
            preserve_intensity: Whether to include intensity column
            
        Returns:
            Result dict with success status, point count, and output file path
        """
        try:
            # Validate points array
            if points is None or len(points) == 0:
                return {
                    "success": False,
                    "error": "No points to convert",
                    "points_converted": 0,
                    "output_file": None
                }
            
            # Validate output directory
            validation = self._validate_output_directory(output_path)
            if not validation["valid"]:
                return {
                    "success": False,
                    "error": validation["error"],
                    "points_converted": 0,
                    "output_file": None
                }
            
            # Write using numpy's savetxt for efficiency
            if preserve_intensity and points.shape[1] >= 4:
                header = "x,y,z,intensity"
                np.savetxt(output_path, points, delimiter=',', 
                          header=header, comments='', fmt='%.6f')
            else:
                header = "x,y,z"
                np.savetxt(output_path, points[:, :3], delimiter=',',
                          header=header, comments='', fmt='%.6f')
            
            # Validate output file was created
            if not Path(output_path).exists():
                return {
                    "success": False,
                    "error": "Output file was not created",
                    "points_converted": 0,
                    "output_file": None
                }
            
            # Check file size is reasonable
            file_size = Path(output_path).stat().st_size
            if file_size == 0:
                return {
                    "success": False,
                    "error": "Output file is empty",
                    "points_converted": 0,
                    "output_file": None
                }
            
            point_count = len(points)
            self.logger.info(f"Successfully converted {point_count} points to CSV format")
            return {
                "success": True,
                "points_converted": point_count,
                "output_file": output_path
            }
            
        except IOError as e:
            return {
                "success": False,
                "error": f"File I/O error: {str(e)}",
                "points_converted": 0,
                "output_file": None
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error during CSV conversion: {str(e)}",
                "points_converted": 0,
                "output_file": None
            }
    
    def _compress_las_to_laz(self, las_path: str, laz_path: str) -> Dict[str, Any]:
        """
        Compress LAS file to LAZ format.
        
        Tries multiple compression methods in order of preference:
        1. laspy built-in compression (version 2.0+)
        2. External laszip command-line tool
        3. Return uncompressed LAS with warning
        
        Args:
            las_path: Input LAS file path
            laz_path: Output LAZ file path
            
        Returns:
            Result dict with success status, compression method, and output file path
        """
        try:
            # Validate input file exists
            validation = self._validate_file_path(las_path, must_exist=True)
            if not validation["valid"]:
                return {
                    "success": False,
                    "error": validation["error"],
                    "output_file": None,
                    "compression_method": None
                }
            
            # Validate output directory
            validation = self._validate_output_directory(laz_path)
            if not validation["valid"]:
                return {
                    "success": False,
                    "error": validation["error"],
                    "output_file": None,
                    "compression_method": None
                }
            
            # Method 1: Try laspy built-in compression
            try:
                import laspy
                
                # Check if laspy version supports LAZ
                laspy_version = getattr(laspy, '__version__', '0.0.0')
                major_version = int(laspy_version.split('.')[0])
                
                if major_version >= 2:
                    self.logger.info("Using laspy built-in LAZ compression")
                    las_file = laspy.read(las_path)
                    las_file.write(laz_path)
                    
                    # Validate output
                    if Path(laz_path).exists() and Path(laz_path).stat().st_size > 0:
                        self.logger.info(f"Successfully compressed to LAZ using laspy")
                        return {
                            "success": True,
                            "output_file": laz_path,
                            "compression_method": "laspy"
                        }
                else:
                    self.logger.warning(f"laspy version {laspy_version} does not support LAZ compression")
                    
            except ImportError:
                self.logger.warning("laspy not available for LAZ compression")
            except Exception as e:
                self.logger.warning(f"laspy compression failed: {str(e)}")
            
            # Method 2: Try external laszip command
            laszip_path = shutil.which("laszip")
            if laszip_path:
                try:
                    self.logger.info("Using external laszip command for compression")
                    result = subprocess.run(
                        [laszip_path, "-i", las_path, "-o", laz_path],
                        capture_output=True,
                        text=True,
                        timeout=300  # 5 minute timeout
                    )
                    
                    if result.returncode == 0 and Path(laz_path).exists():
                        self.logger.info(f"Successfully compressed to LAZ using laszip command")
                        return {
                            "success": True,
                            "output_file": laz_path,
                            "compression_method": "laszip"
                        }
                    else:
                        self.logger.warning(f"laszip command failed: {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    self.logger.warning("laszip command timed out")
                except Exception as e:
                    self.logger.warning(f"laszip command failed: {str(e)}")
            else:
                self.logger.warning("laszip command not found in PATH")
            
            # Method 3: Return uncompressed LAS with warning
            self.logger.warning("LAZ compression not available, returning uncompressed LAS file")
            return {
                "success": True,
                "output_file": las_path,
                "compression_method": "none",
                "warning": "LAZ compression not available. Install laspy 2.0+ or laszip for compression support."
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error during LAZ compression: {str(e)}",
                "output_file": None,
                "compression_method": None
            }

