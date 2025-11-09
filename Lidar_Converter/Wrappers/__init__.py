"""
Vendor-specific wrapper modules for LiDAR conversion.

This package contains vendor-specific wrappers that abstract SDK complexity
behind a unified interface. All wrappers inherit from BaseVendorWrapper and
provide consistent method signatures for seamless integration with converter.py.

Supported Vendors:
    - Ouster: Full implementation
    - Velodyne: Full implementation
    - Hesai: Full implementation
    - Livox: Full implementation
    - RIEGL: Planned
    - SICK: Planned
"""

from .base_wrapper import BaseVendorWrapper
from .ouster_wrapper import OusterWrapper
from .velodyne_wrapper import VelodyneWrapper
from .livox_wrapper import LivoxWrapper
from .hesai_wrapper import HesaiWrapper

__all__ = [
    "BaseVendorWrapper",
    "OusterWrapper",
    "VelodyneWrapper",
    "LivoxWrapper",
    "HesaiWrapper",
]

