# Vendor Wrapper Modules

This directory contains vendor-specific wrappers that provide a unified interface for LiDAR conversion across different manufacturers.

## Architecture

All vendor wrappers follow the same architecture pattern:

```
BaseVendorWrapper (abstract base class)
    ├── OusterWrapper ✅ (implemented - Python SDK)
    ├── VelodyneWrapper ✅ (implemented - dpkt parsing)
    ├── HesaiWrapper (planned)
    ├── LivoxWrapper (planned)
    ├── RIEGLWrapper (planned)
    └── SICKWrapper (planned)
```

## Design Principles

1. **Unified Interface**: All wrappers implement the same abstract methods
2. **Abstraction**: Vendor-specific details are hidden from converter.py
3. **Robustness**: Comprehensive error handling and validation
4. **Extensibility**: Easy to add new vendors following the same pattern
5. **Documentation**: Clear docstrings and usage examples

## Usage

### Basic Usage

```python
from Wrappers import OusterWrapper, VelodyneWrapper

# Ouster conversion (requires ouster-sdk)
ouster_wrapper = OusterWrapper()
if ouster_wrapper.sdk_available:
    result = ouster_wrapper.convert_to_las(
        input_path="ouster_data.pcap",
        output_path="output.las",
        calibration_file="metadata.json"  # Required for Ouster
    )

# Velodyne conversion (uses dpkt fallback)
velodyne_wrapper = VelodyneWrapper()
if velodyne_wrapper.sdk_available:
    result = velodyne_wrapper.convert_to_las(
        input_path="velodyne_data.pcap",
        output_path="output.las",
        sensor_model="VLP-16"  # Optional, auto-detected if not provided
    )

# Check results
if result["success"]:
    print(f"Converted {result['points_converted']} points")
    print(f"Time: {result['conversion_time']:.2f}s")
    print(f"Vendor: {result.get('vendor', 'unknown')}")
else:
    print(f"Error: {result['error']}")
```

### Integration with converter.py

```python
from converter import LiDARConverter

# Automatic vendor detection and conversion
converter = LiDARConverter()
result = converter.convert("file.pcap", "output.las")

# Manual wrapper usage
from detector import VendorDetector
from Wrappers import OusterWrapper, VelodyneWrapper

detector = VendorDetector()
detection_result = detector.detect_vendor("file.pcap")

if detection_result["success"]:
    vendor = detection_result["vendor_name"]
    
    if vendor == "ouster":
        wrapper = OusterWrapper()
    elif vendor == "velodyne":
        wrapper = VelodyneWrapper()
    
    if wrapper.sdk_available:
        result = wrapper.convert_to_las("file.pcap", "output.las")
```

## Wrapper Interface

All wrappers must implement:

- `get_vendor_name()` → str
- `validate_sdk_installation()` → dict
- `convert_to_las(input_path, output_path, **kwargs)` → dict
- `convert(input_path, output_format, output_path, **kwargs)` → dict
- `get_vendor_info()` → dict
- `validate_conversion(input_path, output_path)` → bool

## Adding New Vendors

To add a new vendor wrapper:

1. Create `{vendor}_wrapper.py` in this directory
2. Inherit from `BaseVendorWrapper`
3. Implement all abstract methods
4. Follow the same error handling and logging patterns
5. Add to `__init__.py` exports
6. Document vendor-specific requirements

See `ouster_wrapper.py` and `velodyne_wrapper.py` as reference implementations.

## Current Implementations

### OusterWrapper
- **SDK**: Uses official Ouster Python SDK (ouster-sdk)
- **Requirements**: Requires JSON metadata file
- **Sensors**: OS-0, OS-1, OS-2, OS-Dome series (16/32/64/128 channels)
- **Features**: Full SDK integration with coordinate transformations

### VelodyneWrapper  
- **SDK**: Uses dpkt for PCAP parsing (no vendor SDK required)
- **Requirements**: Only PCAP file needed
- **Sensors**: VLP-16, VLP-32C, HDL-32E, HDL-64E, VLS-128
- **Features**: Manual packet parsing with coordinate conversion

