# Vendor Wrapper Modules

This directory contains vendor-specific wrappers that provide a unified interface for LiDAR conversion across different manufacturers.

## Architecture

All vendor wrappers follow the same architecture pattern:

```
BaseVendorWrapper (abstract base class)
    ├── OusterWrapper ✅ (implemented)
    ├── VelodyneWrapper (planned)
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
from Wrappers import OusterWrapper

# Initialize wrapper
wrapper = OusterWrapper()

# Check SDK availability
if wrapper.sdk_available:
    # Convert PCAP to LAS
    result = wrapper.convert_to_las(
        input_path="data.pcap",
        output_path="output.las",
        calibration_file="metadata.json"  # Optional
    )
    
    if result["success"]:
        print(f"Converted {result['points_converted']} points")
        print(f"Time: {result['conversion_time']:.2f}s")
    else:
        print(f"Error: {result['error']}")
else:
    print("Ouster SDK not installed")
```

### Integration with converter.py

```python
from detector import detect_lidar_vendor
from Wrappers import OusterWrapper, BaseVendorWrapper

# Detect vendor
vendor, detection_info = detect_lidar_vendor("file.pcap")

# Get appropriate wrapper
if vendor == "ouster":
    wrapper = OusterWrapper()
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

See `ouster_wrapper.py` as the reference implementation.

