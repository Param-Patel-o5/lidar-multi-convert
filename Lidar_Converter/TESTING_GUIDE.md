# Testing Guide for LiDAR Converter

## Quick Test Commands

### 1. Test Detection (Works without SDK)

This will work even if SDKs aren't installed:

```bash
python cli.py detect "Sample_Data/ouster/OS-0-128_v3.0.1_1024x10_20230216_172749-000.pcap"
```

### 2. Install Ouster SDK

For conversion to work, you need to install the Ouster SDK:

```bash
pip install ouster-sdk
```

### 3. Test Conversion with Limited Scans

For testing with large files, use `--max-scans` to limit processing time:

**Quick test (100 scans, ~1-2 minutes):**
```bash
python cli.py convert "Sample_Data/ouster/OS-0-128_v3.0.1_1024x10_20230216_172749-000.pcap" ^
  -o Output/test_ouster_100.las ^
  --max-scans 100 ^
  -c "Sample_Data/ouster/OS-0-128_v3.0.1_1024x10_20230216_172749.json"
```

**Medium test (500 scans, ~5-10 minutes):**
```bash
python cli.py convert "Sample_Data/ouster/OS-0-128_v3.0.1_1024x10_20230216_172749-000.pcap" ^
  -o Output/test_ouster_500.las ^
  --max-scans 500 ^
  -c "Sample_Data/ouster/OS-0-128_v3.0.1_1024x10_20230216_172749.json"
```

**Full conversion (all scans, may take 30+ minutes for large files):**
```bash
python cli.py convert "Sample_Data/ouster/OS-0-128_v3.0.1_1024x10_20230216_172749-000.pcap" ^
  -o Output/test_ouster_full.las ^
  -c "Sample_Data/ouster/OS-0-128_v3.0.1_1024x10_20230216_172749.json"
```

### 4. Test with Validation

Add `--validate` to check the output file:

```bash
python cli.py convert "Sample_Data/ouster/OS-0-128_v3.0.1_1024x10_20230216_172749-000.pcap" ^
  -o Output/test_ouster_validated.las ^
  --max-scans 100 ^
  --validate ^
  -c "Sample_Data/ouster/OS-0-128_v3.0.1_1024x10_20230216_172749.json"
```

### 5. Verbose Output for Debugging

Use `--verbose` and `--log-level DEBUG` to see detailed progress:

```bash
python cli.py convert "Sample_Data/ouster/OS-0-128_v3.0.1_1024x10_20230216_172749-000.pcap" ^
  -o Output/test_ouster_debug.las ^
  --max-scans 100 ^
  --verbose ^
  --log-level DEBUG ^
  -c "Sample_Data/ouster/OS-0-128_v3.0.1_1024x10_20230216_172749.json"
```

## Understanding max-scans Parameter

The `--max-scans` parameter limits how many scans (point cloud frames) are processed from the PCAP file:

- **Lower values (50-200)**: Fast testing, completes in 1-5 minutes
- **Medium values (500-1000)**: Good balance, 5-15 minutes
- **Higher values (2000+)**: More complete data, 15-30+ minutes
- **No limit (omit parameter)**: Process entire file, may take hours for large files

**Recommendation for testing**: Start with 100 scans to verify everything works, then increase as needed.

## File Size Information

Your test file:
- **Size**: ~2.5 GB (2,538,547,026 bytes)
- **Vendor**: Ouster OS-0-128
- **Format**: PCAP with JSON metadata

For a 2.5 GB file:
- 100 scans: ~1-2 minutes
- 500 scans: ~5-10 minutes  
- 1000 scans: ~10-20 minutes
- Full file: Could be 30-60+ minutes depending on hardware

## Expected Output

When conversion succeeds, you'll see:

```
Conversion Result
╭──────────────────┬───────────────────────────────╮
│ Status           │ ✓ SUCCESS                     │
│ Vendor           │ ouster                        │
│ Points Converted │ 12,800,000                    │
│ Conversion Time  │ 85.23s                        │
│ Confidence       │ 39.17%                        │
╰──────────────────┴───────────────────────────────╯

Successfully converted ouster file to las
```

## Troubleshooting

### SDK Not Installed Error

If you see "SDK for ouster is not installed":

```bash
pip install ouster-sdk
```

Then verify:
```bash
python cli.py health
```

### File Not Found

Make sure you're in the correct directory:
```bash
cd E:\SOP\Lidar_Converter
```

Or use absolute paths:
```bash
python cli.py convert "E:\SOP\Lidar_Converter\Sample_Data\ouster\OS-0-128_v3.0.1_1024x10_20230216_172749-000.pcap" ...
```

### Conversion Too Slow

Reduce `--max-scans`:
```bash
--max-scans 50   # Very fast, ~30 seconds
--max-scans 100  # Fast, ~1-2 minutes
--max-scans 200  # Medium, ~3-5 minutes
```

### Memory Issues

If you run out of memory, process fewer scans at a time or use a machine with more RAM.

