#!/usr/bin/env python3
"""
LiDAR Vendor Detection Module
Automatically detects the vendor/manufacturer of LiDAR data files based on
file headers, magic bytes, extensions, and metadata analysis.
"""

import os
import json
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class LidarVendorDetector:
    """Detects LiDAR vendor from file analysis."""
    
    def __init__(self):
        # Define vendor signatures and detection patterns
        self.vendor_patterns = {
            "ouster": {
                "extensions": [".pcap"],
                "magic_bytes": [],
                "required_files": [".json"],  # Requires corresponding JSON metadata
                "json_fields": ["ouster", "lidar_mode", "sensor_info"],
                "description": "Ouster LiDAR sensors"
            },
            "hesai": {
                "extensions": [".pcap", ".bin"],
                "magic_bytes": [],
                "required_files": [],
                "pcap_patterns": ["hesai", "pandar", "xt"],
                "description": "Hesai Technology LiDAR sensors"
            },
            "velodyne": {
                "extensions": [".pcap", ".bin"],
                "magic_bytes": [b"\x00\x00\x00\x00"],  # Common Velodyne packet start
                "required_files": [],
                "pcap_patterns": ["velodyne", "vlp", "vls", "hdl"],
                "description": "Velodyne LiDAR sensors"
            },
            "riegl": {
                "extensions": [".rxp", ".rdbx"],
                "magic_bytes": [b"RIEGL"],
                "required_files": [],
                "description": "RIEGL LiDAR sensors"
            },
            "sick": {
                "extensions": [".pcap", ".bin"],
                "magic_bytes": [],
                "required_files": [],
                "pcap_patterns": ["sick", "lds"],
                "description": "SICK LiDAR sensors"
            }
        }
    
    def detect_vendor(self, file_path: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Detect the vendor of a LiDAR file using multiple methods and scoring.
        
        Args:
            file_path: Path to the LiDAR file
            
        Returns:
            Tuple of (vendor_name, detection_info)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return None, {"error": "File not found"}
        
        detection_info = {
            "file_path": str(file_path),
            "file_size": file_path.stat().st_size,
            "file_extension": file_path.suffix.lower(),
            "detection_methods": [],
            "vendor_scores": {},
            "confidence": 0.0
        }
        
        # Initialize vendor scores
        vendor_scores = {vendor: 0.0 for vendor in self.vendor_patterns.keys()}
        
        # Method 1: Check file extension (weight: 1.0)
        extension_matches = self._check_extension_scored(file_path, detection_info)
        for vendor, score in extension_matches.items():
            vendor_scores[vendor] += score
        
        # Method 2: Check magic bytes (weight: 3.0 - high confidence)
        magic_matches = self._check_magic_bytes_scored(file_path, detection_info)
        for vendor, score in magic_matches.items():
            vendor_scores[vendor] += score * 3.0
        
        # Method 3: Check companion files (weight: 2.5 - high confidence)
        companion_matches = self._check_companion_files_scored(file_path, detection_info)
        for vendor, score in companion_matches.items():
            vendor_scores[vendor] += score * 2.5
        
        # Method 4: Analyze file content patterns (weight: 2.0)
        content_matches = self._check_content_patterns_scored(file_path, detection_info)
        for vendor, score in content_matches.items():
            vendor_scores[vendor] += score * 2.0
        
        # Method 5: Check PCAP packet patterns (weight: 2.5 - for PCAP files)
        if file_path.suffix.lower() == ".pcap":
            pcap_matches = self._check_pcap_patterns_scored(file_path, detection_info)
            for vendor, score in pcap_matches.items():
                vendor_scores[vendor] += score * 2.5
        
        # Find the vendor with the highest score
        detection_info["vendor_scores"] = vendor_scores
        
        if vendor_scores:
            best_vendor = max(vendor_scores, key=vendor_scores.get)
            best_score = vendor_scores[best_vendor]
            
            # Only return a vendor if confidence is above threshold
            if best_score >= 1.0:  # Minimum confidence threshold
                detection_info["confidence"] = best_score
                detection_info["detection_methods"].append(f"scored_detection_{best_vendor}")
                logger.info(f"Detected vendor: {best_vendor} (confidence: {best_score:.2f})")
                return best_vendor, detection_info
        
        logger.warning("No vendor detected with sufficient confidence")
        return None, detection_info
    
    def _check_extension(self, file_path: Path, detection_info: Dict) -> Optional[str]:
        """Check if file extension matches known vendor patterns."""
        extension = file_path.suffix.lower()
        
        # For PCAP files, check specific filename patterns first
        if extension == ".pcap":
            # Check for Velodyne files first (more specific patterns)
            if ("velodyne" in file_path.name.lower() or 
                "vlp16" in file_path.name.lower() or 
                "vlp32" in file_path.name.lower() or
                "vls128" in file_path.name.lower()):
                detection_info["detection_methods"].append("extension_match_velodyne")
                logger.info(f"Extension match for velodyne: {extension} (filename hint)")
                return "velodyne"
            # Check for Ouster files (by filename pattern)
            elif "ouster" in file_path.name.lower() or "os-" in file_path.name.lower():
                detection_info["detection_methods"].append("extension_match_ouster")
                logger.info(f"Extension match for ouster: {extension} (filename hint)")
                return "ouster"
        
        # For other extensions or non-PCAP files, check all vendors
        for vendor, patterns in self.vendor_patterns.items():
            if extension in patterns["extensions"]:
                detection_info["detection_methods"].append(f"extension_match_{vendor}")
                logger.info(f"Extension match for {vendor}: {extension}")
                return vendor
        
        # For PCAP files, check other vendors
        if extension == ".pcap":
            for vendor, patterns in self.vendor_patterns.items():
                if extension in patterns["extensions"] and vendor != "velodyne":
                    detection_info["detection_methods"].append(f"extension_match_{vendor}")
                    logger.info(f"Extension match for {vendor}: {extension}")
                    return vendor
        
        return None
    
    def _check_extension_scored(self, file_path: Path, detection_info: Dict) -> Dict[str, float]:
        """Check file extension and return scores for all matching vendors."""
        extension = file_path.suffix.lower()
        scores = {}
        
        # For PCAP files, check specific filename patterns (NO PRIORITY - equal weight)
        if extension == ".pcap":
            # Check for Ouster files (by filename pattern)
            if "ouster" in file_path.name.lower() or "os-" in file_path.name.lower():
                scores["ouster"] = 1.0
                detection_info["detection_methods"].append("extension_match_ouster")
                logger.info(f"Extension match for ouster: {extension} (filename hint)")
            # Check for Hesai files (by filename pattern)
            elif ("hesai" in file_path.name.lower() or 
                  "pandar" in file_path.name.lower() or 
                  "xt" in file_path.name.lower()):
                scores["hesai"] = 1.0
                detection_info["detection_methods"].append("extension_match_hesai")
                logger.info(f"Extension match for hesai: {extension} (filename hint)")
            # Check for Velodyne files (by filename pattern)
            elif ("velodyne" in file_path.name.lower() or 
                  "vlp16" in file_path.name.lower() or 
                  "vlp32" in file_path.name.lower() or
                  "vls128" in file_path.name.lower()):
                scores["velodyne"] = 1.0
                detection_info["detection_methods"].append("extension_match_velodyne")
                logger.info(f"Extension match for velodyne: {extension} (filename hint)")
        
        # For other extensions or non-PCAP files, check all vendors
        for vendor, patterns in self.vendor_patterns.items():
            if extension in patterns["extensions"] and vendor not in scores:
                scores[vendor] = 0.5  # Lower score for generic extension match
                detection_info["detection_methods"].append(f"extension_match_{vendor}")
                logger.info(f"Extension match for {vendor}: {extension}")
        
        return scores
    
    def _check_magic_bytes_scored(self, file_path: Path, detection_info: Dict) -> Dict[str, float]:
        """Check file magic bytes and return scores for all matching vendors."""
        scores = {}
        
        try:
            with open(file_path, 'rb') as f:
                # Read first 1024 bytes for magic byte analysis
                header = f.read(1024)
            
            for vendor, patterns in self.vendor_patterns.items():
                for magic_bytes in patterns.get("magic_bytes", []):
                    if header.startswith(magic_bytes):
                        scores[vendor] = 1.0
                        detection_info["detection_methods"].append(f"magic_bytes_{vendor}")
                        detection_info["magic_bytes_found"] = magic_bytes.hex()
                        logger.info(f"Magic bytes match for {vendor}: {magic_bytes.hex()}")
            
            # Check for RIEGL signature in header
            if b"RIEGL" in header:
                scores["riegl"] = 1.0
                detection_info["detection_methods"].append("magic_bytes_riegl")
                logger.info("RIEGL signature found in header")
                
        except Exception as e:
            logger.warning(f"Error reading magic bytes: {e}")
            detection_info["magic_bytes_error"] = str(e)
        
        return scores
    
    def _check_companion_files_scored(self, file_path: Path, detection_info: Dict) -> Dict[str, float]:
        """Check for required companion files and return scores."""
        scores = {}
        
        for vendor, patterns in self.vendor_patterns.items():
            required_files = patterns.get("required_files", [])
            
            for req_ext in required_files:
                companion_file = file_path.with_suffix(req_ext)
                if companion_file.exists():
                    # For Ouster, also check JSON content
                    if vendor == "ouster" and req_ext == ".json":
                        if self._validate_ouster_json(companion_file):
                            scores[vendor] = 1.0
                            detection_info["detection_methods"].append(f"companion_file_{vendor}")
                            detection_info["companion_file"] = str(companion_file)
                            logger.info(f"Companion file match for {vendor}: {companion_file}")
        
        return scores
    
    def _check_content_patterns_scored(self, file_path: Path, detection_info: Dict) -> Dict[str, float]:
        """Analyze file content for vendor-specific patterns and return scores."""
        scores = {}
        
        try:
            with open(file_path, 'rb') as f:
                # Read first 4096 bytes for pattern analysis
                content = f.read(4096)
            
            content_str = content.decode('utf-8', errors='ignore').lower()
            
            for vendor, patterns in self.vendor_patterns.items():
                json_fields = patterns.get("json_fields", [])
                for field in json_fields:
                    if field in content_str:
                        scores[vendor] = 0.8
                        detection_info["detection_methods"].append(f"content_pattern_{vendor}")
                        detection_info["pattern_found"] = field
                        logger.info(f"Content pattern match for {vendor}: {field}")
        
        except Exception as e:
            logger.warning(f"Error analyzing content patterns: {e}")
            detection_info["content_analysis_error"] = str(e)
        
        return scores
    
    def _check_pcap_patterns_scored(self, file_path: Path, detection_info: Dict) -> Dict[str, float]:
        """Check PCAP file for vendor-specific packet patterns and return scores."""
        scores = {}
        
        try:
            # This is a simplified check - in practice, you'd use pcap parsing
            # For now, we'll check if the file contains vendor-specific strings
            with open(file_path, 'rb') as f:
                content = f.read(8192)  # Read first 8KB
            
            content_str = content.decode('utf-8', errors='ignore').lower()
            
            for vendor, patterns in self.vendor_patterns.items():
                pcap_patterns = patterns.get("pcap_patterns", [])
                for pattern in pcap_patterns:
                    if pattern in content_str:
                        scores[vendor] = 0.7
                        detection_info["detection_methods"].append(f"pcap_pattern_{vendor}")
                        detection_info["pcap_pattern_found"] = pattern
                        logger.info(f"PCAP pattern match for {vendor}: {pattern}")
        
        except Exception as e:
            logger.warning(f"Error analyzing PCAP patterns: {e}")
            detection_info["pcap_analysis_error"] = str(e)
        
        return scores
    
    def _check_magic_bytes(self, file_path: Path, detection_info: Dict) -> Optional[str]:
        """Check file magic bytes against known vendor signatures."""
        try:
            with open(file_path, 'rb') as f:
                # Read first 1024 bytes for magic byte analysis
                header = f.read(1024)
            
            for vendor, patterns in self.vendor_patterns.items():
                for magic_bytes in patterns.get("magic_bytes", []):
                    if header.startswith(magic_bytes):
                        detection_info["detection_methods"].append(f"magic_bytes_{vendor}")
                        detection_info["magic_bytes_found"] = magic_bytes.hex()
                        logger.info(f"Magic bytes match for {vendor}: {magic_bytes.hex()}")
                        return vendor
            
            # Check for RIEGL signature in header
            if b"RIEGL" in header:
                detection_info["detection_methods"].append("magic_bytes_riegl")
                logger.info("RIEGL signature found in header")
                return "riegl"
                
        except Exception as e:
            logger.warning(f"Error reading magic bytes: {e}")
            detection_info["magic_bytes_error"] = str(e)
        
        return None
    
    def _check_companion_files(self, file_path: Path, detection_info: Dict) -> Optional[str]:
        """Check for required companion files (like JSON metadata)."""
        for vendor, patterns in self.vendor_patterns.items():
            required_files = patterns.get("required_files", [])
            
            for req_ext in required_files:
                companion_file = file_path.with_suffix(req_ext)
                if companion_file.exists():
                    # For Ouster, also check JSON content
                    if vendor == "ouster" and req_ext == ".json":
                        if self._validate_ouster_json(companion_file):
                            detection_info["detection_methods"].append(f"companion_file_{vendor}")
                            detection_info["companion_file"] = str(companion_file)
                            logger.info(f"Companion file match for {vendor}: {companion_file}")
                            return vendor
                else:
                    # If Ouster file doesn't have JSON, don't detect as Velodyne
                    if vendor == "ouster" and req_ext == ".json":
                        logger.info(f"No JSON metadata found for Ouster file: {file_path}")
                        return None
        
        return None
    
    def _validate_ouster_json(self, json_path: Path) -> bool:
        """Validate that JSON file contains Ouster-specific metadata."""
        try:
            with open(json_path, 'r') as f:
                metadata = json.load(f)
            
            # Check for Ouster-specific fields
            metadata_str = str(metadata).lower()
            ouster_fields = ["ouster", "lidar_mode", "sensor_info", "beam_altitude_angles"]
            
            for field in ouster_fields:
                if field in metadata_str:
                    return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error validating Ouster JSON: {e}")
            return False
    
    def _check_content_patterns(self, file_path: Path, detection_info: Dict) -> Optional[str]:
        """Analyze file content for vendor-specific patterns."""
        try:
            with open(file_path, 'rb') as f:
                # Read first 4096 bytes for pattern analysis
                content = f.read(4096)
            
            content_str = content.decode('utf-8', errors='ignore').lower()
            
            for vendor, patterns in self.vendor_patterns.items():
                json_fields = patterns.get("json_fields", [])
                for field in json_fields:
                    if field in content_str:
                        detection_info["detection_methods"].append(f"content_pattern_{vendor}")
                        detection_info["pattern_found"] = field
                        logger.info(f"Content pattern match for {vendor}: {field}")
                        return vendor
        
        except Exception as e:
            logger.warning(f"Error analyzing content patterns: {e}")
            detection_info["content_analysis_error"] = str(e)
        
        return None
    
    def _check_pcap_patterns(self, file_path: Path, detection_info: Dict) -> Optional[str]:
        """Check PCAP file for vendor-specific packet patterns."""
        try:
            # This is a simplified check - in practice, you'd use pcap parsing
            # For now, we'll check if the file contains vendor-specific strings
            with open(file_path, 'rb') as f:
                content = f.read(8192)  # Read first 8KB
            
            content_str = content.decode('utf-8', errors='ignore').lower()
            
            for vendor, patterns in self.vendor_patterns.items():
                pcap_patterns = patterns.get("pcap_patterns", [])
                for pattern in pcap_patterns:
                    if pattern in content_str:
                        detection_info["detection_methods"].append(f"pcap_pattern_{vendor}")
                        detection_info["pcap_pattern_found"] = pattern
                        logger.info(f"PCAP pattern match for {vendor}: {pattern}")
                        return vendor
        
        except Exception as e:
            logger.warning(f"Error analyzing PCAP patterns: {e}")
            detection_info["pcap_analysis_error"] = str(e)
        
        return None
    
    def get_supported_vendors(self) -> List[str]:
        """Get list of supported vendors."""
        return list(self.vendor_patterns.keys())
    
    def get_vendor_info(self, vendor: str) -> Dict[str, Any]:
        """Get detailed information about a specific vendor."""
        if vendor not in self.vendor_patterns:
            return {"error": f"Unknown vendor: {vendor}"}
        
        return self.vendor_patterns[vendor]
    
    def add_vendor_pattern(self, vendor: str, patterns: Dict[str, Any]) -> bool:
        """Add a new vendor pattern for detection."""
        try:
            self.vendor_patterns[vendor] = patterns
            logger.info(f"Added vendor pattern for: {vendor}")
            return True
        except Exception as e:
            logger.error(f"Error adding vendor pattern: {e}")
            return False

def detect_lidar_vendor(file_path: str) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Convenience function to detect LiDAR vendor.
    
    Args:
        file_path: Path to the LiDAR file
        
    Returns:
        Tuple of (vendor_name, detection_info)
    """
    detector = LidarVendorDetector()
    return detector.detect_vendor(file_path)

def main():
    """Command-line interface for vendor detection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LiDAR Vendor Detection Tool")
    parser.add_argument("file_path", help="Path to LiDAR file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--list-vendors", action="store_true", help="List supported vendors")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    if args.list_vendors:
        detector = LidarVendorDetector()
        print("Supported vendors:")
        for vendor in detector.get_supported_vendors():
            info = detector.get_vendor_info(vendor)
            print(f"  {vendor}: {info['description']}")
            print(f"    Extensions: {info['extensions']}")
        return 0
    
    vendor, info = detect_lidar_vendor(args.file_path)
    
    if vendor:
        print(f"✅ Detected vendor: {vendor}")
        print(f"Detection methods: {', '.join(info.get('detection_methods', []))}")
    else:
        print("❌ No vendor detected")
        print(f"Detection info: {info}")
    
    return 0 if vendor else 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
