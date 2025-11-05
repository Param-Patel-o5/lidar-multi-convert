#!/usr/bin/env python3
"""
LiDAR Vendor Detection Module
Automatically detects the vendor/manufacturer of LiDAR data files using modern
packet-structure-based analysis. Uses UDP port detection, packet structure analysis,
packet size validation, magic bytes, companion files, and file extensions.

Supports detection of:
- Velodyne: UDP ports 2368/2369, 1206-byte payloads, 0xFFEE magic bytes
- Ouster: UDP ports 7502/7503, 6K-33K byte payloads (by model), 0x0001 magic bytes
- Hesai: UDP port 2368, 1000-1300 byte payloads, 0xEEFF magic bytes
- RIEGL: Proprietary .rxp/.rdbx formats
- SICK: Various formats

Requires dpkt library for PCAP parsing (install with: pip install dpkt).
"""

import os
import json
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

# Module-level logger (for backward compatibility)
logger = logging.getLogger(__name__)

# PCAP parsing dependency
try:
    import dpkt
except ImportError:
    DPKT_AVAILABLE = False
    logger.warning("dpkt not installed. Install with: pip install dpkt")
else:
    DPKT_AVAILABLE = True

class VendorDetector:
    """
    Stateless class for detecting LiDAR vendor from file analysis.
    
    This class encapsulates all vendor detection logic using modern
    packet-structure-based methods. It is designed to be instantiated
    once and reused for detecting multiple files.
    
    Uses multiple detection methods with weighted scoring:
    - UDP Port Detection (weight: 3.5) - Most reliable for PCAP files
    - Packet Structure Detection (weight: 3.0) - Magic bytes in UDP payload
    - Magic Bytes (weight: 3.0) - File header magic bytes
    - Companion Files (weight: 2.5) - Required metadata files (e.g., Ouster JSON)
    - Packet Size Detection (weight: 2.0) - UDP payload size patterns
    - File Extension (weight: 0.5) - Lower confidence, can be misleading
    
    Requires dpkt library for PCAP parsing (install with: pip install dpkt).
    
    Example:
        detector = VendorDetector()
        result = detector.detect_vendor("file.pcap")
        if result["success"]:
            print(f"Detected: {result['vendor_name']} (confidence: {result['confidence']})")
    """
    
    def __init__(self, enable_cache: bool = False, cache_ttl: int = 3600):
        """
        Initialize vendor detector with optional caching.
        
        Args:
            enable_cache: If True, cache detection results for performance
            cache_ttl: Cache time-to-live in seconds (default: 1 hour)
        """
        # Detection cache (optional)
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl
        self._detection_cache = {}  # Maps file_path -> (result, timestamp)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Define vendor signatures registry
        self.vendor_patterns = {
            "ouster": {
                "extensions": [".pcap"],
                "magic_bytes": [b"\x00\x01"],  # 0x0001 at start of UDP payload
                "udp_ports": [7502, 7503],  # Lidar Data Port: 7502, IMU Data Port: 7503
                "packet_size_ranges": [(6400, 8448), (12544, 16640), (24832, 33024)],  # 32/64/128 channel ranges
                "required_files": [".json"],  # Requires corresponding JSON metadata
                "json_fields": ["ouster", "lidar_mode", "sensor_info"],
                "description": "Ouster LiDAR sensors"
            },
            "hesai": {
                "extensions": [".pcap", ".bin"],
                "magic_bytes": [b"\xEE\xFF"],  # 0xEEFF pre-header signature
                "udp_ports": [2368],  # Can vary, but 2368 is common
                "packet_size_ranges": [(1000, 1300)],  # Typical range
                "required_files": [],
                "description": "Hesai Technology LiDAR sensors"
            },
            "velodyne": {
                "extensions": [".pcap", ".bin"],
                "magic_bytes": [b"\xFF\xEE"],  # 0xFFEE at start of UDP payload
                "udp_ports": [2368, 2369],  # Data Port: 2368, Telemetry/Position Port: 2369
                "packet_size_ranges": [(1206, 1206)],  # Exactly 1206 bytes payload (1248 total with headers)
                "required_files": [],
                "description": "Velodyne LiDAR sensors"
            },
            "riegl": {
                "extensions": [".rxp", ".rdbx"],
                "magic_bytes": [b"RIEGL"],
                "udp_ports": [],
                "packet_size_ranges": [],
                "required_files": [],
                "description": "RIEGL LiDAR sensors"
            },
            "sick": {
                "extensions": [".pcap", ".bin"],
                "magic_bytes": [],
                "udp_ports": [],
                "packet_size_ranges": [],
                "required_files": [],
                "description": "SICK LiDAR sensors"
            }
        }
    
    def detect_vendor(self, file_path: str) -> Dict[str, Any]:
        """
        Detect the vendor of a LiDAR file using multiple methods and scoring.
        
        Detection methods (in order of reliability):
        1. UDP Port Detection (weight: 3.5) - Most reliable for PCAP files
        2. Packet Structure Detection (weight: 3.0) - Magic bytes in UDP payload
        3. Magic Bytes (weight: 3.0) - File header magic bytes
        4. Companion Files (weight: 2.5) - Required metadata files
        5. Packet Size Detection (weight: 2.0) - UDP payload size patterns
        6. File Extension (weight: 0.5) - Lower confidence, can be misleading
        
        Args:
            file_path: Path to the LiDAR file
            
        Returns:
            dict: Detection result containing:
                - "success": bool - Whether vendor was detected
                - "vendor_name": str - Detected vendor (lowercase) or None
                - "confidence": float - Detection confidence (0.0-1.0)
                - "file_signature": str - Hex/magic bytes that identified vendor
                - "message": str - Human-readable detection message
                - "metadata": dict - Optional vendor-specific metadata
                - "file_path": str - Input file path
                - "file_size": int - File size in bytes
                - "error": str - Error message if detection failed
        """
        file_path_str = str(file_path)
        file_path = Path(file_path)
        
        # Check cache if enabled
        if self.enable_cache:
            import time
            if file_path_str in self._detection_cache:
                cached_result, cache_time = self._detection_cache[file_path_str]
                if time.time() - cache_time < self.cache_ttl:
                    self.logger.debug(f"Returning cached detection result for: {file_path_str}")
                    return cached_result
        
        # Initialize result dict
        result = {
            "success": False,
            "vendor_name": None,
            "confidence": 0.0,
            "file_signature": None,
            "message": "",
            "metadata": {},
            "file_path": str(file_path),
            "file_size": 0,
            "error": None
        }
        
        # Validate file exists
        if not file_path.exists():
            result["error"] = "File not found"
            result["message"] = f"File does not exist: {file_path}"
            self.logger.error(result["error"])
            return result
        
        # Validate file is readable
        try:
            file_size = file_path.stat().st_size
            result["file_size"] = file_size
            
            if file_size == 0:
                result["error"] = "File is empty"
                result["message"] = "Cannot detect vendor from empty file"
                self.logger.error(result["error"])
                return result
                
        except PermissionError:
            result["error"] = "Permission denied"
            result["message"] = f"Cannot read file: {file_path}"
            self.logger.error(result["error"])
            return result
        except Exception as e:
            result["error"] = f"File access error: {e}"
            result["message"] = f"Cannot access file: {file_path}"
            self.logger.error(result["error"])
            return result
        
        # Log detection attempt
        self.logger.info(f"Detecting vendor for: {file_path} (size: {file_size} bytes)")
        
        # Early return optimization for RIEGL proprietary format
        if file_path.suffix.lower() == ".rxp":
            result.update({
                "success": True,
                "vendor_name": "riegl",
                "confidence": 1.0,
                "file_signature": "RIEGL proprietary format (.rxp)",
                "message": "Detected RIEGL proprietary format from file extension",
                "metadata": {
                    "file_extension": file_path.suffix.lower(),
                    "detection_method": "file_extension"
                }
            })
            self.logger.info(f"RIEGL proprietary format detected: {file_path}")
            if self.enable_cache:
                import time
                self._detection_cache[file_path_str] = (result.copy(), time.time())
            return result
        
        # Internal detection info for scoring
        detection_info = {
            "file_path": str(file_path),
            "file_size": file_size,
            "file_extension": file_path.suffix.lower(),
            "detection_methods": [],
            "vendor_scores": {},
            "confidence": 0.0
        }
        
        # Initialize vendor scores
        vendor_scores = {vendor: 0.0 for vendor in self.vendor_patterns.keys()}
        
        # Method 1: Check file extension (weight: 0.5 - reduced from 1.0)
        extension_matches = self._check_extension_scored(file_path, detection_info)
        for vendor, score in extension_matches.items():
            vendor_scores[vendor] += score * 0.5
        
        # Method 2: Check magic bytes (weight: 3.0 - high confidence)
        magic_matches = self._check_magic_bytes_scored(file_path, detection_info)
        for vendor, score in magic_matches.items():
            vendor_scores[vendor] += score * 3.0
        
        # Method 3: Check companion files (weight: 2.5 - high confidence)
        companion_matches = self._check_companion_files_scored(file_path, detection_info)
        for vendor, score in companion_matches.items():
            vendor_scores[vendor] += score * 2.5
        
        # Methods 4-6: PCAP-specific detection (only for .pcap files)
        if file_path.suffix.lower() == ".pcap" and DPKT_AVAILABLE:
            # Method 4: UDP Port Detection (weight: 3.5 - highest confidence for PCAP)
            udp_port_matches = self._check_udp_port_scored(file_path, detection_info)
            for vendor, score in udp_port_matches.items():
                vendor_scores[vendor] += score * 3.5
            
            # Method 5: Packet Structure Detection (weight: 3.0 - magic bytes in UDP payload)
            packet_struct_matches = self._check_packet_structure_scored(file_path, detection_info)
            for vendor, score in packet_struct_matches.items():
                vendor_scores[vendor] += score * 3.0
            
            # Method 6: Packet Size Detection (weight: 2.0)
            packet_size_matches = self._check_packet_size_scored(file_path, detection_info)
            for vendor, score in packet_size_matches.items():
                vendor_scores[vendor] += score * 2.0
        elif file_path.suffix.lower() == ".pcap" and not DPKT_AVAILABLE:
            logger.warning("dpkt not available - skipping UDP port, packet structure, and packet size detection")
            detection_info["pcap_detection_skipped"] = "dpkt not installed"
        
        # Find the vendor with the highest score
        detection_info["vendor_scores"] = vendor_scores
        
        if vendor_scores:
            best_vendor = max(vendor_scores, key=vendor_scores.get)
            best_score = vendor_scores[best_vendor]
            
            # Normalize confidence to 0.0-1.0 range (scale by max possible score)
            # Max possible score: 3.5 (UDP) + 3.0 (packet struct) + 3.0 (magic) + 2.5 (companion) + 2.0 (size) + 0.5 (ext) = 14.5
            max_possible_score = 14.5
            normalized_confidence = min(best_score / max_possible_score, 1.0)
            
            # Only return a vendor if confidence is above threshold (2.0 raw score ≈ 0.14 normalized)
            if best_score >= 2.0:
                # Extract file signature from detection info
                file_signature = None
                if detection_info.get("magic_bytes_found"):
                    file_signature = f"Magic bytes: {detection_info['magic_bytes_found']}"
                elif detection_info.get("udp_ports_ouster") or detection_info.get("udp_ports_velodyne") or detection_info.get("udp_ports_hesai"):
                    ports = detection_info.get(f"udp_ports_{best_vendor}", [])
                    file_signature = f"UDP ports: {ports}"
                elif detection_info.get(f"packet_structure_matches_{best_vendor}"):
                    file_signature = "Packet structure match"
                else:
                    file_signature = f"File extension: {file_path.suffix}"
                
                # Extract metadata
                metadata = {
                    "file_extension": file_path.suffix.lower(),
                    "detection_methods": detection_info.get("detection_methods", []),
                    "vendor_scores": {k: v for k, v in vendor_scores.items() if v > 0},
                    "best_score": best_score
                }
                
                result.update({
                    "success": True,
                    "vendor_name": best_vendor,
                    "confidence": normalized_confidence,
                    "file_signature": file_signature,
                    "message": f"Detected {best_vendor} LiDAR file (confidence: {normalized_confidence:.2%})",
                    "metadata": metadata
                })
                
                self.logger.info(f"Detected vendor: {best_vendor} (confidence: {normalized_confidence:.2%}, raw score: {best_score:.2f})")
                
                if self.enable_cache:
                    import time
                    self._detection_cache[file_path_str] = (result.copy(), time.time())
                
                return result
        
        # No vendor detected
        result.update({
            "success": False,
            "message": "No vendor detected with sufficient confidence",
            "error": "Vendor detection failed - file may be unsupported or corrupted"
        })
        self.logger.warning(f"No vendor detected for: {file_path}")
        
        if self.enable_cache:
            import time
            self._detection_cache[file_path_str] = (result.copy(), time.time())
        
        return result
    
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
    
    def _check_udp_port_scored(self, file_path: Path, detection_info: Dict) -> Dict[str, float]:
        """
        Check UDP destination ports from PCAP packets to identify vendor.
        
        Parses at least the first 5-10 packets and checks for UDP destination ports
        typical for each vendor. This is the fastest initial filter and highly reliable.
        
        Weight: 3.5 (highest confidence for PCAP files)
        
        Args:
            file_path: Path to PCAP file
            detection_info: Detection information dictionary to update
            
        Returns:
            Dictionary mapping vendor names to scores (1.0 for confident match)
        """
        scores = {}
        
        if not DPKT_AVAILABLE:
            logger.warning("dpkt not available for UDP port detection")
            return scores
        
        try:
            with open(file_path, 'rb') as f:
                pcap = dpkt.pcap.Reader(f)
                
                udp_ports_found = {}  # Track ports found for each vendor
                packet_count = 0
                max_packets = 10  # Check first 10 packets
                
                for ts, buf in pcap:
                    packet_count += 1
                    if packet_count > max_packets:
                        break
                    
                    try:
                        eth = dpkt.ethernet.Ethernet(buf)
                        if eth.type != dpkt.ethernet.ETH_TYPE_IP:
                            continue
                        
                        ip = eth.data
                        if ip.p != dpkt.ip.IP_PROTO_UDP:
                            continue
                        
                        udp = ip.data
                        dst_port = udp.dport
                        
                        # Check each vendor's expected UDP ports
                        for vendor, patterns in self.vendor_patterns.items():
                            udp_ports = patterns.get("udp_ports", [])
                            if udp_ports and dst_port in udp_ports:
                                if vendor not in udp_ports_found:
                                    udp_ports_found[vendor] = []
                                udp_ports_found[vendor].append(dst_port)
                                
                    except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError, AttributeError):
                        # Skip malformed packets
                        continue
                
                # Score vendors based on port matches
                # If we found at least one packet with vendor's port, give full score
                for vendor, ports in udp_ports_found.items():
                    if ports:
                        scores[vendor] = 1.0
                        detection_info["detection_methods"].append(f"udp_port_{vendor}")
                        detection_info[f"udp_ports_{vendor}"] = list(set(ports))
                        logger.info(f"UDP port match for {vendor}: {list(set(ports))}")
        
        except Exception as e:
            logger.warning(f"Error parsing PCAP for UDP ports: {e}")
            detection_info["udp_port_error"] = str(e)
        
        return scores
    
    def _check_packet_structure_scored(self, file_path: Path, detection_info: Dict) -> Dict[str, float]:
        """
        Check UDP payload for vendor-specific magic bytes/structure.
        
        Examines the first bytes of UDP payload after the UDP header for vendor
        "magic bytes" that uniquely identify packet structure.
        
        - Velodyne: 0xFFEE at start of UDP payload
        - Ouster: 0x0001 at start of UDP payload
        - Hesai: 0xEEFF pre-header signature
        
        Weight: 3.0 (high confidence - confirms vendor)
        
        Args:
            file_path: Path to PCAP file
            detection_info: Detection information dictionary to update
            
        Returns:
            Dictionary mapping vendor names to scores (1.0 for strong match)
        """
        scores = {}
        
        if not DPKT_AVAILABLE:
            logger.warning("dpkt not available for packet structure detection")
            return scores
        
        try:
            with open(file_path, 'rb') as f:
                pcap = dpkt.pcap.Reader(f)
                
                magic_byte_matches = {}  # Track matches for each vendor
                packet_count = 0
                max_packets = 10  # Check first 10 packets
                
                for ts, buf in pcap:
                    packet_count += 1
                    if packet_count > max_packets:
                        break
                    
                    try:
                        eth = dpkt.ethernet.Ethernet(buf)
                        if eth.type != dpkt.ethernet.ETH_TYPE_IP:
                            continue
                        
                        ip = eth.data
                        if ip.p != dpkt.ip.IP_PROTO_UDP:
                            continue
                        
                        udp = ip.data
                        udp_payload = udp.data
                        
                        if len(udp_payload) < 2:
                            continue
                        
                        # Check first 2 bytes of UDP payload for magic bytes
                        payload_start = udp_payload[:2]
                        
                        # Check each vendor's magic bytes
                        for vendor, patterns in self.vendor_patterns.items():
                            magic_bytes_list = patterns.get("magic_bytes", [])
                            for magic_bytes in magic_bytes_list:
                                if magic_bytes[:2] == payload_start:
                                    if vendor not in magic_byte_matches:
                                        magic_byte_matches[vendor] = 0
                                    magic_byte_matches[vendor] += 1
                                
                    except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError, AttributeError):
                        # Skip malformed packets
                        continue
                
                # Score vendors based on magic byte matches
                # If at least one packet matches, give full score
                for vendor, match_count in magic_byte_matches.items():
                    if match_count > 0:
                        scores[vendor] = 1.0
                        detection_info["detection_methods"].append(f"packet_structure_{vendor}")
                        detection_info[f"packet_structure_matches_{vendor}"] = match_count
                        logger.info(f"Packet structure match for {vendor}: {match_count} packets")
        
        except Exception as e:
            logger.warning(f"Error checking packet structure: {e}")
            detection_info["packet_structure_error"] = str(e)
        
        return scores
    
    def _check_packet_size_scored(self, file_path: Path, detection_info: Dict) -> Dict[str, float]:
        """
        Check UDP packet payload sizes to identify vendor.
        
        For the first 50 UDP packets, records payload lengths and checks if they
        match vendor-specific size patterns:
        
        - Velodyne: Exactly 1206 bytes
        - Ouster: Typical ranges (~6,000-34,000 bytes by model: 32/64/128 channel)
        - Hesai: 1000-1300 bytes (varies)
        
        Returns score based on fraction of packets matching vendor size ranges.
        Score is 1.0 if at least 80% of packets match.
        
        Weight: 2.0 (moderate confidence - validates consistency)
        
        Args:
            file_path: Path to PCAP file
            detection_info: Detection information dictionary to update
            
        Returns:
            Dictionary mapping vendor names to scores (fraction if >=80% match)
        """
        scores = {}
        
        if not DPKT_AVAILABLE:
            logger.warning("dpkt not available for packet size detection")
            return scores
        
        try:
            with open(file_path, 'rb') as f:
                pcap = dpkt.pcap.Reader(f)
                
                payload_sizes = []
                packet_count = 0
                max_packets = 50  # Check first 50 packets
                
                for ts, buf in pcap:
                    packet_count += 1
                    if packet_count > max_packets:
                        break
                    
                    try:
                        eth = dpkt.ethernet.Ethernet(buf)
                        if eth.type != dpkt.ethernet.ETH_TYPE_IP:
                            continue
                        
                        ip = eth.data
                        if ip.p != dpkt.ip.IP_PROTO_UDP:
                            continue
                        
                        udp = ip.data
                        payload_size = len(udp.data)
                        payload_sizes.append(payload_size)
                                
                    except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError, AttributeError):
                        # Skip malformed packets
                        continue
                
                if not payload_sizes:
                    logger.warning("No UDP packets found for size analysis")
                    return scores
                
                # Check each vendor's packet size ranges
                for vendor, patterns in self.vendor_patterns.items():
                    size_ranges = patterns.get("packet_size_ranges", [])
                    if not size_ranges:
                        continue
                    
                    matching_packets = 0
                    for payload_size in payload_sizes:
                        for min_size, max_size in size_ranges:
                            if min_size <= payload_size <= max_size:
                                matching_packets += 1
                                break  # Count each packet only once
                    
                    # Calculate fraction of matching packets
                    match_fraction = matching_packets / len(payload_sizes) if payload_sizes else 0.0
                    
                    # Only give score if at least 80% of packets match
                    if match_fraction >= 0.8:
                        scores[vendor] = match_fraction  # Return the fraction as score
                        detection_info["detection_methods"].append(f"packet_size_{vendor}")
                        detection_info[f"packet_size_match_fraction_{vendor}"] = match_fraction
                        detection_info[f"packet_size_matching_{vendor}"] = f"{matching_packets}/{len(payload_sizes)}"
                        logger.info(f"Packet size match for {vendor}: {matching_packets}/{len(payload_sizes)} packets ({match_fraction:.1%})")
        
        except Exception as e:
            logger.warning(f"Error checking packet sizes: {e}")
            detection_info["packet_size_error"] = str(e)
        
        return scores
    
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
    Convenience function to detect LiDAR vendor (backward compatibility).
    
    Args:
        file_path: Path to the LiDAR file
        
    Returns:
        Tuple of (vendor_name, detection_info) for backward compatibility
    """
    detector = VendorDetector()
    result = detector.detect_vendor(file_path)
    
    # Convert new dict format to old tuple format
    if result.get("success", False):
        return result.get("vendor_name"), result
    else:
        return None, result

def main():
    """Command-line interface for vendor detection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LiDAR Vendor Detection Tool")
    parser.add_argument("file_path", nargs="?", help="Path to LiDAR file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--list-vendors", action="store_true", help="List supported vendors")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    if args.list_vendors:
        detector = VendorDetector()
        print("Supported vendors:")
        for vendor in detector.get_supported_vendors():
            info = detector.get_vendor_info(vendor)
            print(f"  {vendor}: {info['description']}")
            print(f"    Extensions: {info['extensions']}")
        return 0
    
    if not args.file_path:
        parser.error("file_path is required when not using --list-vendors")
    
    vendor, info = detect_lidar_vendor(args.file_path)
    
    if vendor:
        print(f"✅ Detected vendor: {vendor}")
        print(f"Confidence: {info.get('confidence', 0):.2%}")
        print(f"File signature: {info.get('file_signature', 'N/A')}")
    else:
        print("❌ No vendor detected")
        print(f"Error: {info.get('error', 'Unknown error')}")
    
    return 0 if vendor else 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
