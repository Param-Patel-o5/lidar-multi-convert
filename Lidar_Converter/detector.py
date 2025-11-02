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

logger = logging.getLogger(__name__)

# PCAP parsing dependency
try:
    import dpkt
except ImportError:
    DPKT_AVAILABLE = False
    logger.warning("dpkt not installed. Install with: pip install dpkt")
else:
    DPKT_AVAILABLE = True

class LidarVendorDetector:
    """
    Detects LiDAR vendor from file analysis using modern packet-structure-based methods.
    
    Uses multiple detection methods with weighted scoring:
    - UDP Port Detection (weight: 3.5) - Most reliable for PCAP files
    - Packet Structure Detection (weight: 3.0) - Magic bytes in UDP payload
    - Magic Bytes (weight: 3.0) - File header magic bytes
    - Companion Files (weight: 2.5) - Required metadata files (e.g., Ouster JSON)
    - Packet Size Detection (weight: 2.0) - UDP payload size patterns
    - File Extension (weight: 0.5) - Lower confidence, can be misleading
    
    Requires dpkt library for PCAP parsing (install with: pip install dpkt).
    """
    
    def __init__(self):
        # Define vendor signatures and detection patterns
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
    
    def detect_vendor(self, file_path: str) -> Tuple[Optional[str], Dict[str, Any]]:
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
            Tuple of (vendor_name, detection_info)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return None, {"error": "File not found"}
        
        # Early return optimization for RIEGL proprietary format
        if file_path.suffix.lower() == ".rxp":
            detection_info = {
                "file_path": str(file_path),
                "file_size": file_path.stat().st_size,
                "file_extension": file_path.suffix.lower(),
                "detection_methods": ["riegl_proprietary_format"],
                "vendor_scores": {"riegl": 10.0},
                "confidence": 10.0
            }
            logger.info("RIEGL proprietary format detected (.rxp)")
            return "riegl", detection_info
        
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
            
            # Only return a vendor if confidence is above threshold (raised from 1.0 to 2.0)
            if best_score >= 2.0:  # Minimum confidence threshold
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
