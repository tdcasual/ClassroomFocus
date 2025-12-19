# tools/device_probe.py
# -*- coding: utf-8 -*-
"""
Device detection and performance profile selection for classroom_focus.

Detects runtime environment (Raspberry Pi, ARM, constrained devices) and
returns a profile dict with optimized settings for FaceAnalyzer and capture.

Usage:
    from tools.device_probe import detect_device_profile, PROFILES
    profile = detect_device_profile()
    # profile = {"name": "rpi", "is_constrained": True, ...}
"""
from __future__ import annotations

import os
import platform
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Performance Profiles
# -----------------------------------------------------------------------------

@dataclass
class DeviceProfile:
    """Performance profile for a device class."""
    name: str
    is_constrained: bool = False
    is_rpi: bool = False
    cpu_arch: str = ""
    mem_gb: float = 0.0
    has_edgetpu: bool = False
    has_cuda: bool = False
    
    # FaceMesh parameters
    max_faces: int = 5
    refine_landmarks: bool = False
    model_complexity: int = 1  # 0=lite, 1=full
    min_det_conf: float = 0.5
    min_trk_conf: float = 0.5
    
    # Frame processing
    target_fps: float = 15.0
    process_every_n: int = 1
    input_scale: float = 1.0  # Downscale factor (0.5 = half resolution)
    input_max_width: int = 1920
    input_max_height: int = 1080
    
    # Capture queue settings
    use_capture_queue: bool = False
    capture_queue_size: int = 2
    drop_old_frames: bool = True
    
    # Debug/output settings
    debug_draw: bool = False
    async_video_write: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Predefined profiles
PROFILE_DESKTOP = DeviceProfile(
    name="desktop",
    is_constrained=False,
    max_faces=5,
    refine_landmarks=False,
    model_complexity=1,
    target_fps=30.0,
    process_every_n=1,
    input_scale=1.0,
    input_max_width=1920,
    input_max_height=1080,
    use_capture_queue=False,
)

PROFILE_LAPTOP = DeviceProfile(
    name="laptop",
    is_constrained=False,
    max_faces=4,
    refine_landmarks=False,
    model_complexity=1,
    target_fps=20.0,
    process_every_n=1,
    input_scale=1.0,
    input_max_width=1280,
    input_max_height=720,
    use_capture_queue=False,
)

PROFILE_ARM_GENERIC = DeviceProfile(
    name="arm_generic",
    is_constrained=True,
    max_faces=2,
    refine_landmarks=False,
    model_complexity=0,
    min_det_conf=0.6,
    min_trk_conf=0.6,
    target_fps=10.0,
    process_every_n=2,
    input_scale=0.75,
    input_max_width=640,
    input_max_height=480,
    use_capture_queue=True,
    capture_queue_size=1,
    drop_old_frames=True,
    async_video_write=True,
)

PROFILE_RPI = DeviceProfile(
    name="rpi",
    is_constrained=True,
    is_rpi=True,
    max_faces=1,
    refine_landmarks=False,
    model_complexity=0,
    min_det_conf=0.65,
    min_trk_conf=0.65,
    target_fps=8.0,
    process_every_n=3,
    input_scale=0.5,
    input_max_width=480,
    input_max_height=360,
    use_capture_queue=True,
    capture_queue_size=1,
    drop_old_frames=True,
    async_video_write=True,
)

PROFILE_RPI_EDGETPU = DeviceProfile(
    name="rpi_edgetpu",
    is_constrained=True,
    is_rpi=True,
    has_edgetpu=True,
    max_faces=2,
    refine_landmarks=False,
    model_complexity=0,
    target_fps=12.0,
    process_every_n=2,
    input_scale=0.6,
    input_max_width=640,
    input_max_height=480,
    use_capture_queue=True,
    capture_queue_size=1,
    drop_old_frames=True,
    async_video_write=True,
)

PROFILES: Dict[str, DeviceProfile] = {
    "desktop": PROFILE_DESKTOP,
    "laptop": PROFILE_LAPTOP,
    "arm_generic": PROFILE_ARM_GENERIC,
    "rpi": PROFILE_RPI,
    "rpi_edgetpu": PROFILE_RPI_EDGETPU,
}

# -----------------------------------------------------------------------------
# Device Detection
# -----------------------------------------------------------------------------

def _read_file_safe(path: str) -> str:
    """Read file content safely, return empty string on error."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except Exception:
        return ""


def _get_total_memory_gb() -> float:
    """Get total system memory in GB."""
    # Try psutil first (cross-platform)
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        pass
    
    # Fallback: parse /proc/meminfo on Linux
    if platform.system() == "Linux":
        content = _read_file_safe("/proc/meminfo")
        for line in content.split("\n"):
            if line.startswith("MemTotal:"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        kb = int(parts[1])
                        return kb / (1024 ** 2)
                    except ValueError:
                        pass
    
    return 0.0


def _detect_raspberry_pi() -> bool:
    """Detect if running on Raspberry Pi."""
    if platform.system() != "Linux":
        return False
    
    # Check /proc/device-tree/model
    model = _read_file_safe("/proc/device-tree/model")
    if "raspberry pi" in model.lower():
        return True
    
    # Check /proc/cpuinfo for BCM (Broadcom)
    cpuinfo = _read_file_safe("/proc/cpuinfo")
    if "BCM" in cpuinfo.upper():
        return True
    
    # Check for Raspberry Pi specific files
    if os.path.exists("/sys/firmware/devicetree/base/model"):
        model2 = _read_file_safe("/sys/firmware/devicetree/base/model")
        if "raspberry" in model2.lower():
            return True
    
    return False


def _detect_edgetpu() -> bool:
    """Detect if EdgeTPU (Coral) is available."""
    try:
        # Check if libedgetpu is available
        import ctypes
        ctypes.CDLL("libedgetpu.so.1")
        return True
    except (OSError, ImportError):
        pass
    
    # Check for pycoral/tflite_runtime
    try:
        from pycoral.utils import edgetpu
        devices = edgetpu.list_edge_tpus()
        return len(devices) > 0
    except ImportError:
        pass
    
    # Check USB devices for Coral
    if platform.system() == "Linux":
        try:
            import subprocess
            result = subprocess.run(["lsusb"], capture_output=True, text=True, timeout=2)
            # Google Coral USB Accelerator VID:PID is 1a6e:089a or 18d1:9302
            if "1a6e:089a" in result.stdout or "18d1:9302" in result.stdout:
                return True
        except Exception:
            pass
    
    return False


def _detect_cuda() -> bool:
    """Detect if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        pass
    
    # Fallback: check nvidia-smi
    if platform.system() in ("Linux", "Windows"):
        try:
            import subprocess
            result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
            return result.returncode == 0
        except Exception:
            pass
    
    return False


def detect_device_profile(force_profile: Optional[str] = None) -> DeviceProfile:
    """
    Detect runtime environment and return appropriate DeviceProfile.
    
    Args:
        force_profile: Force a specific profile by name (for testing/override)
        
    Returns:
        DeviceProfile with optimized settings for the detected device
    """
    # Check for forced profile (via arg or env var)
    profile_name = force_profile or os.getenv("CLASSROOM_DEVICE_PROFILE")
    if profile_name and profile_name in PROFILES:
        profile = PROFILES[profile_name]
        logger.info(f"Using forced device profile: {profile.name}")
        return profile
    
    # Gather system info
    system = platform.system()
    machine = platform.machine().lower()
    mem_gb = _get_total_memory_gb()
    is_rpi = _detect_raspberry_pi()
    has_edgetpu = _detect_edgetpu()
    has_cuda = _detect_cuda()
    
    is_arm = machine in ("arm", "armv7l", "armv8l", "aarch64", "arm64")
    is_constrained = (mem_gb > 0 and mem_gb < 4) or is_arm
    
    logger.info(f"Device detection: system={system}, arch={machine}, mem={mem_gb:.1f}GB, "
                f"is_rpi={is_rpi}, is_arm={is_arm}, has_edgetpu={has_edgetpu}, has_cuda={has_cuda}")
    
    # Select profile based on detection
    if is_rpi:
        profile = PROFILE_RPI_EDGETPU if has_edgetpu else PROFILE_RPI
    elif is_arm and is_constrained:
        profile = PROFILE_ARM_GENERIC
    elif is_constrained:
        profile = PROFILE_LAPTOP
    else:
        profile = PROFILE_DESKTOP
    
    # Update profile with detected values
    profile.cpu_arch = machine
    profile.mem_gb = mem_gb
    profile.has_edgetpu = has_edgetpu
    profile.has_cuda = has_cuda
    
    # Apply CUDA optimizations if available on non-constrained devices
    if has_cuda and not is_constrained:
        profile.max_faces = min(profile.max_faces + 2, 10)
        profile.target_fps = min(profile.target_fps * 1.5, 60.0)
    
    logger.info(f"Selected device profile: {profile.name}")
    return profile


def get_profile_by_name(name: str) -> Optional[DeviceProfile]:
    """Get a profile by name, or None if not found."""
    return PROFILES.get(name)


def list_profiles() -> Dict[str, Dict[str, Any]]:
    """List all available profiles as dicts."""
    return {name: profile.to_dict() for name, profile in PROFILES.items()}


# -----------------------------------------------------------------------------
# Runtime Profile File
# -----------------------------------------------------------------------------

RUNTIME_PROFILE_FILE = ".runtime_profile.json"


def save_runtime_profile(profile: DeviceProfile, base_dir: Optional[str] = None) -> str:
    """Save detected profile to a JSON file for other processes to read."""
    import json
    
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    filepath = os.path.join(base_dir, RUNTIME_PROFILE_FILE)
    data = profile.to_dict()
    data["_detected_at"] = __import__("time").time()
    
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved runtime profile to {filepath}")
        return filepath
    except Exception as e:
        logger.warning(f"Failed to save runtime profile: {e}")
        return ""


def load_runtime_profile(base_dir: Optional[str] = None) -> Optional[DeviceProfile]:
    """Load previously detected profile from JSON file."""
    import json
    
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    filepath = os.path.join(base_dir, RUNTIME_PROFILE_FILE)
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Remove internal fields
        data.pop("_detected_at", None)
        
        return DeviceProfile(**data)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Main (for testing)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    profile = detect_device_profile()
    print(f"\n=== Detected Device Profile: {profile.name} ===")
    print(f"  Is Constrained: {profile.is_constrained}")
    print(f"  Is Raspberry Pi: {profile.is_rpi}")
    print(f"  CPU Arch: {profile.cpu_arch}")
    print(f"  Memory: {profile.mem_gb:.1f} GB")
    print(f"  Has EdgeTPU: {profile.has_edgetpu}")
    print(f"  Has CUDA: {profile.has_cuda}")
    print(f"\n  FaceMesh Settings:")
    print(f"    max_faces: {profile.max_faces}")
    print(f"    model_complexity: {profile.model_complexity}")
    print(f"    refine_landmarks: {profile.refine_landmarks}")
    print(f"\n  Frame Processing:")
    print(f"    target_fps: {profile.target_fps}")
    print(f"    process_every_n: {profile.process_every_n}")
    print(f"    input_scale: {profile.input_scale}")
    print(f"    input_max: {profile.input_max_width}x{profile.input_max_height}")
    print(f"\n  Capture Queue:")
    print(f"    use_capture_queue: {profile.use_capture_queue}")
    print(f"    queue_size: {profile.capture_queue_size}")
    print(f"    drop_old_frames: {profile.drop_old_frames}")
    
    # Save for testing
    save_runtime_profile(profile)
    print(f"\nProfile saved to {RUNTIME_PROFILE_FILE}")
