#!/usr/bin/env python3
# tools/benchmark_facemesh.py
# -*- coding: utf-8 -*-
"""
Benchmark script for FaceMesh performance on the current device.

Runs FaceAnalyzer with different configurations and measures:
- Inference time (median, p90, p99, max)
- Processed FPS
- CPU usage
- Memory usage
- Frame drop rate

Outputs results to CSV for tuning baseline.

Usage:
    python tools/benchmark_facemesh.py [--frames 100] [--camera 0] [--profile rpi]
    python tools/benchmark_facemesh.py --output results.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import platform
import statistics
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional, Dict, Any

# Handle optional dependencies
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Lazy import cv2 to avoid import error if not installed
cv2 = None
try:
    import cv2 as _cv2
    cv2 = _cv2
except ImportError:
    pass

import sys
from pathlib import Path

# Add project root to path
PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from cv.face_analyzer import FaceAnalyzer, FaceAnalyzerConfig
from tools.device_probe import detect_device_profile, get_profile_by_name, DeviceProfile


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    profile_name: str
    frames_processed: int
    total_time_sec: float
    infer_times_ms: List[float]
    cpu_percent_samples: List[float]
    mem_percent_samples: List[float] = field(default_factory=list)
    mem_mb_samples: List[float] = field(default_factory=list)
    faces_detected: int = 0
    frames_dropped: int = 0
    timestamp: str = ""
    device_arch: str = ""
    device_mem_gb: float = 0.0
    
    @property
    def fps(self) -> float:
        if self.total_time_sec <= 0:
            return 0.0
        return self.frames_processed / self.total_time_sec
    
    @property
    def median_ms(self) -> float:
        if not self.infer_times_ms:
            return 0.0
        return statistics.median(self.infer_times_ms)
    
    @property
    def p90_ms(self) -> float:
        if len(self.infer_times_ms) < 10:
            return self.median_ms
        sorted_times = sorted(self.infer_times_ms)
        idx = int(len(sorted_times) * 0.9)
        return sorted_times[idx]
    
    @property
    def p99_ms(self) -> float:
        if len(self.infer_times_ms) < 100:
            return self.p90_ms
        sorted_times = sorted(self.infer_times_ms)
        idx = int(len(sorted_times) * 0.99)
        return sorted_times[idx]
    
    @property
    def max_ms(self) -> float:
        if not self.infer_times_ms:
            return 0.0
        return max(self.infer_times_ms)
    
    @property
    def avg_cpu(self) -> float:
        if not self.cpu_percent_samples:
            return 0.0
        return statistics.mean(self.cpu_percent_samples)
    
    @property
    def avg_mem_percent(self) -> float:
        if not self.mem_percent_samples:
            return 0.0
        return statistics.mean(self.mem_percent_samples)
    
    @property
    def avg_mem_mb(self) -> float:
        if not self.mem_mb_samples:
            return 0.0
        return statistics.mean(self.mem_mb_samples)
    
    @property
    def drop_rate(self) -> float:
        total = self.frames_processed + self.frames_dropped
        if total <= 0:
            return 0.0
        return self.frames_dropped / total
    
    def print_report(self):
        print(f"\n{'='*60}")
        print(f"Benchmark Results: {self.profile_name}")
        print(f"{'='*60}")
        print(f"  Timestamp: {self.timestamp}")
        print(f"  Device: {self.device_arch}, {self.device_mem_gb:.1f}GB RAM")
        print(f"  Frames processed: {self.frames_processed}")
        print(f"  Frames dropped: {self.frames_dropped} ({self.drop_rate*100:.1f}%)")
        print(f"  Total time: {self.total_time_sec:.2f}s")
        print(f"  Processed FPS: {self.fps:.2f}")
        print(f"  Faces detected: {self.faces_detected}")
        print()
        print(f"  Inference time (ms):")
        print(f"    Median: {self.median_ms:.1f}")
        print(f"    P90:    {self.p90_ms:.1f}")
        print(f"    P99:    {self.p99_ms:.1f}")
        print(f"    Max:    {self.max_ms:.1f}")
        print()
        if self.cpu_percent_samples:
            print(f"  CPU usage: {self.avg_cpu:.1f}% avg")
        if self.mem_percent_samples:
            print(f"  Memory usage: {self.avg_mem_percent:.1f}% ({self.avg_mem_mb:.1f} MB)")
        print(f"{'='*60}\n")
        
        # Performance thresholds check
        print("Performance Check:")
        checks = [
            ("Median infer <= 120ms", self.median_ms <= 120, f"{self.median_ms:.1f}ms"),
            ("Processed FPS >= 6", self.fps >= 6, f"{self.fps:.1f}"),
            ("CPU avg < 80%", self.avg_cpu < 80 or not self.cpu_percent_samples, f"{self.avg_cpu:.1f}%"),
            ("Drop rate < 20%", self.drop_rate < 0.2, f"{self.drop_rate*100:.1f}%"),
        ]
        for name, passed, value in checks:
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {name}: {value}")
    
    def to_csv_row(self) -> Dict[str, Any]:
        """Convert result to a dict suitable for CSV export."""
        return {
            "timestamp": self.timestamp,
            "profile": self.profile_name,
            "device_arch": self.device_arch,
            "device_mem_gb": self.device_mem_gb,
            "frames_processed": self.frames_processed,
            "frames_dropped": self.frames_dropped,
            "drop_rate": round(self.drop_rate, 4),
            "total_time_sec": round(self.total_time_sec, 2),
            "fps": round(self.fps, 2),
            "faces_detected": self.faces_detected,
            "infer_median_ms": round(self.median_ms, 2),
            "infer_p90_ms": round(self.p90_ms, 2),
            "infer_p99_ms": round(self.p99_ms, 2),
            "infer_max_ms": round(self.max_ms, 2),
            "cpu_avg_percent": round(self.avg_cpu, 2),
            "mem_avg_percent": round(self.avg_mem_percent, 2),
            "mem_avg_mb": round(self.avg_mem_mb, 2),
        }


def run_benchmark(
    profile: DeviceProfile,
    n_frames: int = 100,
    camera_index: int = 0,
    use_test_image: bool = False,
    test_image_path: Optional[str] = None,
) -> BenchmarkResult:
    """Run a benchmark with the given profile."""
    
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for benchmarking")
    
    # Create analyzer with profile settings
    cfg = FaceAnalyzerConfig.from_device_profile(profile)
    analyzer = FaceAnalyzer(cfg)
    
    # Prepare frame source
    if use_test_image or test_image_path:
        # Use a test image (repeated)
        if test_image_path:
            frame = cv2.imread(test_image_path)
        else:
            # Create a synthetic test image
            frame = create_test_frame(640, 480)
        if frame is None:
            raise RuntimeError(f"Could not load test image: {test_image_path}")
        cap = None
    else:
        # Use camera
        current_os = platform.system()
        if current_os == "Windows":
            backend = cv2.CAP_DSHOW
        elif current_os == "Linux":
            backend = cv2.CAP_V4L2
        else:
            backend = cv2.CAP_ANY
        
        cap = cv2.VideoCapture(camera_index, backend)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.input_max_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.input_max_height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        frame = None
    
    # Warm up
    print(f"Warming up with profile: {profile.name}...")
    for _ in range(5):
        if cap:
            ok, frame = cap.read()
            if not ok:
                continue
        if frame is not None:
            analyzer.analyze_frame(frame, time.time())
    
    # Run benchmark
    print(f"Running benchmark ({n_frames} frames)...")
    infer_times: List[float] = []
    cpu_samples: List[float] = []
    mem_percent_samples: List[float] = []
    mem_mb_samples: List[float] = []
    faces_total = 0
    frames_dropped = 0
    
    # Get process for memory tracking
    process = None
    if HAS_PSUTIL:
        process = psutil.Process()
    
    start_time = time.time()
    
    for i in range(n_frames):
        # Read frame
        if cap:
            ok, frame = cap.read()
            if not ok:
                frames_dropped += 1
                continue
        
        # Measure inference
        t0 = time.perf_counter()
        results, events = analyzer.analyze_frame(frame, time.time())
        t1 = time.perf_counter()
        
        infer_ms = (t1 - t0) * 1000
        infer_times.append(infer_ms)
        faces_total += len(results)
        
        # Sample CPU and memory usage periodically
        if HAS_PSUTIL and i % 10 == 0:
            cpu_samples.append(psutil.cpu_percent())
            mem_info = psutil.virtual_memory()
            mem_percent_samples.append(mem_info.percent)
            if process:
                try:
                    proc_mem = process.memory_info()
                    mem_mb_samples.append(proc_mem.rss / (1024 * 1024))
                except Exception:
                    pass
        
        # Progress
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{n_frames} frames, last infer: {infer_ms:.1f}ms")
    
    end_time = time.time()
    
    # Cleanup
    if cap:
        cap.release()
    
    return BenchmarkResult(
        profile_name=profile.name,
        frames_processed=len(infer_times),
        total_time_sec=end_time - start_time,
        infer_times_ms=infer_times,
        cpu_percent_samples=cpu_samples,
        mem_percent_samples=mem_percent_samples,
        mem_mb_samples=mem_mb_samples,
        faces_detected=faces_total,
        frames_dropped=frames_dropped,
        timestamp=datetime.now().isoformat(),
        device_arch=profile.cpu_arch,
        device_mem_gb=profile.mem_gb,
    )


def create_test_frame(width: int, height: int):
    """Create a simple synthetic test frame."""
    import numpy as np
    
    # Create a frame with a circle that looks vaguely like a face
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = (200, 200, 200)  # Light gray background
    
    # Draw a face-like ellipse
    center = (width // 2, height // 2)
    axes = (width // 6, height // 4)
    cv2.ellipse(frame, center, axes, 0, 0, 360, (180, 140, 120), -1)
    
    # Draw eyes
    eye_y = center[1] - axes[1] // 3
    eye_offset = axes[0] // 2
    cv2.circle(frame, (center[0] - eye_offset, eye_y), 10, (50, 50, 50), -1)
    cv2.circle(frame, (center[0] + eye_offset, eye_y), 10, (50, 50, 50), -1)
    
    # Draw mouth
    mouth_y = center[1] + axes[1] // 2
    cv2.ellipse(frame, (center[0], mouth_y), (20, 10), 0, 0, 180, (100, 50, 50), 2)
    
    return frame


def main():
    parser = argparse.ArgumentParser(description="Benchmark FaceMesh performance")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames to process")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--profile", type=str, default=None, help="Force a specific profile (desktop, laptop, arm_generic, rpi)")
    parser.add_argument("--test-image", type=str, default=None, help="Path to test image (instead of camera)")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic test image")
    parser.add_argument("--compare-all", action="store_true", help="Run benchmark with all profiles")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output CSV file for results")
    parser.add_argument("--append", action="store_true", help="Append to existing CSV instead of overwriting")
    args = parser.parse_args()
    
    if cv2 is None:
        print("ERROR: OpenCV (cv2) is required for benchmarking")
        print("Install with: pip install opencv-python")
        return 1
    
    if not HAS_PSUTIL:
        print("WARNING: psutil not installed. Memory metrics will not be collected.")
        print("Install with: pip install psutil")
    
    # Detect device
    detected_profile = detect_device_profile(args.profile)
    print(f"\nDevice Detection:")
    print(f"  Profile: {detected_profile.name}")
    print(f"  Constrained: {detected_profile.is_constrained}")
    print(f"  RPi: {detected_profile.is_rpi}")
    print(f"  Arch: {detected_profile.cpu_arch}")
    print(f"  Memory: {detected_profile.mem_gb:.1f} GB")
    print(f"  EdgeTPU: {detected_profile.has_edgetpu}")
    print(f"  CUDA: {detected_profile.has_cuda}")
    
    results: List[BenchmarkResult] = []
    
    if args.compare_all:
        # Run with all profiles
        from tools.device_probe import PROFILES
        for name, profile in PROFILES.items():
            try:
                result = run_benchmark(
                    profile,
                    n_frames=args.frames,
                    camera_index=args.camera,
                    use_test_image=args.synthetic,
                    test_image_path=args.test_image,
                )
                results.append(result)
                result.print_report()
            except Exception as e:
                print(f"ERROR running {name}: {e}")
        
        # Summary table
        if results:
            print("\n" + "="*80)
            print("SUMMARY COMPARISON")
            print("="*80)
            print(f"{'Profile':<15} {'FPS':>8} {'Median':>10} {'P90':>10} {'CPU':>8} {'Mem':>10} {'Drop':>8}")
            print("-"*80)
            for r in results:
                cpu_str = f"{r.avg_cpu:.1f}%" if r.cpu_percent_samples else "N/A"
                mem_str = f"{r.avg_mem_mb:.0f}MB" if r.mem_mb_samples else "N/A"
                drop_str = f"{r.drop_rate*100:.1f}%"
                print(f"{r.profile_name:<15} {r.fps:>8.1f} {r.median_ms:>9.1f}ms {r.p90_ms:>9.1f}ms {cpu_str:>8} {mem_str:>10} {drop_str:>8}")
    else:
        # Run with detected/specified profile
        result = run_benchmark(
            detected_profile,
            n_frames=args.frames,
            camera_index=args.camera,
            use_test_image=args.synthetic,
            test_image_path=args.test_image,
        )
        results.append(result)
        result.print_report()
    
    # Export to CSV if requested
    if args.output and results:
        export_results_to_csv(results, args.output, append=args.append)
    
    return 0


def export_results_to_csv(results: List[BenchmarkResult], filepath: str, append: bool = False) -> None:
    """Export benchmark results to CSV file."""
    mode = "a" if append and os.path.exists(filepath) else "w"
    write_header = mode == "w" or not os.path.exists(filepath)
    
    fieldnames = [
        "timestamp", "profile", "device_arch", "device_mem_gb",
        "frames_processed", "frames_dropped", "drop_rate",
        "total_time_sec", "fps", "faces_detected",
        "infer_median_ms", "infer_p90_ms", "infer_p99_ms", "infer_max_ms",
        "cpu_avg_percent", "mem_avg_percent", "mem_avg_mb",
    ]
    
    with open(filepath, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in results:
            writer.writerow(r.to_csv_row())
    
    print(f"\nResults exported to: {filepath}")


if __name__ == "__main__":
    sys.exit(main())
