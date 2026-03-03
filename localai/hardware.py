"""
Hardware detection module.

Detects CPU, RAM, GPU/VRAM, and disk information across platforms.
"""

import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field


@dataclass
class GPUInfo:
    """Information about a detected GPU."""
    name: str = "Unknown"
    vram_gb: float = 0.0
    gpu_type: str = "unknown"  # "apple_silicon", "nvidia", "amd", "intel", "integrated"


@dataclass
class HardwareInfo:
    """Complete hardware profile of the system."""
    # CPU
    cpu_name: str = "Unknown"
    cpu_cores: int = 0
    cpu_threads: int = 0
    cpu_arch: str = "unknown"  # "arm64", "x86_64"

    # RAM
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0

    # GPU
    gpus: list = field(default_factory=list)
    is_apple_silicon: bool = False

    # Disk
    disk_available_gb: float = 0.0

    # OS
    os_name: str = "Unknown"
    os_version: str = ""

    @property
    def best_gpu(self) -> GPUInfo | None:
        """Return the GPU with the most VRAM."""
        if not self.gpus:
            return None
        return max(self.gpus, key=lambda g: g.vram_gb)

    @property
    def effective_vram_gb(self) -> float:
        """
        Return the effective memory available for model loading.

        For Apple Silicon, unified memory means most of RAM is usable.
        For discrete GPUs, it's the GPU VRAM.
        For CPU-only, it's available RAM (with a penalty).
        """
        if self.is_apple_silicon:
            # Apple Silicon can use ~75% of unified memory for ML
            return self.ram_total_gb * 0.75

        best = self.best_gpu
        if best and best.vram_gb > 0:
            return best.vram_gb

        # CPU-only fallback: use available RAM but it'll be slower
        return self.ram_available_gb * 0.6

    @property
    def has_gpu(self) -> bool:
        """Check if a usable GPU was detected."""
        return self.is_apple_silicon or (
            self.best_gpu is not None and self.best_gpu.vram_gb > 0
        )

    @property
    def inference_mode(self) -> str:
        """Describe how inference will run."""
        if self.is_apple_silicon:
            return "Apple Metal (Unified Memory)"
        best = self.best_gpu
        if best and best.vram_gb > 0:
            if best.gpu_type == "nvidia":
                return f"NVIDIA CUDA ({best.name})"
            elif best.gpu_type == "amd":
                return f"AMD ROCm ({best.name})"
            return f"GPU ({best.name})"
        return "CPU Only (slower)"


def _run_command(cmd: list[str], timeout: int = 10) -> str:
    """Run a command and return stdout, or empty string on failure."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""


def _detect_cpu() -> tuple[str, int, int, str]:
    """Detect CPU name, physical cores, logical threads, and architecture."""
    arch = platform.machine().lower()
    if arch in ("arm64", "aarch64"):
        arch = "arm64"
    elif arch in ("x86_64", "amd64"):
        arch = "x86_64"

    cores = os.cpu_count() or 1

    # Try to get physical core count
    try:
        import psutil
        physical = psutil.cpu_count(logical=False) or cores
        logical = psutil.cpu_count(logical=True) or cores
    except ImportError:
        physical = cores
        logical = cores

    # Get CPU name
    cpu_name = platform.processor() or "Unknown"

    system = platform.system()
    if system == "Darwin":
        # macOS: use sysctl for a better name
        brand = _run_command(["sysctl", "-n", "machdep.cpu.brand_string"])
        if brand:
            cpu_name = brand
    elif system == "Linux":
        # Linux: parse /proc/cpuinfo
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.startswith("model name"):
                        cpu_name = line.split(":", 1)[1].strip()
                        break
        except (FileNotFoundError, PermissionError):
            pass

    return cpu_name, physical, logical, arch


def _detect_ram() -> tuple[float, float]:
    """Detect total and available RAM in GB."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        total = mem.total / (1024 ** 3)
        available = mem.available / (1024 ** 3)
        return round(total, 1), round(available, 1)
    except ImportError:
        return 0.0, 0.0


def _detect_disk(path: str = "/") -> float:
    """Detect available disk space in GB at the given path."""
    try:
        import psutil
        usage = psutil.disk_usage(path)
        return round(usage.free / (1024 ** 3), 1)
    except (ImportError, OSError):
        # Fallback
        total, used, free = shutil.disk_usage(path)
        return round(free / (1024 ** 3), 1)


def _detect_apple_silicon() -> list[GPUInfo]:
    """Detect Apple Silicon GPU capabilities."""
    gpus = []

    # Check if this is Apple Silicon
    output = _run_command(["sysctl", "-n", "hw.optional.arm64"])
    if output != "1":
        return gpus

    # Get chip name from system_profiler
    sp_output = _run_command([
        "system_profiler", "SPHardwareDataType"
    ])

    chip_name = "Apple Silicon"
    for line in sp_output.splitlines():
        if "Chip" in line and ":" in line:
            chip_name = line.split(":", 1)[1].strip()
            break

    gpus.append(GPUInfo(
        name=chip_name,
        vram_gb=0,  # Unified memory, handled via effective_vram_gb
        gpu_type="apple_silicon",
    ))

    return gpus


def _detect_nvidia_gpus() -> list[GPUInfo]:
    """Detect NVIDIA GPUs using nvidia-smi."""
    gpus = []

    output = _run_command([
        "nvidia-smi",
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ])

    if not output:
        return gpus

    for line in output.splitlines():
        parts = line.split(",")
        if len(parts) >= 2:
            name = parts[0].strip()
            try:
                vram_mb = float(parts[1].strip())
                vram_gb = round(vram_mb / 1024, 1)
            except ValueError:
                vram_gb = 0.0

            gpus.append(GPUInfo(
                name=name,
                vram_gb=vram_gb,
                gpu_type="nvidia",
            ))

    return gpus


def _detect_amd_gpus() -> list[GPUInfo]:
    """Detect AMD GPUs using rocm-smi (Linux only)."""
    gpus = []

    output = _run_command(["rocm-smi", "--showmeminfo", "vram"])
    if not output:
        return gpus

    # Parse rocm-smi output for VRAM
    name_output = _run_command(["rocm-smi", "--showproductname"])

    name = "AMD GPU"
    if name_output:
        for line in name_output.splitlines():
            if "Card" in line or "GPU" in line:
                # Try to extract GPU name
                match = re.search(r":\s*(.+)", line)
                if match:
                    name = match.group(1).strip()
                    break

    vram_gb = 0.0
    for line in output.splitlines():
        match = re.search(r"Total.*?:\s*(\d+)", line)
        if match:
            vram_mb = float(match.group(1))
            vram_gb = round(vram_mb / 1024, 1)
            break

    if vram_gb > 0:
        gpus.append(GPUInfo(
            name=name,
            vram_gb=vram_gb,
            gpu_type="amd",
        ))

    return gpus


def _detect_macos_discrete_gpu() -> list[GPUInfo]:
    """Detect discrete GPUs on macOS (non-Apple-Silicon Macs)."""
    gpus = []

    output = _run_command(["system_profiler", "SPDisplaysDataType"])
    if not output:
        return gpus

    current_name = None
    current_vram = 0.0

    for line in output.splitlines():
        line = line.strip()

        if "Chipset Model:" in line:
            if current_name and current_vram > 0:
                gpus.append(GPUInfo(
                    name=current_name,
                    vram_gb=current_vram,
                    gpu_type="nvidia" if "nvidia" in current_name.lower() else "amd",
                ))
            current_name = line.split(":", 1)[1].strip()
            current_vram = 0.0

        elif "VRAM" in line and ":" in line:
            vram_str = line.split(":", 1)[1].strip()
            match = re.search(r"(\d+)\s*(MB|GB)", vram_str, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                unit = match.group(2).upper()
                current_vram = value if unit == "GB" else value / 1024

    # Don't forget the last one
    if current_name and current_vram > 0:
        gpus.append(GPUInfo(
            name=current_name,
            vram_gb=round(current_vram, 1),
            gpu_type="nvidia" if "nvidia" in current_name.lower() else "amd",
        ))

    return gpus


def _detect_gpus() -> tuple[list[GPUInfo], bool]:
    """Detect all GPUs. Returns (gpus, is_apple_silicon)."""
    system = platform.system()

    if system == "Darwin":
        # Check Apple Silicon first
        apple_gpus = _detect_apple_silicon()
        if apple_gpus:
            return apple_gpus, True

        # Intel Mac with discrete GPU
        discrete = _detect_macos_discrete_gpu()
        return discrete, False

    # Linux / Windows: try NVIDIA, then AMD
    nvidia = _detect_nvidia_gpus()
    if nvidia:
        return nvidia, False

    amd = _detect_amd_gpus()
    if amd:
        return amd, False

    return [], False


def detect_hardware() -> HardwareInfo:
    """
    Detect all hardware information.

    Returns a HardwareInfo dataclass with CPU, RAM, GPU, and disk details.
    """
    cpu_name, cpu_cores, cpu_threads, cpu_arch = _detect_cpu()
    ram_total, ram_available = _detect_ram()
    disk_available = _detect_disk()
    gpus, is_apple_silicon = _detect_gpus()

    system = platform.system()
    os_name = {
        "Darwin": "macOS",
        "Linux": "Linux",
        "Windows": "Windows",
    }.get(system, system)

    os_version = platform.mac_ver()[0] if system == "Darwin" else platform.release()

    return HardwareInfo(
        cpu_name=cpu_name,
        cpu_cores=cpu_cores,
        cpu_threads=cpu_threads,
        cpu_arch=cpu_arch,
        ram_total_gb=ram_total,
        ram_available_gb=ram_available,
        gpus=gpus,
        is_apple_silicon=is_apple_silicon,
        disk_available_gb=disk_available,
        os_name=os_name,
        os_version=os_version,
    )
