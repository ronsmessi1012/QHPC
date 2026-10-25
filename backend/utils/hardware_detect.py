"""
Hardware Inventory Detection — supporting module for Step 7

Probes the host for the resources the Hardware Mapper is allowed to assign.
Without this, the mapper happily recommends "MPI cluster (8 nodes)" on a
laptop with no cluster and "NVIDIA CUDA GPU" on a machine with no NVIDIA
GPU — a hardware mapper that maps to hardware that does not exist.

Two modes, because the framework serves two legitimate purposes:

    PlanningMode.DETECTED  (default)
        Only assign backends actually present on this host. Use for any
        run whose numbers you intend to report as measured.

    PlanningMode.TARGET
        Assign against a declared target deployment, whether or not this
        host has it. Use for "what would this workload map to on an
        8-node cluster?" Every mapping produced this way is marked
        `is_hypothetical=True` so a report can never present a planned
        placement as an observed one.

Mode and target values come from environment variables, so a deployment is
declared rather than hardcoded:

    QHPC_PLANNING_MODE=detected|target     (default: detected)
    QHPC_TARGET_MPI_SLOTS=<int>            (default: 8, TARGET mode only)
    QHPC_TARGET_GPU_NAME=<str>             (default: "NVIDIA A100", TARGET only)
    QHPC_TARGET_GPU_VRAM_GB=<float>        (default: 40.0, TARGET only)

Detection is cached after the first call; pass `refresh=True` to re-probe.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None


# Probe timeout, seconds. nvidia-smi on a machine with a broken driver can
# hang; the mapper must never block on hardware detection.
PROBE_TIMEOUT_S = 5.0

# Fallbacks when psutil is unavailable.
FALLBACK_CPU_CORES = 1
FALLBACK_RAM_GB = 4.0


class PlanningMode(str, Enum):
    DETECTED = "detected"
    TARGET = "target"


@dataclass(frozen=True)
class GPUDevice:
    name: str
    vram_gb: float


@dataclass(frozen=True)
class HardwareInventory:
    """What this host actually has (or what a target deployment declares)."""

    planning_mode: PlanningMode

    cpu_logical_cores: int
    cpu_physical_cores: int
    total_ram_gb: float

    cuda_available: bool
    cuda_devices: List[GPUDevice] = field(default_factory=list)

    mpi_available: bool = False
    mpi_slots: int = 1

    # Human-readable record of how each conclusion was reached. Goes into
    # the mapper's reasoning_trace so a decision is auditable after the fact.
    detection_notes: List[str] = field(default_factory=list)

    @property
    def is_hypothetical(self) -> bool:
        """True when this inventory describes hardware that was not verified."""
        return self.planning_mode is PlanningMode.TARGET

    @property
    def gpu_vram_gb(self) -> float:
        """VRAM of the largest available GPU, or 0.0 when there is none."""
        return max((d.vram_gb for d in self.cuda_devices), default=0.0)

    @property
    def gpu_description(self) -> str:
        if not self.cuda_devices:
            return "no CUDA device"
        if len(self.cuda_devices) == 1:
            d = self.cuda_devices[0]
            return f"{d.name} ({d.vram_gb:.0f} GB VRAM)"
        return f"{len(self.cuda_devices)}x {self.cuda_devices[0].name}"


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def _probe_cpu() -> tuple[int, int, float, List[str]]:
    notes: List[str] = []
    if psutil is None:
        notes.append(
            f"psutil not installed; assuming {FALLBACK_CPU_CORES} core / "
            f"{FALLBACK_RAM_GB} GB RAM"
        )
        return FALLBACK_CPU_CORES, FALLBACK_CPU_CORES, FALLBACK_RAM_GB, notes

    logical = psutil.cpu_count(logical=True) or FALLBACK_CPU_CORES
    physical = psutil.cpu_count(logical=False) or logical
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    notes.append(
        f"CPU: {logical} logical / {physical} physical cores, "
        f"{ram_gb:.1f} GB RAM (psutil)"
    )
    return logical, physical, ram_gb, notes


def _probe_cuda() -> tuple[bool, List[GPUDevice], List[str]]:
    """
    Detect NVIDIA GPUs via nvidia-smi.

    nvidia-smi is used rather than importing torch/cupy because those are
    heavy optional dependencies this project does not require, and their
    absence would say nothing about whether a GPU exists.
    """
    notes: List[str] = []
    binary = shutil.which("nvidia-smi")
    if binary is None:
        notes.append("CUDA: nvidia-smi not on PATH -> no CUDA device")
        return False, [], notes

    try:
        completed = subprocess.run(
            [binary, "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=PROBE_TIMEOUT_S, check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        notes.append(f"CUDA: nvidia-smi probe failed ({type(exc).__name__}) "
                     f"-> treating as no CUDA device")
        return False, [], notes

    if completed.returncode != 0:
        detail = (completed.stderr or "").strip().splitlines()
        reason = detail[0] if detail else f"exit code {completed.returncode}"
        notes.append(f"CUDA: nvidia-smi returned an error ({reason}) "
                     f"-> treating as no CUDA device")
        return False, [], notes

    devices: List[GPUDevice] = []
    for line in completed.stdout.strip().splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            vram_mib = float(parts[1])
        except ValueError:
            continue
        devices.append(GPUDevice(name=parts[0], vram_gb=vram_mib / 1024.0))

    if not devices:
        notes.append("CUDA: nvidia-smi ran but reported no devices")
        return False, [], notes

    notes.append(
        "CUDA: " + ", ".join(f"{d.name} ({d.vram_gb:.1f} GB)" for d in devices)
    )
    return True, devices, notes


def _probe_mpi(cpu_logical_cores: int) -> tuple[bool, int, List[str]]:
    """
    Detect a usable MPI runtime.

    On a single machine, `mpirun -n N` across N cores is real MPI — real
    processes, real message passing, real measurable speedup. It is not a
    multi-node cluster, and the slot count reflects that: cores, not nodes.
    """
    notes: List[str] = []
    binary = shutil.which("mpirun") or shutil.which("mpiexec")
    if binary is None:
        notes.append("MPI: neither mpirun nor mpiexec on PATH -> MPI unavailable")
        return False, 1, notes

    # mpi4py being importable matters as much as the launcher existing,
    # since the runtime engine (Step 8) will drive MPI through it.
    try:
        import mpi4py  # noqa: F401
        mpi4py_ok = True
    except ImportError:
        mpi4py_ok = False

    if not mpi4py_ok:
        notes.append(f"MPI: {os.path.basename(binary)} found but mpi4py is not "
                     f"importable -> MPI unavailable")
        return False, 1, notes

    notes.append(
        f"MPI: {os.path.basename(binary)} + mpi4py available, "
        f"{cpu_logical_cores} local slots (single host, not multi-node)"
    )
    return True, cpu_logical_cores, notes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _planning_mode_from_env() -> PlanningMode:
    raw = os.environ.get("QHPC_PLANNING_MODE", "detected").strip().lower()
    try:
        return PlanningMode(raw)
    except ValueError:
        return PlanningMode.DETECTED


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _build_target_inventory(
    logical: int, physical: int, ram_gb: float, notes: List[str]
) -> HardwareInventory:
    slots = _env_int("QHPC_TARGET_MPI_SLOTS", 8)
    gpu_name = os.environ.get("QHPC_TARGET_GPU_NAME", "NVIDIA A100")
    gpu_vram = _env_float("QHPC_TARGET_GPU_VRAM_GB", 40.0)

    notes.append(
        "PLANNING MODE = target: the backends below are a DECLARED target "
        "deployment, not verified hardware. Mappings are hypothetical."
    )
    notes.append(f"target MPI slots: {slots}")
    notes.append(f"target GPU: {gpu_name} ({gpu_vram:.0f} GB VRAM)")

    return HardwareInventory(
        planning_mode=PlanningMode.TARGET,
        cpu_logical_cores=logical,
        cpu_physical_cores=physical,
        total_ram_gb=ram_gb,
        cuda_available=True,
        cuda_devices=[GPUDevice(name=gpu_name, vram_gb=gpu_vram)],
        mpi_available=True,
        mpi_slots=slots,
        detection_notes=notes,
    )


def detect_hardware(refresh: bool = False) -> HardwareInventory:
    """
    Return the hardware inventory the mapper may assign from.

    Cached after the first call. The cache is intentional: the mapper may
    be invoked many times per pipeline run, and shelling out to nvidia-smi
    each time would dominate its cost.
    """
    if not refresh and getattr(detect_hardware, "_cache", None) is not None:
        return detect_hardware._cache  # type: ignore[attr-defined]

    mode = _planning_mode_from_env()
    logical, physical, ram_gb, notes = _probe_cpu()

    if mode is PlanningMode.TARGET:
        inventory = _build_target_inventory(logical, physical, ram_gb, list(notes))
    else:
        cuda_available, cuda_devices, cuda_notes = _probe_cuda()
        mpi_available, mpi_slots, mpi_notes = _probe_mpi(logical)
        inventory = HardwareInventory(
            planning_mode=PlanningMode.DETECTED,
            cpu_logical_cores=logical,
            cpu_physical_cores=physical,
            total_ram_gb=ram_gb,
            cuda_available=cuda_available,
            cuda_devices=cuda_devices,
            mpi_available=mpi_available,
            mpi_slots=mpi_slots,
            detection_notes=notes + cuda_notes + mpi_notes,
        )

    detect_hardware._cache = inventory  # type: ignore[attr-defined]
    return inventory


def clear_cache() -> None:
    """Drop the cached inventory. Used by tests and after an env change."""
    detect_hardware._cache = None  # type: ignore[attr-defined]


def describe_inventory(inventory: Optional[HardwareInventory] = None) -> str:
    """Human-readable summary, for CLI output and report headers."""
    inv = inventory or detect_hardware()
    lines = [
        f"Planning mode : {inv.planning_mode.value}"
        + ("  (HYPOTHETICAL — hardware not verified)" if inv.is_hypothetical else ""),
        f"CPU           : {inv.cpu_logical_cores} logical / "
        f"{inv.cpu_physical_cores} physical cores",
        f"RAM           : {inv.total_ram_gb:.1f} GB",
        f"CUDA          : {inv.gpu_description}",
        f"MPI           : "
        + (f"available, {inv.mpi_slots} slots" if inv.mpi_available else "unavailable"),
    ]
    lines.append("Detection notes:")
    lines.extend(f"  - {note}" for note in inv.detection_notes)
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe_inventory())
