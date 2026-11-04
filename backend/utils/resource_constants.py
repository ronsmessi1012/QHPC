"""
Constants & Thresholds Registry — Steps 6 & 7

Mirrors the style of backend/utils/constants.py and
backend/utils/compatibility_library.py: deterministic documented priors
only. No decision logic belongs here.

CALIBRATION STATUS
------------------
Every profile carries a `calibrated` flag and a `calibration_host` string.

    calibrated = False  -> the constants are reasoned defaults derived from
                           the algorithm's documented complexity class, NOT
                           measured on real hardware.
    calibrated = True   -> base_* values were fitted from real benchmark
                           timings via backend/tests/benchmark_resources.py.

    calibration_host    -> which machine produced the measurement, or None.

WHY THE HOST MATTERS
--------------------
The three fitted quantities do not transfer between machines equally:

  memory_exp, runtime_exp   TRANSFERABLE. These reflect the algorithm's
                            complexity class, which is a property of the
                            algorithm, not the hardware.

  base_memory_gb            BROADLY TRANSFERABLE. An algorithm allocates
                            roughly what it allocates, regardless of host.

  base_runtime_s            HOST-SPECIFIC. A figure measured on a Colab
                            Xeon says nothing about a laptop's runtime.

So a profile calibrated on a different machine than the one being planned
for is only partially valid, and `calibration_host` is what makes that
visible. Any reported runtime must name the host it was measured on.

Do not describe an uncalibrated profile as "empirically measured" in any
report. Run the benchmark harness and flip the flag first.
"""

# ===========================================================================
# Step 6 — Classical Resource Profiles
# ===========================================================================
# Scaling model, per algorithm:
#
#     scale   = size_driver_value / ref_size
#     memory  = base_memory_gb  * scale ** memory_exp
#     runtime = base_runtime_s  * scale ** runtime_exp
#
# `size_driver` names WHICH quantity drives the cost, because this differs by
# algorithm family and using one global measure distorts both ends:
#   - "dataset_size"   : supervised ML and graph algorithms scale with samples
#                        / nodes (XGBoost on 20k rows).
#   - "dimensionality" : combinatorial optimization and PDE solvers scale with
#                        decision variables / grid size (B&B on 500 cities).
#
# `ref_size` is the size at which base_memory_gb and base_runtime_s hold, so
# each family is anchored at a size that is actually typical for it. This is
# also exactly what the calibration harness measures at.
#
# Exponents approximate the documented complexity class:
#   O(n) -> 1.0,  O(n log n) -> ~1.15,  O(n^2) -> 2.0,  exponential -> 2.5+

CLASSICAL_RESOURCE_PROFILES = {
    # ---- Classification ---------------------------------------------------
    "XGBoost": {
        "size_driver": "dataset_size", "ref_size": 10_000,
        "base_memory_gb": 0.08, "memory_exp": 1.05,
        "base_runtime_s": 3.0, "runtime_exp": 1.15,
        "gpu_capable": True, "mpi_capable": False,
        "calibrated": False, "calibration_host": None,
    },
    "Random Forest": {
        "size_driver": "dataset_size", "ref_size": 10_000,
        "base_memory_gb": 0.15, "memory_exp": 1.10,
        "base_runtime_s": 5.0, "runtime_exp": 1.15,
        "gpu_capable": False, "mpi_capable": True,
        "calibrated": False, "calibration_host": None,
    },
    "SVM": {
        "size_driver": "dataset_size", "ref_size": 10_000,
        "base_memory_gb": 0.10, "memory_exp": 2.00,
        "base_runtime_s": 8.0, "runtime_exp": 2.00,
        "gpu_capable": True, "mpi_capable": False,
        "calibrated": False, "calibration_host": None,
    },
    "Logistic Regression": {
        "size_driver": "dataset_size", "ref_size": 10_000,
        "base_memory_gb": 0.02, "memory_exp": 1.00,
        "base_runtime_s": 0.5, "runtime_exp": 1.00,
        "gpu_capable": False, "mpi_capable": False,
        "calibrated": False, "calibration_host": None,
    },

    # ---- Regression -------------------------------------------------------
    "Linear Regression": {
        "size_driver": "dataset_size", "ref_size": 10_000,
        "base_memory_gb": 0.02, "memory_exp": 1.00,
        "base_runtime_s": 0.3, "runtime_exp": 1.00,
        "gpu_capable": False, "mpi_capable": False,
        "calibrated": False, "calibration_host": None,
    },
    "Random Forest Regressor": {
        "size_driver": "dataset_size", "ref_size": 10_000,
        "base_memory_gb": 0.15, "memory_exp": 1.10,
        "base_runtime_s": 5.0, "runtime_exp": 1.15,
        "gpu_capable": False, "mpi_capable": True,
        "calibrated": False, "calibration_host": None,
    },
    "XGBoost Regressor": {
        "size_driver": "dataset_size", "ref_size": 10_000,
        "base_memory_gb": 0.08, "memory_exp": 1.05,
        "base_runtime_s": 3.0, "runtime_exp": 1.15,
        "gpu_capable": True, "mpi_capable": False,
        "calibrated": False, "calibration_host": None,
    },

    # ---- Optimization (driven by decision variables, not samples) ---------
    "Genetic Algorithm": {
        "size_driver": "dimensionality", "ref_size": 100,
        "base_memory_gb": 0.03, "memory_exp": 1.20,
        "base_runtime_s": 4.0, "runtime_exp": 1.40,
        "gpu_capable": False, "mpi_capable": True,
        "calibrated": False, "calibration_host": None,
    },
    "Simulated Annealing": {
        "size_driver": "dimensionality", "ref_size": 100,
        "base_memory_gb": 0.02, "memory_exp": 1.10,
        "base_runtime_s": 3.0, "runtime_exp": 1.30,
        "gpu_capable": False, "mpi_capable": True,
        "calibrated": False, "calibration_host": None,
    },
    "Branch and Bound": {
        "size_driver": "dimensionality", "ref_size": 100,
        "base_memory_gb": 0.05, "memory_exp": 2.20,
        "base_runtime_s": 2.0, "runtime_exp": 2.60,
        "gpu_capable": False, "mpi_capable": True,
        "calibrated": False, "calibration_host": None,
    },

    # ---- Simulation -------------------------------------------------------
    "Monte Carlo Simulation": {
        "size_driver": "dataset_size", "ref_size": 1_000,
        "base_memory_gb": 0.05, "memory_exp": 1.00,
        "base_runtime_s": 2.0, "runtime_exp": 1.10,
        "gpu_capable": True, "mpi_capable": True,
        "calibrated": False, "calibration_host": None,
    },
    "Finite Difference Solver": {
        "size_driver": "dimensionality", "ref_size": 1_000,
        "base_memory_gb": 0.10, "memory_exp": 1.60,
        "base_runtime_s": 3.0, "runtime_exp": 1.80,
        "gpu_capable": True, "mpi_capable": True,
        "calibrated": False, "calibration_host": None,
    },

    # ---- Search (driven by graph node count) ------------------------------
    "A* Search": {
        "size_driver": "dataset_size", "ref_size": 1_000,
        "base_memory_gb": 0.04, "memory_exp": 1.40,
        "base_runtime_s": 1.0, "runtime_exp": 1.40,
        "gpu_capable": False, "mpi_capable": False,
        "calibrated": False, "calibration_host": None,
    },
    "Dijkstra": {
        "size_driver": "dataset_size", "ref_size": 1_000,
        "base_memory_gb": 0.03, "memory_exp": 1.30,
        "base_runtime_s": 0.8, "runtime_exp": 1.30,
        "gpu_capable": False, "mpi_capable": False,
        "calibrated": False, "calibration_host": None,
    },
}

# ===========================================================================
# Step 6 — Quantum Resource Profiles
# ===========================================================================
# Qubit count:
#   "linear" -> qubit_count = ceil(qubit_coeff * dimensionality)
#               (one qubit per decision variable / feature)
#   "log2"   -> qubit_count = ceil(qubit_coeff * log2(dataset_size))
#               (index register over a search space, e.g. Grover)
#
# circuit_depth = ceil(depth_layers * qubit_count * depth_multiplier)
#
# `sim_strategy` is the decisive field. Classical simulation cost depends far
# more on HOW the circuit is simulated than on the algorithm's name:
#
#   "statevector" : exact, stores all 2^n complex amplitudes. Correct for
#                   algorithms whose whole point is exact state evolution,
#                   and for small index registers. Memory is exponential, so
#                   it is only viable to ~30 qubits.
#
#   "mps"         : matrix-product-state / tensor-network. Correct for
#                   shallow variational circuits (QAOA, VQE, quantum kernels),
#                   which is how a 500-variable QAOA instance would actually
#                   be simulated. Memory is POLYNOMIAL in qubit count and
#                   quadratic in bond dimension, so large n stays tractable.
#
# Applying the statevector model to a 500-qubit QAOA instance yields ~1e142 GB
# — a number that is arithmetically correct and physically meaningless. The
# strategy field exists to prevent reporting it.

QUANTUM_RESOURCE_PROFILES = {
    # ---- Classification (variational / kernel -> MPS) ---------------------
    "Quantum Support Vector Machine (QSVM)": {
        "qubit_mode": "linear", "qubit_coeff": 1.0,
        "depth_layers": 2, "depth_multiplier": 2.0,
        "sim_strategy": "mps", "bond_dimension": 64,
        "calibrated": False, "calibration_host": None,
    },
    "Quantum Kernel Classifier": {
        "qubit_mode": "linear", "qubit_coeff": 1.0,
        "depth_layers": 2, "depth_multiplier": 2.0,
        "sim_strategy": "mps", "bond_dimension": 64,
        "calibrated": False, "calibration_host": None,
    },

    # ---- Regression -------------------------------------------------------
    "Variational Quantum Regressor (VQR)": {
        "qubit_mode": "linear", "qubit_coeff": 1.0,
        "depth_layers": 3, "depth_multiplier": 2.5,
        "sim_strategy": "mps", "bond_dimension": 64,
        "calibrated": False, "calibration_host": None,
    },

    # ---- Optimization -----------------------------------------------------
    "Quantum Approximate Optimization Algorithm (QAOA)": {
        "qubit_mode": "linear", "qubit_coeff": 1.0,
        "depth_layers": 3, "depth_multiplier": 2.0,
        "sim_strategy": "mps", "bond_dimension": 64,
        "calibrated": False, "calibration_host": None,
    },
    "Variational Quantum Eigensolver (VQE)": {
        "qubit_mode": "linear", "qubit_coeff": 1.0,
        "depth_layers": 4, "depth_multiplier": 2.5,
        "sim_strategy": "mps", "bond_dimension": 64,
        "calibrated": False, "calibration_host": None,
    },

    # ---- Simulation (exact state evolution IS the deliverable) ------------
    "Quantum Statevector Simulation": {
        "qubit_mode": "linear", "qubit_coeff": 1.0,
        "depth_layers": 1, "depth_multiplier": 1.0,
        "sim_strategy": "statevector", "bond_dimension": None,
        "calibrated": False, "calibration_host": None,
    },
    "Hamiltonian Simulation": {
        "qubit_mode": "linear", "qubit_coeff": 1.0,
        "depth_layers": 5, "depth_multiplier": 3.0,
        "sim_strategy": "statevector", "bond_dimension": None,
        "calibrated": False, "calibration_host": None,
    },

    # ---- Search (small index register -> statevector is fine) -------------
    "Grover Search": {
        "qubit_mode": "log2", "qubit_coeff": 1.0,
        "depth_layers": 1, "depth_multiplier": 1.0,
        "sim_strategy": "statevector", "bond_dimension": None,
        "calibrated": False, "calibration_host": None,
    },
}

# Fallback profiles for any algorithm not explicitly listed above.
DEFAULT_CLASSICAL_PROFILE = {
    "size_driver": "dataset_size", "ref_size": 10_000,
    "base_memory_gb": 0.10, "memory_exp": 1.30,
    "base_runtime_s": 4.0, "runtime_exp": 1.40,
    "gpu_capable": False, "mpi_capable": False,
        "calibrated": False, "calibration_host": None,
}
DEFAULT_QUANTUM_PROFILE = {
    "qubit_mode": "linear", "qubit_coeff": 1.0,
    "depth_layers": 2, "depth_multiplier": 2.0,
    "sim_strategy": "mps", "bond_dimension": 64,
        "calibrated": False, "calibration_host": None,
}

# ===========================================================================
# Simulation cost constants
# ===========================================================================
BYTES_PER_GB = 1024 ** 3

# complex128 amplitude = 16 bytes
STATEVECTOR_BYTES_PER_AMPLITUDE = 16

# MPS tensor storage: n_qubits sites, each holding ~2 * chi^2 complex128
# elements (two physical indices' worth of bond tensors).
MPS_TENSORS_PER_SITE = 2
MPS_BYTES_PER_ELEMENT = 16

# Effective simulation throughput used to turn gate counts into wall-clock
# seconds. A single commodity core sustains order 1e9 useful flops on this
# workload; this is the crudest constant in the file and the first one the
# calibration harness should replace.
SIM_FLOPS_PER_SECOND = 1.0e9

# Guards so an absurd qubit count cannot overflow a float or print 1e142.
MAX_STATEVECTOR_EXPONENT = 300      # beyond this, don't even compute 2**n
MAX_REPORTED_MEMORY_GB = 1.0e9      # 1 exabyte, reported as the ceiling

# ===========================================================================
# Resource Tier Thresholds (bucketed from estimated_memory_gb)
# ===========================================================================
RESOURCE_TIER_THRESHOLDS = {
    "light": 2.0,       # <= 2 GB
    "moderate": 8.0,    # <= 8 GB
    "heavy": 32.0,      # <= 32 GB
    # > 32 GB -> "extreme"
}

# ===========================================================================
# Step 7 — Hardware Mapping Thresholds
# ===========================================================================
GPU_MEMORY_THRESHOLD_GB = 4.0       # above this, prefer CUDA if supported
MPI_RUNTIME_THRESHOLD_S = 20.0      # above this, prefer MPI if supported

# Practical classical-simulation ceilings, per strategy.
# 2^30 amplitudes * 16 B = 16 GB, about the edge of a single workstation.
STATEVECTOR_QUBIT_LIMIT = 30
# MPS stays polynomial, so the ceiling is far higher and set by bond growth.
MPS_QUBIT_LIMIT = 1_000

# ===========================================================================
# Parallel speedup model
# ===========================================================================
# NOTE: the NUMBER of MPI slots is no longer a constant. It comes from
# backend/utils/hardware_detect.py, which probes the host (or reads a
# declared target deployment). Hardcoding it meant the mapper would
# recommend an 8-node cluster on a machine that had none.
#
# What remains here are the two MODELLING constants — how efficiently
# parallelism converts to speedup — which are properties of the algorithm
# and interconnect, not of a particular machine.

# Fraction of ideal linear speedup achieved across MPI ranks. 0.75 is a
# conventional figure for coarse-grained parallel workloads with modest
# communication; it is NOT measured on this deployment.
MPI_PARALLEL_EFFICIENCY = 0.75

# CUDA speedup over CPU for GPU-capable algorithms.
#
# MEASURED, not assumed: XGBoost device="cpu" vs device="cuda" on a Colab
# Tesla T4 (15 GB), identical data and hyperparameters, test accuracy
# identical on both devices at every size (so the speedup is not bought by
# computing something different):
#
#       n=10,000    2.613s -> 0.901s    2.90x
#       n=100,000   6.916s -> 1.670s    4.14x
#       n=500,000  22.167s -> 5.004s    4.43x
#
# The value below is the median of the large-input regime (n >= 100k).
# Small inputs show less: kernel launch and host-to-device transfer are
# fixed costs that a short run cannot amortise, so a single scalar is a
# simplification of a size-dependent curve.
#
# This figure does NOT transfer. It is XGBoost on a T4 against Colab's
# CPU. Different hardware, or a different algorithm, gives a different
# number — and a machine with no NVIDIA GPU cannot produce one at all,
# which is why hardware_detect.py gates its use.
GPU_SPEEDUP_FACTOR = 4.3
GPU_SPEEDUP_CALIBRATION_HOST = "Colab Tesla T4 (15 GB) vs Colab CPU, XGBoost n>=100k"

# Cap on the modelled MPI speedup regardless of slot count, so a large
# declared cluster cannot imply unbounded scaling (Amdahl's law in spirit:
# the serial fraction eventually dominates).
MAX_MPI_SPEEDUP = 32.0
