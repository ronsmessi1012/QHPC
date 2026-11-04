from backend.utils.compatibility_library import ALGORITHM_COMPATIBILITY

algorithms = [
    "Quantum Approximate Optimization Algorithm (QAOA)",
    "Genetic Algorithm",
    "XGBoost",
    "Grover Search",
]

for algorithm in algorithms:
    print("=" * 60)
    print(algorithm)
    print(ALGORITHM_COMPATIBILITY[algorithm])
    print()