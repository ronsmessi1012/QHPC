"""
Algorithm Compatibility Knowledge Base

These values represent deterministic expert priors used by the
Constraint Compatibility Engine before LLM arbitration.
"""

ALGORITHM_COMPATIBILITY = {

    # ===============================
    # CLASSICAL ALGORITHMS
    # ===============================

    "XGBoost": {
        "constraint_fit": 0.90,
        "complexity_fit": 0.85,
        "preferred_dimension": "very_large",
    },

    "Random Forest": {
        "constraint_fit": 0.85,
        "complexity_fit": 0.80,
        "preferred_dimension": "large",
    },

    "SVM": {
        "constraint_fit": 0.80,
        "complexity_fit": 0.70,
        "preferred_dimension": "medium",
    },

    "Logistic Regression": {
        "constraint_fit": 0.75,
        "complexity_fit": 0.55,
        "preferred_dimension": "small",
    },

    "Linear Regression": {
        "constraint_fit": 0.70,
        "complexity_fit": 0.50,
        "preferred_dimension": "small",
    },

    "Random Forest Regressor": {
        "constraint_fit": 0.82,
        "complexity_fit": 0.78,
        "preferred_dimension": "large",
    },

    "XGBoost Regressor": {
        "constraint_fit": 0.90,
        "complexity_fit": 0.85,
        "preferred_dimension": "very_large",
    },

    "Branch and Bound": {
        "constraint_fit": 0.95,
        "complexity_fit": 0.82,
        "preferred_dimension": "medium",
    },

    "Genetic Algorithm": {
        "constraint_fit": 0.92,
        "complexity_fit": 0.90,
        "preferred_dimension": "large",
    },

    "Simulated Annealing": {
        "constraint_fit": 0.85,
        "complexity_fit": 0.82,
        "preferred_dimension": "large",
    },

    "Monte Carlo Simulation": {
        "constraint_fit": 0.80,
        "complexity_fit": 0.92,
        "preferred_dimension": "large",
    },

    "Finite Difference Solver": {
        "constraint_fit": 0.78,
        "complexity_fit": 0.88,
        "preferred_dimension": "medium",
    },

    "A* Search": {
        "constraint_fit": 0.90,
        "complexity_fit": 0.84,
        "preferred_dimension": "large",
    },

    "Dijkstra": {
        "constraint_fit": 0.82,
        "complexity_fit": 0.78,
        "preferred_dimension": "medium",
    },

    # ===============================
    # QUANTUM ALGORITHMS
    # ===============================

    "Quantum Approximate Optimization Algorithm (QAOA)": {
        "constraint_fit": 0.98,
        "complexity_fit": 0.95,
        "preferred_dimension": "large",
    },

    "Variational Quantum Eigensolver (VQE)": {
        "constraint_fit": 0.75,
        "complexity_fit": 0.90,
        "preferred_dimension": "medium",
    },

    "Quantum Support Vector Machine (QSVM)": {
        "constraint_fit": 0.88,
        "complexity_fit": 0.83,
        "preferred_dimension": "medium",
    },

    "Quantum Kernel Classifier": {
        "constraint_fit": 0.86,
        "complexity_fit": 0.80,
        "preferred_dimension": "medium",
    },

    "Variational Quantum Regressor (VQR)": {
        "constraint_fit": 0.82,
        "complexity_fit": 0.82,
        "preferred_dimension": "medium",
    },

    "Quantum Statevector Simulation": {
        "constraint_fit": 0.90,
        "complexity_fit": 0.99,
        "preferred_dimension": "small",
    },

    "Hamiltonian Simulation": {
        "constraint_fit": 0.90,
        "complexity_fit": 0.96,
        "preferred_dimension": "small",
    },

    "Grover Search": {
        "constraint_fit": 0.95,
        "complexity_fit": 0.92,
        "preferred_dimension": "large",
    },
}