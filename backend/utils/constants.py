"""
Central algorithm registry for the Q-HPC framework.

This file contains deterministic metadata for all supported algorithms.
No decision logic belongs here.
"""

CLASSICAL_ALGORITHMS = {

    "classification": [
        {
            "name": "XGBoost",
            "score": 0.95,
            "complexity": "O(n log n)",
            "justification": "Excellent performance on structured tabular datasets."
        },
        {
            "name": "Random Forest",
            "score": 0.90,
            "complexity": "O(T × n log n)",
            "justification": "Robust ensemble method with low overfitting."
        },
        {
            "name": "SVM",
            "score": 0.86,
            "complexity": "O(n² to n³)",
            "justification": "Effective for high-dimensional classification tasks."
        },
        {
            "name": "Logistic Regression",
            "score": 0.80,
            "complexity": "O(nd)",
            "justification": "Strong interpretable linear baseline."
        }
    ],

    "regression": [
        {
            "name": "Linear Regression",
            "score": 0.82,
            "complexity": "O(nd²)",
            "justification": "Simple baseline for continuous prediction."
        },
        {
            "name": "Random Forest Regressor",
            "score": 0.88,
            "complexity": "O(T × n log n)",
            "justification": "Captures nonlinear relationships."
        },
        {
            "name": "XGBoost Regressor",
            "score": 0.93,
            "complexity": "O(n log n)",
            "justification": "High-performance gradient boosting regression."
        }
    ],

    "optimization": [
        {
            "name": "Genetic Algorithm",
            "score": 0.87,
            "complexity": "O(P × G)",
            "justification": "Population-based optimization baseline."
        },
        {
            "name": "Simulated Annealing",
            "score": 0.84,
            "complexity": "Problem dependent",
            "justification": "Handles nonlinear constrained optimization."
        },
        {
            "name": "Branch and Bound",
            "score": 0.90,
            "complexity": "Exponential worst case",
            "justification": "Classical exact solver for combinatorial optimization."
        }
    ],

    "simulation": [
        {
            "name": "Monte Carlo Simulation",
            "score": 0.91,
            "complexity": "O(N)",
            "justification": "Widely used probabilistic simulation baseline."
        },
        {
            "name": "Finite Difference Solver",
            "score": 0.85,
            "complexity": "Grid dependent",
            "justification": "Numerical simulation for physical systems."
        }
    ],

    "search": [
        {
            "name": "A* Search",
            "score": 0.92,
            "complexity": "O(E log V)",
            "justification": "Efficient informed graph search."
        },
        {
            "name": "Dijkstra",
            "score": 0.88,
            "complexity": "O(E log V)",
            "justification": "Shortest path baseline."
        }
    ]
}

QUANTUM_ALGORITHMS = {

    "classification": [
        {
            "name": "Quantum Support Vector Machine (QSVM)",
            "score": 0.89,
            "complexity": "Kernel evaluation + quantum feature map",
            "justification": "Quantum kernel method for classification in high-dimensional feature spaces."
        },
        {
            "name": "Quantum Kernel Classifier",
            "score": 0.86,
            "complexity": "Quantum kernel estimation",
            "justification": "Maps classical data into quantum feature space."
        }
    ],

    "regression": [
        {
            "name": "Variational Quantum Regressor (VQR)",
            "score": 0.84,
            "complexity": "Parameterized quantum circuit optimization",
            "justification": "Regression using variational quantum circuits."
        }
    ],

    "optimization": [
        {
            "name": "Quantum Approximate Optimization Algorithm (QAOA)",
            "score": 0.94,
            "complexity": "Depends on circuit depth p and qubit count",
            "justification": "Designed for constrained combinatorial optimization problems."
        },
        {
            "name": "Variational Quantum Eigensolver (VQE)",
            "score": 0.83,
            "complexity": "Variational circuit optimization",
            "justification": "Optimization through variational quantum circuits."
        }
    ],

    "simulation": [
        {
            "name": "Quantum Statevector Simulation",
            "score": 0.95,
            "complexity": "O(2^n)",
            "justification": "Exact simulation of quantum state evolution."
        },
        {
            "name": "Hamiltonian Simulation",
            "score": 0.90,
            "complexity": "Circuit-depth dependent",
            "justification": "Simulation of quantum systems and molecular dynamics."
        }
    ],

    "search": [
        {
            "name": "Grover Search",
            "score": 0.94,
            "complexity": "O(√N)",
            "justification": "Quantum search providing quadratic speedup for unstructured search."
        }
    ]
}