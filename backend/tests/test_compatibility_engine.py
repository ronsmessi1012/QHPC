from backend.agents.problem_analyzer import ProblemAnalyzer
from backend.agents.candidate_generator import CandidateGenerator
from backend.agents.compatibility_engine import CompatibilityEngine


analyzer = ProblemAnalyzer()
generator = CandidateGenerator()
engine = CompatibilityEngine()

queries = [
    "Optimize delivery routes for 500 cities under travel constraints.",
    "Classify spam emails using 20000 customer records.",
    "Search the shortest path in a graph containing 1000 nodes.",
    "Simulate a quantum molecule with 32 qubits."
]

for query in queries:

    print("=" * 80)
    print(query)
    print("=" * 80)

    problem = analyzer.analyze(query)
    pool = generator.generate(problem)

    result = engine.evaluate(problem, pool)

    print("Best Classical:")
    print(result.best_classical_algorithm, result.best_classical_score)

    print()

    print("Best Quantum:")
    print(result.best_quantum_algorithm, result.best_quantum_score)

    print()

    print("All Scores")
    for name, score in sorted(
        result.algorithm_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"{name} : {score}")

    print()