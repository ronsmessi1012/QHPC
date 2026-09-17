from backend.agents.problem_analyzer import ProblemAnalyzer
from backend.agents.candidate_generator import CandidateGenerator

problem_analyzer = ProblemAnalyzer()
candidate_generator = CandidateGenerator()

queries = [
    "Classify spam emails using 20000 customer records.",
    "Optimize delivery routes for 500 cities under travel constraints.",
    "Simulate a quantum molecule with 32 qubits.",
    "Search the shortest path in a graph containing 1000 nodes.",
]

for query in queries:

    print("=" * 80)
    print("USER QUERY")
    print(query)
    print("=" * 80)

    problem = problem_analyzer.analyze(query)

    print("\nProblem Specification")
    print(problem.model_dump())

    pool = candidate_generator.generate(problem)

    print("\nClassical Candidates")
    for algo in pool.classical_candidates:
        print(f" • {algo.name} ({algo.prior_score})")

    print("\nQuantum Candidates")
    if len(pool.quantum_candidates) == 0:
        print(" • None")
    else:
        for algo in pool.quantum_candidates:
            print(f" • {algo.name} ({algo.prior_score})")

    print("\n")