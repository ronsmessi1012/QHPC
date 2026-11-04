from backend.agents.problem_analyzer import ProblemAnalyzer
from backend.agents.candidate_generator import CandidateGenerator
from backend.agents.feature_scoring_engine import FeatureScoringEngine


analyzer = ProblemAnalyzer()
generator = CandidateGenerator()
engine = FeatureScoringEngine()

queries = [
    "Optimize delivery routes for 500 cities under travel constraints.",
    "Classify spam emails using 20000 customer records.",
    "Simulate a quantum molecule with 32 qubits.",
]

for query in queries:

    print("=" * 80)
    print(query)
    print("=" * 80)

    problem = analyzer.analyze(query)
    pool = generator.generate(problem)

    scores = engine.score(problem, pool)

    print(scores.model_dump())
    print()