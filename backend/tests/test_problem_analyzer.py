import sys
from pathlib import Path

# Add backend directory to sys.path so modules can be imported directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.problem_analyzer import ProblemAnalyzer


analyzer = ProblemAnalyzer()

sample_queries = [
    "Optimize delivery routes for 500 cities with travel constraints.",
    "Classify spam emails using customer data of 20000 samples.",
    "Predict house prices using 5000 records.",
    "Simulate a quantum molecule with 32 qubits.",
]

for query in sample_queries:
    print("=" * 60)
    print("INPUT:", query)
    print()

    result = analyzer.analyze(query)

    print(result.model_dump_json(indent=4))
    print()