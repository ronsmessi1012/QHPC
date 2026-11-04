from backend.agents.problem_analyzer import ProblemAnalyzer
from backend.agents.candidate_generator import CandidateGenerator
from backend.agents.compatibility_engine import CompatibilityEngine
from backend.routers.borderline_router import BorderlineRouter


analyzer = ProblemAnalyzer()
generator = CandidateGenerator()
compatibility_engine = CompatibilityEngine()
router = BorderlineRouter()

queries = [
    "Optimize delivery routes for 500 cities under travel constraints.",
    "Classify spam emails using 20000 customer records.",
    "Search the shortest path in a graph containing 1000 nodes.",
]

for query in queries:

    print("=" * 80)
    print(query)
    print("=" * 80)

    problem = analyzer.analyze(query)
    pool = generator.generate(problem)

    compatibility = compatibility_engine.evaluate(problem, pool)
    result = router.route(compatibility)

    print(result.model_dump())
    print()