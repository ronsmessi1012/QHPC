from backend.agents.problem_analyzer import ProblemAnalyzer
from backend.agents.candidate_generator import CandidateGenerator
from backend.agents.feature_scoring_engine import FeatureScoringEngine
from backend.agents.compatibility_engine import CompatibilityEngine
from backend.routers.borderline_router import BorderlineRouter
from backend.agents.arbitration_context_builder import ArbitrationContextBuilder
from backend.agents.agent_selector import AgentSelector


query = "Optimize delivery routes for 500 cities under travel constraints."

analyzer = ProblemAnalyzer()
generator = CandidateGenerator()
feature_engine = FeatureScoringEngine()
compatibility_engine = CompatibilityEngine()
router = BorderlineRouter()
builder = ArbitrationContextBuilder()
selector = AgentSelector()

problem = analyzer.analyze(query)
pool = generator.generate(problem)

features = feature_engine.score(problem, pool)
compatibility = compatibility_engine.evaluate(problem, pool)
borderline = router.route(compatibility)

context = builder.build(
    problem,
    features,
    compatibility,
    borderline,
)

decision = selector.select(context)

print(decision.model_dump_json(indent=4))