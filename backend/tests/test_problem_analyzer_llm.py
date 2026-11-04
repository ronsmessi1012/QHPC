from backend.agents.problem_analyzer import ProblemAnalyzer

analyzer = ProblemAnalyzer()

query = "Optimize delivery routes for 500 cities while minimizing fuel cost under travel constraints."

result = analyzer.analyze(query)

print(result.model_dump_json(indent=4))