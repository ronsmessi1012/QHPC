
import plotly.graph_objects as go
import pandas as pd

problems = [
    ("N-Queens", 512, "Classical"),
    ("N-Queens", 1024, "Borderline"),
    ("N-Queens", 2048, "Quantum"),

    ("Travelling Salesperson", 128, "Classical"),
    ("Travelling Salesperson", 512, "Borderline"),
    ("Travelling Salesperson", 1024, "Quantum"),

    ("MAX-CUT", 256, "Classical"),
    ("MAX-CUT", 768, "Borderline"),
    ("MAX-CUT", 1536, "Quantum"),

    ("Graph Partitioning", 500, "Classical"),
    ("Graph Partitioning", 1500, "Borderline"),
    ("Graph Partitioning", 3000, "Quantum"),

    ("Molecular Energy", 24, "Classical"),
    ("Molecular Energy", 48, "Borderline"),
    ("Molecular Energy", 96, "Quantum"),

    ("Classification (Spam)", 20000, "Classical"),
    ("Classification (Spam)", 200000, "Borderline"),
    ("Classification (Spam)", 1000000, "Quantum"),

    ("Portfolio Optimization", 128, "Classical"),
    ("Portfolio Optimization", 1024, "Borderline"),
    ("Portfolio Optimization", 4096, "Quantum"),
]

df = pd.DataFrame(problems, columns=["Problem","Size","Route"])

colors = {
    "Classical":"#22C55E",
    "Borderline":"#F59E0B",
    "Quantum":"#8B5CF6"
}

fig = go.Figure()

for route in colors:
    d = df[df.Route==route]
    fig.add_trace(go.Scatter(
        x=d.Size,
        y=d.Problem,
        mode="markers+lines",
        marker=dict(size=18,color=colors[route],line=dict(width=2,color="white")),
        name=route,
        text=d.Size
    ))

fig.update_layout(
    title="Q-HPC AI Agent Borderline Decision Landscape",
    xaxis_type="log",
    xaxis_title="Problem Size (log scale)",
    yaxis_title="Computational Problems",
    template="plotly_dark",
    height=650,
    title_x=0.5
)

fig.show()