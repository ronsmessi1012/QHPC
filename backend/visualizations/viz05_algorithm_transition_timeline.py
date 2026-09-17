
import plotly.graph_objects as go

algorithms = [
    ("N-Queens",512,1024,2048),
    ("MAX-CUT",256,768,1536),
    ("Travelling Salesperson",128,512,1024),
    ("Graph Partitioning",500,1500,3000),
    ("Vehicle Routing",64,256,1024),
    ("Portfolio Optimization",128,1024,4096),
    ("Protein Folding",300,900,2000),
    ("Molecular Energy",24,48,96),
]

fig = go.Figure()

for i,(name,c,b,q) in enumerate(algorithms):

    fig.add_trace(go.Scatter(
        x=[c,b,q],
        y=[name]*3,
        mode="lines+markers+text",
        marker=dict(
            size=[16,18,20],
            color=["#22C55E","#F59E0B","#8B5CF6"]
        ),
        text=["Classical","Borderline","Quantum"],
        textposition="top center",
        name=name,
        showlegend=False,
        line=dict(width=4,color="#64748B")
    ))

fig.update_layout(
    template="plotly_dark",
    title="Algorithm Transition Timeline in Q-HPC",
    xaxis_type="log",
    xaxis_title="Problem Size / Dataset Size",
    yaxis_title="Algorithm Family",
    height=700,
    title_x=0.5
)

fig.show()