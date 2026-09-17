
import plotly.graph_objects as go
import numpy as np

n = np.linspace(10,4096,300)

classical = n*np.log2(n)

quantum = np.sqrt(n)*70

borderline = np.full_like(n,1500)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=n,
    y=classical,
    name="Classical Complexity",
    line=dict(color="#22C55E",width=4)
))

fig.add_trace(go.Scatter(
    x=n,
    y=quantum,
    name="Quantum Complexity",
    line=dict(color="#8B5CF6",width=4)
))

fig.add_trace(go.Scatter(
    x=n,
    y=borderline,
    name="Borderline Threshold",
    line=dict(color="#F59E0B",dash="dash")
))

fig.update_layout(
    template="plotly_dark",
    title="Computational Scaling in Q-HPC",
    xaxis_title="Problem Size",
    yaxis_title="Estimated Computational Cost",
    height=550,
    title_x=0.5
)

fig.show()