
import plotly.graph_objects as go
import numpy as np

sizes = np.arange(0,4097,64)

threshold = 1536

borderline_low = threshold-300
borderline_high = threshold+300

routes = []

for s in sizes:
    if s < borderline_low:
        routes.append(0)
    elif s <= borderline_high:
        routes.append(1)
    else:
        routes.append(2)

labels = ["Classical","Borderline","Quantum"]

colors = ["#22C55E","#F59E0B","#8B5CF6"]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=sizes,
    y=routes,
    mode="lines",
    line=dict(color="#60A5FA",width=6),
    fill="tozeroy"
))

fig.add_vrect(
    x0=borderline_low,
    x1=borderline_high,
    fillcolor="#F59E0B",
    opacity=0.25,
    annotation_text="AI Agent Arbitration Zone",
    annotation_position="top left"
)

fig.add_vline(
    x=threshold,
    line_dash="dash",
    line_color="#F59E0B",
    line_width=3
)

fig.update_layout(
    template="plotly_dark",
    title="AI Agent Decision Boundary",
    xaxis_title="Problem Dimension / Dataset Size",
    yaxis=dict(
        tickvals=[0,1,2],
        ticktext=labels,
        title="Execution Backend"
    ),
    height=500,
    title_x=0.5
)

fig.show()