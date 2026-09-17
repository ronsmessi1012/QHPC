
import plotly.express as px
import pandas as pd

data = pd.DataFrame({
    "Problem":[
        "MAX-CUT","TSP","N-Queens","Graph Partitioning",
        "Molecular Energy","Classification","Portfolio","Grover Search"
    ],
    "Classical":[0.88,0.90,0.92,0.87,0.80,0.95,0.89,0.83],
    "Quantum":[0.96,0.95,0.94,0.95,0.98,0.86,0.95,0.99]
})

heat = data.set_index("Problem")

fig = px.imshow(
    heat,
    text_auto=".2f",
    color_continuous_scale="Plasma",
    aspect="auto"
)

fig.update_layout(
    template="plotly_dark",
    title="AI Agent Compatibility Heatmap",
    title_x=0.5,
    height=550
)

fig.show()