"""
Step V2 — Borderline Score Decision Surface (Interactive 3D & 2D Contours)
Generates an analytical 3D decision landscape mapping Dimensionality and Constraint/Complexity
to the Borderline Score (B) and Framework Routing Regimes.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go


def generate_decision_surface_data():
    """
    Computes a 2D grid across Dimensionality (10 to 1000) and Constraint/Complexity (0.0 to 1.0),
    evaluating deterministic Classical vs Quantum compatibility margins and Borderline Score B.
    """
    dim_vals = np.linspace(10, 1000, 50)
    complexity_vals = np.linspace(0.1, 1.0, 50)
    D, C = np.meshgrid(dim_vals, complexity_vals)

    # Classical compatibility profile: high for medium/large tabular, decays on extreme constraints/quantum simulation
    classical_score = 0.85 + 0.08 * (1.0 - C**1.5) + 0.05 * np.exp(-((D - 300) / 400) ** 2)
    classical_score = np.clip(classical_score, 0.50, 0.96)

    # Quantum compatibility profile: climbs rapidly with constraint density C and scale D
    quantum_score = 0.70 + 0.26 * (C**1.2) + 0.04 * (D / 1000.0)
    quantum_score = np.clip(quantum_score, 0.50, 0.98)

    margin = np.abs(classical_score - quantum_score)
    borderline_score = 1.0 - margin

    return D, C, classical_score, quantum_score, margin, borderline_score


def generate_static_3d_surface(D, C, borderline_score, output_path: Path, show_gui: bool = False):
    """
    Renders publication-quality 3D decision surface using Matplotlib.
    Ensures zero text occlusion by using clean label padding, high-visibility 3D markers,
    and a 2D HUD overlay for benchmark details.
    """
    fig = plt.figure(figsize=(15, 10.5), dpi=300)
    ax = fig.add_subplot(111, projection="3d")

    bg_color = "#0B0F19"
    card_color = "#111827"
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(card_color)

    # 3D Surface Plot with subtle grid lines
    surf = ax.plot_surface(
        D,
        C,
        borderline_score,
        cmap="viridis",
        edgecolor="#1F2937",
        linewidth=0.3,
        alpha=0.82,
        antialiased=True,
        rstride=2,
        cstride=2,
        zorder=2,
    )

    # 2D contour projection at base plane
    z_min = float(np.min(borderline_score)) - 0.08
    ax.contourf(
        D,
        C,
        borderline_score,
        zdir="z",
        offset=z_min,
        cmap="viridis",
        alpha=0.45,
        zorder=1,
    )

    # Benchmark test problems
    benchmarks = [
        {"name": "500-City Routing (N=500)", "d": 500, "c": 0.95, "b": 0.944, "color": "#00F0FF", "winner": "Quantum (QAOA)"},
        {"name": "20k Spam Clf (N=900)", "d": 900, "c": 0.40, "b": 0.905, "color": "#F59E0B", "winner": "Classical (XGBoost)"},
        {"name": "32-Qubit Molecule (N=32)", "d": 32, "c": 0.90, "b": 0.952, "color": "#EC4899", "winner": "Quantum (Statevector)"},
        {"name": "1000-Node Search (N=1000)", "d": 1000, "c": 0.65, "b": 0.962, "color": "#10B981", "winner": "Quantum (Grover)"},
    ]

    # Plot distinct benchmark scatter spheres
    for bm in benchmarks:
        ax.scatter(
            [bm["d"]],
            [bm["c"]],
            [bm["b"]],
            color=bm["color"],
            s=140,
            edgecolors="#FFFFFF",
            linewidth=2.0,
            depthshade=False,
            zorder=10,
        )

    # 2D HUD Legend overlay on top-right (never gets occluded by 3D mesh)
    hud_text = "BENCHMARK PROBLEM ARCHETYPES:\n" + "\n".join([
        f"• {bm['name']}\n   Winner: {bm['winner']} | B: {bm['b']:.3f}"
        for bm in benchmarks
    ])

    ax.text2D(
        0.03,
        0.88,
        hud_text,
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        color="#E5E7EB",
        bbox=dict(boxstyle="round,pad=0.6", fc="#1F2937", ec="#374151", lw=1.5, alpha=0.92),
        zorder=20,
    )

    # Formatting axis labels with generous padding so they never clip
    ax.set_xlabel("Problem Dimensionality ($D$)", fontsize=11.5, fontweight="bold", color="#E5E7EB", labelpad=14)
    ax.set_ylabel("Constraint & Complexity Fit", fontsize=11.5, fontweight="bold", color="#E5E7EB", labelpad=14)
    ax.set_zlabel("Borderline Score ($B = 1 - \\Delta S$)", fontsize=11.5, fontweight="bold", color="#E5E7EB", labelpad=12)
    ax.set_zlim(z_min, 1.01)

    ax.set_title(
        "Step V2 — 3D Borderline Decision Surface & Multi-Parametric Landscape",
        fontsize=15,
        fontweight="bold",
        color="#FFFFFF",
        pad=22,
    )

    # Tick and Pane styling
    ax.tick_params(colors="#9CA3AF", labelsize=9.5)
    ax.xaxis.pane.set_edgecolor("#374151")
    ax.yaxis.pane.set_edgecolor("#374151")
    ax.zaxis.pane.set_edgecolor("#374151")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Elevated angle for clear visibility
    ax.view_init(elev=26, azim=-50)

    # Colorbar
    cb = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=14, pad=0.08)
    cb.set_label("Borderline Similarity Score ($B$)", color="#E5E7EB", fontsize=11, fontweight="bold", labelpad=10)
    cb.ax.tick_params(colors="#E5E7EB", labelsize=9.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[✓] Saved high-resolution 3D decision surface to: {output_path}")

    if show_gui:
        print("[ℹ] Opening interactive 3D Pyplot window (Rotate with mouse click-and-drag)...")
        plt.show()
    else:
        plt.close()


def generate_interactive_3d_surface(D, C, borderline_score, classical_score, quantum_score, output_path: Path):
    """
    Renders fully interactive 3D WebGL Decision Surface using Plotly.
    """
    fig = go.Figure()

    # 3D Surface
    fig.add_trace(
        go.Surface(
            x=D,
            y=C,
            z=borderline_score,
            colorscale="Viridis",
            name="Borderline Surface",
            colorbar=dict(
                title=dict(text="<b>Borderline Score (B)</b>", font=dict(color="#FFFFFF", size=13)),
                tickfont=dict(color="#FFFFFF"),
                len=0.75,
            ),
            hovertemplate=(
                "<b>Dimensionality:</b> %{x:.0f}<br>"
                "<b>Constraint/Complexity:</b> %{y:.2f}<br>"
                "<b>Borderline Score (B):</b> %{z:.3f}<extra></extra>"
            ),
        )
    )

    # Benchmark test points
    benchmarks = [
        ("500-City Delivery Optimization", 500, 0.95, 0.944, "Quantum Favored (QAOA)", "#00F0FF"),
        ("20,000-Record Spam Classification", 900, 0.40, 0.905, "Classical Favored (XGBoost)", "#F59E0B"),
        ("32-Qubit Molecule Simulation", 32, 0.90, 0.952, "Quantum Favored (Statevector)", "#EC4899"),
        ("1,000-Node Shortest Path Search", 1000, 0.65, 0.962, "Quantum Favored (Grover)", "#10B981"),
    ]

    bx = [b[1] for b in benchmarks]
    by = [b[2] for b in benchmarks]
    bz = [b[3] for b in benchmarks]
    btext = [f"<b>{b[0]}</b><br>Regime: {b[4]}<br>B: {b[3]}" for b in benchmarks]
    bcolor = [b[5] for b in benchmarks]

    fig.add_trace(
        go.Scatter3d(
            x=bx,
            y=by,
            z=bz,
            mode="markers+text",
            marker=dict(size=9, color=bcolor, symbol="diamond", line=dict(color="#FFFFFF", width=2)),
            text=[b[0] for b in benchmarks],
            textposition="top center",
            textfont=dict(color="#FFFFFF", size=11),
            hovertext=btext,
            hoverinfo="text",
            name="Benchmark Problems",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B0F19",
        title=dict(
            text="<b>Step V2 — Interactive 3D Borderline Score Decision Surface & Threshold Planes</b>",
            font=dict(size=20, color="#FFFFFF"),
            x=0.03,
            y=0.96,
        ),
        scene=dict(
            xaxis=dict(
                title="<b>Dimensionality (D)</b>",
                backgroundcolor="#111827",
                gridcolor="#374151",
                showbackground=True,
            ),
            yaxis=dict(
                title="<b>Constraint & Complexity Fit</b>",
                backgroundcolor="#111827",
                gridcolor="#374151",
                showbackground=True,
            ),
            zaxis=dict(
                title="<b>Borderline Score (B)</b>",
                backgroundcolor="#111827",
                gridcolor="#374151",
                showbackground=True,
            ),
            camera=dict(eye=dict(x=-1.5, y=-1.5, z=1.1)),
        ),
        margin=dict(l=40, r=40, t=90, b=40),
        height=850,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    print(f"[✓] Saved interactive 3D decision surface to: {output_path}")


def main():
    import sys
    show_gui = "--gui" in sys.argv or "--show" in sys.argv

    D, C, classical_score, quantum_score, margin, borderline_score = generate_decision_surface_data()
    out_dir = Path("/Users/ronnie/Documents/QHPC/outputs/visualizations")
    out_dir.mkdir(parents=True, exist_ok=True)

    generate_static_3d_surface(D, C, borderline_score, out_dir / "v2_borderline_decision_surface_3d.png", show_gui=show_gui)
    generate_interactive_3d_surface(
        D, C, borderline_score, classical_score, quantum_score, out_dir / "v2_borderline_decision_surface_3d.html"
    )


if __name__ == "__main__":
    main()
