"""
Step V3 — AI Agent Compatibility Heatmap (Conference Paper / Journal Style)
Generates high-density publication-grade matrix heatmaps displaying algorithm compatibility
across multi-dimensional problem archetypes for both Classical and Quantum paradigms.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

from backend.models.problem_schema import ProblemSpecification
from backend.models.candidate_schema import AlgorithmCandidate, CandidatePool
from backend.agents.compatibility_engine import CompatibilityEngine
from backend.utils.constants import CLASSICAL_ALGORITHMS, QUANTUM_ALGORITHMS


def build_benchmark_archetypes():
    """
    Constructs a diverse suite of 12 real-world computational problem archetypes.
    """
    archetypes = [
        ("Small Tabular Clf (D=30)", "classification", 30, 1000, "unconstrained", False),
        ("Med High-Dim Clf (D=150)", "classification", 150, 5000, "constrained", True),
        ("Massive Spam Clf (D=20k)", "classification", 20000, 20000, "unconstrained", False),
        ("Linear Regression (D=20)", "regression", 20, 2000, "unconstrained", False),
        ("Nonlinear Reg (D=500)", "regression", 500, 10000, "mixed", True),
        ("Exact Combinatorial (D=50)", "optimization", 50, 50, "constrained", True),
        ("Routing Opt (D=500)", "optimization", 500, 500, "constrained", True),
        ("Large Discrete Opt (D=1200)", "optimization", 1200, 1200, "constrained", True),
        ("32-Qubit Molecule Sim", "simulation", 32, 32, "constrained", True),
        ("PDE Grid Simulation (D=256)", "simulation", 256, 256, "unconstrained", True),
        ("Unstructured Search (D=1000)", "search", 1000, 1000, "unconstrained", True),
        ("Shortest Path Graph (D=250)", "search", 250, 250, "mixed", False),
    ]

    return archetypes


def compute_compatibility_matrix():
    """
    Evaluates every registered classical and quantum algorithm against all archetypes.
    """
    archetypes = build_benchmark_archetypes()
    engine = CompatibilityEngine()

    # Gather unique algorithm names grouped by family
    classical_names = []
    for algos in CLASSICAL_ALGORITHMS.values():
        for a in algos:
            if a["name"] not in classical_names:
                classical_names.append(a["name"])

    quantum_names = []
    for algos in QUANTUM_ALGORITHMS.values():
        for a in algos:
            if a["name"] not in quantum_names:
                quantum_names.append(a["name"])

    all_algorithms = classical_names + quantum_names
    archetype_labels = [a[0] for a in archetypes]

    score_matrix = np.zeros((len(all_algorithms), len(archetypes)))

    # Dummy pool containing all candidates
    for col_idx, arch in enumerate(archetypes):
        name, p_class, dim, dset, con, q_cand = arch
        problem = ProblemSpecification(
            user_query=name,
            problem_class=p_class,
            confidence=0.90,
            dimensionality=dim,
            dataset_size=dset,
            linearity="unknown",
            sparsity="unknown",
            constraint_structure=con,
            quantum_candidate=q_cand,
            reasoning="Archetype benchmark evaluation.",
        )

        dim_cat = engine._dimension_category(dim)

        for row_idx, algo_name in enumerate(all_algorithms):
            # Check if this algorithm belongs to the problem class
            prior = 0.50
            if algo_name in [a["name"] for a in CLASSICAL_ALGORITHMS.get(p_class, [])]:
                prior = next(a["score"] for a in CLASSICAL_ALGORITHMS[p_class] if a["name"] == algo_name)
            elif algo_name in [a["name"] for a in QUANTUM_ALGORITHMS.get(p_class, [])]:
                prior = next(a["score"] for a in QUANTUM_ALGORITHMS[p_class] if a["name"] == algo_name)
            else:
                # Algorithm is for a different class -> low base fit
                prior = 0.20

            score = engine._compatibility_score(prior, algo_name, dim_cat)
            score_matrix[row_idx, col_idx] = score

    df = pd.DataFrame(score_matrix, index=all_algorithms, columns=archetype_labels)
    return df, len(classical_names), len(quantum_names)


def generate_conference_style_heatmap(df, num_classical, num_quantum, output_png: Path, output_pdf: Path):
    """
    Renders IEEE / ACM / Nature conference paper style matrix heatmap.
    Ensures paradigm section headers and algorithm tick labels are clearly separated and legible.
    """
    fig, ax = plt.subplots(figsize=(16.5, 11.5), dpi=300)

    # Style
    plt.rcParams["font.family"] = "sans-serif"
    bg_color = "#0B0F19"
    card_color = "#111827"
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(card_color)

    # Seaborn heatmap
    cmap = sns.color_palette("mako", as_cmap=True)
    sns.heatmap(
        df,
        ax=ax,
        cmap=cmap,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 9.0, "weight": "bold", "color": "#FFFFFF"},
        linewidths=0.75,
        linecolor="#1F2937",
        cbar_kws={"label": "Compatibility Score ($S_{compat}$)", "shrink": 0.75, "pad": 0.08},
    )

    # Dividing line between classical and quantum
    ax.axhline(num_classical, color="#00F0FF", linewidth=3.0, linestyle="-")

    # Format y-axis tick labels and color-code by paradigm
    y_labels = ax.get_yticklabels()
    for i, label in enumerate(y_labels):
        if i < num_classical:
            label.set_color("#F59E0B")  # Amber for Classical
            label.set_fontweight("bold")
        else:
            label.set_color("#00F0FF")  # Cyan for Quantum
            label.set_fontweight("bold")

    # Dedicated Paradigm Badges on the right margin (between matrix and colorbar)
    total_rows = len(df)
    classical_center = 1.0 - (num_classical / (2.0 * total_rows))
    quantum_center = 1.0 - (num_classical + num_quantum / 2.0) / total_rows

    ax.text(
        1.015,
        classical_center,
        "CLASSICAL ALGORITHMS (14)",
        transform=ax.transAxes,
        rotation=-90,
        va="center",
        ha="left",
        fontsize=10.5,
        fontweight="bold",
        color="#F59E0B",
        bbox=dict(boxstyle="round,pad=0.4", fc="#1F2937", ec="#F59E0B", lw=1.5),
    )

    ax.text(
        1.015,
        quantum_center,
        "QUANTUM ALGORITHMS (8)",
        transform=ax.transAxes,
        rotation=-90,
        va="center",
        ha="left",
        fontsize=10.5,
        fontweight="bold",
        color="#00F0FF",
        bbox=dict(boxstyle="round,pad=0.4", fc="#1F2937", ec="#00F0FF", lw=1.5),
    )

    # Formatting ticks and titles
    ax.set_title(
        "Step V3 — Multi-Paradigm Algorithm Compatibility Matrix (Conference Paper Benchmark)",
        fontsize=15,
        fontweight="bold",
        color="#FFFFFF",
        pad=22,
    )
    ax.tick_params(axis="x", labelrotation=32, labelsize=10, colors="#E5E7EB")
    ax.tick_params(axis="y", labelsize=10, colors="#E5E7EB", pad=8)
    plt.setp(ax.get_xticklabels(), ha="right")

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors="#E5E7EB", labelsize=9.5)
    cbar.set_label("Compatibility Score ($S_{compat}$)", color="#E5E7EB", fontsize=11, fontweight="bold", labelpad=12)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.savefig(output_pdf, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[✓] Saved conference publication heatmap to: {output_png} and {output_pdf}")


def generate_interactive_heatmap(df, num_classical, output_html: Path):
    """
    Renders interactive Plotly heatmap with custom hovercards.
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=df.values,
            x=df.columns,
            y=df.index,
            colorscale="Viridis",
            colorbar=dict(title="<b>Compatibility</b>", tickfont=dict(color="#FFFFFF")),
            hovertemplate="<b>Algorithm:</b> %{y}<br><b>Archetype:</b> %{x}<br><b>Score:</b> %{z:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B0F19",
        plot_bgcolor="#111827",
        title=dict(
            text="<b>Step V3 — Interactive AI Agent Compatibility Matrix & Cross-Archetype Benchmark</b>",
            font=dict(size=18, color="#FFFFFF"),
            x=0.03,
            y=0.97,
        ),
        xaxis=dict(tickangle=-30, tickfont=dict(color="#E5E7EB")),
        yaxis=dict(tickfont=dict(color="#E5E7EB")),
        margin=dict(l=220, r=40, t=90, b=120),
        height=850,
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_html))
    print(f"[✓] Saved interactive heatmap to: {output_html}")


def main():
    df, num_classical, num_quantum = compute_compatibility_matrix()
    out_dir = Path("/Users/ronnie/Documents/QHPC/outputs/visualizations")
    out_dir.mkdir(parents=True, exist_ok=True)

    generate_conference_style_heatmap(
        df,
        num_classical,
        num_quantum,
        out_dir / "v3_agent_compatibility_heatmap.png",
        out_dir / "v3_agent_compatibility_heatmap.pdf",
    )
    generate_interactive_heatmap(df, num_classical, out_dir / "v3_agent_compatibility_heatmap.html")


if __name__ == "__main__":
    main()
