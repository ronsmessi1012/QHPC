"""
Master Visualizations Generator
Executes all visualization modules and produces PNG, PDF, and interactive HTML artifacts in outputs/visualizations.
"""

from pathlib import Path
from rich.console import Console
from rich.table import Table

from backend.visualizations.v1_borderline_switching_curve import main as run_v1
from backend.visualizations.v2_decision_surface_3d import main as run_v2
from backend.visualizations.v3_compatibility_heatmap import main as run_v3
from backend.visualizations.v4_agent_impact_network import main as run_v4

console = Console()


def run_all():
    out_dir = Path("/Users/ronnie/Documents/QHPC/outputs/visualizations")
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold cyan]========================================================[/bold cyan]")
    console.print("[bold cyan]       Q-HPC Master Visualizations Suite Pipeline       [/bold cyan]")
    console.print("[bold cyan]========================================================[/bold cyan]\n")

    console.print("[bold yellow]Generating Step V1: Dynamic Boundary & Fluctuation Curve...[/bold yellow]")
    run_v1()

    console.print("\n[bold yellow]Generating Step V2: Borderline Score Decision Surface (3D)...[/bold yellow]")
    run_v2()

    console.print("\n[bold yellow]Generating Step V3: AI Agent Compatibility Heatmap...[/bold yellow]")
    run_v3()

    console.print("\n[bold yellow]Generating Step V4: AI Agent Impact Network...[/bold yellow]")
    run_v4()

    table = Table(title="Generated Q-HPC Visualizations Registry", border_style="cyan")
    table.add_column("Step", style="bold green", justify="left")
    table.add_column("Artifact Name", style="white")
    table.add_column("Type", style="magenta")
    table.add_column("Description", style="yellow")

    table.add_row(
        "Step V1",
        "v1_borderline_switching_curve.png / .html",
        "Static PNG + Interactive HTML",
        "Fine-grained Classical vs Quantum Switching & NISQ Oscillation Curve",
    )
    table.add_row(
        "Step V2",
        "v2_borderline_decision_surface_3d.png / .html",
        "Static 3D + Interactive WebGL 3D",
        "3D Decision Surface & Threshold Planes across Dimension & Complexity",
    )
    table.add_row(
        "Step V3",
        "v3_agent_compatibility_heatmap.png / .pdf / .html",
        "Conference PNG + Vector PDF + HTML",
        "Multi-Paradigm Algorithm Compatibility Matrix (IEEE/Nature Journal Style)",
    )
    table.add_row(
        "Step V4",
        "v4_agent_impact_network.png / .html",
        "Network Graph + Interactive Sankey",
        "Flagship Multi-Stage Agentic Impact Network & Evidence Propagation Flow",
    )

    console.print("\n")
    console.print(table)
    console.print(f"\n[bold green]✓ All visual artifacts successfully generated at: {out_dir}[/bold green]\n")


if __name__ == "__main__":
    run_all()
