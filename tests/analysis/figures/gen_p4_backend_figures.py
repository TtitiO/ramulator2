"""Generate backend-backed P4 figures from actual Ramulator2 LPDDR5-PIM simulation.

Produces figures with real backend observability data:
  - Command trace timeline (PIM_MAC/SB/HAB/PIM_BCAST over clock cycles)
  - Operator latency/throughput comparison bar chart
  - Stall breakdown by operator or zero-stall summary

Usage:
    PYTHONPATH="ramulator2/python:ramulator2" .venv/bin/python \
        ramulator2/tests/analysis/figures/gen_p4_backend_figures.py \
        --output-dir paper/figures/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parent.parent.parent.parent / "python"))
sys.path.insert(0, str(_here.parent.parent.parent.parent))


def _fig_path(output_dir: Path, name: str, ext: str = "png") -> Path:
    return output_dir / f"{name}.{ext}"


def _save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        p = _fig_path(output_dir, name, ext)
        fig.savefig(str(p), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _paper_table_dir(output_dir: Path) -> Path:
    """Route paper runs to paper/tables while keeping tests local to tmp dirs."""
    return output_dir.parent / "tables" if output_dir.name == "figures" else output_dir


OPCODE_COLORS = {
    "ACT1": "#999999",
    "ACT2": "#999999",
    "SB": "#0072B2",
    "HAB": "#D55E00",
    "HAB_PIM": "#E69F00",
    "PIM_BCAST": "#56B4E9",
    "PIM_MAC": "#009E73",
    "PIM_MAC_AB": "#CC79A7",
    "PREpb": "#D9D9D9",
    "PREab": "#D9D9D9",
}
DEFAULT_COLOR = "#CCCCCC"


# ───────────────────────────────────────────────────────────────────────
# Command Trace Timeline
# ───────────────────────────────────────────────────────────────────────


def gen_figure_7_cmd_trace_timeline(output_dir: Path, *, use_tiny: bool = False) -> None:
    from tests.analysis.figures.p4_backend_data import (
        collect_all_backend_stats,
        collect_all_backend_stats_paper,
    )

    stats = collect_all_backend_stats() if use_tiny else collect_all_backend_stats_paper()
    trace_names = [
        "attention_serialized",
        "ffn_swiglu",
        "moe_top2",
    ]

    fig, axes = plt.subplots(len(trace_names), 1, figsize=(14, 2.5 * len(trace_names)),
                              sharex=False)

    for ax_idx, name in enumerate(trace_names):
        ax = axes[ax_idx] if len(trace_names) > 1 else axes
        data = stats[name]
        trace = data["command_trace"]

        if not trace:
            ax.text(0.5, 0.5, "No command trace data", transform=ax.transAxes,
                    ha="center", va="center")
            continue

        # Build a clock-cycle raster using relative backend cycles.
        clks = [r["clk"] for r in trace]
        cmds = [r["command"] for r in trace]
        min_clk = min(clks) if clks else 0
        max_clk = max(clks) if clks else 1

        # Assign y-level per unique command type
        unique_cmds = list(dict.fromkeys(c for c in cmds))
        cmd_to_y = {c: i for i, c in enumerate(unique_cmds)}

        for clk, cmd in zip(clks, cmds):
            rel_clk = clk - min_clk
            y = cmd_to_y[cmd]
            ax.vlines(
                rel_clk,
                y - 0.38,
                y + 0.38,
                color=OPCODE_COLORS.get(cmd, DEFAULT_COLOR),
                linewidth=1.4 if cmd.startswith("PIM") else 0.9,
                alpha=0.9,
            )

        ax.set_yticks(range(len(unique_cmds)))
        ax.set_yticklabels(unique_cmds, fontsize=8)
        ax.set_ylabel("Command", fontsize=9, fontweight="bold")
        ax.set_xlabel("Backend cycle offset", fontsize=9, fontweight="bold")
        ax.set_xlim(0, max(max_clk - min_clk, 1) + 5)
        ax.set_ylim(-0.8, len(unique_cmds) - 0.2)

        # Summary label
        label = (
            f"{name}\n"
            f"cycles={data['cycles']}, shown_events={len(trace)}, "
            f"PIM_MAC={data['pim_mac_issued']}, "
            f"avg_lat={data['avg_pim_latency_cycles']:.1f}cy"
        )
        ax.text(0.01, 0.95, label, transform=ax.transAxes, fontsize=8,
                va="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5", alpha=0.8))

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _save(fig, output_dir, "fig7_cmd_trace_timeline")


# ───────────────────────────────────────────────────────────────────────
# Operator Latency/Throughput Comparison
# ───────────────────────────────────────────────────────────────────────


def gen_figure_8_operator_latency_throughput(output_dir: Path, *, use_tiny: bool = False) -> None:
    from tests.analysis.figures.p4_backend_data import (
        collect_all_backend_stats,
        collect_all_backend_stats_paper,
    )

    stats = collect_all_backend_stats() if use_tiny else collect_all_backend_stats_paper()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.5, 4.2))

    names = [
        "attention_serialized",
        "attention_overlapped",
        "ffn_swiglu",
        "moe_top2",
        "combined_layer",
    ]
    labels = ["Attention\n(serial)", "Attention\n(overlap)", "FFN/SwiGLU", "MoE (top-2)", "Combined\nLayer"]

    pim_macs = [stats[n]["pim_mac_issued"] for n in names]
    latencies = [stats[n]["avg_pim_latency_cycles"] for n in names]
    cycles = [stats[n]["cycles"] for n in names]
    # Throughput = PIM_MAC / cycles
    throughputs = [m / max(c, 1) for m, c in zip(pim_macs, cycles)]

    colors_ops = ["#0072B2", "#56B4E9", "#D55E00", "#009E73", "#CC79A7"]

    # Panel A: PIM_MAC count
    bars = ax1.bar(range(len(labels)), pim_macs, color=colors_ops, edgecolor="white",
                   linewidth=0.5, alpha=0.85)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("PIM_MAC issued", fontweight="bold")

    for bar, val in zip(bars, pim_macs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 str(val), ha="center", va="bottom", fontsize=7)
    ax1.spines["top"].set_visible(False)

    # Panel B: Avg PIM latency
    bars2 = ax2.bar(range(len(labels)), latencies, color=colors_ops, edgecolor="white",
                    linewidth=0.5, alpha=0.85)
    for bar, lat in zip(bars2, latencies):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{lat:.1f}", ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("Average PIM latency (cycles)", fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Panel C: command density, not closed-loop system throughput
    bars3 = ax3.bar(range(len(labels)), throughputs, color=colors_ops, edgecolor="white",
                    linewidth=0.5, alpha=0.85)
    for bar, tp in zip(bars3, throughputs):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                 f"{tp:.4f}", ha="center", va="bottom", fontsize=8)
    ax3.set_xticks(range(len(labels)))
    ax3.set_xticklabels(labels, fontsize=8)
    ax3.set_ylabel("PIM_MAC density (cmds/cycle)", fontweight="bold")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    fig.tight_layout()
    _save(fig, output_dir, "fig8_operator_latency_throughput")


# ───────────────────────────────────────────────────────────────────────
# Stall Breakdown by Operator
# ───────────────────────────────────────────────────────────────────────


def gen_stall_summary_or_figure(output_dir: Path, stats: dict[str, dict[str, object]]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    names = [
        "attention_serialized",
        "attention_overlapped",
        "ffn_swiglu",
        "moe_top2",
        "combined_layer",
    ]
    labels = ["Attention\n(serial)", "Attention\n(overlap)", "FFN/SwiGLU", "MoE (top-2)", "Combined\nLayer"]

    dependency = [max(0, int(str(stats[n].get("pim_dependency_stalls", 0)))) for n in names]
    capacity = [max(0, int(str(stats[n].get("pim_capacity_stalls", 0)))) for n in names]
    mpu_group = [max(0, int(str(stats[n].get("mpu_group_busy_cycles", 0)))) for n in names]
    total_pim = [max(0, int(str(stats[n].get("pim_mac_issued", 0)))) for n in names]

    if all(v == 0 for series in (dependency, capacity, mpu_group) for v in series):
        for stale in ("fig9_stall_breakdown.png", "fig9_stall_breakdown.pdf"):
            stale_path = output_dir / stale
            if stale_path.exists():
                stale_path.unlink()
        path = output_dir / "p4_stall_summary.tex"
        path.write_text(
            "\\paragraph{Backend stall summary.} "
            "All measured P4 operator rows report zero dependency stalls, "
            "zero capacity stalls, and zero MPU group busy cycles. "
            "These stall categories are tracked independently and may overlap "
            "(they are not mutually exclusive). The all-zero result reflects "
            "the small-scale surrogate traces (\\texttt{past\\_len}=32, "
            "4096\\,\\texttt{PIM\\_MAC} requests per operator) not stressing "
            "the shared-MPU arbitration path, so the stall-breakdown plot is "
            "omitted. Runtimes use the simulator-internal tCK convention "
            "(non-silicon-calibrated).\n"
        )
        return path

    fig, ax = plt.subplots(figsize=(11, 5))

    x = np.arange(len(labels))
    width = 0.25

    bars1 = ax.bar(x, dependency, width, label="Dependency stalls",
                   color="#D55E00", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x, capacity, width, bottom=dependency, label="Capacity stalls",
                   color="#E69F00", edgecolor="white", linewidth=0.5)
    bottom_mpu = [d + c for d, c in zip(dependency, capacity)]
    bars3 = ax.bar(x, mpu_group, width, bottom=bottom_mpu, label="MPU group busy cycles",
                   color="#0072B2", edgecolor="white", linewidth=0.5)

    # Annotate bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                y = bar.get_y() + h
                ax.text(bar.get_x() + bar.get_width() / 2, y + 1,
                        str(int(h)), ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Nonnegative stall/busy cycles", fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.spines["top"].set_visible(False)
    fig.tight_layout()
    _save(fig, output_dir, "fig9_stall_breakdown")
    return _fig_path(output_dir, "fig9_stall_breakdown")


def gen_figure_9_stall_breakdown(output_dir: Path, *, use_tiny: bool = False) -> None:
    from tests.analysis.figures.p4_backend_data import (
        collect_all_backend_stats,
        collect_all_backend_stats_paper,
    )

    stats = collect_all_backend_stats() if use_tiny else collect_all_backend_stats_paper()
    gen_stall_summary_or_figure(_paper_table_dir(output_dir), stats)


def gen_figure_10_llama2_7b_backend(output_dir: Path) -> None:
    from tests.analysis.figures.p4_backend_data import (
        collect_all_backend_stats_llama2_7b,
    )

    stats = collect_all_backend_stats_llama2_7b()
    names = [
        "llama2_7b_32_layer_steady_state",
        "llama2_7b_32_layer_cold_start",
    ]
    labels = ["Steady\nstate", "Cold\nstart"]
    colors = ["#0072B2", "#D55E00"]

    pim_macs = [stats[n]["pim_mac_issued"] for n in names]
    cycles = [stats[n]["cycles"] for n in names]
    latencies = [stats[n]["avg_pim_latency_cycles"] for n in names]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.5, 4.2))

    # Panel A: PIM_MAC issued count
    bars1 = ax1.bar(labels, pim_macs, color=colors, edgecolor="white",
                    linewidth=0.5, alpha=0.85)
    ax1.set_ylabel("PIM_MAC issued", fontweight="bold")
    ax1.set_title("Panel A: PIM_MAC issued", fontsize=10)
    for bar, val in zip(bars1, pim_macs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                 f"{int(val):,}", ha="center", va="bottom", fontsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel B: total simulation cycles
    bars2 = ax2.bar(labels, cycles, color=colors, edgecolor="white",
                    linewidth=0.5, alpha=0.85)
    ax2.set_ylabel("Total simulation cycles", fontweight="bold")
    ax2.set_title("Panel B: Cycles", fontsize=10)
    for bar, val in zip(bars2, cycles):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                 f"{int(val):,}", ha="center", va="bottom", fontsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Panel C: average PIM latency
    bars3 = ax3.bar(labels, latencies, color=colors, edgecolor="white",
                    linewidth=0.5, alpha=0.85)
    ax3.set_ylabel("Average PIM latency (cycles)", fontweight="bold")
    ax3.set_title("Panel C: Average PIM latency", fontsize=10)
    for bar, val in zip(bars3, latencies):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                 f"{float(val):.1f}", ha="center", va="bottom", fontsize=8)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    fig.tight_layout()
    _save(fig, output_dir, "fig10_llama2_7b_backend_comparison")


# ───────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate backend-backed P4 paper figures"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/figures"),
        help="Output directory for figures (default: paper/figures)",
    )
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Use tiny/test manifests instead of paper-scale model configurations",
    )
    parser.add_argument(
        "--skip-llama2",
        action="store_true",
        help="Skip Llama2-7B 32-layer backend simulation (very slow)",
    )
    opts = parser.parse_args()

    os.makedirs(opts.output_dir, exist_ok=True)

    manifest_mode = "TINY (unit-test scale)" if opts.tiny else "PAPER (representative model scale)"
    print(f"Manifest mode: {manifest_mode}")

    # Clean stale fig9 raster artifacts — fig9 is now handled as prose/table
    for stale in ("fig9_stall_breakdown.png", "fig9_stall_breakdown.pdf"):
        stale_path = opts.output_dir / stale
        if stale_path.exists():
            stale_path.unlink()

    print("Generating command trace timeline...")
    gen_figure_7_cmd_trace_timeline(opts.output_dir, use_tiny=opts.tiny)
    print("  OK")

    print("Generating operator latency/throughput...")
    gen_figure_8_operator_latency_throughput(opts.output_dir, use_tiny=opts.tiny)
    print("  OK")

    print("Generating stall breakdown or summary...")
    gen_figure_9_stall_breakdown(opts.output_dir, use_tiny=opts.tiny)
    print("  OK")

    if not opts.skip_llama2:
        print("Generating Llama2-7B backend comparison...")
        gen_figure_10_llama2_7b_backend(opts.output_dir)
        print("  OK")
    else:
        print("  Skipped (--skip-llama2)")

    print(f"\nAll backend figures written to {opts.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
