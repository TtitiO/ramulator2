"""Generate P4 paper figures: attention/FFN/MoE decomposition and replay validation.

Usage:
    PYTHONPATH="ramulator2/python" .venv/bin/python \
        ramulator2/tests/analysis/figures/gen_p4_figures.py \
        --output-dir paper/figures/

Produces:
    fig4_attention_decomposition.png/pdf
    fig5_ffn_moe_decomposition.png/pdf
    p4_replay_validation.tex
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Ensure the ramulator package is importable
_here = Path(__file__).resolve().parent
# Add ramulator2/python for the ramulator package
sys.path.insert(0, str(_here.parent.parent.parent.parent / "python"))
# Add ramulator2/ for tests and other local modules
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


CMD_ORDER = ["SB", "HAB", "HAB_PIM", "PIM_BCAST", "PIM_MAC", "PIM_MAC_AB"]
CMD_COLORS = {
    "SB": "#0072B2",
    "HAB": "#D55E00",
    "HAB_PIM": "#E69F00",
    "PIM_BCAST": "#56B4E9",
    "PIM_MAC": "#009E73",
    "PIM_MAC_AB": "#CC79A7",
}


def _plot_repeat_expanded_counts(ax: plt.Axes, counts: dict[str, int]) -> None:
    present_cmds = [c for c in CMD_ORDER if c in counts]
    cmd_values = [counts.get(c, 0) for c in present_cmds]
    bars = ax.bar(
        present_cmds,
        cmd_values,
        color=[CMD_COLORS[c] for c in present_cmds],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_yscale("log")
    ax.set_ylim(0.8, max(cmd_values + [1]) * 2.5)
    for bar, val in zip(bars, cmd_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(val, 1) * 1.12,
            str(val),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.tick_params(axis="x", labelrotation=30)


# ───────────────────────────────────────────────────────────────────────
# Attention Operator Decomposition
# ───────────────────────────────────────────────────────────────────────


def gen_figure_4(output_dir: Path, *, use_tiny: bool = False) -> None:
    from tests.analysis.figures.p4_figure_data import (
        collect_attention_decomposition,
        collect_attention_sweep,
        collect_attention_paper,
        collect_attention_sweep_paper,
    )

    if use_tiny:
        data = collect_attention_decomposition()
        sweep = collect_attention_sweep(
            num_heads_list=[1, 2, 4, 8],
            past_len_list=[32, 64, 128, 256],
        )
        provenance_label = "TINY MANIFEST (unit-test scale)"
    else:
        data = collect_attention_paper()
        sweep = collect_attention_sweep_paper()
        provenance_label = "OPT-125M / LLaMA-style (representative model scale)"

    fig = plt.figure(figsize=(12, 5))

    # --- Panel A: Command breakdown bar chart ---
    ax1 = fig.add_subplot(1, 2, 1)
    counts = data["concrete_counts"]
    semantic_counts = data["semantic_counts"]

    _plot_repeat_expanded_counts(ax1, counts)
    ax1.set_ylabel("Repeat-expanded command count", fontweight="bold")
    ax1.set_title(
        f"Panel A: LPDDR5-PIM Commands for Attention\n"
        f"({data['num_heads']} head{'s' if data['num_heads'] > 1 else ''}, head_dim={data['head_dim']}, "
        f"past_len={data['past_len']}, {data['datatype']})",
        fontsize=10,
    )
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Annotate semantic-only records
    sem_text = (
        f"{provenance_label}\n"
        f"Counts include repeat factors\n"
        f"Semantic-only (no opcode):\n"
        f"  Softmax: {semantic_counts.get('AttentionSoftmax', 0)} accounting record\n"
        f"  Context reduction: {semantic_counts.get('PIMElementwise', 0)} accounting"
    )
    ax1.text(0.02, 0.98, sem_text, transform=ax1.transAxes, fontsize=8,
             va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5", alpha=0.8))

    # --- Panel B: Parameter sensitivity ---
    ax2 = fig.add_subplot(1, 2, 2)
    head_vals = sorted(set(r["_num_heads"] for r in sweep))
    markers = ["o", "s", "^", "D"]
    colors_line = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]

    for head_idx, num_heads in enumerate(head_vals):
        subset = [r for r in sweep if r["_num_heads"] == num_heads]
        subset.sort(key=lambda r: r["_past_len"])
        x = [r["_past_len"] for r in subset]
        y = [r["concrete_counts"].get("PIM_MAC", 0) for r in subset]
        ax2.plot(x, y, marker=markers[head_idx], color=colors_line[head_idx],
                 label=f"{num_heads} heads", linewidth=1.5, markersize=6)

    ax2.set_xlabel("past_len", fontweight="bold")
    ax2.set_ylabel("PIM_MAC Count", fontweight="bold")
    ax2.set_title("Panel B: PIM_MAC Count vs Parameters", fontsize=10)
    ax2.legend(fontsize=8, framealpha=0.9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    _save(fig, output_dir, "fig4_attention_decomposition")


# ───────────────────────────────────────────────────────────────────────
# FFN/SwiGLU and MoE Decomposition
# ───────────────────────────────────────────────────────────────────────


def gen_figure_5(output_dir: Path, *, use_tiny: bool = False) -> None:
    from tests.analysis.figures.p4_figure_data import (
        collect_ffn_decomposition,
        collect_moe_decomposition,
        collect_ffn_paper,
        collect_moe_paper,
    )

    if use_tiny:
        ffn_data = collect_ffn_decomposition()
        moe_data = collect_moe_decomposition()
        provenance_label = "TINY MANIFEST (unit-test scale)"
    else:
        ffn_data = collect_ffn_paper()
        moe_data = collect_moe_paper()
        provenance_label = "OPT-125M / Mixtral-style (representative model scale)"

    fig = plt.figure(figsize=(12, 5))

    # --- Panel A: FFN ---
    ax1 = fig.add_subplot(1, 2, 1)
    ffn_counts = ffn_data["concrete_counts"]
    ffn_sem = ffn_data["semantic_counts"]

    proj_cmds = ffn_sem.get("FFNProjection", 0)
    _plot_repeat_expanded_counts(ax1, ffn_counts)
    ax1.set_ylabel("Repeat-expanded command count", fontweight="bold")
    ax1.set_title(
        f"Panel A: FFN/SwiGLU Commands\n"
        f"(hidden={ffn_data['hidden_size']}, ffn_hidden={ffn_data['ffn_hidden_size']}, "
        f"{ffn_data['datatype']}, act={ffn_data['activation']})",
        fontsize=10,
    )
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    sem_text = (
        f"{provenance_label}\n"
        f"Counts include repeat factors\n"
        f"Semantic-only:\n"
        f"  Activation ({ffn_data['activation']}): "
        f"{ffn_sem.get('PIMElementwise', 0)} accounting\n"
        f"  FFNProjection (up+gate+down): {proj_cmds}"
    )
    ax1.text(0.02, 0.98, sem_text, transform=ax1.transAxes, fontsize=8,
             va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5", alpha=0.8))

    # --- Panel B: MoE ---
    ax2 = fig.add_subplot(1, 2, 2)
    moe_counts = moe_data["concrete_counts"]
    moe_sem = moe_data["semantic_counts"]

    _plot_repeat_expanded_counts(ax2, moe_counts)
    ax2.set_ylabel("Repeat-expanded command count", fontweight="bold")
    ax2.set_title(
        f"Panel B: MoE Commands\n"
        f"({moe_data['num_experts']} experts, top-{moe_data['top_k']}, "
        f"selected {moe_data['selected_experts']}, {moe_data['datatype']})",
        fontsize=10,
    )
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    inactive = sorted(set(range(moe_data["num_experts"])) - set(moe_data["selected_experts"]))
    sem_text = (
        f"Tiny deterministic routing: selected experts {moe_data['selected_experts']}\n"
        f"Semantic-only:\n"
        f"  TopK: {moe_sem.get('MoETopK', 0)}, "
        f"Dispatch: {moe_sem.get('MoEDispatch', 0)}\n"
        f"  Combine: {moe_sem.get('MoECombine', 0)}\n"
        f"  Active experts: {moe_sem.get('MoEExpertFFN', 0)}× compute records\n"
        f"  Inactive experts {inactive}: 0 MACs"
    )
    ax2.text(0.02, 0.98, sem_text, transform=ax2.transAxes, fontsize=8,
             va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5", alpha=0.8))

    fig.tight_layout()
    _save(fig, output_dir, "fig5_ffn_moe_decomposition")


# ───────────────────────────────────────────────────────────────────────
# End-to-End Replay Validation
# ───────────────────────────────────────────────────────────────────────


def _latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def write_replay_validation_table(rows: list[dict[str, object]], output_dir: Path) -> Path:
    """Write replay validation as a LaTeX table/prose artifact."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("fig6_replay_validation.png", "fig6_replay_validation.pdf"):
        stale_path = output_dir / stale
        if stale_path.exists():
            stale_path.unlink()

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{End-to-end replay validation for generated P4 traces. Traces use tiny surrogate parameters (\texttt{past\_len}=32, \texttt{head\_dim}=32, INT8) for fast replay; runtimes are not representative of production-scale inference. All values are simulator-internal diagnostics (non-silicon-calibrated).}",
        r"\label{tab:p4-replay-validation}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Trace & Semantic records & Concrete opcodes & Status & PIM\_MAC issued & Runtime (ns) \\",
        r"\midrule",
    ]
    for row in rows:
        runtime = row.get("runtime_ns", 0)
        try:
            runtime_text = f"{float(str(runtime)):.1f}"
        except (TypeError, ValueError):
            runtime_text = str(runtime)
        lines.append(
            " & ".join(
                [
                    str(row.get("trace_name", "")),
                    _latex_escape(row.get("semantic_records", "")),
                    _latex_escape(row.get("concrete_records", "")),
                    _latex_escape(row.get("replay_status", "PASS")),
                    _latex_escape(row.get("pim_mac_issued", "")),
                    _latex_escape(runtime_text),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\smallskip",
            r"\footnotesize Status PASS indicates the replay completed successfully (all opcode requests completed). Runtime uses a controller-internal clock (tCK~$=$~0.625\,ns for LPDDR5-6400). \texttt{past\_len}=32 is used to keep replay fast.",
            r"\end{table}",
            "",
        ]
    )
    path = output_dir / "p4_replay_validation.tex"
    path.write_text("\n".join(lines))
    return path


def gen_figure_6(output_dir: Path) -> None:
    from tests.analysis.figures.p4_figure_data import (
        collect_replay_stats,
    )

    rows = collect_replay_stats()
    write_replay_validation_table(rows, _paper_table_dir(output_dir))


# ───────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate P4 paper figures"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/figures"),
        help="Output directory for figures (default: paper/figures)",
    )
    parser.add_argument(
        "--skip-replay",
        action="store_true",
        help="Skip replay validation to save time",
    )
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Use tiny/test manifests instead of paper-scale model configurations",
    )
    opts = parser.parse_args()

    os.makedirs(opts.output_dir, exist_ok=True)

    manifest_mode = "TINY (unit-test scale)" if opts.tiny else "PAPER (representative model scale)"
    print(f"Manifest mode: {manifest_mode}")

    # Clean stale fig6 raster artifacts — fig6 is now exported as LaTeX table
    for stale in ("fig6_replay_validation.png", "fig6_replay_validation.pdf"):
        stale_path = opts.output_dir / stale
        if stale_path.exists():
            stale_path.unlink()

    print("Generating attention decomposition...")
    gen_figure_4(opts.output_dir, use_tiny=opts.tiny)
    print("  OK")

    print("Generating FFN/MoE decomposition...")
    gen_figure_5(opts.output_dir, use_tiny=opts.tiny)
    print("  OK")

    if not opts.skip_replay:
        print("Generating replay validation table (this runs the simulator)...")
        gen_figure_6(opts.output_dir)
        print("  OK")
    else:
        print("  Skipped (--skip-replay)")

    print(f"\nAll figures written to {opts.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
