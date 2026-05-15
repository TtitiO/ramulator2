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
import datetime as _dt
import json
import os
import subprocess
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
OP_ORDER = ["READ", "WRITE", "PIM_MAC"]
CMD_COLORS = {
    "SB": "#0072B2",
    "HAB": "#D55E00",
    "HAB_PIM": "#E69F00",
    "PIM_BCAST": "#56B4E9",
    "PIM_MAC": "#009E73",
    "PIM_MAC_AB": "#CC79A7",
    "READ": "#0072B2",
    "WRITE": "#D55E00",
}
LLAMA2_SCALING_FIGURE_ID = "fig9_llama2_7b_13b_models_latency_breakdown"
DEFAULT_LLAMA2_SCALING_CACHE = Path("paper/data") / f"{LLAMA2_SCALING_FIGURE_ID}.json"
DECODE_CONTEXT_SWEEP_FIGURE_ID = "fig11_llama2_7b_decode_context_length_sweep"
DEFAULT_DECODE_CONTEXT_SWEEP_CACHE = Path("paper/data") / f"{DECODE_CONTEXT_SWEEP_FIGURE_ID}.json"
GENERATED_TOKEN_SWEEP_FIGURE_ID = "fig12_llama2_7b_generated_token_sweep"
DEFAULT_GENERATED_TOKEN_SWEEP_CACHE = Path("paper/data") / f"{GENERATED_TOKEN_SWEEP_FIGURE_ID}.json"
MIXTRAL_BREAKDOWN_FIGURE_ID = "fig13_mixtral_8x7b_32_layer_breakdown"
DEFAULT_MIXTRAL_BREAKDOWN_CACHE = Path("paper/data") / f"{MIXTRAL_BREAKDOWN_FIGURE_ID}.json"
MIXTRAL_VS_LLAMA2_FIGURE_ID = "fig14_mixtral_vs_llama2_decode_comparison"
MOE_SENSITIVITY_FIGURE_ID = "fig15_moe_expert_scaling_sensitivity"
DEFAULT_MOE_SENSITIVITY_CACHE = Path("paper/data") / f"{MOE_SENSITIVITY_FIGURE_ID}.json"
MOE_OPERATOR_DIAG_FIGURE_ID = "fig16_mixtral_operator_diagnostics"


def _plot_repeat_expanded_counts(ax: plt.Axes, counts: dict[str, int]) -> None:
    order = OP_ORDER + [c for c in CMD_ORDER if c not in OP_ORDER]
    present_cmds = [c for c in order if c in counts]
    present_cmds.extend(c for c in counts if c not in present_cmds)
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


def _plot_pim_bcast_modes(ax: plt.Axes, mode_counts: dict[str, int], *, title: str) -> None:
    labels = ["steady-state", "cold-start"]
    keys = ["steady_state", "cold_start"]
    values = [int(mode_counts.get(key, 0)) for key in keys]
    bars = ax.bar(labels, values, color=["#999999", CMD_COLORS["PIM_BCAST"]], edgecolor="white", linewidth=0.5)
    ymax = max(values + [1])
    ax.set_ylim(0, ymax * 1.25)
    ax.set_ylabel("PIM_BCAST repeats", fontweight="bold")
    ax.set_title(title, fontsize=10)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + ymax * 0.04,
            f"{val:,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.tick_params(axis="x", labelrotation=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


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
    else:
        data = collect_attention_paper()
        sweep = collect_attention_sweep_paper()

    fig = plt.figure(figsize=(15, 4.8))

    # --- Panel A: Command breakdown bar chart ---
    ax1 = fig.add_subplot(1, 3, 1)
    counts = data.get("operator_counts", data["concrete_counts"])
    semantic_counts = data["semantic_counts"]

    _plot_repeat_expanded_counts(ax1, counts)
    ax1.set_ylabel("Repeat-expanded op/request count", fontweight="bold")
    ax1.set_title(
        f"Panel A: Attention traffic and PIM MACs\n"
        f"({data['num_heads']} head{'s' if data['num_heads'] > 1 else ''}, head_dim={data['head_dim']}, "
        f"past_len={data['past_len']}, {data['datatype']})",
        fontsize=10,
    )
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # --- Panel B: materialization-mode PIM_BCAST comparison ---
    ax_bcast = fig.add_subplot(1, 3, 2)
    _plot_pim_bcast_modes(
        ax_bcast,
        data.get("pim_bcast_by_mode", {}),
        title="Panel B: PIM_BCAST by materialization mode",
    )

    # --- Panel B: Parameter sensitivity ---
    ax2 = fig.add_subplot(1, 3, 3)
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
    ax2.set_ylabel("PIM_MAC repeats", fontweight="bold")
    ax2.set_title("Panel C: Attention compute scales with context", fontsize=10)
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
    else:
        ffn_data = collect_ffn_paper()
        moe_data = collect_moe_paper()

    fig = plt.figure(figsize=(15, 7.4))

    # --- Panel A: FFN ---
    ax1 = fig.add_subplot(2, 2, 1)
    ffn_counts = ffn_data.get("operator_counts", ffn_data["concrete_counts"])

    _plot_repeat_expanded_counts(ax1, ffn_counts)
    ax1.set_ylabel("Repeat-expanded op/request count", fontweight="bold")
    ax1.set_title(
        f"Panel A: FFN/SwiGLU traffic and PIM MACs\n"
        f"(hidden={ffn_data['hidden_size']}, ffn_hidden={ffn_data['ffn_hidden_size']}, "
        f"{ffn_data['datatype']}, act={ffn_data['activation']})",
        fontsize=10,
    )
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax1_bcast = fig.add_subplot(2, 2, 2)
    _plot_pim_bcast_modes(
        ax1_bcast,
        ffn_data.get("pim_bcast_by_mode", {}),
        title="Panel B: FFN PIM_BCAST by materialization mode",
    )

    # --- Panel B: MoE ---
    ax2 = fig.add_subplot(2, 2, 3)
    moe_counts = moe_data.get("operator_counts", moe_data["concrete_counts"])

    _plot_repeat_expanded_counts(ax2, moe_counts)
    ax2.set_ylabel("Repeat-expanded op/request count", fontweight="bold")
    ax2.set_title(
        f"Panel C: MoE selected-expert traffic and PIM MACs\n"
        f"({moe_data['num_experts']} experts, top-{moe_data['top_k']}, "
        f"selected {moe_data['selected_experts']}, {moe_data['datatype']})",
        fontsize=10,
    )
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    ax2_bcast = fig.add_subplot(2, 2, 4)
    _plot_pim_bcast_modes(
        ax2_bcast,
        moe_data.get("pim_bcast_by_mode", {}),
        title="Panel D: MoE PIM_BCAST by materialization mode",
    )

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
        r"\caption{End-to-end replay validation for generated P4 traces. The tiny surrogate rows use reduced parameters for fast replay; Llama2-7B rows use full-depth surrogate dimensions (32 layers, \texttt{past\_len}=1024, \texttt{head\_dim}=128, INT8). PIM\_MAC and PIM\_BCAST columns are repeat-expanded command counts. One INT8 PIM\_MAC repeat represents 32 scalar MACs (64 primitive multiply/add ops). All values are simulator-internal diagnostics (non-silicon-calibrated).}",
        r"\label{tab:p4-replay-validation}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Trace & Semantic records & Concrete opcodes & Status & PIM\_MAC issued & PIM\_BCAST issued & Runtime (ns) \\",
        r"\midrule",
    ]
    for row in rows:
        runtime = row.get("runtime_ns", 0)
        try:
            runtime_text = f"{float(str(runtime)):.1f}"
        except (TypeError, ValueError):
            runtime_text = str(runtime)
        command_counts = row.get("command_counts", {})
        pim_bcast = row.get("pim_bcast_issued", "")
        if isinstance(command_counts, dict):
            pim_bcast = command_counts.get("PIM_BCAST", pim_bcast)
        lines.append(
            " & ".join(
                [
                    str(row.get("trace_name", "")),
                    _latex_escape(row.get("semantic_records", "")),
                    _latex_escape(row.get("concrete_records", "")),
                    _latex_escape(row.get("replay_status", "PASS")),
                    _latex_escape(row.get("pim_mac_issued", "")),
                    _latex_escape(pim_bcast),
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
            r"\footnotesize Status PASS indicates the replay completed successfully (all opcode requests completed). Runtime uses a controller-internal clock (tCK~$=$~0.625\,ns for LPDDR5-6400). The Llama2 dense-decoder rows are structured workload surrogates, not full end-to-end model execution.",
            r"\end{table}",
            "",
        ]
    )
    path = output_dir / "p4_replay_validation.tex"
    path.write_text("\n".join(lines))
    return path


def gen_figure_6(output_dir: Path) -> None:
    from tests.analysis.figures import p4_figure_data

    rows = p4_figure_data.collect_replay_stats()
    rows.extend(p4_figure_data.collect_llama2_7b_replay_stats())
    rows.extend(p4_figure_data.collect_llama2_13b_replay_stats())
    write_replay_validation_table(rows, _paper_table_dir(output_dir))


def _first_latency(replay_rows: list[dict[str, object]]) -> tuple[float, float]:
    if not replay_rows:
        return 0.0, 0.0
    row = replay_rows[0]
    return float(row.get("runtime_ns", 0) or 0), float(row.get("cycles", 0) or 0)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _normalize_llama2_scaling_model(data: dict[str, object]) -> dict[str, object]:
    replay_rows = data.get("backend_replay_stats", [])
    if isinstance(replay_rows, list):
        replay_stats = {}
        for row in replay_rows:
            if not isinstance(row, dict):
                continue
            trace_name = str(row.get("trace_name", ""))
            mode = "cold_start" if "cold" in trace_name else "steady_state"
            replay_stats[mode] = {
                "runtime_ns": row.get("runtime_ns", 0),
                "cycles": row.get("cycles", 0),
                "command_counts": row.get("command_counts", {}),
            }
    else:
        replay_stats = dict(data.get("backend_replay_stats", {}))

    return {
        "model_name": data.get("model_name", data.get("manifest_name", "Llama2")),
        "dimensions": {
            "num_layers": data.get("num_layers", 0),
            "hidden_size": data.get("hidden_size", 0),
            "num_heads": data.get("num_heads", 0),
            "head_dim": data.get("head_dim", 0),
            "ffn_hidden_size": data.get("ffn_hidden_size", 0),
            "past_len": data.get("past_len", 0),
        },
        "per_layer_pim_mac_buckets": {
            "qkvo_projection": data.get("qkvo_projection_pim_mac_per_layer", 0),
            "attention": data.get("attention_pim_mac_per_layer", 0),
            "ffn": data.get("ffn_pim_mac_per_layer", 0),
        },
        "replay_stats": replay_stats,
        "command_counts": data.get("concrete_counts", {}),
    }


def write_llama2_scaling_cache(cache_path: Path = DEFAULT_LLAMA2_SCALING_CACHE, *, collect_backend: bool = True) -> Path:
    from ramulator.workload_surrogate.generate_full_transformer import FULL_TRANSFORMER_GENERATOR_VERSION
    from tests.analysis.figures import p4_figure_data

    cache_path = Path(cache_path)
    datasets = [
        p4_figure_data.collect_llama2_7b_dense_decoder_data(),
        p4_figure_data.collect_llama2_13b_dense_decoder_data(),
    ]
    payload = {
        "schema_version": 1,
        "figure_id": LLAMA2_SCALING_FIGURE_ID,
        "provenance": {
            "date": _dt.date.today().isoformat(),
            "generator_version": FULL_TRANSFORMER_GENERATOR_VERSION,
            "commit": _git_commit(),
            "replay_mode": "backend" if collect_backend else "precomputed",
        },
        "models": [_normalize_llama2_scaling_model(data) for data in datasets],
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return cache_path


def _llama2_spec_summary(data: dict[str, object]) -> tuple[int, str]:
    """Return approximate FP16 weight size and compact dimension summary for display."""
    model_name = str(data.get("model_name", ""))
    if "13B" in model_name:
        weight_gb = 26
    else:
        weight_gb = 14
    dims = (
        f"{int(data.get('num_layers', 0))}L, "
        f"H={int(data.get('hidden_size', 0))}, "
        f"heads={int(data.get('num_heads', 0))}, "
        f"d={int(data.get('head_dim', 0))}"
    )
    return weight_gb, dims


def _plot_llama2_scaling_payload(payload: dict[str, object]) -> plt.Figure:
    models = [model for model in payload.get("models", []) if isinstance(model, dict)]
    labels = [str(model.get("model_name", "Llama2")) for model in models]
    x = range(len(labels))

    fig = plt.figure(figsize=(12.5, 4.3))
    ax1 = fig.add_subplot(1, 2, 1)
    width = 0.35
    modes = [("steady_state", "Steady-state", "#0072B2"), ("cold_start", "Cold-start", "#D55E00")]
    for idx, (mode_key, mode_label, color) in enumerate(modes):
        raw_values = [model.get("replay_stats", {}).get(mode_key, {}).get("runtime_ns") for model in models]
        values = [float(value) if value is not None else 0.0 for value in raw_values]
        offsets = [pos + (idx - 0.5) * width for pos in x]
        bars = ax1.bar(offsets, values, width=width, label=mode_label, color=color, edgecolor="white", linewidth=0.5, alpha=0.85)
        for bar, value, raw_value in zip(bars, values, raw_values):
            if raw_value is None:
                bar.set_alpha(0.18)
                bar.set_hatch("//")
                label = "not collected"
            else:
                label = f"{value:,.0f}"
            ax1.text(bar.get_x() + bar.get_width() / 2, max(value, 1) * 1.03, label, ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(list(x), labels)
    ax1.set_ylabel("Backend runtime (ns)", fontweight="bold")
    ax1.legend(fontsize=8, framealpha=0.9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2 = fig.add_subplot(1, 2, 2)
    components = [("qkvo_projection", "Q/K/V/O", "#009E73"), ("attention", "Attention", "#0072B2"), ("ffn", "FFN/SwiGLU", "#D55E00")]
    width = 0.24
    for idx, (bucket_key, bucket_label, color) in enumerate(components):
        values = [float(model.get("per_layer_pim_mac_buckets", {}).get(bucket_key, 0)) for model in models]
        offsets = [pos + (idx - 1) * width for pos in x]
        ax2.bar(offsets, values, width=width, label=bucket_label, color=color, edgecolor="white", linewidth=0.5, alpha=0.85)
    ax2.set_xticks(list(x), labels)
    ax2.set_yscale("log")
    ax2.set_ylabel("Per-layer PIM_MAC repeats", fontweight="bold")
    ax2.legend(fontsize=8, framealpha=0.9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def render_llama2_scaling_figure_from_cache(cache_path: Path = DEFAULT_LLAMA2_SCALING_CACHE, output_dir: Path = Path("paper/figures")) -> None:
    payload = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    fig = _plot_llama2_scaling_payload(payload)
    _save(fig, Path(output_dir), LLAMA2_SCALING_FIGURE_ID)


def _validate_decode_context_sweep_payload(payload: dict[str, object]) -> list[dict[str, object]]:
    if payload.get("schema_version") != 1:
        raise ValueError("decode context sweep cache schema_version must be 1")
    if payload.get("figure_id") != DECODE_CONTEXT_SWEEP_FIGURE_ID:
        raise ValueError(f"decode context sweep cache figure_id must be {DECODE_CONTEXT_SWEEP_FIGURE_ID}")
    if payload.get("phase") != "decode" or payload.get("seq_len") != 1:
        raise ValueError("decode context sweep cache must declare phase='decode' and seq_len=1")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("decode context sweep cache must contain non-empty rows")
    required = {"past_len", "mode", "status", "runtime_ns", "cycles", "pim_mac", "pim_bcast", "pim_mac_density"}
    normalized: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"decode context sweep row {index} must be an object")
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"decode context sweep row {index} missing required fields: {', '.join(missing)}")
        normalized.append(row)
    return normalized


def _plot_decode_context_sweep_payload(payload: dict[str, object]) -> plt.Figure:
    rows = _validate_decode_context_sweep_payload(payload)
    modes = sorted({str(row["mode"]) for row in rows})
    colors = {"steady_state": "#0072B2", "cold_start": "#D55E00"}
    labels = {"steady_state": "Steady-state", "cold_start": "Cold-start"}

    fig = plt.figure(figsize=(13.5, 4.6))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    for mode in modes:
        subset = [row for row in rows if str(row["mode"]) == mode]
        subset.sort(key=lambda row: int(row["past_len"]))
        x = [int(row["past_len"]) for row in subset]
        runtime = [float(row["runtime_ns"]) for row in subset]
        pim_mac = [float(row["pim_mac"]) for row in subset]
        density = [float(row["pim_mac_density"]) for row in subset]
        color = colors.get(mode, "#009E73")
        label = labels.get(mode, mode.replace("_", " ").title())
        ax1.plot(x, runtime, marker="o", color=color, label=label, linewidth=1.8)
        ax2.plot(x, pim_mac, marker="s", color=color, label=label, linewidth=1.8)
        ax3.plot(x, density, marker="^", color=color, label=label, linewidth=1.8)

    xlabel = "Decode context length (past_len), seq_len=1"
    for ax in (ax1, ax2, ax3):
        ax.set_xlabel(xlabel, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8, framealpha=0.9)
    ax1.set_ylabel("Backend runtime (ns)", fontweight="bold")
    ax1.set_title("Panel A: Runtime diagnostic", fontsize=10)
    ax2.set_ylabel("Replayed PIM_MAC repeats", fontweight="bold")
    ax2.set_title("Panel B: Concrete opcode count", fontsize=10)
    ax3.set_ylabel("PIM_MAC repeats / cycle", fontweight="bold")
    ax3.set_title("Panel C: Command density", fontsize=10)

    caveat = (
        "Llama2-7B bounded full-depth decode surrogate\n"
        "decode-only, seq_len=1; x-axis varies past_len\n"
        "simulator-internal backend diagnostics"
    )
    ax1.text(0.02, 0.98, caveat, transform=ax1.transAxes, fontsize=8,
             va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5", alpha=0.8))
    fig.tight_layout()
    return fig


def write_decode_context_sweep_cache(
    cache_path: Path = DEFAULT_DECODE_CONTEXT_SWEEP_CACHE,
    *,
    past_len_values: list[int] | None = None,
    modes: tuple[str, ...] | None = None,
) -> Path:
    from tests.analysis.figures import p4_figure_data

    cache_path = Path(cache_path)
    payload = p4_figure_data.collect_llama2_7b_decode_context_length_sweep(
        past_len_values=past_len_values,
        modes=modes,
    )
    payload.setdefault("provenance", {})
    if isinstance(payload["provenance"], dict):
        payload["provenance"].setdefault("date", _dt.date.today().isoformat())
        payload["provenance"].setdefault("commit", _git_commit())
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return cache_path


def render_decode_context_sweep_figure_from_cache(
    cache_path: Path = DEFAULT_DECODE_CONTEXT_SWEEP_CACHE,
    output_dir: Path = Path("paper/figures"),
) -> None:
    payload = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    fig = _plot_decode_context_sweep_payload(payload)
    _save(fig, Path(output_dir), DECODE_CONTEXT_SWEEP_FIGURE_ID)


def _validate_generated_token_sweep_payload(payload: dict[str, object]) -> list[dict[str, object]]:
    if payload.get("schema_version") != 1:
        raise ValueError("generated token sweep cache schema_version must be 1")
    if payload.get("figure_id") != GENERATED_TOKEN_SWEEP_FIGURE_ID:
        raise ValueError(f"generated token sweep cache figure_id must be {GENERATED_TOKEN_SWEEP_FIGURE_ID}")
    if payload.get("phase") != "decode" or payload.get("seq_len_per_step") != 1:
        raise ValueError("generated token sweep cache must declare phase='decode' and seq_len_per_step=1")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("generated token sweep cache must contain non-empty rows")
    required = {
        "generated_token_index",
        "generated_tokens_total",
        "past_len",
        "mode",
        "status",
        "runtime_ns",
        "cumulative_runtime_ns",
        "cycles",
        "pim_mac",
        "pim_bcast",
        "pim_mac_density",
    }
    normalized: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"generated token sweep row {index} must be an object")
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"generated token sweep row {index} missing required fields: {', '.join(missing)}")
        normalized.append(row)
    return normalized


def _plot_generated_token_sweep_payload(payload: dict[str, object]) -> plt.Figure:
    rows = _validate_generated_token_sweep_payload(payload)
    modes = sorted({str(row["mode"]) for row in rows})
    colors = {"steady_state": "#0072B2", "cold_start": "#D55E00"}
    labels = {"steady_state": "Steady-state", "cold_start": "Cold-start"}

    fig = plt.figure(figsize=(13.5, 4.6))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    for mode in modes:
        subset = [row for row in rows if str(row["mode"]) == mode]
        subset.sort(key=lambda row: int(row["generated_tokens_total"]))
        x = [int(row["generated_tokens_total"]) for row in subset]
        cumulative_runtime = [float(row["cumulative_runtime_ns"]) for row in subset]
        per_token_runtime = [float(row["runtime_ns"]) for row in subset]
        pim_mac = [float(row["pim_mac"]) for row in subset]
        color = colors.get(mode, "#009E73")
        label = labels.get(mode, mode.replace("_", " ").title())
        ax1.plot(x, cumulative_runtime, marker="o", color=color, label=label, linewidth=1.8)
        ax2.plot(x, per_token_runtime, marker="s", color=color, label=label, linewidth=1.8)
        ax3.plot(x, pim_mac, marker="^", color=color, label=label, linewidth=1.8)

    xlabel = "Generated tokens total (independent single-token replays, seq_len=1)"
    for ax in (ax1, ax2, ax3):
        ax.set_xlabel(xlabel, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8, framealpha=0.9)
    ax1.set_ylabel("Cumulative backend runtime (ns)", fontweight="bold")
    ax1.set_title("Panel A: Cumulative runtime", fontsize=10)
    ax2.set_ylabel("Per-token backend runtime (ns)", fontweight="bold")
    ax2.set_title("Panel B: Per-token runtime", fontsize=10)
    ax3.set_ylabel("Replayed PIM_MAC repeats / token", fontweight="bold")
    ax3.set_title("Panel C: Per-token command count", fontsize=10)

    caveat = (
        "Llama2-7B bounded decode surrogate\n"
        "independent single-token backend replays\n"
        "past_len grows by token; seq_len=1 per replay"
    )
    ax1.text(0.02, 0.98, caveat, transform=ax1.transAxes, fontsize=8,
             va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5", alpha=0.8))
    fig.tight_layout()
    return fig


def write_generated_token_sweep_cache(
    cache_path: Path = DEFAULT_GENERATED_TOKEN_SWEEP_CACHE,
    *,
    initial_past_len: int = 1024,
    num_generated_tokens: int = 4,
    modes: tuple[str, ...] | None = None,
) -> Path:
    from tests.analysis.figures import p4_figure_data

    cache_path = Path(cache_path)
    payload = p4_figure_data.collect_llama2_7b_generated_token_sweep(
        initial_past_len=initial_past_len,
        num_generated_tokens=num_generated_tokens,
        modes=modes,
    )
    payload.setdefault("provenance", {})
    if isinstance(payload["provenance"], dict):
        payload["provenance"].setdefault("date", _dt.date.today().isoformat())
        payload["provenance"].setdefault("commit", _git_commit())
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return cache_path


def render_generated_token_sweep_figure_from_cache(
    cache_path: Path = DEFAULT_GENERATED_TOKEN_SWEEP_CACHE,
    output_dir: Path = Path("paper/figures"),
) -> None:
    payload = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    fig = _plot_generated_token_sweep_payload(payload)
    _save(fig, Path(output_dir), GENERATED_TOKEN_SWEEP_FIGURE_ID)


def gen_multi_model_llama2_latency_breakdown(output_dir: Path) -> None:
    from tests.analysis.figures.p4_figure_data import (
        collect_llama2_7b_dense_decoder_data,
        collect_llama2_13b_dense_decoder_data,
    )

    datasets = [collect_llama2_7b_dense_decoder_data(), collect_llama2_13b_dense_decoder_data()]
    labels = [data.get("model_name", data.get("manifest_name", "Llama2")) for data in datasets]
    runtimes = [_first_latency(data.get("backend_replay_stats", []))[0] for data in datasets]
    cycles = [_first_latency(data.get("backend_replay_stats", []))[1] for data in datasets]

    weight_sizes = [_llama2_spec_summary(data)[0] for data in datasets]
    spec_summaries = [_llama2_spec_summary(data)[1] for data in datasets]

    fig = plt.figure(figsize=(13.5, 4.6))
    ax1 = fig.add_subplot(1, 3, 1)
    x = range(len(labels))
    bars = ax1.bar(list(x), runtimes, color="#0072B2", edgecolor="white", linewidth=0.5, alpha=0.85)
    ax1.set_xticks(list(x), labels)
    ax1.set_ylabel("Backend runtime (ns)", fontweight="bold")
    for bar, runtime, cyc in zip(bars, runtimes, cycles):
        ax1.text(bar.get_x() + bar.get_width() / 2, max(runtime, 1) * 1.05,
                 f"{runtime:,.1f} ns\n{int(cyc):,} cyc", ha="center", va="bottom", fontsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2 = fig.add_subplot(1, 3, 2)
    components = ["Q/K/V/O", "Attention", "FFN/SwiGLU"]
    width = 0.35
    for idx, data in enumerate(datasets):
        values = [
            data["qkvo_projection_pim_mac_per_layer"],
            data["attention_pim_mac_per_layer"],
            data["ffn_pim_mac_per_layer"],
        ]
        offset = (idx - (len(datasets) - 1) / 2) * width
        ax2.bar([pos + offset for pos in range(len(components))], values, width=width,
                label=str(labels[idx]), edgecolor="white", linewidth=0.5, alpha=0.85)
    ax2.set_xticks(list(range(len(components))), components)
    ax2.set_yscale("log")
    ax2.set_ylabel("Per-layer PIM_MAC repeats", fontweight="bold")
    ax2.legend(fontsize=8, framealpha=0.9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    ax3 = fig.add_subplot(1, 3, 3)
    weight_bars = ax3.bar(list(x), weight_sizes, color="#009E73", edgecolor="white", linewidth=0.5, alpha=0.85)
    ax3.set_xticks(list(x), labels)
    ax3.set_ylabel("Approx. FP16 weight size (GB)", fontweight="bold")
    for bar, weight_gb, summary in zip(weight_bars, weight_sizes, spec_summaries):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            max(weight_gb, 1) * 1.03,
            f"~{weight_gb} GB\n{summary}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    fig.tight_layout()
    _save(fig, output_dir, "fig9_llama2_7b_13b_models_latency_breakdown")


def _cached_or_collect_llama2_scaling(output_dir: Path) -> None:
    if DEFAULT_LLAMA2_SCALING_CACHE.exists():
        render_llama2_scaling_figure_from_cache(DEFAULT_LLAMA2_SCALING_CACHE, output_dir)
    else:
        print(f"  Skipped Llama2 scaling figure; cache missing at {DEFAULT_LLAMA2_SCALING_CACHE}")
        print("  Run with --collect-llama2-scaling-cache first, then --render-llama2-scaling-cache.")


def _cached_or_collect_decode_context_sweep(output_dir: Path) -> None:
    if DEFAULT_DECODE_CONTEXT_SWEEP_CACHE.exists():
        render_decode_context_sweep_figure_from_cache(DEFAULT_DECODE_CONTEXT_SWEEP_CACHE, output_dir)
    else:
        print(f"  Skipped decode context-length sweep figure; cache missing at {DEFAULT_DECODE_CONTEXT_SWEEP_CACHE}")
        print("  Run with --collect-decode-context-sweep-cache first, then --render-decode-context-sweep-cache.")


def _cached_or_collect_generated_token_sweep(output_dir: Path) -> None:
    if DEFAULT_GENERATED_TOKEN_SWEEP_CACHE.exists():
        render_generated_token_sweep_figure_from_cache(DEFAULT_GENERATED_TOKEN_SWEEP_CACHE, output_dir)
    else:
        print(f"  Skipped generated-token sweep figure; cache missing at {DEFAULT_GENERATED_TOKEN_SWEEP_CACHE}")
        print("  Run with --collect-generated-token-sweep-cache first, then --render-generated-token-sweep-cache.")


def gen_figure_9_llama2_7b_32_layer(output_dir: Path) -> None:
    from tests.analysis.figures.p4_figure_data import (
        collect_llama2_7b_dense_decoder_data,
    )

    data = collect_llama2_7b_dense_decoder_data()
    replay_rows = data["backend_replay_stats"]

    fig = plt.figure(figsize=(12, 5))

    # --- Panel A: backend replay command comparison ---
    ax1 = fig.add_subplot(1, 2, 1)
    labels = ["Steady-state", "Cold-start"]
    command_names = ["PIM_MAC", "PIM_BCAST"]
    x = range(len(labels))
    width = 0.35
    for idx, command in enumerate(command_names):
        values = [row["command_counts"].get(command, 0) for row in replay_rows]
        offset = (idx - 0.5) * width
        bars = ax1.bar(
            [pos + offset for pos in x],
            values,
            width=width,
            label=command,
            color=CMD_COLORS[command],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.85,
        )
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width() / 2, max(val, 1) * 1.05,
                     f"{int(val):,}", ha="center", va="bottom", fontsize=8)

    ax1.set_xticks(list(x), labels)
    ax1.set_yscale("log")
    ax1.set_ylim(0.8, max([row["command_counts"].get(cmd, 0) for row in replay_rows for cmd in command_names] + [1]) * 2.5)
    ax1.set_ylabel("Backend command count", fontweight="bold")
    ax1.set_title("Panel A: Backend replay commands", fontsize=10)
    ax1.legend(fontsize=8, framealpha=0.9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    caption = (
        "Backend replay\n"
        "Steady-state vs Cold-start\n"
        "Llama2-7B 32-layer dense-decoder surrogate (hidden=4096, ffn_hidden=11008, "
        "32 heads, head_dim=128, past_len=1024, INT8).\n"
        "Decode-block v2 scope: Q/K/V/O projections plus QK^T + PV attention core; "
        "semantic KV HostRead/HostWrite lower to native READ/WRITE concrete requests; Barrier/Drain remain ordering annotations. "
        "Panel A shows replayed READ/WRITE and PIM opcodes. Excludes RoPE, norm, residual, and softmax hardware cost."
    )
    ax1.text(0.02, 0.98, caption, transform=ax1.transAxes, fontsize=8,
             va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5", alpha=0.8))

    # --- Panel B: per-layer attention vs FFN PIM_MAC ---
    ax2 = fig.add_subplot(1, 2, 2)
    lanes = int(data["pim_mac_lanes"])
    qkvo_per_layer = data["qkvo_projection_pim_mac_per_layer"]
    attention_per_layer = data["attention_pim_mac_per_layer"]
    ffn_per_layer = data["ffn_pim_mac_per_layer"]
    labels = ["Q/K/V/O\nprojections", "Attention\n(QK^T+PV)", "FFN/SwiGLU\n(up+gate+down)"]
    values = [qkvo_per_layer, attention_per_layer, ffn_per_layer]
    bars = ax2.bar(labels, values, color=["#009E73", "#0072B2", "#D55E00"], edgecolor="white",
                   linewidth=0.5, alpha=0.85)
    ax2.set_yscale("log")
    ax2.set_ylabel(f"Per-layer PIM_MAC repeats\n({lanes} scalar INT8 MACs/repeat)", fontweight="bold")
    ax2.set_title("Panel B: Modeled per-layer compute slice", fontsize=10)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, val * 1.12,
                 f"{int(val):,}", ha="center", va="bottom", fontsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    _save(fig, output_dir, "fig9_llama2_7b_32_layer_breakdown")


# ─── Mixtral-8x7B MoE breakdown figure ──────────────────────────────────

def _plot_mixtral_breakdown_payload(payload: dict[str, object]) -> plt.Figure:
    models = [model for model in payload.get("models", []) if isinstance(model, dict)]
    if not models:
        raise ValueError("Mixtral breakdown cache must contain at least one model")
    model = models[0]
    dims = (
        f"{int(model.get('num_layers', 0))}L, "
        f"H={int(model.get('hidden_size', 0))}, "
        f"E={int(model.get('num_experts', 0))}, "
        f"top_k={int(model.get('top_k', 0))}"
    )

    fig = plt.figure(figsize=(12.5, 4.3))
    modes = [("steady_state", "Steady-state", "#0072B2"), ("cold_start", "Cold-start", "#D55E00")]

    # Panel A: backend cycles and PIM_MAC issued
    replay = model.get("replay_stats", {})
    ax1 = fig.add_subplot(1, 2, 1)
    metrics = [("cycles", "Cycles"), ("pim_mac_issued", "PIM_MAC issued")]
    x = range(len(metrics))
    width = 0.35
    for idx, (mode_key, mode_label, color) in enumerate(modes):
        mode_stats = replay.get(mode_key, {}) if isinstance(replay, dict) else {}
        values = []
        for metric_key, _ in metrics:
            val = mode_stats.get(metric_key, 0) if isinstance(mode_stats, dict) else 0
            if metric_key == "cycles":
                val = int(val) if val else 0
            else:
                val = int(val) if val else 0
            values.append(val)
        offsets = [pos + (idx - 0.5) * width for pos in x]
        bars = ax1.bar(offsets, values, width=width, label=mode_label, color=color, edgecolor="white", linewidth=0.5, alpha=0.85)
        for bar, value in zip(bars, values):
            label = f"{value:,.0f}" if value > 1000 else str(value)
            ax1.text(bar.get_x() + bar.get_width() / 2, max(value, 1) * 1.03, label, ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(list(x), [m[1] for m in metrics])
    ax1.set_title(f"Panel A: Backend replay — {dims}", fontsize=10)
    ax1.legend(fontsize=8, framealpha=0.9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel B: per-layer PIM_MAC buckets
    ax2 = fig.add_subplot(1, 2, 2)
    buckets = model.get("per_layer_pim_mac_buckets", {})
    if not isinstance(buckets, dict):
        buckets = {}
    bucket_order = ["qkvo_projection", "attention", "router", "expert_ffn_fused"]
    bucket_labels = ["Q/K/V/O", "Attention", "Router", "Expert FFN\n(fused)"]
    bucket_colors = ["#009E73", "#0072B2", "#E69F00", "#D55E00"]
    present = [(k, l, c) for k, l, c in zip(bucket_order, bucket_labels, bucket_colors) if k in buckets]
    x_buckets = range(len(present))
    values_buckets = [int(buckets.get(k, 0)) for k, _, _ in present]
    bars2 = ax2.bar(x_buckets, values_buckets, color=[c for _, _, c in present], edgecolor="white", linewidth=0.5, alpha=0.85)
    for bar, val in zip(bars2, values_buckets):
        ax2.text(bar.get_x() + bar.get_width() / 2, val * 1.03, f"{val:,}", ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(list(x_buckets), [l for _, l, _ in present], fontsize=8)
    ax2.set_yscale("log")
    ax2.set_title("Panel B: Per-layer PIM_MAC repeats", fontsize=10)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


def write_mixtral_breakdown_cache(cache_path: Path = DEFAULT_MIXTRAL_BREAKDOWN_CACHE, *, collect_backend: bool = True) -> Path:
    from ramulator.workload_surrogate.generate_full_transformer import FULL_TRANSFORMER_GENERATOR_VERSION
    from tests.analysis.figures import p4_figure_data

    cache_path = Path(cache_path)
    data = p4_figure_data.collect_mixtral_8x7b_moe_decoder_data(
        replay_stats_fn=p4_figure_data.collect_mixtral_8x7b_replay_stats if collect_backend else None,
    )
    backend_replay = data.pop("backend_replay_stats", [])
    replay_stats: dict[str, dict[str, object]] = {}
    if isinstance(backend_replay, list):
        for entry in backend_replay:
            if not isinstance(entry, dict):
                continue
            trace_name = str(entry.get("trace_name", ""))
            is_cold = "cold" in trace_name.lower()
            mode = "cold_start" if is_cold else "steady_state"
            replay_stats[mode] = {
                "cycles": int(entry.get("cycles", 0) or 0),
                "runtime_ns": entry.get("runtime_ns"),
                "pim_mac_issued": int(entry.get("pim_mac_issued", 0) or 0),
                "avg_pim_latency_cycles": entry.get("avg_pim_latency_cycles"),
                "command_counts": entry.get("command_counts", {}),
            }

    payload = {
        "schema_version": 1,
        "figure_id": MIXTRAL_BREAKDOWN_FIGURE_ID,
        "provenance": {
            "date": _dt.date.today().isoformat(),
            "generator_version": FULL_TRANSFORMER_GENERATOR_VERSION,
            "commit": _git_commit(),
            "replay_mode": "backend" if collect_backend else "precomputed",
        },
        "models": [
            {
                "model_name": data.get("model_name", "Mixtral-8x7B"),
                "num_layers": int(data.get("num_layers", 32)),
                "hidden_size": int(data.get("hidden_size", 512)),
                "expert_hidden_size": int(data.get("expert_hidden_size", 2048)),
                "real_hidden_size": int(data.get("real_hidden_size", 4096)),
                "real_expert_hidden_size": int(data.get("real_expert_hidden_size", 14336)),
                "num_experts": int(data.get("num_experts", 8)),
                "top_k": int(data.get("top_k", 2)),
                "num_heads": int(data.get("num_heads", 4)),
                "num_kv_heads": int(data.get("num_kv_heads", 1)),
                "head_dim": int(data.get("head_dim", 128)),
                "past_len": int(data.get("past_len", 1024)),
                "datatype": str(data.get("datatype", "int8")),
                "per_layer_pim_mac_buckets": {
                    "qkvo_projection": int(data.get("qkvo_projection_pim_mac_per_layer", 0)),
                    "attention": int(data.get("attention_pim_mac_per_layer", 0)),
                    "router": int(data.get("router_pim_mac_per_layer", 0)),
                    "expert_ffn_fused": int(data.get("expert_ffn_pim_mac_fused_per_layer", 0)),
                    "expert_ffn_real": int(data.get("expert_ffn_pim_mac_real_per_layer", 0)),
                    "expert_ffn_full_dim_real_analytic": int(data.get("expert_ffn_pim_mac_full_dim_real_per_layer", 0)),
                },
                "replay_stats": replay_stats,
                "concrete_counts": data.get("concrete_counts", {}),
                "caveats": [
                    "scaled_dimensions_hidden_512_expert_ffn_2048",
                    "gqa_no_reuse_4to1_ratio",
                    "fused_single_gemv_expert_mac_3x_factor_for_real_swiglu",
                    "deterministic_routing_selected_experts_0_1",
                    "decode_only_seq_len_1",
                    "bounded_surrogate_not_serving",
                    "non_silicon_calibrated",
                ],
            }
        ],
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return cache_path


def render_mixtral_breakdown_figure_from_cache(cache_path: Path = DEFAULT_MIXTRAL_BREAKDOWN_CACHE, output_dir: Path = Path("paper/figures")) -> None:
    payload = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    fig = _plot_mixtral_breakdown_payload(payload)
    _save(fig, Path(output_dir), MIXTRAL_BREAKDOWN_FIGURE_ID)


# ─── P1: Mixtral vs Llama2 comparison ───────────────────────────────────

def gen_figure_mixtral_vs_llama2(output_dir: Path, *, llama2_cache_path: Path = DEFAULT_LLAMA2_SCALING_CACHE, mixtral_cache_path: Path = DEFAULT_MIXTRAL_BREAKDOWN_CACHE) -> None:
    """Render Mixtral vs Llama2 comparison from both caches.

    Backend cycles are carried in the caches for provenance, but this figure
    intentionally plots PIM_MAC repeats only: raw cycles would be misleading
    because the Mixtral trace is dimension-scaled while Llama2-7B is full-size.
    """
    llama2_payload = json.loads(llama2_cache_path.read_text(encoding="utf-8")) if llama2_cache_path.exists() else None
    mixtral_payload = json.loads(mixtral_cache_path.read_text(encoding="utf-8")) if mixtral_cache_path.exists() else None

    fig = plt.figure(figsize=(12.5, 4.3))
    models_data: list[dict[str, object]] = []

    if llama2_payload:
        for m in llama2_payload.get("models", []):
            if isinstance(m, dict) and "Llama2-7B" in str(m.get("model_name", "")):
                dims = m.get("dimensions", {}) if isinstance(m.get("dimensions"), dict) else {}
                replay = m.get("replay_stats", {})
                steady = replay.get("steady_state", {}) if isinstance(replay, dict) else {}
                buckets = m.get("per_layer_pim_mac_buckets", {})
                models_data.append({
                    "label": "Llama2-7B",
                    "cycles": int(steady.get("cycles", 0) or 0),
                    "pim_mac": sum(int(buckets.get(k, 0)) for k in ["qkvo_projection", "attention", "ffn"]),
                    "buckets": {
                        "Q/K/V/O": int(buckets.get("qkvo_projection", 0)),
                        "Attention": int(buckets.get("attention", 0)),
                        "FFN/SwiGLU": int(buckets.get("ffn", 0)),
                    },
                })

    if mixtral_payload:
        for m in mixtral_payload.get("models", []):
            if isinstance(m, dict):
                replay = m.get("replay_stats", {})
                steady = replay.get("steady_state", {}) if isinstance(replay, dict) else {}
                buckets = m.get("per_layer_pim_mac_buckets", {})
                models_data.append({
                    "label": "Mixtral-8x7B\n(scaled)",
                    "cycles": int(steady.get("cycles", 0) or 0),
                    "pim_mac": sum(int(buckets.get(k, 0)) for k in ["qkvo_projection", "attention", "router", "expert_ffn_fused"]),
                    "buckets": {
                        "Q/K/V/O": int(buckets.get("qkvo_projection", 0)),
                        "Attention": int(buckets.get("attention", 0)),
                        "Router+Experts": int(buckets.get("router", 0)) + int(buckets.get("expert_ffn_fused", 0)),
                    },
                })

    if not models_data:
        print("  WARNING: No cached data for Mixtral vs Llama2 comparison — skipping fig14")
        return

    # Panel A: normalized per-layer PIM_MAC composition.  The raw totals differ
    # by construction (scaled Mixtral vs full-dim Llama2), so percentages make
    # the intra-model operator mix visible without implying latency fairness.
    ax1 = fig.add_subplot(1, 2, 1)
    width = 0.35
    x = range(len(models_data))
    bucket_order = ["Q/K/V/O", "Attention", "FFN/SwiGLU", "Router+Experts"]
    all_bucket_keys = [k for k in bucket_order if any(k in m["buckets"] for m in models_data)]
    colors = {"Q/K/V/O": "#009E73", "Attention": "#0072B2", "FFN/SwiGLU": "#D55E00", "Router+Experts": "#E69F00"}
    bottom = [0.0] * len(models_data)
    for bk in all_bucket_keys:
        vals = []
        for m in models_data:
            total = max(1, int(m.get("pim_mac", 0)))
            vals.append(float(m["buckets"].get(bk, 0)) / total * 100.0)
        bars = ax1.bar(x, vals, width, bottom=bottom, label=bk, color=colors.get(bk, "#999999"), edgecolor="white", linewidth=0.5)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax1.set_xticks(list(x), [m["label"] for m in models_data], fontsize=9)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Share of per-layer PIM_MAC repeats (%)", fontweight="bold")
    ax1.set_title("Panel A: Per-layer PIM_MAC composition", fontsize=10)
    ax1.legend(fontsize=7, framealpha=0.9)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    # Panel B: PIM_MAC total
    ax2 = fig.add_subplot(1, 2, 2)
    vals2 = [m["pim_mac"] for m in models_data]
    bars2 = ax2.bar(x, vals2, width * 1.2, color=["#0072B2", "#D55E00"], edgecolor="white", linewidth=0.5, alpha=0.85)
    for bar, val in zip(bars2, vals2):
        ax2.text(bar.get_x() + bar.get_width() / 2, val * 1.03, f"{val:,}", ha="center", va="bottom", fontsize=9)
    ax2.set_xticks(list(x), [m["label"] for m in models_data], fontsize=9)
    ax2.set_yscale("log")
    ax2.set_ylabel("PIM_MAC repeats per layer", fontweight="bold")
    ax2.set_title("Panel B: Total PIM_MAC (1 layer, log scale)", fontsize=10)
    ax2.text(
        0.02,
        0.03,
        "Caveat: Mixtral is scaled (H=512, expert=2048);\n"
        "Llama2-7B is full-dim (H=4096, FFN=11008).\n"
        "Raw backend cycles exist in caches but are not fairness-normalized.",
        transform=ax2.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F5F5F5", edgecolor="#BBBBBB", alpha=0.92),
    )
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    _save(fig, output_dir, MIXTRAL_VS_LLAMA2_FIGURE_ID)


# ─── P2: Expert scaling sensitivity ──────────────────────────────────────

def gen_figure_moe_expert_sensitivity(output_dir: Path, *, use_tiny: bool = False) -> None:
    from tests.analysis.figures import p4_figure_data

    sweep = p4_figure_data.collect_moe_expert_sensitivity_sweep()
    # Aggregate by (num_experts, top_k)
    aggregated: dict[tuple[int, int], dict[str, int]] = {}
    for r in sweep:
        key = (r["_num_experts"], r["_top_k"])
        if key not in aggregated:
            aggregated[key] = {"router_mac": 0, "expert_mac": 0, "total_pim_mac": 0}
        concrete = r.get("concrete_counts", {})
        aggregated[key]["total_pim_mac"] = int(concrete.get("PIM_MAC", 0))
        # Router PIM_MAC is hidden_size × num_experts, ceil-divided by PIM lanes.
        hidden = r.get("hidden_size", 32)
        num_experts = r.get("_num_experts", 4)
        lanes = 32
        aggregated[key]["router_mac"] = max(1, (1 * num_experts * hidden + lanes - 1) // lanes)

    num_experts_keys = sorted(set(k[0] for k in aggregated))
    top_k_keys = sorted(set(k[1] for k in aggregated))

    fig = plt.figure(figsize=(12.5, 4.3))
    # Panel A: total PIM_MAC heatmap/grouped bars
    ax1 = fig.add_subplot(1, 2, 1)
    width = 0.8 / len(top_k_keys)
    colors_tk = ["#0072B2", "#D55E00", "#009E73", "#E69F00"]
    for ti, tk in enumerate(top_k_keys):
        vals = []
        for ne in num_experts_keys:
            vals.append(aggregated.get((ne, tk), {}).get("total_pim_mac", 0))
        offsets = [pos + (ti - (len(top_k_keys) - 1) / 2) * width for pos in range(len(num_experts_keys))]
        ax1.bar(offsets, vals, width=width * 0.9, label=f"top_k={tk}", color=colors_tk[ti % len(colors_tk)], edgecolor="white", linewidth=0.5, alpha=0.85)
    ax1.set_xticks(range(len(num_experts_keys)), [str(ne) for ne in num_experts_keys])
    ax1.set_xlabel("num_experts", fontweight="bold")
    ax1.set_ylabel("PIM_MAC repeats", fontweight="bold")
    ax1.set_title("Panel A: PIM_MAC vs num_experts × top_k", fontsize=10)
    ax1.legend(fontsize=8, framealpha=0.9)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    # Panel B: router overhead fraction
    ax2 = fig.add_subplot(1, 2, 2)
    markers = ["o", "s", "^", "D", "P", "X"]
    for ti, tk in enumerate(top_k_keys):
        fracs = []
        for ne in num_experts_keys:
            a = aggregated.get((ne, tk), {})
            total = a.get("total_pim_mac", 1)
            router = a.get("router_mac", 0)
            fracs.append(router / max(1, total) * 100 if total else 0)
        ax2.plot(
            range(len(num_experts_keys)),
            fracs,
            linestyle="None",
            marker=markers[ti % len(markers)],
            label=f"top_k={tk}",
            color=colors_tk[ti % len(colors_tk)],
            markersize=6,
        )
    ax2.set_xticks(range(len(num_experts_keys)), [str(ne) for ne in num_experts_keys])
    ax2.set_xlabel("num_experts", fontweight="bold")
    ax2.set_ylabel("Router overhead (%)", fontweight="bold")
    ax2.set_ylim(0, 22)
    ax2.set_title("Panel B: Router MAC fraction", fontsize=10)
    ax2.legend(fontsize=8, framealpha=0.9)
    ax2.text(
        0.02,
        0.03,
        "Generator-level sweep only\nTiny manifest: H=32, expert=64\nNo backend timing/cycles",
        transform=ax2.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F5F5F5", edgecolor="#BBBBBB", alpha=0.92),
    )
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    _save(fig, output_dir, MOE_SENSITIVITY_FIGURE_ID)


# ─── P3: Per-operator diagnostics ───────────────────────────────────────

def gen_figure_moe_operator_diagnostics(output_dir: Path) -> None:
    from tests.analysis.figures import p4_figure_data

    data = p4_figure_data.collect_mixtral_8x7b_moe_decoder_data(replay_stats_fn=None)
    op_buckets = {
        "Q/K/V/O": int(data.get("qkvo_projection_pim_mac_per_layer", 0)),
        "Attention\n(QK^T+PV)": int(data.get("attention_pim_mac_per_layer", 0)),
        "Router": int(data.get("router_pim_mac_per_layer", 0)),
        "Expert FFN\n(fused, ×1)": int(data.get("expert_ffn_pim_mac_fused_per_layer", 0)),
        "Expert FFN\n(real, ×3; scaled)": int(data.get("expert_ffn_pim_mac_real_per_layer", 0)),
        "Expert FFN\n(full dim, ×3)\n(analytic)": int(data.get("expert_ffn_pim_mac_full_dim_real_per_layer", 0)),
    }

    fig = plt.figure(figsize=(10.5, 4.3))
    ax1 = fig.add_subplot(1, 1, 1)
    labels = list(op_buckets.keys())
    values = list(op_buckets.values())
    colors = ["#009E73", "#0072B2", "#E69F00", "#D55E00", "#CC79A7", "#999999"]
    bars = ax1.bar(range(len(labels)), values, color=colors, edgecolor="white", linewidth=0.5, alpha=0.85)
    bars[-1].set_hatch("//")
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, val * 1.03, f"{val:,}", ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(range(len(labels)), labels, fontsize=8)
    ax1.set_yscale("log")
    ax1.set_ylabel("PIM_MAC repeats per layer\n(1 repeat = 32 INT8 MACs)", fontweight="bold")
    ax1.set_title("Mixtral-8x7B: per-operator PIM_MAC (scaled, 1 layer)", fontsize=10)
    ax1.text(
        0.02,
        0.97,
        "Traceable bars use scaled dims (H=512, expert=2048).\n"
        "Full-dim ×3 bar is analytical only, not backend-replayed.",
        transform=ax1.transAxes,
        fontsize=8,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F5F5F5", edgecolor="#BBBBBB", alpha=0.92),
    )
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    fig.tight_layout()
    _save(fig, output_dir, MOE_OPERATOR_DIAG_FIGURE_ID)


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
    parser.add_argument(
        "--collect-llama2-scaling-cache",
        type=Path,
        nargs="?",
        const=DEFAULT_LLAMA2_SCALING_CACHE,
        default=None,
        help=f"Run expensive Llama2 7B/13B backend collection and write JSON cache (default: {DEFAULT_LLAMA2_SCALING_CACHE})",
    )
    parser.add_argument(
        "--render-llama2-scaling-cache",
        type=Path,
        nargs="?",
        const=DEFAULT_LLAMA2_SCALING_CACHE,
        default=None,
        help=f"Render Llama2 7B/13B scaling figure from JSON cache without backend simulation (default: {DEFAULT_LLAMA2_SCALING_CACHE})",
    )
    parser.add_argument(
        "--collect-decode-context-sweep-cache",
        type=Path,
        nargs="?",
        const=DEFAULT_DECODE_CONTEXT_SWEEP_CACHE,
        default=None,
        help=f"Run Llama2-7B backend decode context-length sweep and write JSON cache (default: {DEFAULT_DECODE_CONTEXT_SWEEP_CACHE})",
    )
    parser.add_argument(
        "--render-decode-context-sweep-cache",
        type=Path,
        nargs="?",
        const=DEFAULT_DECODE_CONTEXT_SWEEP_CACHE,
        default=None,
        help=f"Render decode context-length sweep figure from JSON cache without backend simulation (default: {DEFAULT_DECODE_CONTEXT_SWEEP_CACHE})",
    )
    parser.add_argument(
        "--collect-generated-token-sweep-cache",
        type=Path,
        nargs="?",
        const=DEFAULT_GENERATED_TOKEN_SWEEP_CACHE,
        default=None,
        help=f"Run Llama2-7B generated-token backend sweep and write JSON cache (default: {DEFAULT_GENERATED_TOKEN_SWEEP_CACHE})",
    )
    parser.add_argument(
        "--render-generated-token-sweep-cache",
        type=Path,
        nargs="?",
        const=DEFAULT_GENERATED_TOKEN_SWEEP_CACHE,
        default=None,
        help=f"Render generated-token sweep figure from JSON cache without backend simulation (default: {DEFAULT_GENERATED_TOKEN_SWEEP_CACHE})",
    )
    parser.add_argument(
        "--collect-mixtral-breakdown-cache",
        type=Path,
        nargs="?",
        const=DEFAULT_MIXTRAL_BREAKDOWN_CACHE,
        default=None,
        help=f"Run Mixtral-8x7B 32-layer backend collection and write JSON cache (default: {DEFAULT_MIXTRAL_BREAKDOWN_CACHE})",
    )
    parser.add_argument(
        "--render-mixtral-breakdown-cache",
        type=Path,
        nargs="?",
        const=DEFAULT_MIXTRAL_BREAKDOWN_CACHE,
        default=None,
        help=f"Render Mixtral-8x7B breakdown figure from JSON cache without backend simulation (default: {DEFAULT_MIXTRAL_BREAKDOWN_CACHE})",
    )
    opts = parser.parse_args()

    os.makedirs(opts.output_dir, exist_ok=True)

    if opts.collect_llama2_scaling_cache is not None:
        path = write_llama2_scaling_cache(opts.collect_llama2_scaling_cache, collect_backend=True)
        print(f"Wrote Llama2 scaling cache to {path.resolve()}")
        return 0

    if opts.render_llama2_scaling_cache is not None:
        render_llama2_scaling_figure_from_cache(opts.render_llama2_scaling_cache, opts.output_dir)
        print(f"Rendered Llama2 scaling figure from {opts.render_llama2_scaling_cache.resolve()}")
        return 0

    if opts.collect_decode_context_sweep_cache is not None:
        path = write_decode_context_sweep_cache(opts.collect_decode_context_sweep_cache)
        print(f"Wrote decode context-length sweep cache to {path.resolve()}")
        return 0

    if opts.render_decode_context_sweep_cache is not None:
        render_decode_context_sweep_figure_from_cache(opts.render_decode_context_sweep_cache, opts.output_dir)
        print(f"Rendered decode context-length sweep figure from {opts.render_decode_context_sweep_cache.resolve()}")
        return 0

    if opts.collect_generated_token_sweep_cache is not None:
        path = write_generated_token_sweep_cache(opts.collect_generated_token_sweep_cache)
        print(f"Wrote generated-token sweep cache to {path.resolve()}")
        return 0

    if opts.render_generated_token_sweep_cache is not None:
        render_generated_token_sweep_figure_from_cache(opts.render_generated_token_sweep_cache, opts.output_dir)
        print(f"Rendered generated-token sweep figure from {opts.render_generated_token_sweep_cache.resolve()}")
        return 0

    if opts.collect_mixtral_breakdown_cache is not None:
        path = write_mixtral_breakdown_cache(opts.collect_mixtral_breakdown_cache, collect_backend=True)
        print(f"Wrote Mixtral breakdown cache to {path.resolve()}")
        return 0

    if opts.render_mixtral_breakdown_cache is not None:
        render_mixtral_breakdown_figure_from_cache(opts.render_mixtral_breakdown_cache, opts.output_dir)
        print(f"Rendered Mixtral breakdown figure from {opts.render_mixtral_breakdown_cache.resolve()}")
        return 0

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

    print("Generating Llama2-7B 32-layer dense decoder breakdown...")
    gen_figure_9_llama2_7b_32_layer(opts.output_dir)
    print("  OK")

    print("Generating Llama2 7B/13B latency and compute comparison from cache...")
    _cached_or_collect_llama2_scaling(opts.output_dir)

    print("Generating Llama2-7B decode context-length sweep from cache...")
    _cached_or_collect_decode_context_sweep(opts.output_dir)

    print("Generating Llama2-7B generated-token sweep from cache...")
    _cached_or_collect_generated_token_sweep(opts.output_dir)

    print("Generating MoE expert scaling sensitivity...")
    gen_figure_moe_expert_sensitivity(opts.output_dir, use_tiny=opts.tiny)
    print("  OK")

    print("Generating MoE per-operator diagnostics...")
    gen_figure_moe_operator_diagnostics(opts.output_dir)
    print("  OK")

    print("Generating Mixtral vs Llama2 comparison from caches...")
    gen_figure_mixtral_vs_llama2(opts.output_dir)
    print("  OK")

    print(f"\nAll figures written to {opts.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
