"""Latency-throughput plot generation for validation tests.

Produces one plot per DRAM standard (PNG at 300 DPI) with:
- Lat-TP curves colored by read ratio (custom blue-red colormap)
- Reference lines for theoretical throughput and unloaded latency
- Summary below the plot with key measured vs expected values
"""

import os

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def plot_lat_tp(
    curves,
    std_name,
    check_results,
    output_dir="tests/analysis/plots/fast",
    verbose=False,
):
    """Generate and save a latency-throughput plot for one DRAM standard.

    Args:
        curves: {read_ratio: {"bw": [...], "lat": [...], "nops": [...]}}
        std_name: e.g. "DDR4"
        check_results: {"latency": lat_result_dict, "bandwidth": bw_result_dict}
        output_dir: directory to save the plot

    Returns:
        Path to the saved PNG plot file.
    """
    os.makedirs(output_dir, exist_ok=True)

    lat_res = check_results["latency"]
    bw_res = check_results["bandwidth"]

    read_ratios = sorted(curves.keys())
    cmap = LinearSegmentedColormap.from_list(
        "custom_blue_red", [(0.0, "#0000AA"), (1.0, "#AA0000")]
    )
    norm = mcolors.Normalize(vmin=50, vmax=100)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Data curves (heaviest visual element)
    for rr in read_ratios:
        c = curves[rr]
        color = cmap(norm(rr))
        ax.plot(c["bw"], c["lat"], "o-", color=color, linewidth=2, markersize=5, zorder=3)

    # Reference lines (only in verbose mode)
    if verbose:
        peak_bw = bw_res["max_theoretical_bw"]
        ax.axvline(
            peak_bw,
            color="#cc7722",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label="Max. Theoretical Throughput",
        )

        streaming_res = check_results.get("streaming")
        if streaming_res:
            streaming_bw = streaming_res["measured_streaming_bw"]
            ax.axvline(
                streaming_bw,
                color="#4477AA",
                linestyle="-.",
                linewidth=1.5,
                alpha=0.7,
                label="Streaming-Only Throughput (Measured)",
            )

        expected_lat = lat_res["expected_ns"]
        ax.axhline(
            expected_lat,
            color="#AA3377",
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
            label="Unloaded Latency (Ideal)",
        )

    # Axis limits and labels
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    from tests.analysis.testcases import STANDARDS

    cfg = STANDARDS[std_name]
    ax.set_title(cfg["timing_preset"].replace("_", "-"), fontsize=20, fontweight="bold", pad=16)
    ax.set_xlabel("Throughput (GB/s)", fontsize=16, labelpad=8)
    ax.set_ylabel("Random Probe Access Latency (ns)", fontsize=16, labelpad=8)

    # Grid (light, behind data)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.4)

    # Ticks
    ax.tick_params(axis="both", which="major", direction="in", length=5, width=1.2, labelsize=13)

    # Spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # Legend (only in verbose mode — reference lines provide the labels)
    if verbose:
        legend = ax.legend(
            fontsize=10,
            frameon=True,
            edgecolor="#666",
            facecolor="white",
            framealpha=0.9,
            loc="upper left",
        )
        legend.get_frame().set_linewidth(1.5)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("% Reads", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    # Summary below the plot (only in verbose mode)
    if verbose:
        peak_bw = bw_res["max_theoretical_bw"]
        streaming_res = check_results.get("streaming")

        left_lines = [
            f"Unloaded Latency (Ideal): {lat_res['expected_ns']:.1f} ns",
            f"Unloaded Latency (Measured): {lat_res['measured_ns']:.1f} ns",
        ]
        right_lines = [
            f"Max. Theoretical Throughput: {peak_bw:.1f} GB/s",
        ]
        if streaming_res:
            streaming_bw = streaming_res["measured_streaming_bw"]
            right_lines.append(
                f"Streaming-Only Throughput (Measured): {streaming_bw:.1f} GB/s"
            )

        fig.subplots_adjust(bottom=0.30, top=0.92)
        pos = ax.get_position()
        cx = (pos.x0 + pos.x1) / 2
        text_bbox = dict(boxstyle="round,pad=0.4", fc="white", ec="#ccc", linewidth=0.8)
        fig.text(
            cx,
            0.15,
            "\n".join(left_lines),
            ha="center",
            va="center",
            fontsize=9.5,
            color="#555",
            family="monospace",
            bbox=text_bbox,
        )
        fig.text(
            cx,
            0.04,
            "\n".join(right_lines),
            ha="center",
            va="center",
            fontsize=9.5,
            color="#555",
            family="monospace",
            bbox=text_bbox,
        )

    png_path = os.path.join(output_dir, f"{std_name}_lat_tp.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return png_path


def plot_pim_lat_tp(
    curves,
    std_name,
    output_dir="tests/analysis/plots/fast",
):
    os.makedirs(output_dir, exist_ok=True)

    read_ratios = sorted(curves.keys())
    cmap = LinearSegmentedColormap.from_list(
        "custom_blue_red", [(0.0, "#0000AA"), (1.0, "#AA0000")]
    )
    norm = mcolors.Normalize(vmin=50, vmax=100)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for rr in read_ratios:
        c = curves[rr]
        color = cmap(norm(rr))
        ax.plot(c["pim_throughput"], c["pim_lat"], "o-", color=color, linewidth=2, markersize=5, zorder=3)

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_title(std_name.upper(), fontsize=20, fontweight="bold", pad=16)
    ax.set_xlabel("PIMCompute Throughput (requests/ns)", fontsize=16, labelpad=8)
    ax.set_ylabel("PIMCompute Latency (ns)", fontsize=16, labelpad=8)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.4)
    ax.tick_params(axis="both", which="major", direction="in", length=5, width=1.2, labelsize=13)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("% Reads", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    png_path = os.path.join(output_dir, f"{std_name}_pim_lat_tp.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return png_path


def plot_pim_dependency_pattern_same_nop(
    dependency_pattern_results,
    std_name,
    output_dir="tests/analysis/plots/fast",
):
    """Generate same-NOP dependency-pattern comparison PNG plots.

    Returns a dict with file paths for latency, throughput, and dependency-stall plots.
    """

    os.makedirs(output_dir, exist_ok=True)

    dep_nop_dict = dependency_pattern_results["same_bank_dependent"]["nop_dict"]
    ind_nop_dict = dependency_pattern_results["same_bank_independent"]["nop_dict"]
    bnd_nop_dict = dependency_pattern_results["bounded_bank_group_round_robin"]["nop_dict"]
    partial_nop_dict = dependency_pattern_results.get("bounded_partial_subsequence", {}).get("nop_dict")
    repeated_nop_dict = dependency_pattern_results.get("bounded_repeated_bank_subsequence", {}).get("nop_dict")
    all_bank_nop_dict = None
    
    common_nops = sorted(set(dep_nop_dict.keys()) & set(ind_nop_dict.keys()) & set(bnd_nop_dict.keys()))
    if partial_nop_dict:
        common_nops = sorted(set(common_nops) & set(partial_nop_dict.keys()))
    if repeated_nop_dict:
        common_nops = sorted(set(common_nops) & set(repeated_nop_dict.keys()))
    if all_bank_nop_dict:
        common_nops = sorted(set(common_nops) & set(all_bank_nop_dict.keys()))

    if not common_nops:
        raise ValueError("No common NOP values found for dependency-pattern comparison")

    dep_lat = [dep_nop_dict[nop]["avg_pim_latency"] for nop in common_nops]
    ind_lat = [ind_nop_dict[nop]["avg_pim_latency"] for nop in common_nops]
    bnd_lat = [bnd_nop_dict[nop]["avg_pim_latency"] for nop in common_nops]
    partial_lat = [partial_nop_dict[nop]["avg_pim_latency"] for nop in common_nops] if partial_nop_dict else None
    repeated_lat = [repeated_nop_dict[nop]["avg_pim_latency"] for nop in common_nops] if repeated_nop_dict else None

    dep_tp = [dep_nop_dict[nop]["measured_throughput"] for nop in common_nops]
    ind_tp = [ind_nop_dict[nop]["measured_throughput"] for nop in common_nops]
    bnd_tp = [bnd_nop_dict[nop]["measured_throughput"] for nop in common_nops]
    partial_tp = [partial_nop_dict[nop]["measured_throughput"] for nop in common_nops] if partial_nop_dict else None
    repeated_tp = [repeated_nop_dict[nop]["measured_throughput"] for nop in common_nops] if repeated_nop_dict else None

    dep_stalls = [dep_nop_dict[nop]["pim_dependency_stalls"] for nop in common_nops]
    ind_stalls = [ind_nop_dict[nop]["pim_dependency_stalls"] for nop in common_nops]
    bnd_stalls = [bnd_nop_dict[nop]["pim_dependency_stalls"] for nop in common_nops]
    partial_stalls = [partial_nop_dict[nop]["pim_dependency_stalls"] for nop in common_nops] if partial_nop_dict else None
    repeated_stalls = [repeated_nop_dict[nop]["pim_dependency_stalls"] for nop in common_nops] if repeated_nop_dict else None

    all_bank_lat = (
        [all_bank_nop_dict[nop]["avg_pim_latency"] for nop in common_nops] if all_bank_nop_dict else None
    )
    all_bank_tp = (
        [all_bank_nop_dict[nop]["all_bank_measured_throughput"] for nop in common_nops] if all_bank_nop_dict else None
    )
    all_bank_stalls = (
        [all_bank_nop_dict[nop]["pim_dependency_stalls"] for nop in common_nops] if all_bank_nop_dict else None
    )

    def make_plot(y_dep, y_ind, y_bnd, title, ylabel, filename, y_all_bank=None, y_partial=None, y_repeated=None):
        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.plot(
            common_nops,
            y_dep,
            "o-",
            color="#aec7e8",
            linewidth=1.5,
            markersize=4.0,
            label="dependent (same-bank)",
            zorder=2,
            alpha=0.6,
        )
        ax.plot(
            common_nops,
            y_ind,
            "s-",
            color="#ffbb78",
            linewidth=1.5,
            markersize=4.0,
            label="independent (same-bank)",
            zorder=2,
            alpha=0.6,
        )
        ax.plot(
            common_nops,
            y_bnd,
            "^-",
            color="#2ca02c",
            linewidth=2.0,
            markersize=5.0,
            label="bounded (multi-bank RR)",
            zorder=4,
        )
        if y_partial is not None:
            ax.plot(
                common_nops,
                y_partial,
                "v-",
                color="#8c564b",
                linewidth=1.8,
                markersize=4.5,
                label="bounded (partial subseq)",
                zorder=3,
            )
        if y_repeated is not None:
            ax.plot(
                common_nops,
                y_repeated,
                "p-",
                color="#9467bd",
                linewidth=1.8,
                markersize=4.5,
                label="bounded (repeated bank)",
                zorder=3,
            )
        if y_all_bank is not None:
            ax.plot(
                common_nops,
                y_all_bank,
                "D-",
                color="#d62728",
                linewidth=2.0,
                markersize=5.0,
                label="all-bank split (logical AB req/ns)",
                zorder=4,
            )
        ax.set_title(title, fontsize=13, pad=10)
        ax.set_xlabel("NOP", fontsize=11, labelpad=6)
        ax.set_ylabel(ylabel, fontsize=11, labelpad=6)
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
        ax.tick_params(axis="both", which="major", labelsize=9.5)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
        ax.margins(x=0.04, y=0.08)
        legend = ax.legend(
            loc="best",
            fontsize=9.5,
            frameon=True,
            framealpha=0.95,
            edgecolor="#888",
            borderpad=0.5,
            handlelength=2.0,
        )
        legend.get_frame().set_linewidth(0.9)
        fig.tight_layout(pad=1.0)
        path = os.path.join(output_dir, filename)
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        return path

    lat_path = make_plot(
        dep_lat,
        ind_lat,
        bnd_lat,
        "PIM Dependency Pattern: Latency vs NOP",
        "Avg PIM Latency (ns)",
        f"{std_name}_dependency_pattern_latency_vs_nop.png",
        y_all_bank=all_bank_lat,
        y_partial=partial_lat,
        y_repeated=repeated_lat,
    )
    tp_path = make_plot(
        dep_tp,
        ind_tp,
        bnd_tp,
        "PIM Dependency Pattern: Throughput vs NOP",
        "Measured Throughput (req/ns)",
        f"{std_name}_dependency_pattern_throughput_vs_nop.png",
        y_all_bank=all_bank_tp,
        y_partial=partial_tp,
        y_repeated=repeated_tp,
    )
    stalls_path = make_plot(
        dep_stalls,
        ind_stalls,
        bnd_stalls,
        "PIM Dependency Pattern: Dependency Stalls vs NOP",
        "PIM Dependency Stalls",
        f"{std_name}_dependency_pattern_dependency_stalls_vs_nop.png",
        y_all_bank=all_bank_stalls,
        y_partial=partial_stalls,
        y_repeated=repeated_stalls,
    )

    return {
        "latency_vs_nop": lat_path,
        "throughput_vs_nop": tp_path,
        "dependency_stalls_vs_nop": stalls_path,
    }


def plot_pim_dependency_pattern_same_nop_plotly(
    dependency_pattern_results,
    std_name,
    output_dir="tests/analysis/plots/fast",
):
    """Generate same-NOP dependency-pattern comparison HTML plots (Plotly).

    Returns a dict with file paths for latency, throughput, and dependency-stall plots.
    """
    import plotly.graph_objects as go

    os.makedirs(output_dir, exist_ok=True)

    dep_nop_dict = dependency_pattern_results["same_bank_dependent"]["nop_dict"]
    ind_nop_dict = dependency_pattern_results["same_bank_independent"]["nop_dict"]
    bnd_nop_dict = dependency_pattern_results["bounded_bank_group_round_robin"]["nop_dict"]
    partial_nop_dict = dependency_pattern_results.get("bounded_partial_subsequence", {}).get("nop_dict")
    repeated_nop_dict = dependency_pattern_results.get("bounded_repeated_bank_subsequence", {}).get("nop_dict")
    all_bank_nop_dict = None
    
    common_nops = sorted(set(dep_nop_dict.keys()) & set(ind_nop_dict.keys()) & set(bnd_nop_dict.keys()))
    if partial_nop_dict:
        common_nops = sorted(set(common_nops) & set(partial_nop_dict.keys()))
    if repeated_nop_dict:
        common_nops = sorted(set(common_nops) & set(repeated_nop_dict.keys()))
    if all_bank_nop_dict:
        common_nops = sorted(set(common_nops) & set(all_bank_nop_dict.keys()))

    if not common_nops:
        raise ValueError("No common NOP values found for dependency-pattern comparison")

    dep_lat = [dep_nop_dict[nop]["avg_pim_latency"] for nop in common_nops]
    ind_lat = [ind_nop_dict[nop]["avg_pim_latency"] for nop in common_nops]
    bnd_lat = [bnd_nop_dict[nop]["avg_pim_latency"] for nop in common_nops]
    partial_lat = [partial_nop_dict[nop]["avg_pim_latency"] for nop in common_nops] if partial_nop_dict else None
    repeated_lat = [repeated_nop_dict[nop]["avg_pim_latency"] for nop in common_nops] if repeated_nop_dict else None

    dep_tp = [dep_nop_dict[nop]["measured_throughput"] for nop in common_nops]
    ind_tp = [ind_nop_dict[nop]["measured_throughput"] for nop in common_nops]
    bnd_tp = [bnd_nop_dict[nop]["measured_throughput"] for nop in common_nops]
    partial_tp = [partial_nop_dict[nop]["measured_throughput"] for nop in common_nops] if partial_nop_dict else None
    repeated_tp = [repeated_nop_dict[nop]["measured_throughput"] for nop in common_nops] if repeated_nop_dict else None

    dep_stalls = [dep_nop_dict[nop]["pim_dependency_stalls"] for nop in common_nops]
    ind_stalls = [ind_nop_dict[nop]["pim_dependency_stalls"] for nop in common_nops]
    bnd_stalls = [bnd_nop_dict[nop]["pim_dependency_stalls"] for nop in common_nops]
    partial_stalls = [partial_nop_dict[nop]["pim_dependency_stalls"] for nop in common_nops] if partial_nop_dict else None
    repeated_stalls = [repeated_nop_dict[nop]["pim_dependency_stalls"] for nop in common_nops] if repeated_nop_dict else None

    all_bank_lat = (
        [all_bank_nop_dict[nop]["avg_pim_latency"] for nop in common_nops] if all_bank_nop_dict else None
    )
    all_bank_tp = (
        [all_bank_nop_dict[nop]["all_bank_measured_throughput"] for nop in common_nops] if all_bank_nop_dict else None
    )
    all_bank_stalls = (
        [all_bank_nop_dict[nop]["pim_dependency_stalls"] for nop in common_nops] if all_bank_nop_dict else None
    )

    def make_plotly(y_dep, y_ind, y_bnd, title, ylabel, filename, y_all_bank=None, y_partial=None, y_repeated=None):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=common_nops,
                y=y_dep,
                mode="lines+markers",
                name="dependent (same-bank)",
                line=dict(color="#aec7e8", width=1.5),
                marker=dict(size=6),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=common_nops,
                y=y_ind,
                mode="lines+markers",
                name="independent (same-bank)",
                line=dict(color="#ffbb78", width=1.5),
                marker=dict(size=6),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=common_nops,
                y=y_bnd,
                mode="lines+markers",
                name="bounded (multi-bank RR)",
                line=dict(color="#2ca02c", width=2.5),
                marker=dict(size=8),
            )
        )
        if y_partial is not None:
            fig.add_trace(
                go.Scatter(
                    x=common_nops,
                    y=y_partial,
                    mode="lines+markers",
                    name="bounded (partial subseq)",
                    line=dict(color="#8c564b", width=2),
                    marker=dict(size=7),
                )
            )
        if y_repeated is not None:
            fig.add_trace(
                go.Scatter(
                    x=common_nops,
                    y=y_repeated,
                    mode="lines+markers",
                    name="bounded (repeated bank)",
                    line=dict(color="#9467bd", width=2),
                    marker=dict(size=7),
                )
            )
        if y_all_bank is not None:
            fig.add_trace(
                go.Scatter(
                    x=common_nops,
                    y=y_all_bank,
                    mode="lines+markers",
                    name="all-bank split (logical AB req/ns)",
                    line=dict(color="#d62728", width=2.5),
                    marker=dict(size=8),
                )
            )
        fig.update_layout(
            title=title,
            xaxis_title="NOP",
            yaxis_title=ylabel,
            template="plotly_white",
            hovermode="x unified",
            legend=dict(x=0.01, y=0.99, borderwidth=1),
        )
        path = os.path.join(output_dir, filename)
        fig.write_html(path)
        return path

    lat_path = make_plotly(
        dep_lat,
        ind_lat,
        bnd_lat,
        "PIM Dependency Pattern: Latency vs NOP",
        "Avg PIM Latency (ns)",
        f"{std_name}_dependency_pattern_latency_vs_nop.html",
        y_all_bank=all_bank_lat,
        y_partial=partial_lat,
        y_repeated=repeated_lat,
    )
    tp_path = make_plotly(
        dep_tp,
        ind_tp,
        bnd_tp,
        "PIM Dependency Pattern: Throughput vs NOP",
        "Measured Throughput (req/ns)",
        f"{std_name}_dependency_pattern_throughput_vs_nop.html",
        y_all_bank=all_bank_tp,
        y_partial=partial_tp,
        y_repeated=repeated_tp,
    )
    stalls_path = make_plotly(
        dep_stalls,
        ind_stalls,
        bnd_stalls,
        "PIM Dependency Pattern: Dependency Stalls vs NOP",
        "PIM Dependency Stalls",
        f"{std_name}_dependency_pattern_dependency_stalls_vs_nop.html",
        y_all_bank=all_bank_stalls,
        y_partial=partial_stalls,
        y_repeated=repeated_stalls,
    )

    return {
        "latency_vs_nop": lat_path,
        "throughput_vs_nop": tp_path,
        "dependency_stalls_vs_nop": stalls_path,
    }


def plot_lpddr5_pim_energy_breakdown_by_case(case_reports, output_dir="tests/analysis/plots/fast"):
    """Generate stacked LPDDR5PIM energy decomposition PNG by case at same NOP."""

    os.makedirs(output_dir, exist_ok=True)

    labels = [report["label"] for report in case_reports]
    memory_background = [report["memory_background_energy_pJ"] for report in case_reports]
    memory_cmd = [report["memory_cmd_energy_pJ"] for report in case_reports]
    pim_incremental = [report["pim_incremental_energy_pJ"] for report in case_reports]
    x = range(len(case_reports))

    fig, ax = plt.subplots(figsize=(11.6, 5.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.bar(x, memory_background, color="#9ecae1", label="Memory background")
    ax.bar(x, memory_cmd, bottom=memory_background, color="#3182bd", label="Memory command")
    ax.bar(
        x,
        pim_incremental,
        bottom=[bg + cmd for bg, cmd in zip(memory_background, memory_cmd)],
        color="#e6550d",
        label="PIM incremental",
    )

    shared_nop = case_reports[0]["nop"]
    ax.set_title(f"LPDDR5PIM Energy Decomposition by Case at Same NOP={shared_nop}", fontsize=13, pad=10)
    ax.set_xlabel("Execution case", fontsize=11, labelpad=6)
    ax.set_ylabel("Energy (pJ)", fontsize=11, labelpad=6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8.8, rotation=18, ha="right")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.tick_params(axis="y", which="major", labelsize=9.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    legend = ax.legend(loc="best", fontsize=9, frameon=True, framealpha=0.95, edgecolor="#888")
    legend.get_frame().set_linewidth(0.9)

    for idx, report in enumerate(case_reports):
        total = report["analytical_energy_estimate_pJ"]
        ax.text(idx, total, f"{total:.1f}", ha="center", va="bottom", fontsize=8.5)

    fig.tight_layout(pad=1.0)
    path = os.path.join(output_dir, "lpddr5_pim_energy_breakdown_same_nop_by_case.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path


def plot_lpddr5_pim_energy_breakdown_by_case_plotly(
    case_reports, output_dir="tests/analysis/plots/fast"
):
    """Generate stacked LPDDR5PIM energy decomposition HTML by case at same NOP."""

    import plotly.graph_objects as go

    os.makedirs(output_dir, exist_ok=True)
    labels = [report["label"] for report in case_reports]
    shared_nop = case_reports[0]["nop"]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(name="Memory background", x=labels, y=[report["memory_background_energy_pJ"] for report in case_reports], marker_color="#9ecae1")
    )
    fig.add_trace(
        go.Bar(name="Memory command", x=labels, y=[report["memory_cmd_energy_pJ"] for report in case_reports], marker_color="#3182bd")
    )
    fig.add_trace(
        go.Bar(name="PIM incremental", x=labels, y=[report["pim_incremental_energy_pJ"] for report in case_reports], marker_color="#e6550d")
    )
    fig.update_layout(
        barmode="stack",
        title=f"LPDDR5PIM Energy Decomposition by Case at Same NOP={shared_nop}",
        xaxis_title="Execution case",
        yaxis_title="Energy (pJ)",
        template="plotly_white",
        legend=dict(x=0.01, y=0.99, borderwidth=1),
    )

    path = os.path.join(output_dir, "lpddr5_pim_energy_breakdown_same_nop_by_case.html")
    fig.write_html(path)
    return path


def plot_lpddr5_pim_energy_performance_by_case(case_reports, output_dir="tests/analysis/plots/fast"):
    """Generate LPDDR5PIM energy-performance companion PNG by case at same NOP."""

    os.makedirs(output_dir, exist_ok=True)
    labels = [report["label"] for report in case_reports]
    throughput = [report["measured_throughput"] for report in case_reports]
    energy_per_req = [report["analytical_energy_per_pim_request_pJ"] for report in case_reports]
    shared_nop = case_reports[0]["nop"]
    x = list(range(len(case_reports)))

    fig, ax1 = plt.subplots(figsize=(11.6, 5.0))
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("white")

    bars = ax1.bar(x, throughput, color="#74c476", width=0.58, label="Throughput")
    ax1.set_xlabel("Execution case", fontsize=11, labelpad=6)
    ax1.set_ylabel("Throughput (req/ns)", fontsize=11, labelpad=6, color="#2ca02c")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.8, rotation=18, ha="right")
    ax1.tick_params(axis="y", labelsize=9.5, colors="#2ca02c")
    ax1.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)

    ax2 = ax1.twinx()
    line = ax2.plot(x, energy_per_req, "o-", color="#d62728", linewidth=2.2, markersize=6, label="Energy / PIM req")[0]
    ax2.set_ylabel("Energy per PIM request (pJ)", fontsize=11, labelpad=6, color="#d62728")
    ax2.tick_params(axis="y", labelsize=9.5, colors="#d62728")

    ax1.set_title(f"LPDDR5PIM Energy-Performance by Case at Same NOP={shared_nop}", fontsize=13, pad=10)
    for spine in ax1.spines.values():
        spine.set_linewidth(1.0)
    for spine in ax2.spines.values():
        spine.set_linewidth(1.0)

    ax1.legend([bars, line], ["Throughput", "Energy / PIM req"], loc="best", fontsize=9, frameon=True, framealpha=0.95, edgecolor="#888")

    fig.tight_layout(pad=1.0)
    path = os.path.join(output_dir, "lpddr5_pim_energy_performance_same_nop_by_case.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path


def plot_lpddr5_pim_energy_performance_by_case_plotly(
    case_reports, output_dir="tests/analysis/plots/fast"
):
    """Generate LPDDR5PIM energy-performance companion HTML by case at same NOP."""

    import plotly.graph_objects as go

    os.makedirs(output_dir, exist_ok=True)
    labels = [report["label"] for report in case_reports]
    shared_nop = case_reports[0]["nop"]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=[report["measured_throughput"] for report in case_reports],
            name="Throughput",
            marker_color="#74c476",
            yaxis="y",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=[report["analytical_energy_per_pim_request_pJ"] for report in case_reports],
            mode="lines+markers",
            name="Energy / PIM req",
            line=dict(color="#d62728", width=2.2),
            marker=dict(size=7),
            yaxis="y2",
        )
    )
    fig.update_layout(
        title=f"LPDDR5PIM Energy-Performance by Case at Same NOP={shared_nop}",
        xaxis_title="Execution case",
        yaxis=dict(title="Throughput (req/ns)", color="#2ca02c"),
        yaxis2=dict(title="Energy per PIM request (pJ)", overlaying="y", side="right", color="#d62728"),
        template="plotly_white",
        legend=dict(x=0.01, y=0.99, borderwidth=1),
    )

    path = os.path.join(output_dir, "lpddr5_pim_energy_performance_same_nop_by_case.html")
    fig.write_html(path)
    return path


def plot_lpddr5_pim_avg_power_breakdown_by_case(case_reports, output_dir="tests/analysis/plots/fast"):
    """Generate stacked LPDDR5PIM average simulated power decomposition PNG by case at same NOP."""

    os.makedirs(output_dir, exist_ok=True)

    labels = [report["label"] for report in case_reports]
    memory_background = [report["memory_background_power_mW"] for report in case_reports]
    memory_cmd = [report["memory_cmd_power_mW"] for report in case_reports]
    pim_incremental = [report["pim_incremental_power_mW"] for report in case_reports]
    shared_nop = case_reports[0]["nop"]
    x = range(len(case_reports))

    fig, ax = plt.subplots(figsize=(11.6, 5.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.bar(x, memory_background, color="#c7e9c0", label="Memory background")
    ax.bar(x, memory_cmd, bottom=memory_background, color="#41ab5d", label="Memory command")
    ax.bar(
        x,
        pim_incremental,
        bottom=[bg + cmd for bg, cmd in zip(memory_background, memory_cmd)],
        color="#cb181d",
        label="PIM incremental",
    )

    ax.set_title(f"LPDDR5PIM Average Simulated Power by Case at Same NOP={shared_nop}", fontsize=13, pad=10)
    ax.set_xlabel("Execution case", fontsize=11, labelpad=6)
    ax.set_ylabel("Average simulated power (mW)", fontsize=11, labelpad=6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8.8, rotation=18, ha="right")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.tick_params(axis="y", which="major", labelsize=9.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    legend = ax.legend(loc="best", fontsize=9, frameon=True, framealpha=0.95, edgecolor="#888")
    legend.get_frame().set_linewidth(0.9)

    for idx, report in enumerate(case_reports):
        total = report["analytical_avg_power_mW"]
        ax.text(idx, total, f"{total:.3f}", ha="center", va="bottom", fontsize=8.5)

    fig.tight_layout(pad=1.0)
    path = os.path.join(output_dir, "lpddr5_pim_avg_simulated_power_breakdown_same_nop_by_case.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path


def plot_lpddr5_pim_avg_power_breakdown_by_case_plotly(
    case_reports, output_dir="tests/analysis/plots/fast"
):
    """Generate stacked LPDDR5PIM average simulated power decomposition HTML by case at same NOP."""

    import plotly.graph_objects as go

    os.makedirs(output_dir, exist_ok=True)
    labels = [report["label"] for report in case_reports]
    shared_nop = case_reports[0]["nop"]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(name="Memory background", x=labels, y=[report["memory_background_power_mW"] for report in case_reports], marker_color="#c7e9c0")
    )
    fig.add_trace(
        go.Bar(name="Memory command", x=labels, y=[report["memory_cmd_power_mW"] for report in case_reports], marker_color="#41ab5d")
    )
    fig.add_trace(
        go.Bar(name="PIM incremental", x=labels, y=[report["pim_incremental_power_mW"] for report in case_reports], marker_color="#cb181d")
    )
    fig.update_layout(
        barmode="stack",
        title=f"LPDDR5PIM Average Simulated Power by Case at Same NOP={shared_nop}",
        xaxis_title="Execution case",
        yaxis_title="Average simulated power (mW)",
        template="plotly_white",
        legend=dict(x=0.01, y=0.99, borderwidth=1),
    )

    path = os.path.join(output_dir, "lpddr5_pim_avg_simulated_power_breakdown_same_nop_by_case.html")
    fig.write_html(path)
    return path


def plot_lpddr5_pim_avg_power_vs_nop_representative_cases(
    case_series, output_dir="tests/analysis/plots/fast"
):
    """Generate LPDDR5PIM average simulated power vs NOP PNG for representative cases."""

    os.makedirs(output_dir, exist_ok=True)

    subplot_cases = [
        ("1-bank", "single_bank_1block", "single_bank_2block"),
        ("2-bank RR", "rr_2bank_1block", "rr_2bank_2block"),
        ("4-bank RR", "rr_4bank_1block", "rr_4bank_2block"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), sharey=False)
    fig.patch.set_facecolor("white")

    colors = {
        "1b": "#1f77b4",
        "2b": "#d62728",
    }

    for ax, (title, case_1b, case_2b) in zip(axes, subplot_cases):
        ax.set_facecolor("white")
        series_1b = case_series[case_1b]
        series_2b = case_series[case_2b]
        nops = [n for n in sorted(set(series_1b.keys()) & set(series_2b.keys())) if n <= 20]
        y1 = [series_1b[n]["analytical_avg_power_mW"] for n in nops]
        y2 = [series_2b[n]["analytical_avg_power_mW"] for n in nops]

        ax.plot(nops, y1, "o-", color=colors["1b"], linewidth=2.0, markersize=4.5, label="1 block/bank")
        ax.plot(nops, y2, "s-", color=colors["2b"], linewidth=2.0, markersize=4.5, label="2 blocks/bank")
        ax.set_title(title, fontsize=12, pad=8)
        ax.set_xlabel("NOP", fontsize=11, labelpad=6)
        ax.set_xlim(min(nops), max(nops))
        ymin = min(y1 + y2)
        ymax = max(y1 + y2)
        margin = max((ymax - ymin) * 0.12, ymax * 0.02, 0.0002)
        ax.set_ylim(ymin - margin, ymax + margin)
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
        ax.tick_params(axis="both", which="major", labelsize=9.5)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

    axes[0].set_ylabel("Average simulated power (mW)", fontsize=11, labelpad=6)
    legend = axes[-1].legend(loc="best", fontsize=9, frameon=True, framealpha=0.95, edgecolor="#888")
    legend.get_frame().set_linewidth(0.9)

    fig.suptitle("LPDDR5PIM Average Simulated Power vs NOP (NOP ≤ 20)", fontsize=14, y=1.02)
    fig.tight_layout(pad=1.0)
    path = os.path.join(output_dir, "lpddr5_pim_avg_simulated_power_vs_nop_representative_cases.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path


def plot_lpddr5_pim_latency_vs_nop_representative_cases(
    case_series, output_dir="tests/analysis/plots/fast"
):
    """Generate LPDDR5PIM average latency vs NOP PNG for representative cases."""

    os.makedirs(output_dir, exist_ok=True)

    subplot_cases = [
        ("1-bank", "single_bank_1block", "single_bank_2block"),
        ("2-bank RR", "rr_2bank_1block", "rr_2bank_2block"),
        ("4-bank RR", "rr_4bank_1block", "rr_4bank_2block"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), sharey=False)
    fig.patch.set_facecolor("white")

    colors = {
        "1b": "#1f77b4",
        "2b": "#d62728",
    }

    for ax, (title, case_1b, case_2b) in zip(axes, subplot_cases):
        ax.set_facecolor("white")
        series_1b = case_series[case_1b]
        series_2b = case_series[case_2b]
        nops = [n for n in sorted(set(series_1b.keys()) & set(series_2b.keys())) if n <= 20]
        y1 = [series_1b[n]["avg_pim_latency_ns"] for n in nops]
        y2 = [series_2b[n]["avg_pim_latency_ns"] for n in nops]

        ax.plot(nops, y1, "o-", color=colors["1b"], linewidth=2.0, markersize=4.5, label="1 block/bank")
        ax.plot(nops, y2, "s-", color=colors["2b"], linewidth=2.0, markersize=4.5, label="2 blocks/bank")
        ax.set_title(title, fontsize=12, pad=8)
        ax.set_xlabel("NOP", fontsize=11, labelpad=6)
        ax.set_xlim(min(nops), max(nops))
        ymin = min(y1 + y2)
        ymax = max(y1 + y2)
        margin = max((ymax - ymin) * 0.12, ymax * 0.02, 0.2)
        ax.set_ylim(ymin - margin, ymax + margin)
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
        ax.tick_params(axis="both", which="major", labelsize=9.5)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

    axes[0].set_ylabel("Average PIM latency (ns)", fontsize=11, labelpad=6)
    legend = axes[-1].legend(loc="best", fontsize=9, frameon=True, framealpha=0.95, edgecolor="#888")
    legend.get_frame().set_linewidth(0.9)

    fig.suptitle("LPDDR5PIM Average PIM Latency vs NOP (NOP ≤ 20)", fontsize=14, y=1.02)
    fig.tight_layout(pad=1.0)
    path = os.path.join(output_dir, "lpddr5_pim_avg_pim_latency_vs_nop_representative_cases.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path


def plot_lpddr5_pim_edp_vs_nop_representative_cases(
    case_series, output_dir="tests/analysis/plots/fast"
):
    """Generate LPDDR5PIM EDP vs NOP PNG for representative cases."""

    os.makedirs(output_dir, exist_ok=True)

    subplot_cases = [
        ("1-bank", "single_bank_1block", "single_bank_2block"),
        ("2-bank RR", "rr_2bank_1block", "rr_2bank_2block"),
        ("4-bank RR", "rr_4bank_1block", "rr_4bank_2block"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), sharey=False)
    fig.patch.set_facecolor("white")

    colors = {
        "1b": "#1f77b4",
        "2b": "#d62728",
    }

    for ax, (title, case_1b, case_2b) in zip(axes, subplot_cases):
        ax.set_facecolor("white")
        series_1b = case_series[case_1b]
        series_2b = case_series[case_2b]
        nops = [n for n in sorted(set(series_1b.keys()) & set(series_2b.keys())) if n <= 20]
        y1 = [series_1b[n]["analytical_energy_estimate_pJ"] * series_1b[n]["avg_pim_latency_ns"] for n in nops]
        y2 = [series_2b[n]["analytical_energy_estimate_pJ"] * series_2b[n]["avg_pim_latency_ns"] for n in nops]

        ax.plot(nops, y1, "o-", color=colors["1b"], linewidth=2.0, markersize=4.5, label="1 block/bank")
        ax.plot(nops, y2, "s-", color=colors["2b"], linewidth=2.0, markersize=4.5, label="2 blocks/bank")
        ax.set_title(title, fontsize=12, pad=8)
        ax.set_xlabel("NOP", fontsize=11, labelpad=6)
        ax.set_xlim(min(nops), max(nops))
        ymin = min(y1 + y2)
        ymax = max(y1 + y2)
        margin = max((ymax - ymin) * 0.12, ymax * 0.02, 0.2)
        ax.set_ylim(ymin - margin, ymax + margin)
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
        ax.tick_params(axis="both", which="major", labelsize=9.5)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

    axes[0].set_ylabel("EDP (pJ·ns)", fontsize=11, labelpad=6)
    legend = axes[-1].legend(loc="best", fontsize=9, frameon=True, framealpha=0.95, edgecolor="#888")
    legend.get_frame().set_linewidth(0.9)

    fig.suptitle("LPDDR5PIM EDP vs NOP (NOP ≤ 20)", fontsize=14, y=1.02)
    fig.tight_layout(pad=1.0)
    path = os.path.join(output_dir, "lpddr5_pim_edp_vs_nop_representative_cases.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path


def plot_lpddr5_pim_avg_power_vs_nop_representative_cases_plotly(
    case_series, output_dir="tests/analysis/plots/fast"
):
    """Generate LPDDR5PIM average simulated power vs NOP HTML for representative cases."""

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    os.makedirs(output_dir, exist_ok=True)

    subplot_cases = [
        ("1-bank", "single_bank_1block", "single_bank_2block"),
        ("2-bank RR", "rr_2bank_1block", "rr_2bank_2block"),
        ("4-bank RR", "rr_4bank_1block", "rr_4bank_2block"),
    ]

    fig = make_subplots(rows=1, cols=3, subplot_titles=[title for title, _, _ in subplot_cases], shared_yaxes=False)
    colors = {
        "1b": "#1f77b4",
        "2b": "#d62728",
    }

    for idx, (_, case_1b, case_2b) in enumerate(subplot_cases, start=1):
        series_1b = case_series[case_1b]
        series_2b = case_series[case_2b]
        nops = [n for n in sorted(set(series_1b.keys()) & set(series_2b.keys())) if n <= 20]
        fig.add_trace(
            go.Scatter(
                x=nops,
                y=[series_1b[n]["analytical_avg_power_mW"] for n in nops],
                mode="lines+markers",
                name="1 block/bank",
                legendgroup="1b",
                showlegend=(idx == 1),
                line=dict(color=colors["1b"], width=2.0),
                marker=dict(size=7),
            ),
            row=1,
            col=idx,
        )
        fig.add_trace(
            go.Scatter(
                x=nops,
                y=[series_2b[n]["analytical_avg_power_mW"] for n in nops],
                mode="lines+markers",
                name="2 blocks/bank",
                legendgroup="2b",
                showlegend=(idx == 1),
                line=dict(color=colors["2b"], width=2.0),
                marker=dict(size=7),
            ),
            row=1,
            col=idx,
        )
        fig.update_xaxes(title_text="NOP", row=1, col=idx)
        ymin = min([series_1b[n]["analytical_avg_power_mW"] for n in nops] + [series_2b[n]["analytical_avg_power_mW"] for n in nops])
        ymax = max([series_1b[n]["analytical_avg_power_mW"] for n in nops] + [series_2b[n]["analytical_avg_power_mW"] for n in nops])
        margin = max((ymax - ymin) * 0.12, ymax * 0.02, 0.0002)
        fig.update_yaxes(range=[ymin - margin, ymax + margin], row=1, col=idx)

    fig.update_yaxes(title_text="Average simulated power (mW)", row=1, col=1)
    fig.update_layout(
        title="LPDDR5PIM Average Simulated Power vs NOP (NOP ≤ 20)",
        template="plotly_white",
        legend=dict(x=0.01, y=0.99, borderwidth=1),
    )

    path = os.path.join(output_dir, "lpddr5_pim_avg_simulated_power_vs_nop_representative_cases.html")
    fig.write_html(path)
    return path


def plot_lpddr5_pim_latency_vs_nop_representative_cases_plotly(
    case_series, output_dir="tests/analysis/plots/fast"
):
    """Generate LPDDR5PIM average latency vs NOP HTML for representative cases."""

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    os.makedirs(output_dir, exist_ok=True)

    subplot_cases = [
        ("1-bank", "single_bank_1block", "single_bank_2block"),
        ("2-bank RR", "rr_2bank_1block", "rr_2bank_2block"),
        ("4-bank RR", "rr_4bank_1block", "rr_4bank_2block"),
    ]

    fig = make_subplots(rows=1, cols=3, subplot_titles=[title for title, _, _ in subplot_cases], shared_yaxes=False)
    colors = {
        "1b": "#1f77b4",
        "2b": "#d62728",
    }

    for idx, (_, case_1b, case_2b) in enumerate(subplot_cases, start=1):
        series_1b = case_series[case_1b]
        series_2b = case_series[case_2b]
        nops = [n for n in sorted(set(series_1b.keys()) & set(series_2b.keys())) if n <= 20]
        fig.add_trace(
            go.Scatter(
                x=nops,
                y=[series_1b[n]["avg_pim_latency_ns"] for n in nops],
                mode="lines+markers",
                name="1 block/bank",
                legendgroup="1b",
                showlegend=(idx == 1),
                line=dict(color=colors["1b"], width=2.0),
                marker=dict(size=7),
            ),
            row=1,
            col=idx,
        )
        fig.add_trace(
            go.Scatter(
                x=nops,
                y=[series_2b[n]["avg_pim_latency_ns"] for n in nops],
                mode="lines+markers",
                name="2 blocks/bank",
                legendgroup="2b",
                showlegend=(idx == 1),
                line=dict(color=colors["2b"], width=2.0),
                marker=dict(size=7),
            ),
            row=1,
            col=idx,
        )
        fig.update_xaxes(title_text="NOP", row=1, col=idx)
        ymin = min([series_1b[n]["avg_pim_latency_ns"] for n in nops] + [series_2b[n]["avg_pim_latency_ns"] for n in nops])
        ymax = max([series_1b[n]["avg_pim_latency_ns"] for n in nops] + [series_2b[n]["avg_pim_latency_ns"] for n in nops])
        margin = max((ymax - ymin) * 0.12, ymax * 0.02, 0.2)
        fig.update_yaxes(range=[ymin - margin, ymax + margin], row=1, col=idx)

    fig.update_yaxes(title_text="Average PIM latency (ns)", row=1, col=1)
    fig.update_layout(
        title="LPDDR5PIM Average PIM Latency vs NOP (NOP ≤ 20)",
        template="plotly_white",
        legend=dict(x=0.01, y=0.99, borderwidth=1),
    )

    path = os.path.join(output_dir, "lpddr5_pim_avg_pim_latency_vs_nop_representative_cases.html")
    fig.write_html(path)
    return path


def plot_lpddr5_pim_edp_vs_nop_representative_cases_plotly(
    case_series, output_dir="tests/analysis/plots/fast"
):
    """Generate LPDDR5PIM EDP vs NOP HTML for representative cases."""

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    os.makedirs(output_dir, exist_ok=True)

    subplot_cases = [
        ("1-bank", "single_bank_1block", "single_bank_2block"),
        ("2-bank RR", "rr_2bank_1block", "rr_2bank_2block"),
        ("4-bank RR", "rr_4bank_1block", "rr_4bank_2block"),
    ]

    fig = make_subplots(rows=1, cols=3, subplot_titles=[title for title, _, _ in subplot_cases], shared_yaxes=False)
    colors = {
        "1b": "#1f77b4",
        "2b": "#d62728",
    }

    for idx, (_, case_1b, case_2b) in enumerate(subplot_cases, start=1):
        series_1b = case_series[case_1b]
        series_2b = case_series[case_2b]
        nops = [n for n in sorted(set(series_1b.keys()) & set(series_2b.keys())) if n <= 20]
        fig.add_trace(
            go.Scatter(
                x=nops,
                y=[series_1b[n]["analytical_energy_estimate_pJ"] * series_1b[n]["avg_pim_latency_ns"] for n in nops],
                mode="lines+markers",
                name="1 block/bank",
                legendgroup="1b",
                showlegend=(idx == 1),
                line=dict(color=colors["1b"], width=2.0),
                marker=dict(size=7),
            ),
            row=1,
            col=idx,
        )
        fig.add_trace(
            go.Scatter(
                x=nops,
                y=[series_2b[n]["analytical_energy_estimate_pJ"] * series_2b[n]["avg_pim_latency_ns"] for n in nops],
                mode="lines+markers",
                name="2 blocks/bank",
                legendgroup="2b",
                showlegend=(idx == 1),
                line=dict(color=colors["2b"], width=2.0),
                marker=dict(size=7),
            ),
            row=1,
            col=idx,
        )
        fig.update_xaxes(title_text="NOP", row=1, col=idx)
        ymin = min([
            series_1b[n]["analytical_energy_estimate_pJ"] * series_1b[n]["avg_pim_latency_ns"] for n in nops
        ] + [
            series_2b[n]["analytical_energy_estimate_pJ"] * series_2b[n]["avg_pim_latency_ns"] for n in nops
        ])
        ymax = max([
            series_1b[n]["analytical_energy_estimate_pJ"] * series_1b[n]["avg_pim_latency_ns"] for n in nops
        ] + [
            series_2b[n]["analytical_energy_estimate_pJ"] * series_2b[n]["avg_pim_latency_ns"] for n in nops
        ])
        margin = max((ymax - ymin) * 0.12, ymax * 0.02, 0.2)
        fig.update_yaxes(range=[ymin - margin, ymax + margin], row=1, col=idx)

    fig.update_yaxes(title_text="EDP (pJ·ns)", row=1, col=1)
    fig.update_layout(
        title="LPDDR5PIM EDP vs NOP (NOP ≤ 20)",
        template="plotly_white",
        legend=dict(x=0.01, y=0.99, borderwidth=1),
    )

    path = os.path.join(output_dir, "lpddr5_pim_edp_vs_nop_representative_cases.html")
    fig.write_html(path)
    return path
