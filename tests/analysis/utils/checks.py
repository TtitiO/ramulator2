"""Formula-based validation checks for DRAM simulation results.

All expected values are computed from JEDEC timing parameters via
DRAMStandard.resolve(). No stored baselines — formulas are the
single source of truth.

Each check returns a result dict with expected, measured, and % deviation.
"""

import json
from pathlib import Path

from tests.analysis.utils.spec import resolve_spec
from tests.analysis.utils.sweep import build_pim_energy_report


def check_unloaded_latency(curves, std_name):
    """Compare measured unloaded latency vs theoretical.

    Theoretical unloaded latency for a random (row-miss) probe:
        (nRP + nRCD + nCL) * time_unit_ns

    Measured: average probe latency at the lowest-BW point (highest NOP)
    on the 100% read curve.

    Returns dict with expected_ns, measured_ns, deviation_pct.
    """
    spec = resolve_spec(std_name)
    expected_ns = spec["unloaded_latency_ns"]

    # Use lowest-BW point on 100% read curve (first element, sorted by NOP desc)
    curve_100 = curves[100]
    measured_ns = curve_100["lat"][0]  # lowest BW = highest NOP = first element

    deviation_pct = (measured_ns - expected_ns) / expected_ns * 100

    return {
        "expected_ns": expected_ns,
        "measured_ns": measured_ns,
        "deviation_pct": deviation_pct,
        "nRP": spec["nRP"],
        "nRCD": spec["nRCD"],
        "nCL": spec["nCL"],
        "tCK_ns": spec["tCK_ns"],
    }


def check_peak_bandwidth(curves, std_name):
    """Compare measured max BW vs theoretical peak BW."""
    spec = resolve_spec(std_name)

    curve_100 = curves[100]
    measured_max_bw = max(curve_100["bw"])
    deviation_pct = (measured_max_bw - spec["max_theoretical_bw"]) / spec["max_theoretical_bw"] * 100

    return {
        "max_theoretical_bw": spec["max_theoretical_bw"],
        "measured_max_bw": measured_max_bw,
        "deviation_from_theoretical_pct": deviation_pct,
    }


def check_max_bandwidth(curves, std_name):
    """Compare measured max BW vs theoretical max achievable BW.

    Max theoretical BW = channel_width * rate / 8 / 1000 (GB/s)
    (doubled for pseudo-channel standards with 2 PCs)

    Refresh penalty per refresh event (cycles):
        nRTP + nRP + nRFC + nRCD
    This is the minimum unavailable time between two READs when a refresh
    intervenes: close row (nRTP + nRP), refresh (nRFC), reopen row (nRCD).

    Refresh overhead = refresh_penalty / nREFI
    Max achievable BW = max_theoretical * (1 - refresh_overhead)

    Measured: max BW on the 100% read curve (last element).

    Returns dict with all intermediate values and % deviation.
    """
    spec = resolve_spec(std_name)

    curve_100 = curves[100]
    measured_max_bw = max(curve_100["bw"])  # highest BW point

    deviation_pct = (measured_max_bw - spec["max_achievable_bw"]) / spec["max_achievable_bw"] * 100

    return {
        "max_theoretical_bw": spec["max_theoretical_bw"],
        "refresh_penalty_cycles": spec["refresh_penalty_cycles"],
        "refresh_overhead_pct": spec["refresh_overhead"] * 100,
        "max_achievable_bw": spec["max_achievable_bw"],
        "measured_max_bw": measured_max_bw,
        "deviation_from_achievable_pct": deviation_pct,
        "nRTP": spec["nRTP"],
        "nRP": spec["nRP"],
        "nRFC": spec["nRFC"],
        "nRCD": spec["nRCD"],
        "nREFI": spec["nREFI"],
    }


def check_streaming_peak_bandwidth(std_name, streaming_stats):
    """Compare pure streaming BW (no probe interference) vs theoretical peak BW."""
    spec = resolve_spec(std_name)
    time_unit_ns = spec["time_unit_ns"]
    bytes_per_req = spec["bytes_per_req"]

    ms = streaming_stats["memory_system"]
    fe = streaming_stats["frontend"]
    cycles = ms["controller"]["cycles"]
    streaming = fe["streaming_requests_sent"]

    measured_bw = (streaming * bytes_per_req) / (cycles * time_unit_ns)
    deviation_pct = (measured_bw - spec["max_theoretical_bw"]) / spec["max_theoretical_bw"] * 100

    return {
        "max_theoretical_bw": spec["max_theoretical_bw"],
        "measured_streaming_bw": measured_bw,
        "deviation_from_theoretical_pct": deviation_pct,
    }


def check_streaming_bandwidth(std_name, streaming_stats):
    """Compare pure streaming BW (no probe interference) vs max achievable BW.

    Runs the LatencyThroughputTrace in streaming-only mode (no probes, no NOP
    rate-limiting) and compares the achieved bandwidth against the theoretical
    max achievable BW (accounting for refresh overhead).

    Returns dict with theoretical, achievable, measured, and % deviation.
    """
    spec = resolve_spec(std_name)
    time_unit_ns = spec["time_unit_ns"]
    bytes_per_req = spec["bytes_per_req"]

    ms = streaming_stats["memory_system"]
    fe = streaming_stats["frontend"]
    cycles = ms["controller"]["cycles"]
    streaming = fe["streaming_requests_sent"]

    measured_bw = (streaming * bytes_per_req) / (cycles * time_unit_ns)
    deviation_pct = (measured_bw - spec["max_achievable_bw"]) / spec["max_achievable_bw"] * 100

    return {
        "max_theoretical_bw": spec["max_theoretical_bw"],
        "max_achievable_bw": spec["max_achievable_bw"],
        "measured_streaming_bw": measured_bw,
        "deviation_from_achievable_pct": deviation_pct,
    }


def check_pim_latency_throughput(curves):
    curve_100 = curves[100]
    max_idx = max(
        range(len(curve_100["pim_throughput"])),
        key=lambda idx: curve_100["pim_throughput"][idx],
    )
    measured_latency_ns = curve_100["pim_lat"][max_idx]
    measured_throughput = curve_100["pim_throughput"][max_idx]
    capacity_stalls = curve_100["pim_capacity_stalls"][max_idx]
    dependency_stalls = curve_100["pim_dependency_stalls"][max_idx]
    inflight_peak = curve_100["pim_inflight_peak"][max_idx]
    num_pim_reqs_served = curve_100["num_pim_reqs_served"][max_idx]
    analytical_energy_estimate_pJ = curve_100["analytical_energy_estimate_pJ"][max_idx]
    analytical_energy_per_pim_request_pJ = curve_100["analytical_energy_per_pim_request_pJ"][max_idx]
    total_command_count = curve_100["total_command_count"][max_idx]

    assert measured_latency_ns > 0
    assert measured_throughput > 0
    assert num_pim_reqs_served > 0

    return {
        "avg_pim_latency": measured_latency_ns,
        "measured_latency_ns": measured_latency_ns,
        "measured_throughput": measured_throughput,
        "num_pim_reqs_served": num_pim_reqs_served,
        "pim_capacity_stalls": capacity_stalls,
        "pim_dependency_stalls": dependency_stalls,
        "pim_inflight_peak": inflight_peak,
        "analytical_energy_estimate_pJ": analytical_energy_estimate_pJ,
        "analytical_energy_per_pim_request_pJ": analytical_energy_per_pim_request_pJ,
        "total_command_count": total_command_count,
        "nop": curve_100["nops"][max_idx],
    }


def check_pim_all_bank_split_throughput(curves):
    curve_100 = curves[100]
    max_idx = max(
        range(len(curve_100["pim_all_bank_throughput"])),
        key=lambda idx: curve_100["pim_all_bank_throughput"][idx],
    )
    measured_latency_ns = curve_100["pim_lat"][max_idx]
    measured_throughput = curve_100["pim_all_bank_throughput"][max_idx]
    num_pim_reqs_served = curve_100["num_pim_reqs_served"][max_idx]
    num_pim_ab_reqs_served = curve_100["num_pim_ab_reqs_served"][max_idx]

    assert measured_latency_ns > 0
    assert measured_throughput > 0
    assert num_pim_ab_reqs_served > 0

    return {
        "avg_pim_latency": measured_latency_ns,
        "measured_latency_ns": measured_latency_ns,
        "measured_throughput": measured_throughput,
        "num_pim_reqs_served": num_pim_reqs_served,
        "num_pim_ab_reqs_served": num_pim_ab_reqs_served,
        "pim_capacity_stalls": curve_100["pim_capacity_stalls"][max_idx],
        "pim_dependency_stalls": curve_100["pim_dependency_stalls"][max_idx],
        "pim_inflight_peak": curve_100["pim_inflight_peak"][max_idx],
        "pim_ab_inflight_peak": curve_100["pim_ab_inflight_peak"][max_idx],
        "pim_mode_stalls": curve_100["pim_mode_stalls"][max_idx],
        "pim_load_stalls": curve_100["pim_load_stalls"][max_idx],
        "nop": curve_100["nops"][max_idx],
    }


def check_pim_dependency_pattern_movement(dependent_result, independent_result):
    latency_improved = independent_result["avg_pim_latency"] < dependent_result["avg_pim_latency"]
    throughput_improved = independent_result["measured_throughput"] > dependent_result["measured_throughput"]
    assert latency_improved or throughput_improved

    return {
        "latency_improved": latency_improved,
        "throughput_improved": throughput_improved,
        "latency_delta_ns": independent_result["avg_pim_latency"] - dependent_result["avg_pim_latency"],
        "throughput_delta": independent_result["measured_throughput"] - dependent_result["measured_throughput"],
        "dependency_stall_delta": independent_result["pim_dependency_stalls"]
        - dependent_result["pim_dependency_stalls"],
        "capacity_stall_delta": independent_result["pim_capacity_stalls"]
        - dependent_result["pim_capacity_stalls"],
        "inflight_peak_delta": independent_result["pim_inflight_peak"]
        - dependent_result["pim_inflight_peak"],
        "served_delta": independent_result["num_pim_reqs_served"] - dependent_result["num_pim_reqs_served"],
    }


def curves_to_nop_dict(curves, read_ratio=100):
    curve = curves[read_ratio]
    nop_dict = {}
    for i, nop in enumerate(curve["nops"]):
        nop_dict[nop] = {
            "avg_pim_latency": curve["avg_pim_latency_ns"][i],
            "measured_throughput": curve["pim_throughput"][i],
            "all_bank_measured_throughput": curve.get("pim_all_bank_throughput", [0.0] * len(curve["nops"]))[i],
            "pim_dependency_stalls": curve["pim_dependency_stalls"][i],
            "pim_capacity_stalls": curve["pim_capacity_stalls"][i],
            "pim_inflight_peak": curve["pim_inflight_peak"][i],
            "num_pim_reqs_served": curve["num_pim_reqs_served"][i],
            "num_pim_ab_reqs_served": curve.get("num_pim_ab_reqs_served", [0] * len(curve["nops"]))[i],
            "pim_ab_inflight_peak": curve.get("pim_ab_inflight_peak", [0] * len(curve["nops"]))[i],
            "pim_mode_stalls": curve.get("pim_mode_stalls", [0] * len(curve["nops"]))[i],
            "pim_load_stalls": curve.get("pim_load_stalls", [0] * len(curve["nops"]))[i],
            "nop": nop,
        }
    return nop_dict


def compare_at_same_nop(dep_point, ind_point):
    """Compare dependent vs independent at the same NOP value.

    Returns dict with both values and deltas.
    """
    return {
        "dependent": dep_point,
        "independent": ind_point,
        "latency_delta_ns": ind_point["avg_pim_latency"] - dep_point["avg_pim_latency"],
        "throughput_delta": ind_point["measured_throughput"] - dep_point["measured_throughput"],
        "dependency_stall_delta": ind_point["pim_dependency_stalls"] - dep_point["pim_dependency_stalls"],
        "capacity_stall_delta": ind_point["pim_capacity_stalls"] - dep_point["pim_capacity_stalls"],
        "inflight_peak_delta": ind_point["pim_inflight_peak"] - dep_point["pim_inflight_peak"],
        "served_delta": ind_point["num_pim_reqs_served"] - dep_point["num_pim_reqs_served"],
    }


def build_pim_energy_evidence(stats, std_name):
    energy = build_pim_energy_report(stats, std_name)
    modeled = energy["modeled"]
    derived = energy["derived"]
    assumed = energy["assumed"]

    command_counts = modeled.get("command_counts", {})
    traces = modeled.get("command_traces", [])
    controller_stats = modeled.get("controller_stats", {})

    evidence = {
        "modeled": modeled,
        "derived": derived,
        "assumed": assumed,
        "reproducibility": {
            "command count source": "CommandCounter CSV emitted by latency-throughput runner",
            "command trace source": "CmdTraceRecorder per-channel trace emitted by latency-throughput runner",
            "controller stat source": "sim.stats['memory_system']['controller']",
            "replay rule": (
                "Recompute derived analytical estimates from emitted command counts, controller stats, time_unit_ns, "
                "and explicit analytical parameters; no hidden constants are used."
            ),
            "trace count matches counter total": derived["trace_count_matches_counter_total"],
            "trace command counts match counter": derived["trace_command_counts_match_counter"],
        },
        "summary_lines": [
            f"modeled command count total: {derived['total_command_count']}",
            f"modeled distinct command count entries: {len(command_counts)}",
            f"modeled trace channels: {len(traces)}",
            f"modeled pim requests served: {controller_stats.get('num_pim_reqs_served', 0)}",
            f"derived analytical energy estimate (pJ): {derived['analytical_energy_estimate_pJ']:.3f}",
            f"derived analytical energy per pim request (pJ): {derived['analytical_energy_per_pim_request_pJ']:.6f}",
            f"derived trace count matches counter total: {derived['trace_count_matches_counter_total']}",
            f"derived trace command counts match counter: {derived['trace_command_counts_match_counter']}",
            f"assumed parameter units: {assumed['parameter_units']}",
        ],
    }
    return evidence


def write_pim_energy_summary(evidence, summary_path, json_path=None, tuple_description=None):
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    tuple_description = tuple_description or "PIM_MAC requests (bank_group_size=4, burst_length=16)"

    lines = [
        "# Bounded Slice Energy and Datatype Summary",
        "",
        "## Workload & Datatype Labels",
        "- Workload class: Decode/GEMV-style microbenchmark",
        f"- Tuple: {tuple_description}",
        "- Baseline datatype: int8 (modeled baseline)",
        "- Assumed capability: fp16 (assumed contrast only)",
        "- Framing: Bounded multi-bank distribution (frontend-shaped, not full parallel issue claim)",
        "- Energy method: Trace-driven analytical estimate from observed commands and runtime",
        "",
        "## modeled",
        "",
        "### command count",
    ]
    for command, count in evidence["modeled"].get("command_counts", {}).items():
        lines.append(f"- {command}: {count}")

    lines.extend(["", "### command trace", ""])
    for trace in evidence["modeled"].get("command_traces", []):
        preview = ", ".join(trace.get("commands_preview", []))
        lines.append(
            f"- channel {trace['channel']}: path={trace['path']}, command count={trace['command_count']}, "
            f"preview=[{preview}]"
        )
        for command, count in trace.get("command_counts", {}).items():
            lines.append(f"  - counted from trace: {command}={count}")
        lines.append("  ```")
        lines.extend(f"  {line}" for line in trace.get("raw_text", "").splitlines()[:16])
        lines.append("  ```")

    lines.extend(["", "### controller stats", ""])
    for key, value in evidence["modeled"].get("controller_stats", {}).items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## derived", ""])
    for key, value in evidence["derived"].items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## assumed", ""])
    for key, value in evidence["assumed"].items():
        lines.append(f"- {key}: {value}")

    lines.extend(
        [
            "",
            "## interpretation",
            "",
            "- Analytical estimates here are parameterized comparative outputs derived from observed command activity and runtime.",
            "- They are not silicon-calibrated LPDDR5-PIM joules and must not be presented as validated physical energy.",
        ]
    )

    lines.extend(["", "## reproducibility", ""])
    for key, value in evidence["reproducibility"].items():
        lines.append(f"- {key}: {value}")

    summary_path.write_text("\n".join(lines) + "\n")

    if json_path is not None:
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
