"""Trace-driven analytical energy estimation for analysis tests.

This module replaces opaque command-weight proxies with named analytical
parameters driven by observed command counts and runtime duration. The
resulting outputs remain parameterized comparative estimates, not
silicon-calibrated joules.
"""

import copy

from tests.analysis.utils.spec import resolve_spec


PROXY_COMMAND_WEIGHTS = {
    "ACT1": 2.0,
    "ACT2": 2.0,
    "CAS_RD": 1.0,
    "CAS_WR": 1.0,
    "RD": 1.0,
    "WR": 1.0,
    "RDA": 1.0,
    "WRA": 1.0,
    "PIM_MAC": 6.0,
    "PREpb": 1.5,
    "PREab": 2.0,
    "REFab": 8.0,
}


ANALYTICAL_EVENT_ENERGY = {
    "PIM_MAC": ("IDD_MAC_mA", "nMAC"),
    "PIM_MAC_AB": ("IDD_MAC_mA", "nMAC"),
    "PIM_BCAST": ("IDD_CTRL_mA", "nCTRL"),
    "HAB": ("IDD_CTRL_mA", "nCTRL"),
    "HAB_PIM": ("IDD_CTRL_mA", "nCTRL"),
    "SB": ("IDD_CTRL_mA", "nCTRL"),
}


DEFAULT_ANALYTICAL_PARAMS = {
    "VDD": 1.0,
    "IDD_MAC_mA": 150.0,
    "IDD_CTRL_mA": 40.0,
    "nMAC": 8.0,
    "nCTRL": 1.0,
}


ANALYTICAL_PARAMS = {
    "lpddr5_pim": {
        **DEFAULT_ANALYTICAL_PARAMS,
        "VDD": 1.05,
        "IDD_MAC_mA": 140.0,
        "IDD_CTRL_mA": 35.0,
        "nMAC": 8.0,
        "nCTRL": 1.0,
    }
}


def _safe_div(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _sum_counts(command_counts, commands):
    return sum(command_counts.get(command, 0) for command in commands)


def _build_trace_reproducibility(traces):
    reproducibility = []
    for trace in traces:
        reproducibility.append(
            {
                "channel": trace.get("channel"),
                "path": trace.get("path"),
                "command_count": trace.get("command_count", 0),
                "command_counts": copy.deepcopy(trace.get("command_counts", {})),
                "replay_rule": (
                    "Count commands from the recorded CmdTraceRecorder CSV rows for this channel and apply "
                    "the same explicit analytical energy parameters used in the derived formulas."
                ),
            }
        )
    return reproducibility


def _rollup_trace_command_counts(command_traces, command_keys):
    return dict(
        sorted(
            {
                command: sum(trace.get("command_counts", {}).get(command, 0) for trace in command_traces)
                for command in command_keys
            }.items()
        )
    )


def _resolve_cycle_count(spec, params, cycle_spec):
    if isinstance(cycle_spec, (int, float)):
        return float(cycle_spec)
    if cycle_spec in spec:
        return float(spec.get(cycle_spec, 0.0))
    return float(params.get(cycle_spec, 0.0))


def compute_command_energy(std_name, spec):
    params = ANALYTICAL_PARAMS.get(std_name.lower(), DEFAULT_ANALYTICAL_PARAMS)
    tck_ns = spec["tCK_ns"]
    command_energy = {}
    for command, (current_key, cycle_key) in ANALYTICAL_EVENT_ENERGY.items():
        current_ma = params[current_key]
        cycles = _resolve_cycle_count(spec, params, cycle_key)
        command_energy[command] = current_ma * params["VDD"] * cycles * tck_ns * 1000.0
    return command_energy, params


def build_pim_energy_report(stats, std_name):
    spec = resolve_spec(std_name)
    time_unit_ns = spec["time_unit_ns"]
    ctrl = stats["memory_system"]["controller"]
    observability = stats.get("evidence", {}).get("pim_energy_observability", {})
    modeled = copy.deepcopy(observability.get("modeled", {}))
    command_counts = modeled.get("command_counts", {})
    command_traces = modeled.get("command_traces", [])
    command_keys = sorted(
        set(command_counts)
        | {
            command_name
            for trace in command_traces
            for command_name in trace.get("command_counts", {})
        }
    )
    trace_command_rollup = _rollup_trace_command_counts(command_traces, command_keys)
    total_commands = sum(command_counts.values())
    cycles = ctrl["cycles"]
    total_time_ns = cycles * time_unit_ns
    num_pim_reqs_served = ctrl.get("num_pim_reqs_served", 0)

    # Built-in drampower stats represent standard memory energy terms.
    memory_background_energy = ctrl.get("total_background_energy")
    memory_cmd_energy = ctrl.get("total_cmd_energy")
    memory_total_energy = ctrl.get("total_energy")
    has_builtin_memory_energy = (
        memory_background_energy is not None
        and memory_cmd_energy is not None
        and memory_total_energy is not None
    )

    event_energy_table, params = compute_command_energy(std_name, spec)
    event_breakdown = {
        command: command_counts.get(command, 0) * event_energy_table.get(command, 0.0)
        for command in sorted(command_keys)
    }
    pim_incremental_cmd_energy = sum(event_breakdown.values())

    builtin_incremental_cmd_energy = ctrl.get("total_incremental_cmd_energy")
    has_builtin_incremental_energy = builtin_incremental_cmd_energy is not None
    incremental_cmd_energy = (
        float(builtin_incremental_cmd_energy)
        if has_builtin_incremental_energy
        else pim_incremental_cmd_energy
    )

    pim_incremental_energy = incremental_cmd_energy

    memory_background_energy_value = float(memory_background_energy) if has_builtin_memory_energy else 0.0
    memory_cmd_energy_value = float(memory_cmd_energy) if has_builtin_memory_energy else 0.0
    memory_total_energy_value = (
        float(memory_total_energy)
        if has_builtin_memory_energy
        else (memory_background_energy_value + memory_cmd_energy_value)
    )

    analytical_event_energy = memory_cmd_energy_value + incremental_cmd_energy
    background_energy = memory_background_energy_value
    analytical_total = memory_total_energy_value + pim_incremental_energy

    weighted_score = 0.0
    for command, weight in PROXY_COMMAND_WEIGHTS.items():
        weighted_score += command_counts.get(command, 0) * weight

    derived = {
        "total_command_count": total_commands,
        "pim_mac_share": _safe_div(command_counts.get("PIM_MAC", 0), total_commands),
        "act_share": _safe_div(_sum_counts(command_counts, ["ACT1", "ACT2"]), total_commands),
        "maintenance_share": _safe_div(
            _sum_counts(command_counts, ["PREpb", "PREab", "REFab"]), total_commands
        ),
        "capacity_stall_rate": _safe_div(ctrl.get("pim_capacity_stalls", 0), num_pim_reqs_served),
        "dependency_stall_rate": _safe_div(ctrl.get("pim_dependency_stalls", 0), num_pim_reqs_served),
        "pim_inflight_peak": ctrl.get("pim_inflight_peak", 0),
        "avg_pim_latency_ns": ctrl.get("avg_pim_latency", 0) * time_unit_ns,
        "pim_requests_per_ns": _safe_div(num_pim_reqs_served, total_time_ns),
        "total_time_ns": total_time_ns,
        "analytical_event_energy_pJ": analytical_event_energy,
        "background_energy_pJ": background_energy,
        "memory_background_energy_pJ": memory_background_energy_value,
        "memory_cmd_energy_pJ": memory_cmd_energy_value,
        "memory_total_energy_pJ": memory_total_energy_value,
        "memory_energy_source": "builtin_stats" if has_builtin_memory_energy else "missing_builtin_stats",
        "pim_incremental_cmd_energy_pJ": incremental_cmd_energy,
        "pim_incremental_cmd_energy_source": (
            "builtin_incremental_stats" if has_builtin_incremental_energy else "analytical_command_fallback"
        ),
        "pim_incremental_energy_pJ": pim_incremental_energy,
        "analytical_energy_estimate_pJ": analytical_total,
        "analytical_energy_per_pim_request_pJ": _safe_div(analytical_total, num_pim_reqs_served),
        "memory_background_power_mW": _safe_div(memory_background_energy_value, total_time_ns),
        "memory_cmd_power_mW": _safe_div(memory_cmd_energy_value, total_time_ns),
        "pim_incremental_power_mW": _safe_div(pim_incremental_energy, total_time_ns),
        "analytical_avg_power_mW": _safe_div(analytical_total, total_time_ns),
        "event_energy_share": _safe_div(analytical_event_energy, analytical_total),
        "background_energy_share": _safe_div(background_energy, analytical_total),
        "pim_incremental_energy_share": _safe_div(pim_incremental_energy, analytical_total),
        "command_energy_breakdown_pJ": event_breakdown,
        "legacy_command_weighted_energy_proxy": weighted_score,
        "legacy_energy_proxy_per_pim_request": _safe_div(weighted_score, num_pim_reqs_served),
        "command_trace_channels": len(command_traces),
        "command_trace_total_records": sum(item.get("command_count", 0) for item in command_traces),
        "trace_count_matches_counter_total": total_commands
        == sum(item.get("command_count", 0) for item in command_traces),
        "trace_command_rollup": trace_command_rollup,
        "trace_command_counts_match_counter": dict(
            sorted({command: command_counts.get(command, 0) for command in command_keys}.items())
        )
        == trace_command_rollup,
        "command_counter_reproducibility": {
            "counter_csv_total_commands": total_commands,
            "trace_rollup_total_commands": sum(item.get("command_count", 0) for item in command_traces),
            "trace_rollup_command_counts": trace_command_rollup,
            "per_trace_rollup": _build_trace_reproducibility(command_traces),
        },
        "formulas": {
            "analytical_event_energy_pJ": "memory_cmd_energy_pJ + pim_incremental_cmd_energy_pJ",
            "background_energy_pJ": "memory_background_energy_pJ",
            "memory_total_energy_pJ": "total_energy (built-in drampower stat)",
            "pim_incremental_cmd_energy_pJ": (
                "total_incremental_cmd_energy (built-in) or sum(command_count[pim_cmd] * analytical_event_energy_per_command_pJ[pim_cmd])"
            ),
            "pim_incremental_energy_pJ": "pim_incremental_cmd_energy_pJ",
            "analytical_energy_estimate_pJ": (
                "memory_total_energy_pJ + pim_incremental_energy_pJ"
            ),
            "analytical_energy_per_pim_request_pJ": (
                "analytical_energy_estimate_pJ / num_pim_reqs_served"
            ),
            "memory_background_power_mW": "memory_background_energy_pJ / (cycles * time_unit_ns)",
            "memory_cmd_power_mW": "memory_cmd_energy_pJ / (cycles * time_unit_ns)",
            "pim_incremental_power_mW": "pim_incremental_energy_pJ / (cycles * time_unit_ns)",
            "analytical_avg_power_mW": "analytical_energy_estimate_pJ / (cycles * time_unit_ns)",
            "pim_requests_per_ns": "num_pim_reqs_served / (cycles * time_unit_ns)",
            "avg_pim_latency_ns": "avg_pim_latency * time_unit_ns",
            "trace_count_matches_counter_total": "sum(trace.command_count) == total_command_count",
            "trace_command_counts_match_counter": (
                "sum(trace.command_counts[cmd] for trace in traces) == command_count[cmd] for every cmd"
            ),
        },
    }

    assumed = {
        "analytical_event_energy_per_command_pJ": event_energy_table,
        "memory_energy_terms_source": (
            "Built-in drampower stats (total_background_energy, total_cmd_energy, total_energy)"
        ),
        "parameter_units": "parameterized analytical units (pJ / mW), not silicon-calibrated truth",
        "normalization_note": "Outputs are comparative analytical estimates derived from observed commands and runtime.",
        "interpretation_guardrail": (
            "These estimates are trace-driven and parameterized; they must not be read as silicon-calibrated LPDDR5-PIM joules or validated device power."
        ),
        "legacy_proxy_weights": copy.deepcopy(PROXY_COMMAND_WEIGHTS),
        "legacy_proxy_note": "Legacy proxy values are preserved only for backward comparison during migration.",
    }

    return {
        "modeled": modeled,
        "derived": derived,
        "assumed": assumed,
    }
