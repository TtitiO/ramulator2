"""Trace-driven analytical energy estimation for analysis tests.

This module replaces opaque command-weight proxies with named analytical
parameters driven by observed command counts and runtime duration. The
resulting outputs remain parameterized comparative estimates, not
silicon-calibrated joules.
"""

import copy

from tests.analysis.utils.spec import resolve_spec


def _safe_div(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _sum_counts(command_counts, commands):
    return sum(command_counts.get(command, 0) for command in commands)


def _find_numeric_stat(container, names):
    if not isinstance(container, dict):
        return None, None
    for name in names:
        value = container.get(name)
        if isinstance(value, (int, float)):
            return value, name
    for value in container.values():
        found_value, found_name = _find_numeric_stat(value, names)
        if found_name is not None:
            return found_value, found_name
    return None, None


def _classify_pim_datatype(ctrl, modeled):
    datatype = (
        ctrl.get("pim_datatype")
        or modeled.get("pim_datatype")
        or ctrl.get("pim_datatype_class")
        or modeled.get("pim_datatype_class")
        or "unknown"
    )
    datatype_class = (
        ctrl.get("pim_datatype_class")
        or modeled.get("pim_datatype_class")
        or datatype
    )
    normalized_class = str(datatype_class).lower()
    floating_datatypes = {"fp16", "bf16", "fp32", "fp64"}
    if normalized_class in floating_datatypes:
        return str(datatype).lower(), normalized_class, "floating", "GFLOP/s"
    if normalized_class.startswith("int") or normalized_class.startswith("uint"):
        return str(datatype).lower(), normalized_class, "integer", "GOPS"
    return str(datatype).lower(), normalized_class, "unknown", "GOPS"


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


def _nonnegative(value):
    return max(0.0, float(value))


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
    total_completion_time_ns = total_time_ns
    num_pim_reqs_served = ctrl.get("num_pim_reqs_served", 0)
    frontend_pim_request_names = (
        "num_pim_reqs",
        "num_pim_requests",
        "num_pim_reqs_issued",
        "num_pim_requests_issued",
        "num_frontend_pim_reqs",
        "num_frontend_pim_requests",
        "total_num_pim_requests",
        "total_num_pim_reqs",
    )
    frontend_pim_requests, frontend_pim_request_source = _find_numeric_stat(
        stats.get("frontend", {}), frontend_pim_request_names
    )
    if frontend_pim_request_source is None:
        frontend_pim_requests = num_pim_reqs_served
        frontend_pim_request_source = "fallback:num_pim_reqs_served"
    pim_elements_per_request = ctrl.get("pim_lanes", 0)
    pim_mac_ops_per_element = float(ctrl.get("pim_ops_per_mac", 2.0))
    pim_ops_per_request = float(ctrl.get("pim_ops_per_request", 1.0))
    simulated_pim_gops = _safe_div(num_pim_reqs_served * pim_ops_per_request, total_time_ns)
    pim_datatype, pim_datatype_class, pim_datatype_kind, simulated_pim_display_unit = _classify_pim_datatype(
        ctrl, modeled
    )
    simulated_pim_display_label = f"simulated PIM arithmetic throughput ({simulated_pim_display_unit})"

    # Built-in drampower stats represent standard memory energy terms.
    memory_background_energy = ctrl.get("total_background_energy")
    memory_cmd_energy = ctrl.get("total_cmd_energy")
    memory_total_energy = ctrl.get("total_energy")
    has_builtin_memory_energy = (
        memory_background_energy is not None
        and memory_cmd_energy is not None
        and memory_total_energy is not None
    )

    event_energy_table = {}
    event_breakdown = {}

    count_act = _sum_counts(command_counts, ["ACT1", "ACT2"])
    count_pre = _sum_counts(command_counts, ["PREpb", "PREab"])
    count_rd = _sum_counts(command_counts, ["CAS_RD", "RD", "RDA"])
    count_wr = _sum_counts(command_counts, ["CAS_WR", "WR", "WRA"])
    count_ref = command_counts.get("REFab", 0)
    count_pim_mac = command_counts.get("PIM_MAC", 0)
    count_pim_mac_ab = command_counts.get("PIM_MAC_AB", 0)
    count_pim_setup_commands = _sum_counts(command_counts, ["SB", "HAB", "HAB_PIM", "PIM_BCAST"])

    builtin_incremental_cmd_energy = ctrl.get("total_incremental_cmd_energy")
    has_builtin_incremental_energy = builtin_incremental_cmd_energy is not None
    builtin_incremental_cmd_energy_value = (
        float(builtin_incremental_cmd_energy)
        if has_builtin_incremental_energy
        else 0.0
    )

    memory_background_energy_value = float(memory_background_energy) if has_builtin_memory_energy else 0.0
    memory_cmd_energy_value = float(memory_cmd_energy) if has_builtin_memory_energy else 0.0
    memory_total_energy_value = (
        float(memory_total_energy)
        if has_builtin_memory_energy
        else (memory_background_energy_value + memory_cmd_energy_value)
    )

    background_energy = memory_background_energy_value

    energy_background_pj = _nonnegative(memory_background_energy_value)
    energy_lpddr_base_cmd_pj = _nonnegative(memory_cmd_energy_value)
    pim_compute_energy_per_mac = float(ctrl.get("pim_compute_energy_pJ_per_mac", 0.0))
    count_pim_mac_commands = count_pim_mac + count_pim_mac_ab
    movement_energy_per_request = float(ctrl.get("pim_cell_to_pim_energy_pJ_per_256b", 0.0)) + float(
        ctrl.get("pim_interconnect_energy_pJ_per_256b", 0.0)
    )
    writeback_energy_per_request = float(ctrl.get("pim_vrf_access_energy_pJ", 0.0)) + float(
        ctrl.get("pim_srf_access_energy_pJ", 0.0)
    )
    mode_switch_energy_per_command = float(ctrl.get("pim_mode_switch_energy_pJ", 0.0))
    has_user_pim_coefficients = any(
        value > 0.0
        for value in (
            pim_compute_energy_per_mac,
            movement_energy_per_request,
            writeback_energy_per_request,
            mode_switch_energy_per_command,
        )
    )

    # Two mutually exclusive attribution modes are supported.
    #
    # builtin-total mode: built-in simulator totals are authoritative. Placeholder
    # analytical PIM command coefficients are excluded from displayed components
    # instead of being added and canceled by a negative residual.
    #
    # analytical-component mode: explicit user/source coefficients are present, or
    # no built-in total exists. If only placeholder analytical coefficients are
    # available, the row is sensitivity-only and not paper-facing.
    has_builtin_total_authority = has_builtin_memory_energy and has_builtin_incremental_energy
    use_analytical_component_mode = has_user_pim_coefficients or not has_builtin_total_authority
    uses_placeholder_analytical_components = use_analytical_component_mode and not has_user_pim_coefficients

    if use_analytical_component_mode:
        pim_lanes = float(ctrl.get("pim_lanes", 0.0))
        if uses_placeholder_analytical_components:
            energy_pim_setup_cmd_pj = _nonnegative(
                sum(event_breakdown.get(command, 0.0) for command in ["SB", "HAB", "HAB_PIM", "PIM_BCAST"])
            )
            energy_pim_mac_cmd_pj = _nonnegative(
                sum(event_breakdown.get(command, 0.0) for command in ["PIM_MAC", "PIM_MAC_AB"])
            )
            energy_pim_movement_pj = 0.0
            energy_pim_writeback_pj = 0.0
            energy_mode_switch_pj = 0.0
            attribution_mode = "analytical_component_placeholder"
        else:
            energy_pim_setup_cmd_pj = 0.0
            energy_pim_mac_cmd_pj = _nonnegative(count_pim_mac_commands * pim_lanes * pim_compute_energy_per_mac)
            energy_pim_movement_pj = _nonnegative(count_pim_mac_commands * movement_energy_per_request)
            energy_pim_writeback_pj = _nonnegative(count_pim_mac_commands * writeback_energy_per_request)
            energy_mode_switch_pj = _nonnegative(count_pim_setup_commands * mode_switch_energy_per_command)
            attribution_mode = "analytical_component"
        energy_incremental_pim_builtin_pj = 0.0
        energy_total_pj = (
            energy_background_pj
            + energy_lpddr_base_cmd_pj
            + energy_pim_setup_cmd_pj
            + energy_pim_mac_cmd_pj
            + energy_pim_movement_pj
            + energy_pim_writeback_pj
            + energy_mode_switch_pj
        )
        energy_fallback_or_unattributed_pj = 0.0
    else:
        energy_pim_setup_cmd_pj = 0.0
        energy_pim_mac_cmd_pj = 0.0
        energy_pim_movement_pj = 0.0
        energy_pim_writeback_pj = 0.0
        energy_mode_switch_pj = 0.0
        energy_incremental_pim_builtin_pj = _nonnegative(builtin_incremental_cmd_energy_value)
        energy_total_pj = _nonnegative(memory_total_energy_value) + energy_incremental_pim_builtin_pj
        energy_fallback_or_unattributed_pj = _nonnegative(
            energy_total_pj
            - (energy_background_pj + energy_lpddr_base_cmd_pj + energy_incremental_pim_builtin_pj)
        )
        attribution_mode = "builtin_total"

    incremental_cmd_energy = (
        energy_total_pj - energy_background_pj - energy_lpddr_base_cmd_pj
        if use_analytical_component_mode
        else energy_incremental_pim_builtin_pj
    )
    pim_incremental_energy = incremental_cmd_energy
    analytical_event_energy = memory_cmd_energy_value + incremental_cmd_energy
    analytical_total = energy_total_pj
    attributed_without_fallback_pj = (
        energy_background_pj
        + energy_lpddr_base_cmd_pj
        + energy_incremental_pim_builtin_pj
        + energy_pim_setup_cmd_pj
        + energy_pim_mac_cmd_pj
        + energy_pim_movement_pj
        + energy_pim_writeback_pj
        + energy_mode_switch_pj
    )
    energy_attribution_sum_pj = attributed_without_fallback_pj + energy_fallback_or_unattributed_pj
    request_throughput = _safe_div(num_pim_reqs_served, total_time_ns)
    avg_pim_response_latency_ns = ctrl.get("avg_pim_response_latency", 0) * time_unit_ns
    average_power_mw = _safe_div(energy_total_pj, total_time_ns)
    coefficient_sources = {
        "lpddr_base_memory": "LPDDR5-derived/builtin_stats" if has_builtin_memory_energy else "LPDDR5-derived/missing_builtin_stats",
        "pim_setup_commands": (
            "placeholder_analytical_command_coefficients"
            if uses_placeholder_analytical_components
            else "excluded_placeholder_analytical_command_coefficients"
        ),
        "pim_mac_commands": (
            "user_provided_coefficients"
            if pim_compute_energy_per_mac
            else (
                "placeholder_analytical_command_coefficients"
                if uses_placeholder_analytical_components
                else "excluded_placeholder_analytical_command_coefficients"
            )
        ),
        "pim_movement": "user_provided_coefficients" if movement_energy_per_request else "placeholder_zero_coefficients",
        "pim_writeback": "user_provided_coefficients" if writeback_energy_per_request else "placeholder_zero_coefficients",
        "mode_switch": "user_provided_coefficients" if mode_switch_energy_per_command else "placeholder_zero_coefficients",
        "incremental_pim_builtin": "builtin_incremental_stats" if has_builtin_incremental_energy else "missing_builtin_incremental_stats",
        "fallback_or_unattributed": "non_negative_residual",
        "total_energy": attribution_mode,
    }
    paper_facing_energy_total = not uses_placeholder_analytical_components
    coefficient_source_summary = "; ".join(
        f"{name}={source}" for name, source in sorted(coefficient_sources.items())
    )

    weighted_score = 0.0

    derived = {
        "total_command_count": total_commands,
        "count_ACT": count_act,
        "count_PRE": count_pre,
        "count_RD": count_rd,
        "count_WR": count_wr,
        "count_REF": count_ref,
        "count_PIM_MAC": count_pim_mac,
        "count_PIM_MAC_AB": count_pim_mac_ab,
        "count_PIM_setup_commands": count_pim_setup_commands,
        "count_completed_pim_requests": num_pim_reqs_served,
        "count_frontend_pim_requests": frontend_pim_requests,
        "count_frontend_pim_requests_source": frontend_pim_request_source,
        "pim_mac_share": _safe_div(command_counts.get("PIM_MAC", 0), total_commands),
        "act_share": _safe_div(_sum_counts(command_counts, ["ACT1", "ACT2"]), total_commands),
        "maintenance_share": _safe_div(
            _sum_counts(command_counts, ["PREpb", "PREab", "REFab"]), total_commands
        ),
        "capacity_stall_rate": _safe_div(ctrl.get("pim_capacity_stalls", 0), num_pim_reqs_served),
        "mpu_group_stall_rate": _safe_div(ctrl.get("pim_mpu_group_stalls", 0), num_pim_reqs_served),
        "dependency_stall_rate": _safe_div(ctrl.get("pim_dependency_stalls", 0), num_pim_reqs_served),
        "global_issue_stall_per_request": _safe_div(ctrl.get("num_global_issue_blocked_cycles", 0), num_pim_reqs_served),
        "mpu_group_busy_stall_per_request": _safe_div(ctrl.get("num_mpu_group_busy_blocked_cycles", 0), num_pim_reqs_served),
        "bank_timing_stall_per_request": _safe_div(ctrl.get("num_bank_timing_blocked_cycles", 0), num_pim_reqs_served),
        "rr_head_of_line_stall_per_request": _safe_div(ctrl.get("num_rr_head_of_line_blocked_cycles", 0), num_pim_reqs_served),
        "queue_empty_cycles_per_request": _safe_div(ctrl.get("num_queue_empty_cycles", 0), num_pim_reqs_served),
        "pim_inflight_peak": ctrl.get("pim_inflight_peak", 0),
        "pim_banks_per_mpu": ctrl.get("pim_banks_per_mpu", 0),
        "pim_mpu_group_count": ctrl.get("pim_mpu_group_count", 0),
        "total_banks": ctrl.get("total_banks", 0),
        "effective_mpu_groups": ctrl.get("effective_mpu_groups", ctrl.get("pim_mpu_group_count", 0)),
        "pim_capacity_stalls": ctrl.get("pim_capacity_stalls", 0),
        "pim_mpu_group_stalls": ctrl.get("pim_mpu_group_stalls", 0),
        "num_global_issue_blocked_cycles": ctrl.get("num_global_issue_blocked_cycles", 0),
        "num_mpu_group_busy_blocked_cycles": ctrl.get("num_mpu_group_busy_blocked_cycles", 0),
        "num_bank_timing_blocked_cycles": ctrl.get("num_bank_timing_blocked_cycles", 0),
        "num_rr_head_of_line_blocked_cycles": ctrl.get("num_rr_head_of_line_blocked_cycles", 0),
        "num_queue_empty_cycles": ctrl.get("num_queue_empty_cycles", 0),
        "num_issued_pim_mac": ctrl.get("num_issued_pim_mac", 0),
        "pim_simultaneous_active_banks_peak": ctrl.get("pim_simultaneous_active_banks_peak", 0),
        "avg_pim_latency_ns": ctrl.get("avg_pim_latency", 0) * time_unit_ns,
        "avg_pim_service_latency_ns": ctrl.get("avg_pim_service_latency", 0) * time_unit_ns,
        "avg_pim_launch_wait_ns": ctrl.get("avg_pim_launch_wait", 0) * time_unit_ns,
        "avg_pim_response_latency_ns": avg_pim_response_latency_ns,
        "pim_service_latency": ctrl.get("pim_service_latency", 0),
        "pim_launch_wait": ctrl.get("pim_launch_wait", 0),
        "pim_response_latency": ctrl.get("pim_response_latency", ctrl.get("pim_latency", 0)),
        "pim_requests_per_ns": _safe_div(num_pim_reqs_served, total_time_ns),
        "pim_datatype_behavior_enabled": bool(ctrl.get("pim_datatype_behavior_enabled", 0)),
        "pim_datatype": pim_datatype,
        "pim_datatype_class": pim_datatype_class,
        "pim_datatype_kind": pim_datatype_kind,
        "pim_datatype_bits": ctrl.get("pim_datatype_bits", 0),
        "pim_simd_width_bits": ctrl.get("pim_simd_width_bits", 0),
        "pim_lanes": ctrl.get("pim_lanes", 0),
        "pim_elements_per_request": pim_elements_per_request,
        "pim_mac_ops_per_element": pim_mac_ops_per_element,
        "pim_ops_per_mac": ctrl.get("pim_ops_per_mac", 0.0),
        "pim_ops_per_block_issue": ctrl.get("pim_ops_per_block_issue", 0.0),
        "pim_ops_per_request": pim_ops_per_request,
        "simulated_pim_gops": simulated_pim_gops,
        "simulated_pim_gflops": simulated_pim_gops if pim_datatype_kind == "floating" else None,
        "simulated_pim_display_value": simulated_pim_gops,
        "simulated_pim_display_unit": simulated_pim_display_unit,
        "simulated_pim_display_label": simulated_pim_display_label,
        "pim_mac_issue_interval_cycles": ctrl.get("pim_mac_issue_interval_cycles", 0),
        "pim_mac_pipeline_latency_cycles": ctrl.get("pim_mac_pipeline_latency_cycles", 0),
        "pim_movement_cycles": ctrl.get("pim_movement_cycles", 0),
        "pim_writeback_cycles": ctrl.get("pim_writeback_cycles", 0),
        "pim_completion_latency_cycles": ctrl.get("pim_completion_latency_cycles", 0),
        "pim_slots_per_request": ctrl.get("pim_slots_per_request", ctrl.get("pim_slot_cost", 1)),
        "pim_slot_cost": ctrl.get("pim_slot_cost", ctrl.get("pim_slots_per_request", 1)),
        "pim_compute_energy_pJ_per_mac": ctrl.get("pim_compute_energy_pJ_per_mac", 0.0),
        "pim_cell_to_pim_energy_pJ_per_256b": ctrl.get("pim_cell_to_pim_energy_pJ_per_256b", 0.0),
        "pim_interconnect_energy_pJ_per_256b": ctrl.get("pim_interconnect_energy_pJ_per_256b", 0.0),
        "pim_vrf_access_energy_pJ": ctrl.get("pim_vrf_access_energy_pJ", 0.0),
        "pim_srf_access_energy_pJ": ctrl.get("pim_srf_access_energy_pJ", 0.0),
        "pim_mode_switch_energy_pJ": ctrl.get("pim_mode_switch_energy_pJ", 0.0),
        "total_time_ns": total_time_ns,
        "total_simulation_time": total_time_ns,
        "total_simulation_time_ns": total_time_ns,
        "total_completion_time": total_completion_time_ns,
        "total_completion_time_ns": total_completion_time_ns,
        "analytical_event_energy_pJ": analytical_event_energy,
        "background_energy_pJ": background_energy,
        "memory_background_energy_pJ": memory_background_energy_value,
        "memory_cmd_energy_pJ": memory_cmd_energy_value,
        "memory_total_energy_pJ": memory_total_energy_value,
        "memory_energy_source": "builtin_stats" if has_builtin_memory_energy else "missing_builtin_stats",
        "pim_incremental_cmd_energy_pJ": incremental_cmd_energy,
        "pim_incremental_cmd_energy_source": (
            "analytical_component_sum" if use_analytical_component_mode else (
                "builtin_incremental_stats" if has_builtin_incremental_energy else "missing_builtin_incremental_stats"
            )
        ),
        "pim_incremental_energy_pJ": pim_incremental_energy,
        "analytical_energy_estimate_pJ": analytical_total,
        "energy_attribution_mode": attribution_mode,
        "paper_facing_energy_total": paper_facing_energy_total,
        "sensitivity_only_energy_total": uses_placeholder_analytical_components,
        "placeholder_analytical_fields_excluded_from_paper_total": not uses_placeholder_analytical_components,
        "energy_total_pJ": energy_total_pj,
        "energy_background_pJ": energy_background_pj,
        "energy_lpddr_base_cmd_pJ": energy_lpddr_base_cmd_pj,
        "energy_incremental_pim_builtin_pJ": energy_incremental_pim_builtin_pj,
        "energy_pim_setup_cmd_pJ": energy_pim_setup_cmd_pj,
        "energy_pim_mac_cmd_pJ": energy_pim_mac_cmd_pj,
        "energy_pim_movement_pJ": energy_pim_movement_pj,
        "energy_pim_writeback_pJ": energy_pim_writeback_pj,
        "energy_mode_switch_pJ": energy_mode_switch_pj,
        "energy_fallback_or_unattributed_pJ": energy_fallback_or_unattributed_pj,
        "energy_attribution_sum_pJ": energy_attribution_sum_pj,
        "total_energy": energy_total_pj,
        "total_energy_pJ": energy_total_pj,
        "full_system_edp": energy_total_pj * total_completion_time_ns,
        "full_system_edp_pJ_ns": energy_total_pj * total_completion_time_ns,
        "runtime_full_system_edp_pJ_ns": energy_total_pj * total_completion_time_ns,
        "legacy_full_system_edp_pJ_ns": energy_total_pj * total_completion_time_ns,
        "latency_full_system_edp_pJ_ns": energy_total_pj * avg_pim_response_latency_ns,
        "throughput_edp_pJ_ns": _safe_div(energy_total_pj, request_throughput) if request_throughput > 0 else 0.0,
        "analytical_energy_per_pim_request_pJ": _safe_div(energy_total_pj, num_pim_reqs_served),
        "memory_background_power_mW": _safe_div(memory_background_energy_value, total_time_ns),
        "memory_cmd_power_mW": _safe_div(memory_cmd_energy_value, total_time_ns),
        "pim_incremental_power_mW": _safe_div(pim_incremental_energy, total_time_ns),
        "analytical_avg_power_mW": average_power_mw,
        "average_power": average_power_mw,
        "average_power_mW": average_power_mw,
        "request_throughput": request_throughput,
        "average_pim_latency": ctrl.get("avg_pim_latency", 0) * time_unit_ns,
        "average_pim_service_latency": ctrl.get("avg_pim_service_latency", 0) * time_unit_ns,
        "average_pim_launch_wait": ctrl.get("avg_pim_launch_wait", 0) * time_unit_ns,
        "average_pim_response_latency": avg_pim_response_latency_ns,
        "event_energy_share": _safe_div(analytical_event_energy, energy_total_pj),
        "background_energy_share": _safe_div(background_energy, energy_total_pj),
        "pim_incremental_energy_share": _safe_div(pim_incremental_energy, energy_total_pj),
        "command_energy_breakdown_pJ": {},
        "coefficient_source_summary": coefficient_source_summary,
        "coefficient_sources": coefficient_sources,
        "coefficient_is_silicon_calibrated": False,
        "coefficient_guardrail": (
            "LPDDR5 base-memory energy comes from built-in simulator stats when present; PIM event terms are "
            "placeholder analytical coefficients unless explicitly supplied by user stats, and none of these "
            "coefficients should be reported as silicon-calibrated device energy."
        ),
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
            "count_ACT": "ACT1 + ACT2",
            "count_PRE": "PREpb + PREab",
            "count_RD": "CAS_RD + RD + RDA",
            "count_WR": "CAS_WR + WR + WRA",
            "count_REF": "REFab",
            "count_PIM_MAC": "PIM_MAC command count only; num_issued_pim_mac is not added",
            "count_PIM_MAC_AB": "PIM_MAC_AB",
            "count_PIM_setup_commands": "SB + HAB + HAB_PIM + PIM_BCAST",
            "count_completed_pim_requests": "controller num_pim_reqs_served",
            "count_frontend_pim_requests": "first numeric frontend PIM request stat found, else count_completed_pim_requests",
            "analytical_event_energy_pJ": "memory_cmd_energy_pJ + pim_incremental_cmd_energy_pJ",
            "background_energy_pJ": "memory_background_energy_pJ",
            "memory_total_energy_pJ": "total_energy (built-in drampower stat)",
            "pim_incremental_cmd_energy_pJ": (
                "total_incremental_cmd_energy (built-in) in builtin-total mode, or explicit component sum in analytical-component mode"
            ),
            "pim_incremental_energy_pJ": "pim_incremental_cmd_energy_pJ",
            "analytical_energy_estimate_pJ": (
                "memory_total_energy_pJ + pim_incremental_energy_pJ"
            ),
            "energy_total_pJ": "memory_total_energy_pJ + pim_incremental_energy_pJ",
            "energy_background_pJ": "memory_background_energy_pJ",
            "energy_lpddr_base_cmd_pJ": "memory_cmd_energy_pJ",
            "energy_incremental_pim_builtin_pJ": "total_incremental_cmd_energy in builtin-total mode, else 0",
            "energy_pim_setup_cmd_pJ": "0 unless source-backed setup coefficients are modeled separately",
            "energy_pim_mac_cmd_pJ": (
                "(count_PIM_MAC + count_PIM_MAC_AB) * pim_lanes * pim_compute_energy_pJ_per_mac in analytical-component mode; "
                "0 in builtin-total mode when only placeholder analytical command coefficients are available"
            ),
            "energy_pim_movement_pJ": (
                "(count_PIM_MAC + count_PIM_MAC_AB) * "
                "(pim_cell_to_pim_energy_pJ_per_256b + pim_interconnect_energy_pJ_per_256b)"
            ),
            "energy_pim_writeback_pJ": (
                "(count_PIM_MAC + count_PIM_MAC_AB) * (pim_vrf_access_energy_pJ + pim_srf_access_energy_pJ)"
            ),
            "energy_mode_switch_pJ": "count_PIM_setup_commands * pim_mode_switch_energy_pJ",
            "energy_fallback_or_unattributed_pJ": (
                "max(0, energy_total_pJ - sum(explicit attribution buckets)); never negative"
            ),
            "energy_attribution_sum_pJ": (
                "energy_background_pJ + energy_lpddr_base_cmd_pJ + energy_pim_setup_cmd_pJ + "
                "energy_incremental_pim_builtin_pJ + energy_pim_mac_cmd_pJ + energy_pim_movement_pJ + "
                "energy_pim_writeback_pJ + energy_mode_switch_pJ + energy_fallback_or_unattributed_pJ"
            ),
            "analytical_energy_per_pim_request_pJ": (
                "energy_total_pJ / num_pim_reqs_served"
            ),
            "memory_background_power_mW": "memory_background_energy_pJ / (cycles * time_unit_ns)",
            "memory_cmd_power_mW": "memory_cmd_energy_pJ / (cycles * time_unit_ns)",
            "pim_incremental_power_mW": "pim_incremental_energy_pJ / (cycles * time_unit_ns)",
            "analytical_avg_power_mW": "energy_total_pJ / (cycles * time_unit_ns)",
            "average_power_mW": "energy_total_pJ / total_time_ns",
            "average_power": "same as average_power_mW",
            "full_system_edp_pJ_ns": "total_energy_pJ * total_completion_time_ns",
            "runtime_full_system_edp_pJ_ns": "energy_total_pJ * total_completion_time_ns",
            "legacy_full_system_edp_pJ_ns": "energy_total_pJ * total_completion_time_ns",
            "latency_full_system_edp_pJ_ns": "energy_total_pJ * avg_pim_response_latency_ns",
            "throughput_edp_pJ_ns": "energy_total_pJ / request_throughput when request_throughput > 0, else 0",
            "pim_requests_per_ns": "num_pim_reqs_served / (cycles * time_unit_ns)",
            "pim_elements_per_request": "pim_lanes (lanes/request for one 256-bit PIM_MAC command)",
            "pim_mac_ops_per_element": "2 arithmetic ops per element for MAC",
            "pim_ops_per_request": "pim_elements_per_request * pim_mac_ops_per_element",
            "simulated_pim_gops": "neutral arithmetic op-rate: num_pim_reqs_served * pim_ops_per_request / (cycles * time_unit_ns)",
            "simulated_pim_display_value": "same numeric arithmetic op-rate as simulated_pim_gops, displayed with datatype-aware unit",
            "avg_pim_latency_ns": "avg_pim_latency * time_unit_ns",
            "avg_pim_service_latency_ns": "avg_pim_service_latency * time_unit_ns",
            "avg_pim_launch_wait_ns": "avg_pim_launch_wait * time_unit_ns",
            "avg_pim_response_latency_ns": "avg_pim_response_latency * time_unit_ns",
            "trace_count_matches_counter_total": "sum(trace.command_count) == total_command_count",
            "trace_command_counts_match_counter": (
                "sum(trace.command_counts[cmd] for trace in traces) == command_count[cmd] for every cmd"
            ),
        },
    }

    assumed = {
        "analytical_event_energy_per_command_pJ": {},
        "coefficient_sources": coefficient_sources,
        "coefficient_is_silicon_calibrated": False,
        "coefficient_metadata": {
            "lpddr_base_memory": {
                "source": coefficient_sources["lpddr_base_memory"],
                "fields": ["total_background_energy", "total_cmd_energy", "total_energy"],
            },
            "pim_event_terms": {
                "source": "placeholder_analytical_command_coefficients unless overridden by nonzero user coefficients",
                "fields": ["IDD_MAC_mA", "IDD_CTRL_mA", "nMAC", "nCTRL", "VDD", "tCK_ns"],
            },
            "pim_optional_user_coefficients": {
                "source": "user_provided_coefficients when nonzero, otherwise placeholder_zero_coefficients",
                "fields": [
                    "pim_cell_to_pim_energy_pJ_per_256b",
                    "pim_interconnect_energy_pJ_per_256b",
                    "pim_vrf_access_energy_pJ",
                    "pim_srf_access_energy_pJ",
                    "pim_mode_switch_energy_pJ",
                ],
            },
            "fallback_or_unattributed": {
                "source": coefficient_sources["fallback_or_unattributed"],
                "note": "Residual bucket that makes explicit attribution buckets sum to energy_total_pJ.",
            },
        },
        "memory_energy_terms_source": (
            "Built-in drampower stats (total_background_energy, total_cmd_energy, total_energy)"
        ),
        "parameter_units": "parameterized analytical units (pJ / mW), not silicon-calibrated truth",
        "normalization_note": "Outputs are comparative analytical estimates derived from observed commands and runtime.",
        "interpretation_guardrail": (
            "These estimates are trace-driven and parameterized; they must not be read as silicon-calibrated LPDDR5-PIM joules or validated device power."
        ),
        "legacy_proxy_weights": {},
        "legacy_proxy_note": "Legacy proxy values are preserved only for backward comparison during migration.",
    }

    return {
        "modeled": modeled,
        "derived": derived,
        "assumed": assumed,
    }
