"""Fast latency-throughput: no-refresh checks and plots."""

import json
from pathlib import Path

import pytest

from tests.analysis.testcases import STANDARDS
from tests.analysis.utils.checks import (
    build_pim_energy_evidence,
    check_peak_bandwidth,
    check_pim_all_bank_split_throughput,
    check_pim_latency_throughput,
    check_streaming_peak_bandwidth,
    check_unloaded_latency,
    compare_at_same_nop,
    curves_to_nop_dict,
    write_pim_energy_summary,
)
from tests.analysis.utils.plot import (
    plot_lat_tp,
    plot_lpddr5_pim_avg_power_breakdown_by_case,
    plot_lpddr5_pim_avg_power_breakdown_by_case_plotly,
    plot_lpddr5_pim_bank_count_sensitivity_same_nop,
    plot_lpddr5_pim_edp_vs_nop_representative_cases,
    plot_lpddr5_pim_edp_vs_nop_representative_cases_plotly,
    plot_lpddr5_pim_latency_vs_nop_representative_cases,
    plot_lpddr5_pim_latency_vs_nop_representative_cases_plotly,
    plot_lpddr5_pim_mpu_group_stalls_vs_nop_representative_cases,
    plot_lpddr5_pim_avg_power_vs_nop_representative_cases,
    plot_lpddr5_pim_avg_power_vs_nop_representative_cases_plotly,
    plot_lpddr5_pim_energy_breakdown_by_case,
    plot_lpddr5_pim_energy_breakdown_by_case_plotly,
    plot_lpddr5_pim_energy_performance_by_case,
    plot_lpddr5_pim_energy_performance_by_case_plotly,
    plot_pim_dependency_pattern_same_nop,
    plot_pim_dependency_pattern_same_nop_plotly,
    plot_pim_lat_tp,
)
from tests.analysis.utils.energy import build_pim_energy_report
from tests.analysis.utils.sweep import extract_curves, extract_pim_curves, run_sweep
from tests.analysis.runner import run_streaming_only

# Sweep parameters
CI_READ_RATIOS = [100, 90, 80, 70, 60, 50]
CI_NUM_PROBES = 10000


# Cache sweep results per standard (expensive to compute)
_sweep_cache = {}
_raw_sweep_cache = {}
_streaming_cache = {}
_dependency_pattern_cache = {}
_lpddr5_pim_case_report_cache = {}
_lpddr5_pim_power_vs_nop_cache = {}


LPDDR5_ANALYSIS_POWER = {
    "enabled": True,
    "VDD1": 1.80,
    "VDD2H": 1.05,
    "VDD2L": 0.90,
    "VDDQ": 0.50,
    "IDD01": 2.80,
    "IDD02H": 32.00,
    "IDD02L": 0.25,
    "IDD0Q": 0.75,
    "IDD2N1": 1.20,
    "IDD2N2H": 16.00,
    "IDD2N2L": 0.25,
    "IDD2NQ": 0.75,
    "IDD3N1": 1.20,
    "IDD3N2H": 16.00,
    "IDD3N2L": 0.25,
    "IDD3NQ": 0.75,
    "IDD4R1": 2.00,
    "IDD4R2H": 18.00,
    "IDD4R2L": 0.30,
    "IDD4RQ": 0.85,
    "IDD4W1": 2.10,
    "IDD4W2H": 19.00,
    "IDD4W2L": 0.35,
    "IDD4WQ": 0.90,
    "IDD5AB1": 2.20,
    "IDD5AB2H": 35.00,
    "IDD5AB2L": 0.25,
    "IDD5ABQ": 0.75,
}


def _get_sweep(std_name):
    """Run sweep once per standard, cache for reuse."""
    if std_name not in _sweep_cache:
        nops = STANDARDS[std_name]["nop_counters"]
        raw = run_sweep(std_name, nops, CI_READ_RATIOS, CI_NUM_PROBES, full=False)
        _raw_sweep_cache[std_name] = raw
        if STANDARDS[std_name].get("pim_mode", False):
            curves = extract_pim_curves(raw, std_name)
        else:
            curves = extract_curves(raw, std_name)
        _sweep_cache[std_name] = curves
    return _sweep_cache[std_name]


def _get_raw_sweep(std_name):
    _get_sweep(std_name)
    return _raw_sweep_cache[std_name]


def _get_streaming(std_name):
    """Run streaming-only once per standard, cache for reuse."""
    if STANDARDS[std_name].get("pim_mode", False):
        return None
    if std_name not in _streaming_cache:
        _streaming_cache[std_name] = run_streaming_only(std_name, full=False)
    return _streaming_cache[std_name]


def _get_dependency_pattern_results(std_name):
    cfg = STANDARDS[std_name]
    patterns = cfg.get("dependency_patterns")
    if not patterns:
        return None

    if std_name not in _dependency_pattern_cache:
        nops = cfg.get("dependency_pattern_nop_counters", cfg["nop_counters"])
        results = {}
        for pattern_name, cfg_override in sorted(patterns.items()):
            raw = run_sweep(
                std_name,
                nops,
                CI_READ_RATIOS,
                CI_NUM_PROBES,
                full=False,
                cfg_override=cfg_override,
            )
            curves = extract_pim_curves(raw, std_name)
            nop_dict = curves_to_nop_dict(curves)
            summary = check_pim_latency_throughput(curves)
            results[pattern_name] = {
                "curves": curves,
                "nop_dict": nop_dict,
                "summary": summary,
            }
        _dependency_pattern_cache[std_name] = results

    return _dependency_pattern_cache[std_name]


def _build_lpddr5_pim_case_report(raw_results, case_key, label, target_nop, std_name="lpddr5_pim", case_cfg=None):
    curves = extract_pim_curves(raw_results, std_name)
    nop_dict = curves_to_nop_dict(curves)
    point = nop_dict[target_nop]
    selected_stats = raw_results[(target_nop, 100)]
    energy_report = build_pim_energy_report(selected_stats, std_name)
    derived = energy_report["derived"]
    report = {
        "case_key": case_key,
        "label": label,
        "nop": target_nop,
        "measured_throughput": point["measured_throughput"],
        "simulated_pim_gops": point["simulated_pim_gops"],
        "simulated_pim_gflops": derived["simulated_pim_gflops"],
        "simulated_pim_display_value": point["simulated_pim_display_value"],
        "simulated_pim_display_unit": point["simulated_pim_display_unit"],
        "simulated_pim_display_label": point["simulated_pim_display_label"],
        "pim_datatype": point["pim_datatype"],
        "pim_datatype_class": point["pim_datatype_class"],
        "pim_datatype_kind": point["pim_datatype_kind"],
        "pim_elements_per_request": point["pim_elements_per_request"],
        "pim_ops_per_request": point["pim_ops_per_request"],
        "avg_pim_latency_ns": point["avg_pim_latency"],
        "avg_pim_service_latency_ns": derived["avg_pim_service_latency_ns"],
        "avg_pim_launch_wait_ns": derived["avg_pim_launch_wait_ns"],
        "avg_pim_response_latency_ns": derived["avg_pim_response_latency_ns"],
        "pim_service_latency": derived["pim_service_latency"],
        "pim_launch_wait": derived["pim_launch_wait"],
        "pim_response_latency": derived["pim_response_latency"],
        "memory_background_energy_pJ": derived["memory_background_energy_pJ"],
        "memory_cmd_energy_pJ": derived["memory_cmd_energy_pJ"],
        "pim_incremental_energy_pJ": derived["pim_incremental_energy_pJ"],
        "analytical_energy_estimate_pJ": derived["analytical_energy_estimate_pJ"],
        "energy_total_pJ": derived["energy_total_pJ"],
        "energy_background_pJ": derived["energy_background_pJ"],
        "energy_lpddr_base_cmd_pJ": derived["energy_lpddr_base_cmd_pJ"],
        "energy_incremental_pim_builtin_pJ": derived["energy_incremental_pim_builtin_pJ"],
        "energy_pim_setup_cmd_pJ": derived["energy_pim_setup_cmd_pJ"],
        "energy_pim_mac_cmd_pJ": derived["energy_pim_mac_cmd_pJ"],
        "energy_pim_movement_pJ": derived["energy_pim_movement_pJ"],
        "energy_pim_writeback_pJ": derived["energy_pim_writeback_pJ"],
        "energy_mode_switch_pJ": derived["energy_mode_switch_pJ"],
        "energy_fallback_or_unattributed_pJ": derived["energy_fallback_or_unattributed_pJ"],
        "energy_attribution_sum_pJ": derived["energy_attribution_sum_pJ"],
        "total_energy_pJ": derived["total_energy_pJ"],
        "total_simulation_time_ns": derived["total_simulation_time_ns"],
        "total_completion_time_ns": derived["total_completion_time_ns"],
        "full_system_edp_pJ_ns": derived["full_system_edp_pJ_ns"],
        "runtime_full_system_edp_pJ_ns": derived["runtime_full_system_edp_pJ_ns"],
        "legacy_full_system_edp_pJ_ns": derived["legacy_full_system_edp_pJ_ns"],
        "latency_full_system_edp_pJ_ns": derived["latency_full_system_edp_pJ_ns"],
        "throughput_edp_pJ_ns": derived["throughput_edp_pJ_ns"],
        "analytical_energy_per_pim_request_pJ": derived["analytical_energy_per_pim_request_pJ"],
        "memory_background_power_mW": derived["memory_background_power_mW"],
        "memory_cmd_power_mW": derived["memory_cmd_power_mW"],
        "pim_incremental_power_mW": derived["pim_incremental_power_mW"],
        "analytical_avg_power_mW": derived["analytical_avg_power_mW"],
        "average_power_mW": derived["average_power_mW"],
        "pim_datatype_behavior_enabled": derived["pim_datatype_behavior_enabled"],
        "pim_mac_ops_per_element": derived["pim_mac_ops_per_element"],
        "pim_datatype_bits": derived["pim_datatype_bits"],
        "pim_simd_width_bits": derived["pim_simd_width_bits"],
        "pim_lanes": derived["pim_lanes"],
        "pim_ops_per_block_issue": derived["pim_ops_per_block_issue"],
        "pim_mac_issue_interval_cycles": derived["pim_mac_issue_interval_cycles"],
        "pim_mac_pipeline_latency_cycles": derived["pim_mac_pipeline_latency_cycles"],
        "pim_movement_cycles": derived["pim_movement_cycles"],
        "pim_completion_latency_cycles": derived["pim_completion_latency_cycles"],
        "pim_slots_per_request": derived["pim_slots_per_request"],
        "pim_slot_cost": derived["pim_slot_cost"],
        "pim_compute_energy_pJ_per_mac": derived["pim_compute_energy_pJ_per_mac"],
        "pim_cell_to_pim_energy_pJ_per_256b": derived["pim_cell_to_pim_energy_pJ_per_256b"],
        "pim_interconnect_energy_pJ_per_256b": derived["pim_interconnect_energy_pJ_per_256b"],
        "pim_vrf_access_energy_pJ": derived["pim_vrf_access_energy_pJ"],
        "pim_srf_access_energy_pJ": derived["pim_srf_access_energy_pJ"],
        "pim_mode_switch_energy_pJ": derived["pim_mode_switch_energy_pJ"],
        "pim_banks_per_mpu": derived["pim_banks_per_mpu"],
        "pim_mpu_group_count": derived["pim_mpu_group_count"],
        "total_banks": derived["total_banks"],
        "effective_mpu_groups": derived["effective_mpu_groups"],
        "pim_capacity_stalls": derived["pim_capacity_stalls"],
        "pim_mpu_group_stalls": derived["pim_mpu_group_stalls"],
        "num_mpu_group_busy_blocked_cycles": derived["num_mpu_group_busy_blocked_cycles"],
        "global_issue_stall_per_request": derived["global_issue_stall_per_request"],
        "mpu_group_busy_stall_per_request": derived["mpu_group_busy_stall_per_request"],
        "bank_timing_stall_per_request": derived["bank_timing_stall_per_request"],
        "rr_head_of_line_stall_per_request": derived["rr_head_of_line_stall_per_request"],
        "queue_empty_cycles_per_request": derived["queue_empty_cycles_per_request"],
        "pim_simultaneous_active_banks_peak": derived["pim_simultaneous_active_banks_peak"],
        "num_pim_reqs_served": derived["count_completed_pim_requests"],
        "count_ACT": derived["count_ACT"],
        "count_PRE": derived["count_PRE"],
        "count_RD": derived["count_RD"],
        "count_WR": derived["count_WR"],
        "count_REF": derived["count_REF"],
        "count_PIM_MAC": derived["count_PIM_MAC"],
        "count_PIM_MAC_AB": derived["count_PIM_MAC_AB"],
        "count_PIM_setup_commands": derived["count_PIM_setup_commands"],
        "count_completed_pim_requests": derived["count_completed_pim_requests"],
        "count_frontend_pim_requests": derived["count_frontend_pim_requests"],
        "coefficient_source_summary": derived["coefficient_source_summary"],
    }
    if case_cfg is not None:
        report.update(
            _lpddr5_pim_sequence_provenance(
                case_cfg,
                derived["total_banks"],
                derived["pim_banks_per_mpu"],
            )
        )
    return report


def _lpddr5_pim_cfg_override(case_cfg):
    cfg_override = {
        "dram_kwargs": {**case_cfg["dram_kwargs"], "power": LPDDR5_ANALYSIS_POWER},
        "pim_distribution_mode": case_cfg["pim_distribution_mode"],
        "pim_same_bank": case_cfg["pim_same_bank"],
        "pim_dependency_count": case_cfg["pim_dependency_count"],
        "pim_bank_sequence": case_cfg["pim_bank_sequence"],
        "pim_row_start": case_cfg["pim_row_start"],
        "pim_row_count": case_cfg["pim_row_count"],
    }
    for optional_key in ("pim_bank_group_size", "pim_burst_length", "pim_bank_sequence_order"):
        if optional_key in case_cfg:
            cfg_override[optional_key] = case_cfg[optional_key]
    return cfg_override


def _resolve_lpddr5_pim_controller_bank_sequence(
    input_bank_sequence,
    pim_bank_sequence_order,
    bank_group_size,
    total_banks,
):
    input_bank_sequence = list(input_bank_sequence)
    if pim_bank_sequence_order == "controller":
        return input_bank_sequence
    if pim_bank_sequence_order != "frontend":
        raise ValueError(f"Unsupported pim_bank_sequence_order={pim_bank_sequence_order!r}")

    if not input_bank_sequence:
        return []
    if not bank_group_size:
        return input_bank_sequence

    bank_group_count = max(1, total_banks // bank_group_size)
    return [
        (frontend_bank % bank_group_count) * bank_group_size
        + (frontend_bank // bank_group_count)
        for frontend_bank in input_bank_sequence
    ]


def _lpddr5_pim_sequence_provenance(case_cfg, total_banks, pim_banks_per_mpu):
    input_bank_sequence = list(case_cfg.get("pim_bank_sequence", []))
    pim_bank_sequence_order = case_cfg.get("pim_bank_sequence_order", "frontend")
    resolved_controller_bank_sequence = _resolve_lpddr5_pim_controller_bank_sequence(
        input_bank_sequence,
        pim_bank_sequence_order,
        case_cfg.get("pim_bank_group_size", 0),
        total_banks,
    )
    resolved_mpu_group_sequence = [
        bank // pim_banks_per_mpu for bank in resolved_controller_bank_sequence
    ]
    active_banks = len(set(resolved_controller_bank_sequence))
    if not active_banks and case_cfg.get("pim_same_bank", False):
        active_banks = 1
    return {
        "active_banks": active_banks,
        "pim_bank_sequence_order": pim_bank_sequence_order,
        "input_bank_sequence": input_bank_sequence,
        "resolved_controller_bank_sequence": resolved_controller_bank_sequence,
        "resolved_mpu_group_sequence": resolved_mpu_group_sequence,
    }


def _assert_lpddr5_pim_controller_order_workload(point, expected_active_banks):
    expected_sequence = list(range(expected_active_banks))
    expected_groups = [
        bank // point["pim_banks_per_mpu"] for bank in expected_sequence
    ]
    assert point["active_banks"] == expected_active_banks
    assert point["pim_bank_sequence_order"] == "controller"
    assert point["input_bank_sequence"] == expected_sequence
    assert point["resolved_controller_bank_sequence"] == expected_sequence
    assert point["resolved_mpu_group_sequence"] == expected_groups
    assert point["total_banks"] >= expected_active_banks


def _get_lpddr5_pim_energy_case_reports():
    key = "lpddr5_pim_energy_case_reports"
    if key in _lpddr5_pim_case_report_cache:
        return _lpddr5_pim_case_report_cache[key]

    cfg = STANDARDS["lpddr5_pim"]
    nops = cfg.get("dependency_pattern_nop_counters", cfg["nop_counters"])
    explicit_cases = cfg["dependency_patterns"]["explicit_bank_count_cases"]

    case_specs = [
        ("single_bank_1block", "1-bank\nDedicated MPU, banks/MPU=1"),
        ("single_bank_2block", "1-bank\nShared MPU, banks/MPU=2"),
        ("rr_2bank_1block", "2-bank RR\nDedicated MPU, banks/MPU=1"),
        ("rr_2bank_2block", "2-bank RR\nShared MPU, banks/MPU=2"),
        ("rr_4bank_1block", "4-bank RR\nDedicated MPU, banks/MPU=1"),
        ("rr_4bank_2block", "4-bank RR\nShared MPU, banks/MPU=2"),
        ("forced_conflict_4bank_dedicated_1block", "4-bank forced conflict\nDedicated MPU, banks/MPU=1"),
        ("forced_conflict_4bank_shared_2block", "4-bank forced conflict\nShared MPU, banks/MPU=2"),
        ("forced_conflict_4bank_shared_4block", "4-bank forced conflict\nShared MPU, banks/MPU=4"),
        ("rr_8bank_1block", "8-bank RR\nDedicated MPU, banks/MPU=1"),
        ("rr_8bank_2block", "8-bank RR\nShared MPU, banks/MPU=2"),
    ]

    raw_by_case = {}
    nop_sets = []
    for case_key, _ in case_specs:
        case_cfg = explicit_cases[case_key]
        cfg_override = _lpddr5_pim_cfg_override(case_cfg)

        raw = run_sweep("lpddr5_pim", nops, CI_READ_RATIOS, CI_NUM_PROBES, full=False, cfg_override=cfg_override)
        raw_by_case[case_key] = raw
        nop_sets.append(set(curves_to_nop_dict(extract_pim_curves(raw, "lpddr5_pim")).keys()))

    anchor_curves = extract_pim_curves(raw_by_case["single_bank_2block"], "lpddr5_pim")
    shared_nop = check_pim_latency_throughput(anchor_curves)["nop"]
    common_nops = set.intersection(*nop_sets)
    assert shared_nop in common_nops

    reports = [
        _build_lpddr5_pim_case_report(
            raw_by_case[case_key],
            case_key,
            label,
            shared_nop,
            case_cfg=explicit_cases[case_key],
        )
        for case_key, label in case_specs
    ]

    _lpddr5_pim_case_report_cache[key] = reports
    return reports


def _get_lpddr5_pim_datatype_behavior_reports():
    key = "lpddr5_pim_datatype_behavior_reports"
    if key in _lpddr5_pim_case_report_cache:
        return _lpddr5_pim_case_report_cache[key]

    cfg = STANDARDS["lpddr5_pim"]
    nops = cfg.get("dependency_pattern_nop_counters", cfg["nop_counters"])
    explicit_cases = cfg["dependency_patterns"]["explicit_bank_count_cases"]
    case_specs = [
        ("datatype_int8_2block", "INT8 behavior\n2 banks / MPU group"),
        ("datatype_fp16_2block", "FP16 behavior\n2 banks / MPU group"),
    ]

    raw_by_case = {}
    nop_sets = []
    for case_key, _ in case_specs:
        case_cfg = explicit_cases[case_key]
        cfg_override = _lpddr5_pim_cfg_override(case_cfg)
        raw = run_sweep("lpddr5_pim", nops, CI_READ_RATIOS, CI_NUM_PROBES, full=False, cfg_override=cfg_override)
        raw_by_case[case_key] = raw
        nop_sets.append(set(curves_to_nop_dict(extract_pim_curves(raw, "lpddr5_pim")).keys()))

    anchor_curves = extract_pim_curves(raw_by_case["datatype_int8_2block"], "lpddr5_pim")
    shared_nop = check_pim_latency_throughput(anchor_curves)["nop"]
    common_nops = set.intersection(*nop_sets)
    assert shared_nop in common_nops

    reports = [
        _build_lpddr5_pim_case_report(
            raw_by_case[case_key],
            case_key,
            label,
            shared_nop,
            case_cfg=explicit_cases[case_key],
        )
        for case_key, label in case_specs
    ]
    _lpddr5_pim_case_report_cache[key] = reports
    return reports


def _get_lpddr5_pim_forced_conflict_reports():
    key = "lpddr5_pim_forced_conflict_reports"
    if key in _lpddr5_pim_case_report_cache:
        return _lpddr5_pim_case_report_cache[key]

    explicit_cases = STANDARDS["lpddr5_pim"]["dependency_patterns"]["explicit_bank_count_cases"]
    case_specs = [
        ("forced_conflict_4bank_dedicated_1block", "4-bank forced conflict, dedicated MPU"),
        ("forced_conflict_4bank_shared_2block", "4-bank forced conflict, shared 2 banks/MPU"),
        ("forced_conflict_4bank_shared_4block", "4-bank forced conflict, shared 4 banks/MPU"),
    ]

    reports = []
    for case_key, label in case_specs:
        raw = run_sweep(
            "lpddr5_pim",
            [1],
            CI_READ_RATIOS,
            CI_NUM_PROBES,
            full=False,
            cfg_override=_lpddr5_pim_cfg_override(explicit_cases[case_key]),
        )
        reports.append(
            _build_lpddr5_pim_case_report(
                raw,
                case_key,
                label,
                1,
                case_cfg=explicit_cases[case_key],
            )
        )

    _lpddr5_pim_case_report_cache[key] = reports
    return reports


def _get_lpddr5_pim_avg_power_vs_nop_series():
    key = "lpddr5_pim_avg_power_vs_nop_series"
    if key in _lpddr5_pim_power_vs_nop_cache:
        return _lpddr5_pim_power_vs_nop_cache[key]

    cfg = STANDARDS["lpddr5_pim"]
    nops = cfg.get("dependency_pattern_nop_counters", cfg["nop_counters"])
    explicit_cases = cfg["dependency_patterns"]["explicit_bank_count_cases"]

    representative_case_keys = [
        "rr_4bank_1block",
        "rr_4bank_2block",
        "rr_8bank_1block",
        "rr_8bank_2block",
        "rr_16bank_1block",
        "rr_16bank_2block",
    ]

    case_series = {}
    for case_key in representative_case_keys:
        case_cfg = {**explicit_cases[case_key], "pim_bank_sequence_order": "controller"}
        expected_active_banks = int(case_key.split("bank_", 1)[0].rsplit("_", 1)[1])
        cfg_override = _lpddr5_pim_cfg_override(case_cfg)

        raw = run_sweep("lpddr5_pim", nops, CI_READ_RATIOS, CI_NUM_PROBES, full=False, cfg_override=cfg_override)
        per_nop = {}
        for (nop, rr), stats in raw.items():
            if rr != 100:
                continue
            energy_report = build_pim_energy_report(stats, "lpddr5_pim")
            derived = energy_report["derived"]
            provenance = _lpddr5_pim_sequence_provenance(
                case_cfg,
                derived["total_banks"],
                derived["pim_banks_per_mpu"],
            )
            per_nop[nop] = {
                "avg_pim_latency_ns": derived["avg_pim_latency_ns"],
                "avg_pim_response_latency_ns": derived["avg_pim_response_latency_ns"],
                "analytical_energy_estimate_pJ": derived["analytical_energy_estimate_pJ"],
                "total_energy_pJ": derived["total_energy_pJ"],
                "energy_total_pJ": derived["energy_total_pJ"],
                "energy_attribution_sum_pJ": derived["energy_attribution_sum_pJ"],
                "total_simulation_time_ns": derived["total_simulation_time_ns"],
                "total_completion_time_ns": derived["total_completion_time_ns"],
                "full_system_edp_pJ_ns": derived["full_system_edp_pJ_ns"],
                "runtime_full_system_edp_pJ_ns": derived["runtime_full_system_edp_pJ_ns"],
                "latency_full_system_edp_pJ_ns": derived["latency_full_system_edp_pJ_ns"],
                "analytical_avg_power_mW": derived["analytical_avg_power_mW"],
                "average_power_mW": derived["average_power_mW"],
                "memory_background_power_mW": derived["memory_background_power_mW"],
                "memory_cmd_power_mW": derived["memory_cmd_power_mW"],
                "pim_incremental_power_mW": derived["pim_incremental_power_mW"],
                "measured_throughput": derived["pim_requests_per_ns"],
                "simulated_pim_gops": derived["simulated_pim_gops"],
                "simulated_pim_display_value": derived["simulated_pim_display_value"],
                "simulated_pim_display_unit": derived["simulated_pim_display_unit"],
                "pim_banks_per_mpu": derived["pim_banks_per_mpu"],
                "pim_mpu_group_count": derived["pim_mpu_group_count"],
                "total_banks": derived["total_banks"],
                "effective_mpu_groups": derived["effective_mpu_groups"],
                "pim_capacity_stalls": derived["pim_capacity_stalls"],
                "pim_mpu_group_stalls": derived["pim_mpu_group_stalls"],
                "num_mpu_group_busy_blocked_cycles": derived["num_mpu_group_busy_blocked_cycles"],
                "global_issue_stall_per_request": derived["global_issue_stall_per_request"],
                "mpu_group_busy_stall_per_request": derived["mpu_group_busy_stall_per_request"],
                "bank_timing_stall_per_request": derived["bank_timing_stall_per_request"],
                "pim_simultaneous_active_banks_peak": derived["pim_simultaneous_active_banks_peak"],
                **provenance,
            }
            _assert_lpddr5_pim_controller_order_workload(
                per_nop[nop],
                expected_active_banks,
            )
            if expected_active_banks == 4 and derived["pim_banks_per_mpu"] == 2 and nop == 1:
                assert per_nop[nop]["pim_mpu_group_stalls"] > 0
        case_series[case_key] = dict(sorted(per_nop.items()))

    _lpddr5_pim_power_vs_nop_cache[key] = case_series
    return case_series


def _print_pim_pattern_summary(label, result):
    print(
        f"    {label}: "
        f"avg_pim_latency={result['avg_pim_latency']:.1f} ns, "
        f"throughput={result['measured_throughput']:.6f} req/ns, "
        f"pim_dependency_stalls={result['pim_dependency_stalls']}, "
        f"pim_capacity_stalls={result['pim_capacity_stalls']}, "
        f"pim_inflight_peak={result['pim_inflight_peak']}, "
        f"num_pim_reqs_served={result['num_pim_reqs_served']}, "
        f"nop={result['nop']}"
    )


def _print_pim_pattern_point(label, point):
    print(
        f"      {label}: "
        f"avg_pim_latency={point['avg_pim_latency']:.1f} ns, "
        f"throughput={point['measured_throughput']:.6f} req/ns, "
        f"pim_dependency_stalls={point['pim_dependency_stalls']}, "
        f"pim_capacity_stalls={point['pim_capacity_stalls']}, "
        f"pim_inflight_peak={point['pim_inflight_peak']}, "
        f"num_pim_reqs_served={point['num_pim_reqs_served']}"
    )


def _print_pim_all_bank_summary(label, result):
    print(
        f"    {label}: "
        f"avg_pim_latency={result['avg_pim_latency']:.1f} ns, "
        f"logical_all_bank_throughput={result['measured_throughput']:.6f} req/ns, "
        f"pim_mode_stalls={result['pim_mode_stalls']}, "
        f"pim_load_stalls={result['pim_load_stalls']}, "
        f"pim_ab_inflight_peak={result['pim_ab_inflight_peak']}, "
        f"num_pim_ab_reqs_served={result['num_pim_ab_reqs_served']}, "
        f"nop={result['nop']}"
    )


def _export_task_7_energy_evidence(standard, pim_result):
    if standard.lower() != "lpddr5_pim":
        return None

    raw = _get_raw_sweep(standard)
    best_stats = raw[(pim_result["nop"], 100)]
    energy_evidence = build_pim_energy_evidence(best_stats, standard)
    cfg = STANDARDS[standard]
    tuple_description = (
        f"{cfg.get('num_pim_requests', 0)} PIM_MAC requests "
        f"(bank_group_size={cfg.get('pim_bank_group_size', 4)}, burst_length={cfg.get('pim_burst_length', 16)})"
    )
    summary_path = Path(".sisyphus/evidence/task-7-energy-summary.md")
    json_path = Path(".sisyphus/evidence/task-7-energy-summary.json")
    write_pim_energy_summary(
        energy_evidence,
        summary_path,
        json_path=json_path,
        tuple_description=tuple_description,
    )
    return energy_evidence


@pytest.mark.analysis_fast
@pytest.mark.parametrize("standard", sorted(STANDARDS.keys()))
def test_latency_throughput_fast(request, standard):
    """Run no-refresh formula checks, print % deviations, generate lat-tp plot."""
    verbose = request.config.getoption("--verbose-plot")
    curves = _get_sweep(standard)
    pim_mode = STANDARDS[standard].get("pim_mode", False)

    if pim_mode:
        pim_result = check_pim_latency_throughput(curves)
        energy_evidence = _export_task_7_energy_evidence(standard, pim_result)
        dependency_pattern_results = _get_dependency_pattern_results(standard)

        output_dir = "tests/analysis/plots/fast"
        if verbose:
            output_dir = "tests/analysis/plots/fast_verbose"

        print(f"\n{'=' * 60}")
        print(f"  {standard} Fast PIM Latency-Throughput Results")
        print(f"{'=' * 60}")
        print(f"  Measured PIMCompute latency = {pim_result['measured_latency_ns']:.1f} ns")
        print(
            f"  Measured PIMCompute throughput = {pim_result['measured_throughput']:.6f} requests/ns"
        )
        assert energy_evidence is not None
        print(
            f"  Derived simulated PIM arithmetic throughput = "
            f"{pim_result['simulated_pim_display_value']:.6f} {pim_result['simulated_pim_display_unit']}"
        )
        print(
            f"    datatype={pim_result['pim_datatype']} "
            f"({pim_result['pim_datatype_kind']}), "
            f"SIMD width={energy_evidence['derived']['pim_simd_width_bits']} bits, "
            f"lanes/elements per request={pim_result['pim_elements_per_request']}, "
            f"MAC convention={energy_evidence['derived']['pim_mac_ops_per_element']:.0f} ops/element, "
            f"arithmetic ops/request={pim_result['pim_ops_per_request']:.1f}"
        )
        print(f"  PIM dependency stalls = {pim_result['pim_dependency_stalls']}")
        print(f"  PIM capacity stalls = {pim_result['pim_capacity_stalls']}")
        print(f"  PIM inflight peak = {pim_result['pim_inflight_peak']}")
        print(f"  PIM requests served = {pim_result['num_pim_reqs_served']}")

        cfg = STANDARDS[standard]
        datatype_matrix = cfg.get("datatype_capability_matrix", {})
        baseline_dt = cfg.get("dram_kwargs", {}).get("pim_datatype_class", "int8")
        dt_status = datatype_matrix.get(baseline_dt, {}).get("status", "modeled baseline")

        print("\n  [Workload & Tuple Labels]")
        print("    Workload class: Decode/GEMV-style microbenchmark")
        print(
            f"    Tuple: {cfg.get('num_pim_requests', 0)} PIM_MAC requests, "
            f"bank_group_size={cfg.get('pim_bank_group_size', 4)}, "
            f"burst_length={cfg.get('pim_burst_length', 16)}"
        )
        print("  [Bank Scope Labels]")
        print("    Same-bank: default shared-MPU model serializes same-bank PIM_MAC requests")
        print("    Same-bank independent: dependency-control case under the commodity shared-MPU model")
        print("    Bounded multi-bank (RR): Frontend-shaped bank distribution (does not imply command-level parallelism)")
        print("    Bounded partial subsequence [0,2]: Strict subset of bank group, cross-BG timing only")
        print("    Bounded repeated-bank [0,0,1,1]: Repeated bank visits within distribution cycle")
        print("  [Datatype Labels]")
        print(f"    Baseline capability: {baseline_dt} ({dt_status})")
        print("    Behavior model: datatype behavior can explicitly change PIM latency, slot cost, and PIM incremental energy")
        print("  [Energy Provenance Labels]")
        print("    Modeled: Command counts (ACT, PIM_MAC, etc.) and execution cycles")
        print("    Assumed: Analytical event/background parameters (parameterized, not silicon-calibrated)")
        print("    Derived: Analytical energy estimate per request from observed commands and runtime")

        print(f"\n  Analytical energy estimate = {pim_result['analytical_energy_estimate_pJ']:.3f} pJ")
        print(
            f"  Analytical energy per PIM request = {pim_result['analytical_energy_per_pim_request_pJ']:.6f} pJ"
        )
        print(f"  Total modeled command count = {pim_result['total_command_count']}")

        assert "modeled" in energy_evidence
        assert "derived" in energy_evidence
        assert "assumed" in energy_evidence
        assert energy_evidence["modeled"]["command_counts"]["PIM_MAC"] > 0
        assert energy_evidence["derived"]["analytical_energy_estimate_pJ"] >= 0
        assert energy_evidence["derived"]["analytical_energy_per_pim_request_pJ"] >= 0
        if energy_evidence["derived"].get("sensitivity_only_energy_total", False):
            assert energy_evidence["derived"]["paper_facing_energy_total"] is False
            assert energy_evidence["derived"]["command_energy_breakdown_pJ"] == {}
            assert energy_evidence["assumed"]["analytical_event_energy_per_command_pJ"] == {}
        else:
            assert energy_evidence["derived"]["analytical_energy_estimate_pJ"] > 0
            assert energy_evidence["derived"]["analytical_energy_per_pim_request_pJ"] > 0
        assert energy_evidence["derived"]["simulated_pim_gops"] > 0
        assert energy_evidence["derived"]["simulated_pim_gflops"] is None
        assert energy_evidence["derived"]["simulated_pim_display_value"] == energy_evidence["derived"]["simulated_pim_gops"]
        assert energy_evidence["derived"]["pim_elements_per_request"] == energy_evidence["derived"]["pim_lanes"]
        assert energy_evidence["derived"]["pim_mac_ops_per_element"] == 2.0
        assert energy_evidence["derived"]["pim_ops_per_request"] == (
            energy_evidence["derived"]["pim_elements_per_request"]
            * energy_evidence["derived"]["pim_mac_ops_per_element"]
        )
        assert energy_evidence["derived"]["command_trace_total_records"] > 0
        assert energy_evidence["derived"]["trace_count_matches_counter_total"] is True
        assert energy_evidence["derived"]["trace_command_counts_match_counter"] is True
        assert Path(".sisyphus/evidence/task-7-energy-summary.md").exists()
        summary_text = Path(".sisyphus/evidence/task-7-energy-summary.md").read_text()
        for token in (
            "modeled",
            "derived",
            "assumed",
            "command count",
            "trace count matches counter total",
            "trace command counts match counter",
        ):
            assert token in summary_text

        print("  Energy evidence summary written to .sisyphus/evidence/task-7-energy-summary.md")
        print(
            "  Energy evidence JSON written to .sisyphus/evidence/task-7-energy-summary.json"
        )
        print(
            "  Energy evidence provenance labels = "
            f"{json.dumps(sorted([key for key in energy_evidence if key in {'modeled', 'derived', 'assumed'}]))}"
        )

        if dependency_pattern_results:
            dep_data = dependency_pattern_results["same_bank_dependent"]
            ind_data = dependency_pattern_results["same_bank_independent"]
            bounded_data = dependency_pattern_results["bounded_bank_group_round_robin"]
            all_bank_data = dependency_pattern_results["all_bank_split_mode"]
            dependent = dep_data["summary"]
            independent = ind_data["summary"]
            bounded = bounded_data["summary"]
            all_bank = check_pim_all_bank_split_throughput(all_bank_data["curves"])
            comparison = compare_at_same_nop(dependent, independent)

            dep_nop_dict = dep_data["nop_dict"]
            ind_nop_dict = ind_data["nop_dict"]
            common_nops = sorted(set(dep_nop_dict.keys()) & set(ind_nop_dict.keys()), reverse=True)
            bounded_nop_dict = bounded_data["nop_dict"]
            all_bank_nop_dict = all_bank_data["nop_dict"]

            print(
                "  Bounded multi-bank distribution evidence "
                "(shared_mpu_serial commodity/default, banks-per-MPU=2):"
            )
            print("    This bounded slice reports observed throughput movement and internal mechanism counters.")
            print(
                "    Do not interpret banks-per-MPU as bank-command multi-issue unless command history proves it; "
                "this fast-path evidence only claims frontend-shaped bank distribution."
            )
            print()
            print("  [Best-throughput bounded multi-bank point]")
            _print_pim_pattern_summary("bounded_bank_group_round_robin", bounded)
            assert bounded["num_pim_reqs_served"] > 0
            assert bounded["pim_capacity_stalls"] >= 0
            assert bounded["pim_inflight_peak"] <= 2

            bounded_same_nop = bounded_nop_dict[independent["nop"]]
            bounded_cmp = compare_at_same_nop(ind_nop_dict[independent["nop"]], bounded_same_nop)
            print("  [Bounded multi-bank vs same-bank independent at the same NOP]")
            print(f"    nop={independent['nop']}:")
            _print_pim_pattern_point("same_bank_independent", ind_nop_dict[independent["nop"]])
            _print_pim_pattern_point("bounded_bank_group_round_robin", bounded_same_nop)
            print(
                "      delta (bounded - same_bank_independent): "
                f"avg_pim_latency={bounded_cmp['latency_delta_ns']:+.3f} ns, "
                f"throughput={bounded_cmp['throughput_delta']:+.9f} req/ns, "
                f"pim_dependency_stalls={bounded_cmp['dependency_stall_delta']:+d}, "
                f"pim_capacity_stalls={bounded_cmp['capacity_stall_delta']:+d}, "
                f"pim_inflight_peak={bounded_cmp['inflight_peak_delta']:+d}, "
                f"num_pim_reqs_served={bounded_cmp['served_delta']:+d}"
            )
            assert bounded_same_nop["pim_capacity_stalls"] >= 0
            assert bounded_same_nop["pim_inflight_peak"] <= 2

            # Bounded partial subsequence [0, 2]: strict subset of bank group
            partial_data = dependency_pattern_results["bounded_partial_subsequence"]
            partial = partial_data["summary"]
            partial_nop_dict = partial_data["nop_dict"]
            print()
            print("  [Best-throughput bounded partial subsequence [0, 2] point]")
            _print_pim_pattern_summary("bounded_partial_subsequence", partial)
            assert partial["num_pim_reqs_served"] > 0
            assert partial["pim_capacity_stalls"] >= 0
            assert partial["pim_inflight_peak"] <= 2

            partial_same_nop = partial_nop_dict[independent["nop"]]
            partial_cmp = compare_at_same_nop(ind_nop_dict[independent["nop"]], partial_same_nop)
            print("  [Bounded partial subsequence vs same-bank independent at the same NOP]")
            print(f"    nop={independent['nop']}:")
            _print_pim_pattern_point("same_bank_independent", ind_nop_dict[independent["nop"]])
            _print_pim_pattern_point("bounded_partial_subsequence", partial_same_nop)
            print(
                "      delta (partial - same_bank_independent): "
                f"avg_pim_latency={partial_cmp['latency_delta_ns']:+.3f} ns, "
                f"throughput={partial_cmp['throughput_delta']:+.9f} req/ns, "
                f"pim_dependency_stalls={partial_cmp['dependency_stall_delta']:+d}, "
                f"pim_capacity_stalls={partial_cmp['capacity_stall_delta']:+d}, "
                f"pim_inflight_peak={partial_cmp['inflight_peak_delta']:+d}, "
                f"num_pim_reqs_served={partial_cmp['served_delta']:+d}"
            )
            assert partial_same_nop["pim_capacity_stalls"] >= 0
            assert partial_same_nop["pim_inflight_peak"] <= 2

            # Bounded repeated-bank subsequence [0, 0, 1, 1]: repeated bank visits
            repeated_data = dependency_pattern_results["bounded_repeated_bank_subsequence"]
            repeated = repeated_data["summary"]
            repeated_nop_dict = repeated_data["nop_dict"]
            print()
            print("  [Best-throughput bounded repeated-bank subsequence [0, 0, 1, 1] point]")
            _print_pim_pattern_summary("bounded_repeated_bank_subsequence", repeated)
            assert repeated["num_pim_reqs_served"] > 0
            assert repeated["pim_capacity_stalls"] >= 0
            assert repeated["pim_inflight_peak"] <= 2

            repeated_same_nop = repeated_nop_dict[independent["nop"]]
            repeated_cmp = compare_at_same_nop(ind_nop_dict[independent["nop"]], repeated_same_nop)
            print("  [Bounded repeated-bank subsequence vs same-bank independent at the same NOP]")
            print(f"    nop={independent['nop']}:")
            _print_pim_pattern_point("same_bank_independent", ind_nop_dict[independent["nop"]])
            _print_pim_pattern_point("bounded_repeated_bank_subsequence", repeated_same_nop)
            print(
                "      delta (repeated - same_bank_independent): "
                f"avg_pim_latency={repeated_cmp['latency_delta_ns']:+.3f} ns, "
                f"throughput={repeated_cmp['throughput_delta']:+.9f} req/ns, "
                f"pim_dependency_stalls={repeated_cmp['dependency_stall_delta']:+d}, "
                f"pim_capacity_stalls={repeated_cmp['capacity_stall_delta']:+d}, "
                f"pim_inflight_peak={repeated_cmp['inflight_peak_delta']:+d}, "
                f"num_pim_reqs_served={repeated_cmp['served_delta']:+d}"
            )
            assert repeated_same_nop["pim_capacity_stalls"] >= 0
            assert repeated_same_nop["pim_inflight_peak"] <= 2

            bounded_scaling = bounded["measured_throughput"] / independent["measured_throughput"]
            partial_scaling = partial["measured_throughput"] / independent["measured_throughput"]
            repeated_scaling = repeated["measured_throughput"] / independent["measured_throughput"]

            print("  [Negative Result: Fairness, Starvation, and Interference Scaling Limits]")
            print(
                "    Distributing requests across 4 banks does not imply 4x throughput scaling "
                f"(observed ~{bounded_scaling:.2f}x in this commodity shared-MPU slice)."
            )
            print("    The bounded partial subsequence [0,2] and repeated-bank [0,0,1,1] show severe interference:")
            print(
                "      - They achieve only "
                f"{partial_scaling:.2f}x and {repeated_scaling:.2f}x scaling respectively."
            )
            print("      - The controller-visible pim_inflight_peak caps at 2, not 4.")
            print(
                "    This reports that single-bank PIM_MAC semantics and bank-group timing constraints "
                "limit multi-bank parallelism in the commodity shared-MPU model."
            )

            print()
            print(
                "  Same-bank dependency-pattern comparison "
                "(shared_mpu_serial commodity/default, banks-per-MPU=2):"
            )
            print()
            print("  [Best-throughput points per pattern]")
            _print_pim_pattern_summary("dependent", dependent)
            _print_pim_pattern_summary("independent", independent)
            print(
                "    delta (independent - dependent): "
                f"avg_pim_latency={comparison['latency_delta_ns']:+.3f} ns, "
                f"throughput={comparison['throughput_delta']:+.9f} req/ns, "
                f"pim_dependency_stalls={comparison['dependency_stall_delta']:+d}, "
                f"pim_capacity_stalls={comparison['capacity_stall_delta']:+d}, "
                f"pim_inflight_peak={comparison['inflight_peak_delta']:+d}, "
                f"num_pim_reqs_served={comparison['served_delta']:+d}"
            )
            print()

            print()
            if verbose:
                print("  [Point-by-point at same NOP]")
                for nop in common_nops:
                    dep_pt = dep_nop_dict[nop]
                    ind_pt = ind_nop_dict[nop]
                    pt_cmp = compare_at_same_nop(dep_pt, ind_pt)
                    print(f"    nop={nop}:")
                    _print_pim_pattern_point("dependent", dep_pt)
                    _print_pim_pattern_point("independent", ind_pt)
                    print(
                        f"      delta: "
                        f"avg_pim_latency={pt_cmp['latency_delta_ns']:+.3f} ns, "
                        f"throughput={pt_cmp['throughput_delta']:+.9f} req/ns, "
                        f"pim_dependency_stalls={pt_cmp['dependency_stall_delta']:+d}, "
                        f"pim_capacity_stalls={pt_cmp['capacity_stall_delta']:+d}, "
                        f"pim_inflight_peak={pt_cmp['inflight_peak_delta']:+d}, "
                        f"num_pim_reqs_served={pt_cmp['served_delta']:+d}"
                    )




            print()

            print()
            print("  All-bank split-mode evidence (load-all then compute-all):")
            print("    This path demonstrates the all-bank state machine rather than bank-distributed single-bank PIM issue.")
            print("    The frontend alternates PIMLoadAll and PIMComputeAll and waits for completion between phases.")
            print("    Throughput for this series is counted from logical all-bank completions, not per-bank PIM request retirements.")
            print()
            print("  [Best-throughput all-bank split point]")
            _print_pim_all_bank_summary("all_bank_split_mode", all_bank)
            assert all_bank["num_pim_ab_reqs_served"] > 0
            assert all_bank["pim_ab_inflight_peak"] == 1
            assert all_bank["pim_mode_stalls"] == 0
            assert all_bank["pim_load_stalls"] == 0

            all_bank_same_nop = all_bank_nop_dict[all_bank["nop"]]
            print("  [All-bank split point at its best-throughput NOP]")
            print(f"    nop={all_bank['nop']}:")
            _print_pim_pattern_point("all_bank_split_mode", all_bank_same_nop)
            print(
                "      all-bank extras: "
                f"logical_all_bank_throughput={all_bank_same_nop['all_bank_measured_throughput']:.6f} req/ns, "
                f"num_pim_ab_reqs_served={all_bank_same_nop['num_pim_ab_reqs_served']}, "
                f"pim_ab_inflight_peak={all_bank_same_nop['pim_ab_inflight_peak']}, "
                f"pim_mode_stalls={all_bank_same_nop['pim_mode_stalls']}, "
                f"pim_load_stalls={all_bank_same_nop['pim_load_stalls']}"
            )

            plot_paths = plot_pim_dependency_pattern_same_nop(
                dependency_pattern_results, standard, output_dir=output_dir
            )
            print()
            print("  [Dependency-pattern same-NOP plots (PNG)]")
            print(f"    latency_vs_nop: {plot_paths['latency_vs_nop']}")
            print(f"    throughput_vs_nop: {plot_paths['throughput_vs_nop']}")
            print(f"    dependency_stalls_vs_nop: {plot_paths['dependency_stalls_vs_nop']}")

            if verbose:
                try:
                    plotly_paths = plot_pim_dependency_pattern_same_nop_plotly(
                        dependency_pattern_results, standard, output_dir=output_dir
                    )
                    print()
                    print("  [Dependency-pattern same-NOP plots (Plotly HTML)]")
                    print(f"    latency_vs_nop: {plotly_paths['latency_vs_nop']}")
                    print(f"    throughput_vs_nop: {plotly_paths['throughput_vs_nop']}")
                    print(f"    dependency_stalls_vs_nop: {plotly_paths['dependency_stalls_vs_nop']}")
                except ModuleNotFoundError:
                    print()
                    print("  [Dependency-pattern same-NOP plots (Plotly HTML)]")
                    print("    skipped: plotly is not installed in this environment")
        print(f"{'=' * 60}")

        png_path = plot_pim_lat_tp(curves, standard, output_dir=output_dir)
        print(f"  Baseline PIM plot saved: {png_path}")

        if standard.lower() == "lpddr5_pim":
            for obsolete_name in (
                "lpddr5_pim_pressure_energy_vs_nop.png",
                "lpddr5_pim_pressure_energy_vs_nop.html",
            ):
                obsolete_path = Path(output_dir) / obsolete_name
                if obsolete_path.exists():
                    obsolete_path.unlink()

            case_reports = _get_lpddr5_pim_energy_case_reports()
            assert len(case_reports) == 11
            shared_nops = {report["nop"] for report in case_reports}
            assert len(shared_nops) == 1
            for report in case_reports:
                assert report["memory_background_energy_pJ"] >= 0
                assert report["memory_cmd_energy_pJ"] >= 0
                assert report["pim_incremental_energy_pJ"] >= 0
                assert report["analytical_energy_estimate_pJ"] > 0
                assert report["total_energy_pJ"] == report["analytical_energy_estimate_pJ"]
                energy_component_sum = (
                    report["energy_background_pJ"]
                    + report["energy_lpddr_base_cmd_pJ"]
                    + report["energy_incremental_pim_builtin_pJ"]
                    + report["energy_pim_setup_cmd_pJ"]
                    + report["energy_pim_mac_cmd_pJ"]
                    + report["energy_pim_movement_pJ"]
                    + report["energy_pim_writeback_pJ"]
                    + report["energy_mode_switch_pJ"]
                    + report["energy_fallback_or_unattributed_pJ"]
                )
                assert report["energy_fallback_or_unattributed_pJ"] >= 0
                assert not (
                    report["energy_pim_mac_cmd_pJ"] > 1e6
                    and report["energy_fallback_or_unattributed_pJ"] < 0
                )
                assert report["energy_total_pJ"] == pytest.approx(energy_component_sum)
                assert report["energy_total_pJ"] == pytest.approx(report["total_energy_pJ"])
                assert report["total_simulation_time_ns"] > 0
                assert report["total_completion_time_ns"] > 0
                assert report["full_system_edp_pJ_ns"] == report["total_energy_pJ"] * report["total_completion_time_ns"]
                assert report["runtime_full_system_edp_pJ_ns"] == report["energy_total_pJ"] * report["total_completion_time_ns"]
                assert report["latency_full_system_edp_pJ_ns"] == report["energy_total_pJ"] * report["avg_pim_response_latency_ns"]
                assert report["analytical_energy_per_pim_request_pJ"] > 0
                assert report["memory_background_power_mW"] >= 0
                assert report["memory_cmd_power_mW"] >= 0
                assert report["pim_incremental_power_mW"] >= 0
                assert report["average_power_mW"] == pytest.approx(report["energy_total_pJ"] / report["total_simulation_time_ns"])
                for count_field in (
                    "count_ACT",
                    "count_PRE",
                    "count_RD",
                    "count_WR",
                    "count_REF",
                    "count_PIM_MAC",
                    "count_PIM_MAC_AB",
                    "count_PIM_setup_commands",
                    "count_completed_pim_requests",
                    "count_frontend_pim_requests",
                ):
                    assert report[count_field] >= 0
                assert report["count_completed_pim_requests"] == report["num_pim_reqs_served"]
                assert report["num_mpu_group_busy_blocked_cycles"] >= 0
                assert report["analytical_avg_power_mW"] > 0
                assert report["measured_throughput"] > 0
                assert report["avg_pim_latency_ns"] == pytest.approx(report["avg_pim_response_latency_ns"])
                assert report["avg_pim_response_latency_ns"] == pytest.approx(
                    report["avg_pim_launch_wait_ns"] + report["avg_pim_service_latency_ns"]
                )
                assert report["pim_response_latency"] == report["pim_launch_wait"] + report["pim_service_latency"]
            reports_by_key = {report["case_key"]: report for report in case_reports}
            for one_bank_key, two_bank_key in (
                ("single_bank_1block", "single_bank_2block"),
                ("rr_2bank_1block", "rr_2bank_2block"),
                ("rr_4bank_1block", "rr_4bank_2block"),
                ("forced_conflict_4bank_dedicated_1block", "forced_conflict_4bank_shared_2block"),
                ("rr_8bank_1block", "rr_8bank_2block"),
            ):
                assert reports_by_key[one_bank_key]["pim_banks_per_mpu"] == 1
                assert reports_by_key[two_bank_key]["pim_banks_per_mpu"] == 2
                assert reports_by_key[one_bank_key]["pim_mpu_group_count"] > reports_by_key[two_bank_key]["pim_mpu_group_count"]
                assert reports_by_key[one_bank_key]["effective_mpu_groups"] > reports_by_key[two_bank_key]["effective_mpu_groups"]
            assert reports_by_key["rr_4bank_1block"]["pim_mpu_group_stalls"] >= 0
            assert reports_by_key["rr_4bank_2block"]["pim_mpu_group_stalls"] >= 0
            assert reports_by_key["rr_4bank_1block"]["mpu_group_busy_stall_per_request"] >= 0
            assert reports_by_key["rr_4bank_2block"]["mpu_group_busy_stall_per_request"] >= 0
            assert reports_by_key["rr_4bank_1block"]["avg_pim_response_latency_ns"] == pytest.approx(
                reports_by_key["rr_4bank_1block"]["avg_pim_latency_ns"]
            )
            assert reports_by_key["rr_4bank_2block"]["avg_pim_response_latency_ns"] == pytest.approx(
                reports_by_key["rr_4bank_2block"]["avg_pim_latency_ns"]
            )
            forced_dedicated = reports_by_key["forced_conflict_4bank_dedicated_1block"]
            forced_shared_2block = reports_by_key["forced_conflict_4bank_shared_2block"]
            forced_shared_4block = reports_by_key["forced_conflict_4bank_shared_4block"]
            for forced_shared in (forced_shared_2block, forced_shared_4block):
                assert forced_shared["pim_mpu_group_stalls"] > 0
                assert forced_shared["num_mpu_group_busy_blocked_cycles"] > 0
            assert forced_shared_4block["pim_banks_per_mpu"] == 4
            assert forced_dedicated["effective_mpu_groups"] > forced_shared_4block["effective_mpu_groups"]

            forced_conflict_reports = _get_lpddr5_pim_forced_conflict_reports()
            assert len(forced_conflict_reports) == 3
            forced_conflict_by_key = {report["case_key"]: report for report in forced_conflict_reports}
            forced_conflict_dedicated = forced_conflict_by_key[
                "forced_conflict_4bank_dedicated_1block"
            ]
            assert forced_conflict_dedicated["pim_mpu_group_stalls"] == 0
            assert forced_conflict_dedicated["num_mpu_group_busy_blocked_cycles"] == 0
            assert forced_conflict_dedicated["pim_simultaneous_active_banks_peak"] == 4
            assert forced_conflict_dedicated["measured_throughput"] == pytest.approx(0.354, rel=0.01)

            forced_conflict_expected = {
                "forced_conflict_4bank_shared_2block": (2, 0.178),
                "forced_conflict_4bank_shared_4block": (1, 0.0889),
            }
            for forced_conflict_shared_key in (
                "forced_conflict_4bank_shared_2block",
                "forced_conflict_4bank_shared_4block",
            ):
                forced_conflict_shared = forced_conflict_by_key[forced_conflict_shared_key]
                expected_peak, expected_throughput = forced_conflict_expected[forced_conflict_shared_key]
                assert forced_conflict_shared["pim_mpu_group_stalls"] > 0
                assert forced_conflict_shared["num_mpu_group_busy_blocked_cycles"] > 0
                assert forced_conflict_shared["pim_simultaneous_active_banks_peak"] == expected_peak
                assert forced_conflict_shared["measured_throughput"] == pytest.approx(
                    expected_throughput, rel=0.01
                )
                assert forced_conflict_shared["pim_simultaneous_active_banks_peak"] < forced_conflict_dedicated[
                    "pim_simultaneous_active_banks_peak"
                ]
                assert forced_conflict_shared["measured_throughput"] < forced_conflict_dedicated[
                    "measured_throughput"
                ]

            datatype_reports = _get_lpddr5_pim_datatype_behavior_reports()
            assert len(datatype_reports) == 2
            dt_shared_nops = {report["nop"] for report in datatype_reports}
            assert len(dt_shared_nops) == 1
            dt_by_key = {report["case_key"]: report for report in datatype_reports}
            int8_report = dt_by_key["datatype_int8_2block"]
            fp16_report = dt_by_key["datatype_fp16_2block"]
            assert int8_report["pim_datatype_behavior_enabled"] is True
            assert fp16_report["pim_datatype_behavior_enabled"] is True
            assert int8_report["pim_datatype"] == "int8"
            assert fp16_report["pim_datatype"] == "fp16"
            assert int8_report["pim_datatype_class"] == "int8"
            assert fp16_report["pim_datatype_class"] == "fp16"
            assert int8_report["pim_datatype_kind"] == "integer"
            assert fp16_report["pim_datatype_kind"] == "floating"
            assert int8_report["pim_datatype_bits"] == 8
            assert fp16_report["pim_datatype_bits"] == 16
            assert int8_report["pim_simd_width_bits"] == 256
            assert fp16_report["pim_simd_width_bits"] == 256
            assert int8_report["pim_lanes"] == 32
            assert fp16_report["pim_lanes"] == 16
            assert int8_report["pim_elements_per_request"] == 32
            assert fp16_report["pim_elements_per_request"] == 16
            assert int8_report["pim_mac_ops_per_element"] == 2.0
            assert fp16_report["pim_mac_ops_per_element"] == 2.0
            assert int8_report["pim_ops_per_request"] == 64.0
            assert fp16_report["pim_ops_per_request"] == 32.0
            assert int8_report["simulated_pim_display_unit"] == "GOPS"
            assert fp16_report["simulated_pim_display_unit"] == "GFLOP/s"
            assert int8_report["simulated_pim_gflops"] is None
            assert fp16_report["simulated_pim_gflops"] == fp16_report["simulated_pim_gops"]
            assert int8_report["simulated_pim_display_value"] == int8_report["simulated_pim_gops"]
            assert fp16_report["simulated_pim_display_value"] == fp16_report["simulated_pim_gops"]
            assert int8_report["pim_ops_per_block_issue"] == 64.0
            assert fp16_report["pim_ops_per_block_issue"] == 32.0
            assert int8_report["pim_mac_issue_interval_cycles"] == 4
            assert fp16_report["pim_mac_issue_interval_cycles"] == 4
            assert int8_report["pim_mac_pipeline_latency_cycles"] == 8
            assert fp16_report["pim_mac_pipeline_latency_cycles"] == 8
            assert int8_report["pim_movement_cycles"] == 1
            assert fp16_report["pim_movement_cycles"] == 1
            assert int8_report["pim_completion_latency_cycles"] == 9
            assert fp16_report["pim_completion_latency_cycles"] == 9
            assert int8_report["pim_slots_per_request"] == 1
            assert fp16_report["pim_slots_per_request"] == 1
            assert int8_report["pim_compute_energy_pJ_per_mac"] == 0.0
            assert fp16_report["pim_compute_energy_pJ_per_mac"] == 0.0
            assert fp16_report["avg_pim_latency_ns"] == int8_report["avg_pim_latency_ns"]
            assert fp16_report["measured_throughput"] == int8_report["measured_throughput"]
            assert fp16_report["simulated_pim_gops"] <= int8_report["simulated_pim_gops"]
            assert fp16_report["pim_incremental_energy_pJ"] == int8_report["pim_incremental_energy_pJ"]

            energy_breakdown_png_path = plot_lpddr5_pim_energy_breakdown_by_case(
                case_reports,
                output_dir=output_dir,
            )
            energy_perf_png_path = plot_lpddr5_pim_energy_performance_by_case(
                case_reports,
                output_dir=output_dir,
            )
            power_breakdown_png_path = plot_lpddr5_pim_avg_power_breakdown_by_case(
                case_reports,
                output_dir=output_dir,
            )
            power_vs_nop_series = _get_lpddr5_pim_avg_power_vs_nop_series()
            assert set(power_vs_nop_series.keys()) == {
                "rr_4bank_1block",
                "rr_4bank_2block",
                "rr_8bank_1block",
                "rr_8bank_2block",
                "rr_16bank_1block",
                "rr_16bank_2block",
            }
            for series in power_vs_nop_series.values():
                assert set(series.keys()) == set(STANDARDS["lpddr5_pim"].get("dependency_pattern_nop_counters", STANDARDS["lpddr5_pim"]["nop_counters"]))
                for point in series.values():
                    assert point["avg_pim_latency_ns"] > 0
                    assert point["analytical_energy_estimate_pJ"] >= 0
                    assert point["energy_total_pJ"] == pytest.approx(point["total_energy_pJ"])
                    assert point["energy_attribution_sum_pJ"] == pytest.approx(point["energy_total_pJ"])
                    assert point["full_system_edp_pJ_ns"] == point["total_energy_pJ"] * point["total_completion_time_ns"]
                    assert point["runtime_full_system_edp_pJ_ns"] == point["energy_total_pJ"] * point["total_completion_time_ns"]
                    assert point["latency_full_system_edp_pJ_ns"] == point["energy_total_pJ"] * point["avg_pim_response_latency_ns"]
                    assert point["average_power_mW"] == pytest.approx(point["energy_total_pJ"] / point["total_simulation_time_ns"])
                    assert point["analytical_avg_power_mW"] > 0
                    assert point["pim_mpu_group_stalls"] >= 0
                    assert point["pim_bank_sequence_order"] == "controller"
                    assert point["resolved_controller_bank_sequence"] == point["input_bank_sequence"]
            representative_pairs = (
                ("rr_4bank_1block", "rr_4bank_2block"),
                ("rr_8bank_1block", "rr_8bank_2block"),
                ("rr_16bank_1block", "rr_16bank_2block"),
            )
            for one_bank_key, two_bank_key in representative_pairs:
                assert all(point["pim_banks_per_mpu"] == 1 for point in power_vs_nop_series[one_bank_key].values())
                assert all(point["pim_banks_per_mpu"] == 2 for point in power_vs_nop_series[two_bank_key].values())
                assert all(point["effective_mpu_groups"] > 0 for point in power_vs_nop_series[one_bank_key].values())
                assert all(point["effective_mpu_groups"] > 0 for point in power_vs_nop_series[two_bank_key].values())

            four_bank_shared_nop1 = power_vs_nop_series["rr_4bank_2block"][1]
            assert four_bank_shared_nop1["active_banks"] == 4
            assert four_bank_shared_nop1["pim_banks_per_mpu"] == 2
            assert four_bank_shared_nop1["input_bank_sequence"] == [0, 1, 2, 3]
            assert four_bank_shared_nop1["resolved_controller_bank_sequence"] == [0, 1, 2, 3]
            assert four_bank_shared_nop1["resolved_mpu_group_sequence"] == [0, 0, 1, 1]
            assert four_bank_shared_nop1["pim_mpu_group_stalls"] > 0

            latency_vs_nop_png_path = plot_lpddr5_pim_latency_vs_nop_representative_cases(
                power_vs_nop_series,
                output_dir=output_dir,
            )
            edp_vs_nop_png_path = plot_lpddr5_pim_edp_vs_nop_representative_cases(
                power_vs_nop_series,
                output_dir=output_dir,
            )
            power_vs_nop_png_path = plot_lpddr5_pim_avg_power_vs_nop_representative_cases(
                power_vs_nop_series,
                output_dir=output_dir,
            )
            mpu_group_stalls_png_path = plot_lpddr5_pim_mpu_group_stalls_vs_nop_representative_cases(
                power_vs_nop_series,
                output_dir=output_dir,
            )
            bank_count_sensitivity_png_path = plot_lpddr5_pim_bank_count_sensitivity_same_nop(
                power_vs_nop_series,
                output_dir=output_dir,
                nop=1,
            )
            print(f"  LPDDR5PIM energy breakdown at same-NOP plot saved: {energy_breakdown_png_path}")
            print(f"  LPDDR5PIM energy-performance at same-NOP plot saved: {energy_perf_png_path}")
            print(f"  LPDDR5PIM avg simulated power at same-NOP plot saved: {power_breakdown_png_path}")
            print(f"  LPDDR5PIM avg PIM latency vs NOP plot saved: {latency_vs_nop_png_path}")
            print(f"  LPDDR5PIM EDP vs NOP plot saved: {edp_vs_nop_png_path}")
            print(f"  LPDDR5PIM avg simulated power vs NOP plot saved: {power_vs_nop_png_path}")
            print(f"  LPDDR5PIM MPU-group stalls vs NOP plot saved: {mpu_group_stalls_png_path}")
            print(f"  LPDDR5PIM bank-count shared-MPU sensitivity plot saved: {bank_count_sensitivity_png_path}")

            if verbose:
                try:
                    energy_breakdown_html_path = plot_lpddr5_pim_energy_breakdown_by_case_plotly(
                        case_reports,
                        output_dir=output_dir,
                    )
                    energy_perf_html_path = plot_lpddr5_pim_energy_performance_by_case_plotly(
                        case_reports,
                        output_dir=output_dir,
                    )
                    power_breakdown_html_path = plot_lpddr5_pim_avg_power_breakdown_by_case_plotly(
                        case_reports,
                        output_dir=output_dir,
                    )
                    latency_vs_nop_html_path = plot_lpddr5_pim_latency_vs_nop_representative_cases_plotly(
                        power_vs_nop_series,
                        output_dir=output_dir,
                    )
                    edp_vs_nop_html_path = plot_lpddr5_pim_edp_vs_nop_representative_cases_plotly(
                        power_vs_nop_series,
                        output_dir=output_dir,
                    )
                    power_vs_nop_html_path = plot_lpddr5_pim_avg_power_vs_nop_representative_cases_plotly(
                        power_vs_nop_series,
                        output_dir=output_dir,
                    )
                    print(f"  LPDDR5PIM energy breakdown at same-NOP Plotly HTML saved: {energy_breakdown_html_path}")
                    print(f"  LPDDR5PIM energy-performance at same-NOP Plotly HTML saved: {energy_perf_html_path}")
                    print(f"  LPDDR5PIM avg simulated power at same-NOP Plotly HTML saved: {power_breakdown_html_path}")
                    print(f"  LPDDR5PIM avg PIM latency vs NOP Plotly HTML saved: {latency_vs_nop_html_path}")
                    print(f"  LPDDR5PIM EDP vs NOP Plotly HTML saved: {edp_vs_nop_html_path}")
                    print(f"  LPDDR5PIM avg simulated power vs NOP Plotly HTML saved: {power_vs_nop_html_path}")
                except ModuleNotFoundError:
                    print("  LPDDR5PIM same-NOP Plotly HTML skipped: plotly is not installed")

        return

    streaming_stats = _get_streaming(standard)

    lat_result = check_unloaded_latency(curves, standard)
    bw_result = check_peak_bandwidth(curves, standard)
    streaming_result = check_streaming_peak_bandwidth(standard, streaming_stats)

    print(f"\n{'=' * 60}")
    print(f"  {standard} Fast Latency-Throughput Results")
    print(f"{'=' * 60}")
    print("  Unloaded Latency:")
    print(
        f"    Expected (nRP+nRCD+nCL)*tCK = "
        f"({lat_result['nRP']}+{lat_result['nRCD']}+{lat_result['nCL']})"
        f" * {lat_result['tCK_ns']:.3f} ns = "
        f"{lat_result['expected_ns']:.1f} ns"
    )
    print(f"    Measured = {lat_result['measured_ns']:.1f} ns")
    print(f"    Deviation = {lat_result['deviation_pct']:+.1f}%")
    print()
    print("  Max Bandwidth (probed sweep):")
    print(f"    Theoretical peak = {bw_result['max_theoretical_bw']:.1f} GB/s")
    print(f"    Measured max = {bw_result['measured_max_bw']:.1f} GB/s")
    print(f"    Deviation = {bw_result['deviation_from_theoretical_pct']:+.1f}%")
    print()
    print("  Streaming-Only Bandwidth (no probes):")
    print(f"    Theoretical peak = {streaming_result['max_theoretical_bw']:.1f} GB/s")
    print(f"    Measured = {streaming_result['measured_streaming_bw']:.1f} GB/s")
    print(f"    Deviation = {streaming_result['deviation_from_theoretical_pct']:+.1f}%")
    print(f"{'=' * 60}")

    output_dir = "tests/analysis/plots/fast"
    if verbose:
        output_dir = "tests/analysis/plots/fast_verbose"

    png_path = plot_lat_tp(
        curves,
        standard,
        {"latency": lat_result, "bandwidth": bw_result, "streaming": streaming_result},
        output_dir=output_dir,
        verbose=verbose,
    )
    print(f"  Plot saved: {png_path}")
