"""LPDDR5-PIM shared-MPU validation sweep.

Runs the diagnostic matrix requested for shared-MPU sensitivity:

  active/total banks: 4, 8, 16, 32
  pim_banks_per_mpu: 1, 2, 4
  NOP: 1, 3, 5, 8, 10

The output CSV/JSON focuses on effective MPU groups, throughput/latency, full-system
EDP, and split stall reasons. It intentionally does not transform or separate curves;
overlap in the 4-bank regime is preserved as a model result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.analysis.runner import run_single
from tests.analysis.test_fast import (
    LPDDR5_ANALYSIS_POWER,
    _assert_lpddr5_pim_controller_order_workload,
    _lpddr5_pim_sequence_provenance,
)
from tests.analysis.utils.energy import build_pim_energy_report


BANK_COUNTS = (4, 8, 16, 32)
PIM_BANKS_PER_MPU = (1, 2, 4)
NOP_VALUES = (1, 3, 5, 8, 10)

OUTPUT_FIELDS = (
    "total_banks",
    "active_banks",
    "pim_bank_sequence_order",
    "input_bank_sequence",
    "resolved_controller_bank_sequence",
    "resolved_mpu_group_sequence",
    "pim_banks_per_mpu",
    "effective_mpu_groups",
    "nop",
    "NOP",
    "average_pim_latency",
    "avg_pim_latency_ns",
    "average_pim_service_latency",
    "average_pim_launch_wait",
    "average_pim_response_latency",
    "request_throughput",
    "pim_throughput",
    "completed_pim_requests",
    "total_cycles",
    "runtime_ns",
    "average_power",
    "average_power_mW",
    "total_energy",
    "total_energy_pJ",
    "energy_total_pJ",
    "energy_per_completed_pim_request_pJ",
    "energy_background_pJ",
    "energy_lpddr_base_cmd_pJ",
    "energy_incremental_pim_builtin_pJ",
    "energy_pim_setup_cmd_pJ",
    "energy_pim_mac_cmd_pJ",
    "energy_pim_movement_pJ",
    "energy_pim_writeback_pJ",
    "energy_mode_switch_pJ",
    "energy_fallback_or_unattributed_pJ",
    "energy_attribution_sum_pJ",
    "energy_attribution_mode",
    "full_system_edp",
    "edp_pJ_ns",
    "full_system_edp_pJ_ns",
    "throughput_edp_pJ_ns",
    "direct_total_energy_comparison_valid",
    "direct_total_energy_comparison_group",
    "coefficient_source_summary",
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
    "total_simulation_time",
    "total_completion_time",
    "global_issue_stall_per_request",
    "mpu_group_busy_stall_per_request",
    "bank_timing_stall_per_request",
    "rr_head_of_line_stall_per_request",
    "queue_empty_cycles_per_request",
    "pim_mpu_group_stalls",
    "num_mpu_group_busy_blocked_cycles",
    "pim_simultaneous_active_banks_peak",
    "num_issued_pim_mac",
)

ENERGY_COMPONENT_FIELDS = (
    "energy_background_pJ",
    "energy_lpddr_base_cmd_pJ",
    "energy_incremental_pim_builtin_pJ",
    "energy_pim_setup_cmd_pJ",
    "energy_pim_mac_cmd_pJ",
    "energy_pim_movement_pJ",
    "energy_pim_writeback_pJ",
    "energy_mode_switch_pJ",
    "energy_fallback_or_unattributed_pJ",
)


def _safe_energy_per_request(total_energy_pj: float, completed_pim_requests: int) -> float:
    assert completed_pim_requests > 0
    return total_energy_pj / completed_pim_requests


def _case_override(active_banks: int, pim_banks_per_mpu: int) -> dict:
    rank_count = 2 if active_banks > 16 else 1
    return {
        "dram_kwargs": {
            "pim_enabled": True,
            "pim_mode": "bank",
            "pim_banks_per_mpu": pim_banks_per_mpu,
            "pim_mac_execution_model": "shared_mpu_serial",
            "pim_datatype": "int8",
            "rank": rank_count,
            "power": LPDDR5_ANALYSIS_POWER,
        },
        "pim_distribution_mode": "bank_sequence",
        "pim_same_bank": False,
        "pim_bank_sequence": list(range(active_banks)),
        "pim_bank_sequence_order": "controller",
        "pim_bank_group_size": active_banks,
        "pim_dependency_count": min(active_banks, 4),
        "pim_burst_length": 1,
        "pim_row_start": 0,
        "pim_row_count": 1,
    }


def _assert_forced_conflict_expectations(rows: list[dict]) -> None:
    rows_by_key = {
        (row["active_banks"], row["pim_banks_per_mpu"], row["nop"]): row
        for row in rows
    }
    for nop in (1,):
        dedicated = rows_by_key[(4, 1, nop)]
        assert dedicated["pim_mpu_group_stalls"] == 0
        assert dedicated["num_mpu_group_busy_blocked_cycles"] == 0
        assert dedicated["pim_simultaneous_active_banks_peak"] == 4
        assert math.isclose(dedicated["request_throughput"], 0.354, rel_tol=0.01)
        expected_shared = {
            2: (2, 0.178),
            4: (1, 0.0889),
        }
        for pim_banks_per_mpu in (2, 4):
            shared = rows_by_key[(4, pim_banks_per_mpu, nop)]
            expected_peak, expected_throughput = expected_shared[pim_banks_per_mpu]
            assert shared["pim_bank_sequence_order"] == "controller"
            assert shared["input_bank_sequence"] == [0, 1, 2, 3]
            assert shared["resolved_controller_bank_sequence"] == [0, 1, 2, 3]
            assert shared["resolved_mpu_group_sequence"] == [
                bank // pim_banks_per_mpu for bank in [0, 1, 2, 3]
            ]
            assert shared["pim_mpu_group_stalls"] > 0
            assert shared["num_mpu_group_busy_blocked_cycles"] > 0
            assert shared["pim_simultaneous_active_banks_peak"] == expected_peak
            assert math.isclose(shared["request_throughput"], expected_throughput, rel_tol=0.01)
            assert shared["pim_simultaneous_active_banks_peak"] < dedicated[
                "pim_simultaneous_active_banks_peak"
            ]
            assert shared["request_throughput"] < dedicated["request_throughput"]


def _validate_energy_normalization(rows: list[dict]) -> None:
    rows_by_nop: dict[int, list[dict]] = {}
    for row in rows:
        rows_by_nop.setdefault(row["nop"], []).append(row)

    for nop, group_rows in rows_by_nop.items():
        completed_counts = {row["completed_pim_requests"] for row in group_rows}
        group_valid = len(completed_counts) == 1
        for row in group_rows:
            row["direct_total_energy_comparison_valid"] = group_valid
            row["direct_total_energy_comparison_group"] = f"same_nop={nop}"

    for row in rows:
        completed_pim_requests = row["completed_pim_requests"]
        total_cycles = row["total_cycles"]
        runtime_ns = row["runtime_ns"]
        total_energy_pj = row["total_energy_pJ"]
        component_sum_pj = sum(row[field] for field in ENERGY_COMPONENT_FIELDS)

        assert completed_pim_requests > 0
        assert math.isfinite(total_cycles) and total_cycles > 0
        assert math.isfinite(runtime_ns) and runtime_ns > 0.0
        assert math.isfinite(total_energy_pj) and total_energy_pj >= 0.0
        assert math.isfinite(row["energy_per_completed_pim_request_pJ"])
        assert row["energy_per_completed_pim_request_pJ"] >= 0.0
        assert math.isfinite(row["average_power_mW"])
        assert math.isfinite(row["edp_pJ_ns"])
        assert row["edp_pJ_ns"] >= 0.0
        assert math.isclose(row["average_power_mW"], total_energy_pj / runtime_ns, rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(row["edp_pJ_ns"], total_energy_pj * runtime_ns, rel_tol=1e-9, abs_tol=1e-6)
        assert row["energy_fallback_or_unattributed_pJ"] >= -1e-9
        assert all(row[field] >= -1e-9 for field in ENERGY_COMPONENT_FIELDS)
        assert not (
            row["energy_pim_mac_cmd_pJ"] > 1e6 and row["energy_fallback_or_unattributed_pJ"] < 0.0
        )
        assert math.isclose(total_energy_pj, component_sum_pj, rel_tol=1e-9, abs_tol=1e-6)
        assert math.isclose(row["energy_attribution_sum_pJ"], total_energy_pj, rel_tol=1e-9, abs_tol=1e-6)


def _print_energy_normalization_summary(rows: list[dict]) -> None:
    print("LPDDR5PIM normalized energy representative rows:")
    for index in (0, len(rows) // 2, len(rows) - 1):
        row = rows[index]
        print(
            "  "
            f"banks={row['active_banks']} "
            f"mpu={row['pim_banks_per_mpu']} "
            f"nop={row['nop']} "
            f"completed_pim_requests={row['completed_pim_requests']} "
            f"total_cycles={row['total_cycles']} "
            f"runtime_ns={row['runtime_ns']:.6f} "
            f"total_energy_pJ={row['total_energy_pJ']:.6f} "
            f"energy_per_completed_pim_request_pJ={row['energy_per_completed_pim_request_pJ']:.9f} "
            f"average_power_mW={row['average_power_mW']:.9f} "
            f"edp_pJ_ns={row['edp_pJ_ns']:.6f} "
            f"coefficient_source_summary={row['coefficient_source_summary']}"
        )


def run_validation_sweep(num_probes: int, warmup: int) -> list[dict]:
    rows = []
    for active_banks in BANK_COUNTS:
        for pim_banks_per_mpu in PIM_BANKS_PER_MPU:
            for nop in NOP_VALUES:
                case_override = _case_override(active_banks, pim_banks_per_mpu)
                stats = run_single(
                    "lpddr5_pim",
                    nop,
                    read_ratio=100,
                    num_probes=num_probes,
                    warmup=warmup,
                    full=False,
                    cfg_override=case_override,
                )
                derived = build_pim_energy_report(stats, "lpddr5_pim")["derived"]
                ctrl = stats["memory_system"]["controller"]
                completed_pim_requests = int(derived["count_completed_pim_requests"])
                total_cycles = int(ctrl["cycles"])
                runtime_ns = float(derived["total_simulation_time_ns"])
                total_energy_pj = float(derived["energy_total_pJ"])
                energy_per_completed_pim_request_pj = _safe_energy_per_request(
                    total_energy_pj, completed_pim_requests
                )
                edp_pj_ns = total_energy_pj * runtime_ns
                provenance = _lpddr5_pim_sequence_provenance(
                    case_override,
                    derived["total_banks"],
                    derived["pim_banks_per_mpu"],
                )
                rows.append(
                    row := {
                        "total_banks": derived["total_banks"],
                        **provenance,
                        "pim_banks_per_mpu": derived["pim_banks_per_mpu"],
                        "effective_mpu_groups": derived["effective_mpu_groups"],
                        "nop": nop,
                        "NOP": nop,
                        "average_pim_latency": derived["average_pim_latency"],
                        "avg_pim_latency_ns": derived["avg_pim_latency_ns"],
                        "average_pim_service_latency": derived["average_pim_service_latency"],
                        "average_pim_launch_wait": derived["average_pim_launch_wait"],
                        "average_pim_response_latency": derived["average_pim_response_latency"],
                        "request_throughput": derived["request_throughput"],
                        "pim_throughput": derived.get("pim_throughput", derived["request_throughput"]),
                        "completed_pim_requests": completed_pim_requests,
                        "total_cycles": total_cycles,
                        "runtime_ns": runtime_ns,
                        "average_power": derived["average_power"],
                        "average_power_mW": derived["average_power_mW"],
                        "total_energy": derived["total_energy"],
                        "total_energy_pJ": total_energy_pj,
                        "energy_total_pJ": derived["energy_total_pJ"],
                        "energy_per_completed_pim_request_pJ": energy_per_completed_pim_request_pj,
                        "energy_background_pJ": derived["energy_background_pJ"],
                        "energy_lpddr_base_cmd_pJ": derived["energy_lpddr_base_cmd_pJ"],
                        "energy_incremental_pim_builtin_pJ": derived[
                            "energy_incremental_pim_builtin_pJ"
                        ],
                        "energy_pim_setup_cmd_pJ": derived["energy_pim_setup_cmd_pJ"],
                        "energy_pim_mac_cmd_pJ": derived["energy_pim_mac_cmd_pJ"],
                        "energy_pim_movement_pJ": derived["energy_pim_movement_pJ"],
                        "energy_pim_writeback_pJ": derived["energy_pim_writeback_pJ"],
                        "energy_mode_switch_pJ": derived["energy_mode_switch_pJ"],
                        "energy_fallback_or_unattributed_pJ": derived[
                            "energy_fallback_or_unattributed_pJ"
                        ],
                        "energy_attribution_sum_pJ": derived["energy_attribution_sum_pJ"],
                        "energy_attribution_mode": derived["energy_attribution_mode"],
                        "full_system_edp": derived["full_system_edp"],
                        "edp_pJ_ns": edp_pj_ns,
                        "full_system_edp_pJ_ns": derived["full_system_edp_pJ_ns"],
                        "throughput_edp_pJ_ns": derived["throughput_edp_pJ_ns"],
                        "direct_total_energy_comparison_valid": True,
                        "direct_total_energy_comparison_group": "pending_validation",
                        "coefficient_source_summary": derived["coefficient_source_summary"],
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
                        "total_simulation_time": derived["total_simulation_time"],
                        "total_completion_time": derived["total_completion_time"],
                        "global_issue_stall_per_request": derived["global_issue_stall_per_request"],
                        "mpu_group_busy_stall_per_request": derived["mpu_group_busy_stall_per_request"],
                        "bank_timing_stall_per_request": derived["bank_timing_stall_per_request"],
                        "rr_head_of_line_stall_per_request": derived["rr_head_of_line_stall_per_request"],
                        "queue_empty_cycles_per_request": derived["queue_empty_cycles_per_request"],
                        "pim_mpu_group_stalls": derived["pim_mpu_group_stalls"],
                        "num_mpu_group_busy_blocked_cycles": derived[
                            "num_mpu_group_busy_blocked_cycles"
                        ],
                        "pim_simultaneous_active_banks_peak": derived[
                            "pim_simultaneous_active_banks_peak"
                        ],
                        "num_issued_pim_mac": derived["num_issued_pim_mac"],
                    }
                )
                _assert_lpddr5_pim_controller_order_workload(row, active_banks)
    _validate_energy_normalization(rows)
    _assert_forced_conflict_expectations(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="tests/analysis/plots/fast")
    parser.add_argument("--num-probes", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=10000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = run_validation_sweep(args.num_probes, args.warmup)
    _print_energy_normalization_summary(rows)

    csv_path = output_dir / "lpddr5_pim_shared_mpu_validation_sweep.csv"
    json_path = output_dir / "lpddr5_pim_shared_mpu_validation_sweep.json"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
