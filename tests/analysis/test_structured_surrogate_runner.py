"""Analysis-runner tests for structured workload-surrogate traces."""

from __future__ import annotations

import json
from pathlib import Path

from tests.analysis.runner import run_structured_surrogate


def _provenance() -> dict:
    return {
        "source_kind": "generated",
        "tuple_manifest": "decode_only_tuple",
        "literature_anchor": ["manual:runner_smoke"],
        "generator_version": "v0.1",
        "claim_boundary": [
            "structured workload-surrogate",
            "simulator-diagnostic",
            "non-silicon-calibrated",
            "decode-only-first",
        ],
        "non_claims": [
            "not_runtime_replay",
            "not_vllm_replay",
            "not_mixed_prefill_decode",
            "not_all_bank_fidelity",
        ],
    }


def _mapping_policy() -> dict:
    return {
        "host_policy": "bounded_sequential_host_requests",
        "pim_policy": "per_bank_pimcompute_using_shared_mpu_serial",
        "bank_sequence_policy": "controller_visible_round_robin_4bank",
        "mpu_grouping_policy": "pim_banks_per_mpu=2_shared_mpu_serial",
        "controller_bank_order": "controller",
    }


def _write_runner_trace(path: Path) -> None:
    records = [
        {
            "schema_version": "v0.1",
            "record_id": "rec_0000",
            "kind": "PIMCompute",
            "phase": "decode",
            "layer": "layer_00",
            "op": "q_projection_gemv",
            "repeat": 1,
            "provenance": _provenance(),
            "mapping_policy": _mapping_policy(),
            "num_requests": 4,
            "bank_sequence": [0, 1],
            "dependency_context": {
                "kind": "column_dependency_count",
                "dependency_count": 2,
                "dependency_id": 0,
            },
            "row_policy": {
                "kind": "bounded_row_window_decode_only",
                "row_start": 0,
                "row_count": 2,
                "resolved_row": 0,
            },
            "column_policy": {
                "kind": "dependency_column_round_robin",
                "column_start": 0,
                "resolved_column": 0,
            },
            "datatype_metadata": {"datatype": "int8"},
            "burst_length": 1,
        },
        {
            "schema_version": "v0.1",
            "record_id": "rec_0001",
            "kind": "Drain",
            "phase": "decode",
            "layer": "decode_tail",
            "op": "final_drain",
            "repeat": 1,
            "provenance": _provenance(),
            "mapping_policy": _mapping_policy(),
            "drain_scope": "trace",
        },
    ]
    path.write_text("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n", encoding="utf-8")


def test_structured_surrogate_runner_returns_simulator_diagnostic_evidence(tmp_path: Path):
    trace_path = tmp_path / "structured_runner.jsonl"
    _write_runner_trace(trace_path)

    stats = run_structured_surrogate(
        "lpddr5_pim",
        trace_path,
        cfg_override={"dram_kwargs": {"pim_banks_per_mpu": 2, "pim_mac_execution_model": "shared_mpu_serial"}},
        observability_dir=tmp_path,
    )

    frontend = stats["frontend"]
    assert frontend["records_loaded"] == 2
    assert frontend["host_records"] == 0
    assert frontend["pim_records"] == 1
    assert frontend["barriers_retired"] == 0
    assert frontend["drains_retired"] == 1
    assert frontend["pim_requests_sent"] == frontend["pim_requests_completed"] == 4

    evidence = stats["evidence"]["structured_workload_surrogate"]
    modeled = evidence["modeled"]
    assert evidence["trace_path"] == str(trace_path)
    assert evidence["frontend_stats"]["records_loaded"] == 2
    assert modeled["controller_stats"]["num_pim_reqs_served"] == 4
    assert modeled["command_counts"].get("PIM_MAC", 0) > 0
    assert modeled["command_traces"]
