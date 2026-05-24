"""Integration tests for the structured workload-surrogate replay frontend."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

from tests.analysis.testcases.lpddr5_pim import CONFIG as LPDDR5_PIM_CONFIG
from tests.utils.dram import create_dram
from tests.utils.sim import extract_dram_layout

ramulator = importlib.import_module("ramulator")


def _provenance() -> dict:
    return {
        "source_kind": "handwritten",
        "tuple_manifest": "handwritten_smoke",
        "literature_anchor": ["manual:smoke"],
        "generator_version": "manual",
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


def _mapping_policy(*, include_controller_bank_order: bool = True) -> dict:
    policy = {
        "host_policy": "bounded_sequential_host_requests",
        "pim_policy": "per_bank_pimcompute_using_shared_mpu_serial",
        "bank_sequence_policy": "controller_visible_round_robin_4bank",
        "mpu_grouping_policy": "pim_banks_per_mpu=2_shared_mpu_serial",
    }
    if include_controller_bank_order:
        policy["controller_bank_order"] = "controller"
    return policy


def _make_mem(dram):
    ctrl = ramulator.controller.LPDDR5PIM(
        dram=dram,
        scheduler=ramulator.scheduler.FRFCFS(),
        refresh_manager=ramulator.refresh_manager.NoRefresh(),
        row_policy=ramulator.row_policy.Open(),
        addr_mapper=ramulator.addr_mapper.PassThroughAddrMapper(),
    )
    return ramulator.memory_system.GenericDRAM(
        clock_ratio=1,
        controllers=[ctrl],
        channel_mapper=ramulator.channel_mapper.CacheLineInterleave(),
    )


def _assert_frontend_rejects(trace_path: Path, expected: str) -> None:
    dram = create_dram(LPDDR5_PIM_CONFIG)
    layout = extract_dram_layout(dram)
    frontend = ramulator.frontend.StructuredWorkloadSurrogateTrace(
        clock_ratio=LPDDR5_PIM_CONFIG["frontend_clock_ratio"],
        path=str(trace_path),
        pim_request_type_id=2,
        **layout,
    )
    try:
        ramulator.Simulation(frontend, _make_mem(dram))
    except RuntimeError as exc:
        assert expected in str(exc)
    else:
        raise AssertionError(f"Expected frontend rejection containing: {expected}")


def _minimal_pim_record(*, provenance: dict | None = None) -> dict:
    return {
        "schema_version": "v0.1",
        "record_id": "rec_0000",
        "kind": "PIMCompute",
        "phase": "decode",
        "layer": "layer_00",
        "op": "q_projection_gemv",
        "repeat": 1,
        "provenance": _provenance() if provenance is None else provenance,
        "mapping_policy": _mapping_policy(),
        "num_requests": 1,
        "bank_sequence": [0],
        "dependency_context": {
            "kind": "column_dependency_count",
            "dependency_count": 1,
            "dependency_id": 0,
        },
        "row_policy": {
            "kind": "bounded_row_window_decode_only",
            "row_start": 0,
            "row_count": 1,
            "resolved_row": 0,
        },
        "column_policy": {
            "kind": "dependency_column_round_robin",
            "column_start": 0,
            "resolved_column": 0,
        },
        "datatype_metadata": {"datatype": "int8"},
        "burst_length": 1,
    }


def _write_single_record_trace(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")


def _write_smoke_trace(path: Path) -> None:
    records = [
        {
            "schema_version": "v0.1",
            "record_id": "rec_0000",
            "kind": "HostRead",
            "phase": "decode",
            "layer": "layer_00",
            "op": "attention_k_cache_read",
            "repeat": 1,
            "provenance": _provenance(),
            "mapping_policy": _mapping_policy(),
            "bytes": 128,
            "address_policy": {
                "kind": "bounded_sequential_host_requests",
                "base_byte": 0,
                "stride_bytes": 64,
                "count": 2,
            },
        },
        {
            "schema_version": "v0.1",
            "record_id": "rec_0001",
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
            "record_id": "rec_0002",
            "kind": "Barrier",
            "phase": "decode",
            "layer": "layer_00",
            "op": "layer_transition_barrier",
            "repeat": 1,
            "provenance": _provenance(),
            "mapping_policy": _mapping_policy(),
            "barrier_scope": "layer",
        },
        {
            "schema_version": "v0.1",
            "record_id": "rec_0003",
            "kind": "HostWrite",
            "phase": "decode",
            "layer": "layer_00",
            "op": "kv_cache_append_accounting",
            "repeat": 1,
            "provenance": _provenance(),
            "mapping_policy": _mapping_policy(),
            "bytes": 64,
            "address_policy": {
                "kind": "bounded_sequential_host_requests",
                "base_byte": 4096,
                "stride_bytes": 64,
                "count": 1,
            },
        },
        {
            "schema_version": "v0.1",
            "record_id": "rec_0004",
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


def _write_default_order_trace(path: Path) -> None:
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
            "mapping_policy": _mapping_policy(include_controller_bank_order=False),
            "num_requests": 1,
            "bank_sequence": [0],
            "dependency_context": {
                "kind": "column_dependency_count",
                "dependency_count": 1,
                "dependency_id": 0,
            },
            "row_policy": {
                "kind": "bounded_row_window_decode_only",
                "row_start": 0,
                "row_count": 1,
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
            "mapping_policy": _mapping_policy(include_controller_bank_order=False),
            "drain_scope": "trace",
        },
    ]
    path.write_text("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n", encoding="utf-8")


def _write_all_bank_trace(path: Path) -> None:
    records = [
        {
            "schema_version": "v0.1",
            "record_id": "rec_0000",
            "kind": "PIMLoadAll",
            "phase": "decode",
            "layer": "layer_00",
            "op": "q_projection_gemv_all_bank_load",
            "repeat": 1,
            "provenance": _provenance(),
            "mapping_policy": _mapping_policy(),
            "num_requests": 1,
            "all_bank_scope": "rank",
            "dependency_context": {"kind": "column_dependency_count", "dependency_count": 1, "dependency_id": 0},
            "row_policy": {"kind": "bounded_row_window_all_bank_semantic", "row_start": 0, "row_count": 1, "resolved_row": 0},
            "column_policy": {"kind": "dependency_column_round_robin", "column_start": 0, "resolved_column": 0},
            "datatype_metadata": {"datatype": "int8"},
        },
        {
            "schema_version": "v0.1",
            "record_id": "rec_0001",
            "kind": "PIMComputeAll",
            "phase": "decode",
            "layer": "layer_00",
            "op": "q_projection_gemv_all_bank_compute",
            "repeat": 1,
            "provenance": _provenance(),
            "mapping_policy": _mapping_policy(),
            "num_requests": 1,
            "all_bank_scope": "rank",
            "dependency_context": {"kind": "column_dependency_count", "dependency_count": 1, "dependency_id": 0},
            "row_policy": {"kind": "bounded_row_window_all_bank_semantic", "row_start": 0, "row_count": 1, "resolved_row": 0},
            "column_policy": {"kind": "dependency_column_round_robin", "column_start": 0, "resolved_column": 0},
            "datatype_metadata": {"datatype": "int8"},
        },
        {
            "schema_version": "v0.1",
            "record_id": "rec_0002",
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


def _request_type_ids(dram) -> dict[str, int]:
    return {name: index for index, name in enumerate(type(dram).supported_requests.keys())}


def test_structured_trace_frontend_replays_smoke_trace(tmp_path: Path):
    dram = create_dram(LPDDR5_PIM_CONFIG)
    layout = extract_dram_layout(dram)
    trace_path = tmp_path / "smoke.jsonl"
    _write_smoke_trace(trace_path)

    frontend = ramulator.frontend.StructuredWorkloadSurrogateTrace(
        clock_ratio=LPDDR5_PIM_CONFIG["frontend_clock_ratio"],
        path=str(trace_path),
        pim_request_type_id=2,
        **layout,
    )
    sim = ramulator.Simulation(frontend, _make_mem(dram))
    sim.run()

    stats = sim.stats["frontend"]
    assert stats["records_loaded"] == 5
    assert stats["host_records"] == 2
    assert stats["pim_records"] == 1
    assert stats["barriers_retired"] == 1
    assert stats["drains_retired"] == 1
    assert stats["host_requests_sent"] == 3
    assert stats["host_requests_completed"] == 3
    assert stats["pim_requests_sent"] == 4
    assert stats["pim_requests_completed"] == 4


def test_structured_trace_frontend_defaults_to_controller_bank_order(tmp_path: Path):
    dram = create_dram(LPDDR5_PIM_CONFIG)
    layout = extract_dram_layout(dram)
    trace_path = tmp_path / "default_order.jsonl"
    _write_default_order_trace(trace_path)

    frontend = ramulator.frontend.StructuredWorkloadSurrogateTrace(
        clock_ratio=LPDDR5_PIM_CONFIG["frontend_clock_ratio"],
        path=str(trace_path),
        pim_request_type_id=2,
        **layout,
    )
    sim = ramulator.Simulation(frontend, _make_mem(dram))
    sim.run()

    stats = sim.stats["frontend"]
    assert stats["records_loaded"] == 2
    assert stats["pim_requests_sent"] == 1
    assert stats["pim_requests_completed"] == 1
    assert stats["drains_retired"] == 1


def test_structured_trace_frontend_rejects_all_bank_records_without_request_ids(tmp_path: Path):
    dram = create_dram(LPDDR5_PIM_CONFIG)
    layout = extract_dram_layout(dram)
    trace_path = tmp_path / "bad.jsonl"
    _write_all_bank_trace(trace_path)

    frontend = ramulator.frontend.StructuredWorkloadSurrogateTrace(
        clock_ratio=LPDDR5_PIM_CONFIG["frontend_clock_ratio"],
        path=str(trace_path),
        pim_request_type_id=2,
        **layout,
    )
    try:
        ramulator.Simulation(frontend, _make_mem(dram))
    except RuntimeError as exc:
        assert "PIMLoadAll requires pim_load_request_type_id" in str(exc)
    else:
        raise AssertionError("Expected all-bank request-id rejection")


def test_structured_trace_frontend_replays_all_bank_records_when_enabled(tmp_path: Path):
    dram = create_dram(LPDDR5_PIM_CONFIG)
    layout = extract_dram_layout(dram)
    request_type_ids = _request_type_ids(dram)
    trace_path = tmp_path / "all_bank.jsonl"
    _write_all_bank_trace(trace_path)

    frontend = ramulator.frontend.StructuredWorkloadSurrogateTrace(
        clock_ratio=LPDDR5_PIM_CONFIG["frontend_clock_ratio"],
        path=str(trace_path),
        pim_request_type_id=request_type_ids["PIMCompute"],
        pim_load_request_type_id=request_type_ids["PIMLoadAll"],
        pim_compute_all_request_type_id=request_type_ids["PIMComputeAll"],
        **layout,
    )
    sim = ramulator.Simulation(frontend, _make_mem(dram))
    sim.run()

    stats = sim.stats["frontend"]
    assert stats["records_loaded"] == 3
    assert stats["pim_records"] == 2
    assert stats["pim_requests_sent"] == 2
    assert stats["pim_requests_completed"] == 2
    assert stats["drains_retired"] == 1


def test_structured_trace_frontend_rejects_missing_decode_only_claim(tmp_path: Path):
    provenance = _provenance()
    provenance["claim_boundary"] = [
        claim for claim in provenance["claim_boundary"] if claim != "decode-only-first"
    ]
    trace_path = tmp_path / "missing_decode_only_claim.jsonl"
    _write_single_record_trace(trace_path, _minimal_pim_record(provenance=provenance))

    _assert_frontend_rejects(trace_path, "decode-only-first")


def test_structured_trace_frontend_rejects_missing_non_claims(tmp_path: Path):
    provenance = _provenance()
    del provenance["non_claims"]
    trace_path = tmp_path / "missing_non_claims.jsonl"
    _write_single_record_trace(trace_path, _minimal_pim_record(provenance=provenance))

    _assert_frontend_rejects(trace_path, "non_claims")


def test_structured_trace_frontend_rejects_incomplete_non_claims(tmp_path: Path):
    provenance = _provenance()
    provenance["non_claims"] = [
        claim for claim in provenance["non_claims"] if claim != "not_all_bank_fidelity"
    ]
    trace_path = tmp_path / "incomplete_non_claims.jsonl"
    _write_single_record_trace(trace_path, _minimal_pim_record(provenance=provenance))

    _assert_frontend_rejects(trace_path, "not_all_bank_fidelity")


def test_structured_trace_frontend_rejects_pimcompute_missing_explicit_replay_fields(tmp_path: Path):
    record = _minimal_pim_record()
    del record["burst_length"]
    trace_path = tmp_path / "missing_burst_length.jsonl"
    _write_single_record_trace(trace_path, record)

    _assert_frontend_rejects(trace_path, "burst_length")


def test_structured_trace_frontend_rejects_missing_required_provenance_and_mapping_fields(tmp_path: Path):
    dram = create_dram(LPDDR5_PIM_CONFIG)
    layout = extract_dram_layout(dram)
    trace_path = tmp_path / "bad_contract.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "schema_version": "v0.1",
                "record_id": "rec_0000",
                "kind": "HostRead",
                "phase": "decode",
                "layer": "layer_00",
                "op": "attention_k_cache_read",
                "repeat": 1,
                "provenance": {"source_kind": "handwritten"},
                "mapping_policy": {"controller_bank_order": "controller"},
                "bytes": 64,
                "address_policy": {
                    "kind": "bounded_sequential_host_requests",
                    "base_byte": 0,
                    "stride_bytes": 64,
                    "count": 1,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    frontend = ramulator.frontend.StructuredWorkloadSurrogateTrace(
        clock_ratio=LPDDR5_PIM_CONFIG["frontend_clock_ratio"],
        path=str(trace_path),
        pim_request_type_id=2,
        **layout,
    )
    try:
        ramulator.Simulation(frontend, _make_mem(dram))
    except RuntimeError as exc:
        assert "tuple_manifest" in str(exc)
    else:
        raise AssertionError("Expected frozen-contract field rejection")
