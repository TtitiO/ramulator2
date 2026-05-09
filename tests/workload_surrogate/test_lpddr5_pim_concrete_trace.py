"""Tests for native LPDDR5-PIM concrete opcode replay."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

from tests.analysis.testcases.lpddr5_pim import CONFIG as LPDDR5_PIM_CONFIG
from tests.utils.dram import create_dram
from tests.utils.sim import extract_dram_layout

ramulator = importlib.import_module("ramulator")
schema_mod = importlib.import_module("ramulator.workload_surrogate.lpddr5_pim_concrete_trace")
generator_mod = importlib.import_module("ramulator.workload_surrogate.generate_lpddr5_pim_concrete")
decode_manifest_mod = importlib.import_module("ramulator.workload_surrogate.decode_only_manifest")
decode_generator_mod = importlib.import_module("ramulator.workload_surrogate.generate_decode_only")


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


def _request_type_ids(dram) -> dict[str, int]:
    return {name: index for index, name in enumerate(type(dram).supported_requests.keys())}


def _command_ids(dram) -> dict[str, int]:
    return {name: index for index, name in enumerate(type(dram).commands)}


def _frontend_kwargs(trace_path: Path, dram) -> dict:
    request_type_ids = _request_type_ids(dram)
    command_ids = _command_ids(dram)
    layout = extract_dram_layout(dram)
    return {
        "clock_ratio": LPDDR5_PIM_CONFIG["frontend_clock_ratio"],
        "path": str(trace_path),
        "pim_compute_request_type_id": request_type_ids["PIMCompute"],
        "pim_load_all_request_type_id": request_type_ids["PIMLoadAll"],
        "pim_compute_all_request_type_id": request_type_ids["PIMComputeAll"],
        "sb_command_id": command_ids["SB"],
        "hab_command_id": command_ids["HAB"],
        "hab_pim_command_id": command_ids["HAB_PIM"],
        "addr_vec_size": layout["addr_vec_size"],
    }


def _frontend(trace_path: Path, dram):
    return ramulator.frontend.LPDDR5PIMConcreteTrace(**_frontend_kwargs(trace_path, dram))


def test_concrete_schema_accepts_minimal_legal_sequence():
    records = generator_mod.generate_minimal_concrete_records()
    schema_mod.validate_sequence(records)
    assert {record["opcode"] for record in records} == {"SB", "HAB", "HAB_PIM", "PIM_BCAST", "PIM_MAC", "PIM_MAC_AB"}


def test_concrete_schema_rejects_raw_attacc_opcode():
    record = generator_mod.generate_minimal_concrete_records()[0]
    record["opcode"] = "PIM_MV_BA"
    try:
        schema_mod.validate_record(record)
    except ValueError as exc:
        assert "Raw AttAcc opcode" in str(exc)
    else:
        raise AssertionError("Expected raw AttAcc opcode rejection")


def test_concrete_schema_rejects_illegal_mode_order():
    records = generator_mod.generate_minimal_concrete_records()
    records = [record for record in records if record["opcode"] != "PIM_BCAST"]
    try:
        schema_mod.validate_sequence(records)
    except ValueError as exc:
        assert "HAB_PIM requires" in str(exc)
    else:
        raise AssertionError("Expected illegal mode ordering rejection")


def test_concrete_schema_rejects_non_integer_repeat():
    record = generator_mod.generate_minimal_concrete_records()[0]
    record["repeat"] = "2"
    try:
        schema_mod.validate_record(record)
    except ValueError as exc:
        assert "repeat must be an integer" in str(exc)
    else:
        raise AssertionError("Expected non-integer repeat rejection")


def test_concrete_generator_cli_writes_bounded_artifacts(tmp_path: Path):
    output_dir = tmp_path / "concrete"
    result = subprocess.run(
        [sys.executable, "-m", "ramulator.workload_surrogate.generate_lpddr5_pim_concrete", "--output-dir", str(output_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    trace_path = output_dir / "concrete_opcode_trace.jsonl"
    summary_path = output_dir / "provenance_summary.json"
    assert trace_path.exists()
    assert summary_path.exists()
    assert f"Generated: {trace_path}" in result.stdout
    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    schema_mod.validate_sequence(records)
    assert summary["total_expanded_records"] == schema_mod.expanded_record_count(records)
    assert "not_raw_attacc_schema" in json.dumps(summary, sort_keys=True)


def test_concrete_generator_accepts_bounded_workload_parameters():
    records = generator_mod.generate_minimal_concrete_records(
        single_bank_mac_repeats=3,
        post_all_bank_mac_repeats=2,
        include_all_bank_sequence=False,
    )
    schema_mod.validate_sequence(records)
    assert [record["opcode"] for record in records] == ["SB", "PIM_MAC", "PIM_MAC"]
    assert schema_mod.expanded_record_count(records) == 6


def test_concrete_generator_lowers_semantic_records_with_source_provenance():
    semantic = [
        {
            "schema_version": "v0.1",
            "record_id": "rec_sem_0000",
            "kind": "PIMCompute",
            "phase": "decode",
            "layer": "layer_00",
            "op": "q_projection_gemv",
            "repeat": 1,
            "provenance": {"tuple_manifest": "unit_semantic"},
            "mapping_policy": {"controller_bank_order": "controller"},
            "num_requests": 4,
            "bank_sequence": [0, 1],
            "dependency_context": {"dependency_count": 2, "dependency_id": 0},
            "row_policy": {"row_start": 3, "row_count": 2, "resolved_row": 3},
            "column_policy": {"column_start": 5, "resolved_column": 5},
            "datatype_metadata": {"datatype": "int8"},
            "burst_length": 1,
        }
    ]
    records = generator_mod.lower_semantic_records_to_concrete(semantic)
    schema_mod.validate_sequence(records)
    assert [record["opcode"] for record in records] == ["SB", "PIM_MAC", "PIM_MAC"]
    assert [record["addr_vec"][3] for record in records[1:]] == [0, 1]
    assert [record["repeat"] for record in records[1:]] == [2, 2]
    assert records[1]["addr_vec"][4] == 3
    assert records[1]["addr_vec"][5] == 5
    assert records[1]["provenance"]["semantic_source"]["record_id"] == "rec_sem_0000"
    assert records[1]["provenance"]["semantic_source"]["op"] == "q_projection_gemv"


def test_concrete_generator_is_semantic_parameter_sensitive():
    manifest_a = decode_manifest_mod.get_decode_only_manifest()
    manifest_a.update({"num_layers": 1, "hidden_size": 64, "ffn_hidden_size": 64, "pim_compute_operator_classes": ["q_projection_gemv"]})
    manifest_a["pim_operator_request_widths"] = {"q_projection_gemv": "hidden_size"}
    manifest_a["ramulator_visible_defaults"].update({"bank_sequence": [0, 1], "burst_length": 1, "row_count": 2, "dependency_count": 2})

    manifest_b = decode_manifest_mod.get_decode_only_manifest()
    manifest_b.update({"num_layers": 1, "hidden_size": 128, "ffn_hidden_size": 64, "pim_compute_operator_classes": ["q_projection_gemv"]})
    manifest_b["pim_operator_request_widths"] = {"q_projection_gemv": "hidden_size"}
    manifest_b["ramulator_visible_defaults"].update({"bank_sequence": [0, 1, 2, 3], "burst_length": 1, "row_count": 2, "dependency_count": 2})

    concrete_a = generator_mod.lower_semantic_records_to_concrete(decode_generator_mod.generate_decode_only_records(manifest_a))
    concrete_b = generator_mod.lower_semantic_records_to_concrete(decode_generator_mod.generate_decode_only_records(manifest_b))

    schema_mod.validate_sequence(concrete_a)
    schema_mod.validate_sequence(concrete_b)
    assert [record["opcode"] for record in concrete_a].count("PIM_MAC") == 2
    assert [record["opcode"] for record in concrete_b].count("PIM_MAC") == 4
    assert [record["addr_vec"][3] for record in concrete_a if record["opcode"] == "PIM_MAC"] == [0, 1]
    assert [record["addr_vec"][3] for record in concrete_b if record["opcode"] == "PIM_MAC"] == [0, 1, 2, 3]


def test_concrete_generator_lowers_all_bank_semantic_records():
    manifest = decode_manifest_mod.get_decode_only_manifest()
    manifest.update(
        {
            "enable_all_bank_records": True,
            "all_bank_records_approval": "P2.9b",
            "num_layers": 1,
            "hidden_size": 64,
            "ffn_hidden_size": 64,
            "past_len": 4,
            "pim_compute_operator_classes": ["q_projection_gemv"],
            "pim_operator_request_widths": {"q_projection_gemv": "hidden_size"},
        }
    )
    records = generator_mod.lower_semantic_records_to_concrete(decode_generator_mod.generate_decode_only_records(manifest))
    opcodes = [record["opcode"] for record in records]
    schema_mod.validate_sequence(records)
    assert "PIM_BCAST" in opcodes
    assert "HAB_PIM" in opcodes
    assert "PIM_MAC_AB" in opcodes
    assert "SB" in opcodes


def test_concrete_lowering_splits_all_bank_load_repeats_above_schema_limit():
    semantic = [
        {
            "schema_version": "v0.1",
            "record_id": "rec_big_bcast",
            "kind": "PIMDataMove",
            "phase": "decode",
            "layer": "layer_00",
            "op": "large_weight_residency",
            "repeat": 1,
            "provenance": {"tuple_manifest": "unit_semantic"},
            "mapping_policy": {"controller_bank_order": "controller"},
            "num_requests": schema_mod.MAX_REPEAT + 7,
            "movement_policy": {"movement_kind": "broadcast_or_accounted_tile_load"},
            "dependency_context": {"dependency_count": 1, "dependency_id": 0},
            "row_policy": {"row_start": 0, "row_count": 1, "resolved_row": 0},
            "column_policy": {"column_start": 0, "resolved_column": 0},
            "datatype_metadata": {"datatype": "int8"},
            "burst_length": 1,
        }
    ]
    records = generator_mod.lower_semantic_records_to_concrete(semantic)
    schema_mod.validate_sequence(records)
    bcasts = [record for record in records if record["opcode"] == "PIM_BCAST"]
    assert [record["repeat"] for record in bcasts] == [schema_mod.MAX_REPEAT, 7]


def test_concrete_frontend_replays_native_opcode_trace(tmp_path: Path):
    dram = create_dram(LPDDR5_PIM_CONFIG)
    trace_path = tmp_path / "concrete_opcode_trace.jsonl"
    schema_mod.write_jsonl(generator_mod.generate_minimal_concrete_records(), trace_path)
    sim = ramulator.Simulation(_frontend(trace_path, dram), _make_mem(dram))
    sim.run()
    stats = sim.stats["frontend"]
    assert stats["records_loaded"] == 8
    assert stats["records_expanded"] == 9
    assert stats["opcode_requests_sent"] == 9
    assert stats["opcode_requests_completed"] == 9
    ctrl_stats = sim.stats["memory_system"]["controller"]
    assert ctrl_stats["num_pim_reqs_served"] >= 3
    assert ctrl_stats["num_pim_ab_reqs_served"] == 1


def test_concrete_frontend_replays_semantic_lowered_opcode_trace(tmp_path: Path):
    dram = create_dram(LPDDR5_PIM_CONFIG)
    manifest = decode_manifest_mod.get_decode_only_manifest()
    manifest.update({"num_layers": 1, "hidden_size": 64, "ffn_hidden_size": 64, "past_len": 4, "pim_compute_operator_classes": ["q_projection_gemv"]})
    manifest["pim_operator_request_widths"] = {"q_projection_gemv": "hidden_size"}
    manifest["ramulator_visible_defaults"].update({"bank_sequence": [0, 1], "burst_length": 1, "row_count": 2, "dependency_count": 2})
    records = generator_mod.lower_semantic_records_to_concrete(decode_generator_mod.generate_decode_only_records(manifest))
    trace_path = tmp_path / "semantic_lowered_concrete_opcode_trace.jsonl"
    schema_mod.write_jsonl(records, trace_path)

    sim = ramulator.Simulation(_frontend(trace_path, dram), _make_mem(dram))
    sim.run()
    stats = sim.stats["frontend"]
    assert stats["opcode_requests_sent"] == len(records)
    assert stats["opcode_requests_completed"] == len(records)
    assert stats["pim_mac_records"] == 2
    assert sim.stats["memory_system"]["controller"]["num_pim_reqs_served"] >= 2


def test_concrete_generator_cli_lowers_semantic_trace(tmp_path: Path):
    semantic_dir = tmp_path / "semantic"
    concrete_dir = tmp_path / "concrete"
    semantic_trace, _ = decode_generator_mod.generate_decode_only_artifacts(semantic_dir)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ramulator.workload_surrogate.generate_lpddr5_pim_concrete",
            "--semantic-trace",
            str(semantic_trace),
            "--output-dir",
            str(concrete_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    trace_path = concrete_dir / "concrete_opcode_trace.jsonl"
    summary_path = concrete_dir / "provenance_summary.json"
    assert f"Generated: {trace_path}" in result.stdout
    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    schema_mod.validate_sequence(records)
    assert summary["lowering_mode"] == "phase2-semantic-jsonl-to-native-lpddr5-pim-opcodes"
    assert summary["semantic_sources"]


def test_concrete_frontend_rejects_invalid_provenance(tmp_path: Path):
    dram = create_dram(LPDDR5_PIM_CONFIG)
    records = generator_mod.generate_minimal_concrete_records()
    records[0]["provenance"]["non_claims"] = [claim for claim in records[0]["provenance"]["non_claims"] if claim != "not_raw_attacc_schema"]
    trace_path = tmp_path / "bad_provenance.jsonl"
    trace_path.write_text("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n", encoding="utf-8")
    try:
        ramulator.Simulation(_frontend(trace_path, dram), _make_mem(dram))
    except RuntimeError as exc:
        assert "not_raw_attacc_schema" in str(exc)
    else:
        raise AssertionError("Expected concrete frontend provenance rejection")


def test_concrete_frontend_rejects_invalid_mode_order(tmp_path: Path):
    dram = create_dram(LPDDR5_PIM_CONFIG)
    records = [record for record in generator_mod.generate_minimal_concrete_records() if record["opcode"] != "PIM_BCAST"]
    trace_path = tmp_path / "bad_order.jsonl"
    trace_path.write_text("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n", encoding="utf-8")
    try:
        ramulator.Simulation(_frontend(trace_path, dram), _make_mem(dram))
    except RuntimeError as exc:
        assert "HAB_PIM requires" in str(exc)
    else:
        raise AssertionError("Expected concrete frontend mode-order rejection")


def test_concrete_frontend_rejects_repeat_above_limit(tmp_path: Path):
    dram = create_dram(LPDDR5_PIM_CONFIG)
    records = generator_mod.generate_minimal_concrete_records()
    records[0]["repeat"] = 2
    trace_path = tmp_path / "repeat_limit.jsonl"
    trace_path.write_text("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n", encoding="utf-8")
    try:
        ramulator.Simulation(
            ramulator.frontend.LPDDR5PIMConcreteTrace(
                **_frontend_kwargs(trace_path, dram),
                max_repeat=1,
            ),
            _make_mem(dram),
        )
    except RuntimeError as exc:
        assert "max_repeat" in str(exc)
    else:
        raise AssertionError("Expected concrete frontend repeat cap rejection")


def test_semantic_frontend_wrapper_remains_available():
    assert hasattr(ramulator.frontend, "StructuredWorkloadSurrogateTrace")
    assert hasattr(ramulator.frontend, "LPDDR5PIMConcreteTrace")
