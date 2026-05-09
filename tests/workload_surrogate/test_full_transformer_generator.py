"""Tests for Phase 4 full-transformer semantic dataflow generation."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.analysis.testcases.lpddr5_pim import CONFIG as LPDDR5_PIM_CONFIG
from tests.utils.dram import create_dram
from tests.utils.sim import extract_dram_layout

ramulator = importlib.import_module("ramulator")

generator_mod = importlib.import_module("ramulator.workload_surrogate.generate_full_transformer")
lowering_mod = importlib.import_module("ramulator.workload_surrogate.generate_lpddr5_pim_concrete")
concrete_schema_mod = importlib.import_module("ramulator.workload_surrogate.lpddr5_pim_concrete_trace")
trace_mod = importlib.import_module("ramulator.workload_surrogate.structured_trace")


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


def _frontend(trace_path: Path, dram):
    request_type_ids = {name: index for index, name in enumerate(type(dram).supported_requests.keys())}
    command_ids = {name: index for index, name in enumerate(type(dram).commands)}
    layout = extract_dram_layout(dram)
    return ramulator.frontend.LPDDR5PIMConcreteTrace(
        clock_ratio=LPDDR5_PIM_CONFIG["frontend_clock_ratio"],
        path=str(trace_path),
        pim_compute_request_type_id=request_type_ids["PIMCompute"],
        pim_load_all_request_type_id=request_type_ids["PIMLoadAll"],
        pim_compute_all_request_type_id=request_type_ids["PIMComputeAll"],
        sb_command_id=command_ids["SB"],
        hab_command_id=command_ids["HAB"],
        hab_pim_command_id=command_ids["HAB_PIM"],
        addr_vec_size=layout["addr_vec_size"],
    )


def _p2_frontend(trace_path: Path, dram):
    layout = extract_dram_layout(dram)
    return ramulator.frontend.StructuredWorkloadSurrogateTrace(
        clock_ratio=LPDDR5_PIM_CONFIG["frontend_clock_ratio"],
        path=str(trace_path),
        pim_request_type_id=2,
        **layout,
    )


def _p2_replay_provenance() -> dict:
    return {
        "source_kind": "handwritten",
        "tuple_manifest": "p4_offline_boundary_unit",
        "literature_anchor": ["manual:p4-offline-boundary"],
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


def test_p4_schema_accepts_full_transformer_record_kinds():
    assert {
        "PIMDataMove",
        "AttentionScore",
        "AttentionSoftmax",
        "AttentionContext",
        "FFNProjection",
        "PIMElementwise",
        "MoERouter",
        "MoETopK",
        "MoEDispatch",
        "MoEExpertFFN",
        "MoECombine",
        "PIMOperandResidency",
        "PIMOperandReuse",
    } <= trace_mod.SEMANTIC_RECORD_KINDS


def test_p4_schema_rejects_missing_tensor_io():
    record = generator_mod.generate_attention_records()[0]
    del record["tensor_io"]
    try:
        trace_mod.validate_record(record)
    except ValueError as exc:
        assert "tensor_io" in str(exc)
    else:
        raise AssertionError("Expected missing tensor_io rejection")


def test_p4_schema_requires_offline_ir_metadata():
    record = generator_mod.generate_attention_records()[0]
    assert record["operator_context"].get("record_family") == "p4_offline_transformer_dataflow_ir"
    del record["operator_context"]["record_family"]
    try:
        trace_mod.validate_record(record)
    except ValueError as exc:
        assert "record_family" in str(exc)
    else:
        raise AssertionError("Expected missing P4 record_family rejection")


def test_p4_schema_enforces_required_claim_boundary_and_non_claims():
    record = generator_mod.generate_attention_records()[0]
    record["provenance"]["claim_boundary"] = [claim for claim in record["provenance"]["claim_boundary"] if claim != "operator-internal-dataflow-first"]
    try:
        trace_mod.validate_record(record)
    except ValueError as exc:
        assert "operator-internal-dataflow-first" in str(exc)
    else:
        raise AssertionError("Expected P4 claim-boundary rejection")


def test_p4_schema_rejects_generated_record_missing_required_non_claim():
    record = generator_mod.generate_attention_records()[0]
    record["provenance"]["non_claims"] = [claim for claim in record["provenance"]["non_claims"] if claim != "not_vllm_replay"]
    try:
        trace_mod.validate_record(record)
    except ValueError as exc:
        assert "not_vllm_replay" in str(exc)
    else:
        raise AssertionError("Expected P4 non-claim rejection")


def test_p4_semantic_records_are_offline_only_not_structured_frontend_replay(tmp_path: Path):
    dram = create_dram(LPDDR5_PIM_CONFIG)
    record = [r for r in generator_mod.generate_attention_records() if r["kind"] == "AttentionScore"][0]
    record["provenance"] = _p2_replay_provenance()
    trace_path = tmp_path / "p4_semantic_ir.jsonl"
    trace_path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")

    try:
        ramulator.Simulation(_p2_frontend(trace_path, dram), _make_mem(dram))
    except RuntimeError as exc:
        assert "unsupported kind 'AttentionScore'" in str(exc)
    else:
        raise AssertionError("Expected structured frontend to reject P4 offline IR records")


@pytest.mark.parametrize(
    "record_kind,record_factory",
    [
        ("PIMDataMove", lambda: [r for r in generator_mod.generate_attention_records() if r["kind"] == "PIMDataMove"][0]),
        ("AttentionSoftmax", lambda: [r for r in generator_mod.generate_attention_records() if r["kind"] == "AttentionSoftmax"][0]),
        ("AttentionContext", lambda: [r for r in generator_mod.generate_attention_records() if r["kind"] == "AttentionContext"][0]),
        ("FFNProjection", lambda: [r for r in generator_mod.generate_ffn_records() if r["kind"] == "FFNProjection"][0]),
        ("PIMElementwise", lambda: [r for r in generator_mod.generate_ffn_records() if r["kind"] == "PIMElementwise"][0]),
        ("MoERouter", lambda: [r for r in generator_mod.generate_moe_records() if r["kind"] == "MoERouter"][0]),
        ("MoETopK", lambda: [r for r in generator_mod.generate_moe_records() if r["kind"] == "MoETopK"][0]),
        ("MoEDispatch", lambda: [r for r in generator_mod.generate_moe_records() if r["kind"] == "MoEDispatch"][0]),
        ("MoEExpertFFN", lambda: [r for r in generator_mod.generate_moe_records() if r["kind"] == "MoEExpertFFN"][0]),
        ("MoECombine", lambda: [r for r in generator_mod.generate_moe_records() if r["kind"] == "MoECombine"][0]),
        ("PIMOperandResidency", lambda: [r for r in generator_mod.generate_ffn_records() if r["kind"] == "PIMOperandResidency"][0]),
        ("PIMOperandReuse", lambda: [r for r in generator_mod.generate_ffn_records() if r["kind"] == "PIMOperandReuse"][0]),
    ],
)
def test_all_p4_semantic_record_kinds_are_offline_only_not_structured_frontend_replay(tmp_path: Path, record_kind: str, record_factory):
    dram = create_dram(LPDDR5_PIM_CONFIG)
    record = record_factory()
    assert record["kind"] == record_kind
    trace_mod.validate_record(record)
    record = dict(record)
    record["provenance"] = _p2_replay_provenance()
    trace_path = tmp_path / f"{record_kind}.jsonl"
    trace_path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    try:
        ramulator.Simulation(_p2_frontend(trace_path, dram), _make_mem(dram))
    except RuntimeError as exc:
        assert f"unsupported kind '{record_kind}'" in str(exc)
    else:
        raise AssertionError(f"Expected structured frontend to reject P4 offline IR record kind {record_kind}")


def test_attention_generator_emits_score_softmax_context_ordering():
    manifest = generator_mod.get_tiny_attention_manifest()
    manifest.update({"num_layers": 1, "num_heads": 1, "past_len": 32, "score_tile_tokens": 32})
    records = generator_mod.generate_attention_records(manifest)

    assert [record["kind"] for record in records] == [
        "PIMDataMove",
        "AttentionScore",
        "AttentionSoftmax",
        "PIMDataMove",
        "AttentionContext",
    ]
    score = records[1]
    softmax = records[2]
    context = records[4]
    assert score["record_id"] in softmax["logical_dependencies"]
    assert softmax["record_id"] in context["logical_dependencies"]
    assert score["tensor_io"]["outputs"] == softmax["tensor_io"]["inputs"]
    assert context["operator_context"]["stage"] == "context"


def test_attention_generator_is_parameter_sensitive():
    manifest_a = generator_mod.get_tiny_attention_manifest()
    manifest_a.update({"num_heads": 1, "past_len": 32, "score_tile_tokens": 32, "head_dim": 32})
    records_a = generator_mod.generate_attention_records(manifest_a)

    manifest_b = generator_mod.get_tiny_attention_manifest()
    manifest_b.update({"num_heads": 2, "past_len": 64, "score_tile_tokens": 32, "head_dim": 64})
    records_b = generator_mod.generate_attention_records(manifest_b)

    assert len(records_a) == 5
    assert len(records_b) == 20
    mac_a = sum(record["num_requests"] for record in records_a if record["kind"] in {"AttentionScore", "AttentionContext"})
    mac_b = sum(record["num_requests"] for record in records_b if record["kind"] in {"AttentionScore", "AttentionContext"})
    assert mac_b > mac_a


def test_attention_scales_with_num_layers():
    manifest_one = generator_mod.get_tiny_attention_manifest()
    manifest_one.update({"num_layers": 1, "num_heads": 1, "past_len": 32, "score_tile_tokens": 32})
    manifest_two = generator_mod.get_tiny_attention_manifest()
    manifest_two.update({"num_layers": 2, "num_heads": 1, "past_len": 32, "score_tile_tokens": 32})
    assert len(generator_mod.generate_attention_records(manifest_two)) == 2 * len(generator_mod.generate_attention_records(manifest_one))


def test_attention_with_fp16_datatype_uses_16_lanes():
    manifest = generator_mod.get_tiny_attention_manifest()
    manifest.update({"num_heads": 1, "past_len": 32, "score_tile_tokens": 32, "head_dim": 32, "datatype": "fp16"})
    score = [record for record in generator_mod.generate_attention_records(manifest) if record["kind"] == "AttentionScore"][0]
    assert score["num_requests"] == 64


def test_attention_generator_uses_global_softmax_across_score_tiles():
    manifest = generator_mod.get_tiny_attention_manifest()
    manifest.update({"num_heads": 1, "past_len": 64, "score_tile_tokens": 32})
    records = generator_mod.generate_attention_records(manifest)
    score_ids = [record["record_id"] for record in records if record["kind"] == "AttentionScore"]
    softmax = [record for record in records if record["kind"] == "AttentionSoftmax"][0]
    context_records = [record for record in records if record["kind"] == "AttentionContext"]

    assert softmax["logical_dependencies"] == score_ids
    assert len(softmax["tensor_io"]["inputs"]) == 2
    assert all(softmax["record_id"] in record["logical_dependencies"] for record in context_records)


def test_attention_generator_emits_final_context_reduction_for_multi_tile_heads():
    manifest = generator_mod.get_tiny_attention_manifest()
    manifest.update({"num_layers": 1, "num_heads": 1, "past_len": 64, "score_tile_tokens": 32})
    records = generator_mod.generate_attention_records(manifest)
    context_ids = [record["record_id"] for record in records if record["kind"] == "AttentionContext"]
    reductions = [record for record in records if record["kind"] == "PIMElementwise" and record["op"] == "attention_context_reduction_accounting"]

    assert len(reductions) == 1
    reduction = reductions[0]
    assert reduction["logical_dependencies"] == context_ids
    assert reduction["operator_context"]["stage"] == "context_reduction_accounting"
    assert reduction["accounting_metadata"]["lowering"] == "not_lowered_to_native_opcode_in_p4_2"


def test_attention_context_shape_models_probability_times_v():
    manifest = generator_mod.get_tiny_attention_manifest()
    manifest.update({"num_heads": 1, "past_len": 32, "score_tile_tokens": 32, "head_dim": 64})
    records = generator_mod.generate_attention_records(manifest)
    context = [record for record in records if record["kind"] == "AttentionContext"][0]
    assert context["compute_shape"]["n"] == 64
    assert context["compute_shape"]["k"] == 32


def test_attention_manifest_requires_guardrail_non_claims():
    manifest = generator_mod.get_tiny_attention_manifest()
    manifest["non_claims"] = [claim for claim in manifest["non_claims"] if claim != "not_raw_attacc_schema"]
    try:
        generator_mod.generate_attention_records(manifest)
    except ValueError as exc:
        assert "not_raw_attacc_schema" in str(exc)
    else:
        raise AssertionError("Expected required P4 non-claim rejection")


def test_attention_manifest_rejects_unknown_datatype():
    manifest = generator_mod.get_tiny_attention_manifest()
    manifest["datatype"] = "fp8"
    try:
        generator_mod.generate_attention_records(manifest)
    except ValueError as exc:
        assert "Unsupported attention datatype" in str(exc)
    else:
        raise AssertionError("Expected unknown datatype rejection")


def test_attention_manifest_rejects_invalid_ramulator_defaults():
    for field in ("burst_length", "row_count", "dependency_count"):
        manifest = generator_mod.get_tiny_attention_manifest()
        manifest["ramulator_visible_defaults"][field] = 0
        try:
            generator_mod.generate_attention_records(manifest)
        except ValueError as exc:
            assert field in str(exc)
            assert "positive" in str(exc)
        else:
            raise AssertionError(f"Expected invalid default rejection for {field}")


def test_attention_first_slice_rejects_distinct_context_tile_size():
    manifest = generator_mod.get_tiny_attention_manifest()
    manifest["context_tile_tokens"] = manifest["score_tile_tokens"] // 2
    try:
        generator_mod.generate_attention_records(manifest)
    except ValueError as exc:
        assert "context_tile_tokens == score_tile_tokens" in str(exc)
    else:
        raise AssertionError("Expected unsupported distinct context tile size rejection")


def test_attention_lowering_uses_only_native_lpddr5_pim_opcodes_and_skips_softmax():
    manifest = generator_mod.get_tiny_attention_manifest()
    manifest.update({"num_heads": 1, "past_len": 32, "score_tile_tokens": 32, "head_dim": 32})
    semantic = generator_mod.generate_attention_records(manifest)
    concrete = lowering_mod.lower_semantic_records_to_concrete(semantic, manifest_name=manifest["manifest_name"])
    concrete_schema_mod.validate_sequence(concrete)

    opcodes = [record["opcode"] for record in concrete]
    assert set(opcodes) <= concrete_schema_mod.CONCRETE_OPCODES
    assert "PIM_BCAST" in opcodes
    assert "PIM_MAC" in opcodes
    assert "AttentionSoftmax" not in {record["provenance"]["semantic_source"]["kind"] for record in concrete}
    assert not {"PIM_WR_GB", "PIM_MV_BA", "PIM_SFM"} & set(opcodes)


def test_attention_lowering_rejects_unsupported_data_movement_kind():
    semantic = generator_mod.generate_attention_records()[0:1]
    semantic[0]["movement_policy"]["movement_kind"] = "semantic_only_unmodeled_move"
    try:
        lowering_mod.lower_semantic_records_to_concrete(semantic)
    except ValueError as exc:
        assert "not supported by native PIM_BCAST lowering" in str(exc)
    else:
        raise AssertionError("Expected unsupported PIMDataMove lowering rejection")


def test_concrete_lowering_honors_semantic_repeat_for_data_movement_records():
    semantic = generator_mod.generate_attention_records()[0:1]
    semantic[0]["repeat"] = 3
    concrete = lowering_mod.lower_semantic_records_to_concrete(semantic)
    bcasts = [record for record in concrete if record["opcode"] == "PIM_BCAST"]
    assert len(bcasts) == 1
    assert bcasts[0]["repeat"] == 3


def test_concrete_lowering_matches_bank_distribution_before_dependency_phase():
    semantic = [
        {
            "schema_version": "v0.1",
            "record_id": "rec_sem_0000",
            "kind": "AttentionScore",
            "phase": "decode",
            "layer": "layer_00",
            "op": "attention_score_gemv",
            "repeat": 1,
            "provenance": {"tuple_manifest": "unit_semantic"},
            "mapping_policy": {"controller_bank_order": "controller"},
            "tensor_io": {"inputs": ["q", "k"], "outputs": ["score"]},
            "logical_dependencies": [],
            "operator_context": {"operator_family": "attention", "stage": "score"},
            "residency": {"q": "pim", "k": "pim", "score": "pim"},
            "compute_shape": {"m": 1, "n": 4, "k": 32, "output_elements": 4, "datatype": "int8"},
            "num_requests": 4,
            "bank_sequence": [0, 1],
            "dependency_context": {"dependency_count": 4, "dependency_id": 1},
            "row_policy": {"row_start": 10, "row_count": 8, "resolved_row": 10},
            "column_policy": {"column_start": 20, "resolved_column": 21},
            "datatype_metadata": {"datatype": "int8"},
            "burst_length": 1,
        }
    ]
    concrete = lowering_mod.lower_semantic_records_to_concrete(semantic)
    macs = [record for record in concrete if record["opcode"] == "PIM_MAC"]
    assert [record["addr_vec"][3] for record in macs] == [0, 1]
    assert [record["addr_vec"][5] for record in macs] == [21, 21]
    assert [record["repeat"] for record in macs] == [2, 2]


def test_concrete_lowering_honors_semantic_repeat_for_compute_records():
    semantic = [
        {
            "schema_version": "v0.1",
            "record_id": "rec_sem_0000",
            "kind": "AttentionScore",
            "phase": "decode",
            "layer": "layer_00",
            "op": "attention_score_gemv",
            "repeat": 3,
            "provenance": {"tuple_manifest": "unit_semantic"},
            "mapping_policy": {"controller_bank_order": "controller"},
            "tensor_io": {"inputs": ["q", "k"], "outputs": ["score"]},
            "logical_dependencies": [],
            "operator_context": {"operator_family": "attention", "stage": "score"},
            "residency": {"q": "pim", "k": "pim", "score": "pim"},
            "compute_shape": {"m": 1, "n": 2, "k": 32, "output_elements": 2, "datatype": "int8"},
            "num_requests": 2,
            "bank_sequence": [0, 1],
            "dependency_context": {"dependency_count": 2, "dependency_id": 0},
            "row_policy": {"row_start": 0, "row_count": 2, "resolved_row": 0},
            "column_policy": {"column_start": 0, "resolved_column": 0},
            "datatype_metadata": {"datatype": "int8"},
            "burst_length": 1,
        }
    ]
    concrete = lowering_mod.lower_semantic_records_to_concrete(semantic)
    macs = [record for record in concrete if record["opcode"] == "PIM_MAC"]
    assert len(macs) == 2
    assert sum(record["repeat"] for record in macs) == 6  # 3 repeats per bank × 2 banks


def test_concrete_lowering_decomposes_flat_bank_like_structured_frontend():
    semantic = [
        {
            "schema_version": "v0.1",
            "record_id": "rec_sem_0000",
            "kind": "AttentionScore",
            "phase": "decode",
            "layer": "layer_00",
            "op": "attention_score_gemv",
            "repeat": 1,
            "provenance": {"tuple_manifest": "unit_semantic"},
            "mapping_policy": {"controller_bank_order": "frontend"},
            "tensor_io": {"inputs": ["q", "k"], "outputs": ["score"]},
            "logical_dependencies": [],
            "operator_context": {"operator_family": "attention", "stage": "score"},
            "residency": {"q": "pim", "k": "pim", "score": "pim"},
            "compute_shape": {"m": 1, "n": 1, "k": 32, "output_elements": 1, "datatype": "int8"},
            "num_requests": 1,
            "bank_sequence": [5],
            "dependency_context": {"dependency_count": 1, "dependency_id": 0},
            "row_policy": {"row_start": 0, "row_count": 1, "resolved_row": 0},
            "column_policy": {"column_start": 0, "resolved_column": 0},
            "datatype_metadata": {"datatype": "int8"},
            "burst_length": 1,
        }
    ]
    concrete = lowering_mod.lower_semantic_records_to_concrete(
        semantic,
        addr_vec_size=6,
        bank_positions=[1, 2, 3],
        bank_counts=[2, 4, 4],
        row_level=4,
        col_level=5,
    )
    mac = [record for record in concrete if record["opcode"] == "PIM_MAC"][0]
    assert [mac["addr_vec"][level] for level in [1, 2, 3]] == [0, 1, 1]


def test_concrete_lowering_controller_order_decomposes_unsorted_bank_positions():
    semantic = [
        {
            "schema_version": "v0.1",
            "record_id": "rec_sem_0000",
            "kind": "AttentionScore",
            "phase": "decode",
            "layer": "layer_00",
            "op": "attention_score_gemv",
            "repeat": 1,
            "provenance": {"tuple_manifest": "unit_semantic"},
            "mapping_policy": {"controller_bank_order": "controller"},
            "tensor_io": {"inputs": ["q", "k"], "outputs": ["score"]},
            "logical_dependencies": [],
            "operator_context": {"operator_family": "attention", "stage": "score", "record_family": "p4_offline_transformer_dataflow_ir"},
            "residency": {"q": "pim", "k": "pim", "score": "pim"},
            "compute_shape": {"m": 1, "n": 1, "k": 32, "output_elements": 1, "datatype": "int8"},
            "num_requests": 1,
            "bank_sequence": [5],
            "dependency_context": {"dependency_count": 1, "dependency_id": 0},
            "row_policy": {"row_start": 0, "row_count": 1, "resolved_row": 0},
            "column_policy": {"column_start": 0, "resolved_column": 0},
            "datatype_metadata": {"datatype": "int8"},
            "burst_length": 1,
        }
    ]
    concrete = lowering_mod.lower_semantic_records_to_concrete(
        semantic,
        addr_vec_size=6,
        bank_positions=[3, 1, 2],
        bank_counts=[4, 2, 4],
        row_level=4,
        col_level=5,
    )
    mac = [record for record in concrete if record["opcode"] == "PIM_MAC"][0]
    assert [mac["addr_vec"][level] for level in [1, 2, 3]] == [0, 1, 1]


def test_ffn_swiglu_generator_emits_true_dag_ordering():
    manifest = generator_mod.get_tiny_ffn_manifest()
    records = generator_mod.generate_ffn_records(manifest)
    ffn_ops = [record["op"] for record in records if record["op"].startswith("ffn_")]
    assert ffn_ops == [
        "ffn_hidden_activation_tile_setup",
        "ffn_hidden_activation_tile_setup",
        "ffn_up_weight_residency",
        "ffn_gate_weight_residency",
        "ffn_down_weight_residency",
        "ffn_hidden_reuse_for_up_gate",
        "ffn_up_projection",
        "ffn_gate_projection",
        "ffn_gate_activation_accounting",
        "ffn_gated_multiply_accounting",
        "ffn_intermediate_bank_local_residency",
        "ffn_down_projection",
    ]
    by_id = {record["record_id"]: record for record in records}
    by_op_single = {}
    multi = {}
    for record in records:
        op = record["op"]
        if op in by_op_single:
            multi.setdefault(op, [by_op_single.pop(op)])
            multi[op].append(record)
        else:
            by_op_single[op] = record
    assert len(multi.get("ffn_hidden_activation_tile_setup", [])) == 2
    assert by_op_single["ffn_gate_projection"]["record_id"] in by_op_single["ffn_gate_activation_accounting"]["logical_dependencies"]
    assert by_op_single["ffn_gate_activation_accounting"]["record_id"] in by_op_single["ffn_gated_multiply_accounting"]["logical_dependencies"]
    assert by_op_single["ffn_up_projection"]["record_id"] in by_op_single["ffn_gated_multiply_accounting"]["logical_dependencies"]
    assert by_op_single["ffn_intermediate_bank_local_residency"]["record_id"] in by_op_single["ffn_down_projection"]["logical_dependencies"]
    # Per-tile dependency chain: tile setups → PIMOperandReuse → FFNProjection
    tile_setup_ids = [r["record_id"] for r in records if r["op"] == "ffn_hidden_activation_tile_setup"]
    reuse_deps = by_op_single["ffn_hidden_reuse_for_up_gate"]["logical_dependencies"]
    for tid in tile_setup_ids:
        assert tid in reuse_deps, f"PIMOperandReuse should depend on tile setup {tid}"


def test_ffn_generator_emits_operand_setup_residency_and_reuse_semantics():
    manifest = generator_mod.get_tiny_ffn_manifest()
    assert manifest["operand_movement_policy"] == {
        "weights": "preloaded_stationary",
        "dynamic_activation_setup": "materialized",
        "ffn_intermediate": "bank_local_capacity_controlled",
    }
    records = generator_mod.generate_ffn_records(manifest)
    by_id = {record["record_id"]: record for record in records}
    by_op_single = {}
    multi_ops = {}
    for record in records:
        op = record["op"]
        if op in by_op_single:
            multi_ops.setdefault(op, [by_op_single.pop(op)])
            multi_ops[op].append(record)
        else:
            by_op_single[op] = record

    assert [record["op"] for record in records] == [
        "ffn_hidden_activation_tile_setup",
        "ffn_hidden_activation_tile_setup",
        "ffn_up_weight_residency",
        "ffn_gate_weight_residency",
        "ffn_down_weight_residency",
        "ffn_hidden_reuse_for_up_gate",
        "ffn_up_projection",
        "ffn_gate_projection",
        "ffn_gate_activation_accounting",
        "ffn_gated_multiply_accounting",
        "ffn_intermediate_bank_local_residency",
        "ffn_down_projection",
    ]
    # Per-tile PIMDataMove records carry full metadata
    tile_setups = multi_ops["ffn_hidden_activation_tile_setup"]
    assert len(tile_setups) == 2
    for ts in tile_setups:
        assert ts["kind"] == "PIMDataMove"
        mp = ts["movement_policy"]
        assert mp["operand_role"] == "activation_input"
        assert mp["residency"] == "dynamic_activation_tile"
        assert mp["materialized"] is True
        assert mp["reuse_scope"] == "ffn_layer"
        assert mp["lowering"] == "native_pim_bcast_when_supported"
        assert "tile_index" in mp
        assert "tile_elements" in mp
        assert "tile_start" in mp
        assert mp["distribution_scope"] == "broadcast"
    assert by_op_single["ffn_up_weight_residency"]["kind"] == "PIMDataMove"
    assert by_op_single["ffn_up_weight_residency"]["movement_policy"]["materialized"] is True
    assert by_op_single["ffn_up_weight_residency"]["movement_policy"]["residency"] == "preloaded_stationary"
    assert by_op_single["ffn_hidden_reuse_for_up_gate"]["kind"] == "PIMOperandReuse"
    assert by_op_single["ffn_up_projection"]["record_id"] not in by_op_single["ffn_hidden_reuse_for_up_gate"].get("logical_dependencies", [])
    assert by_op_single["ffn_hidden_reuse_for_up_gate"]["record_id"] in by_op_single["ffn_up_projection"]["logical_dependencies"]
    assert by_op_single["ffn_hidden_reuse_for_up_gate"]["record_id"] in by_op_single["ffn_gate_projection"]["logical_dependencies"]
    # PIMOperandReuse depends on all per-tile setup records
    reuse_deps = by_op_single["ffn_hidden_reuse_for_up_gate"]["logical_dependencies"]
    for ts in tile_setups:
        assert ts["record_id"] in reuse_deps
    assert by_op_single["ffn_intermediate_bank_local_residency"]["accounting_metadata"]["residency"] == "bank_local_capacity_controlled"
    assert by_op_single["ffn_intermediate_bank_local_residency"]["accounting_metadata"]["materialized"] is False
    assert by_op_single["ffn_intermediate_bank_local_residency"]["record_id"] in by_op_single["ffn_down_projection"]["logical_dependencies"]


def test_ffn_weight_residency_is_materialized_data_move():
    manifest = generator_mod.get_tiny_ffn_manifest()
    records = generator_mod.generate_ffn_records(manifest)
    by_op = {record["op"]: record for record in records}

    for op, projection_op in [
        ("ffn_up_weight_residency", "ffn_up_projection"),
        ("ffn_gate_weight_residency", "ffn_gate_projection"),
        ("ffn_down_weight_residency", "ffn_down_projection"),
    ]:
        weight_move = by_op[op]
        expected = generator_mod._num_requests(
            int(manifest["hidden_size"]) * int(manifest["ffn_hidden_size"]),
            manifest["datatype"],
        )
        assert weight_move["kind"] == "PIMDataMove"
        assert weight_move["num_requests"] == expected
        assert weight_move["movement_policy"]["operand_role"] == "weight"
        assert weight_move["movement_policy"]["residency"] == "preloaded_stationary"
        assert weight_move["movement_policy"]["materialized"] is True
        assert weight_move["movement_policy"]["lowering"] == "native_pim_bcast_when_supported"
        assert weight_move["datatype_metadata"]["behavior_claim"] == "semantic_movement_volume_proportional_not_tiled_or_silicon_faithful"
        assert weight_move["record_id"] in by_op[projection_op]["logical_dependencies"]
        assert weight_move["tensor_io"]["outputs"][0] in by_op[projection_op]["tensor_io"]["inputs"]


def test_ffn_projection_shapes_are_direction_specific():
    manifest = generator_mod.get_tiny_ffn_manifest()
    manifest.update({"seq_len": 2, "hidden_size": 64, "ffn_hidden_size": 128})
    records = generator_mod.generate_ffn_records(manifest)
    by_op = {record["op"]: record for record in records}
    assert by_op["ffn_up_projection"]["compute_shape"] == {"m": 2, "n": 128, "k": 64, "output_elements": 256, "datatype": "int8"}
    assert by_op["ffn_gate_projection"]["compute_shape"] == {"m": 2, "n": 128, "k": 64, "output_elements": 256, "datatype": "int8"}
    assert by_op["ffn_down_projection"]["compute_shape"] == {"m": 2, "n": 64, "k": 128, "output_elements": 128, "datatype": "int8"}


def test_ffn_manifest_validation_rejects_missing_and_invalid_fields():
    manifest = generator_mod.get_tiny_ffn_manifest()
    del manifest["activation"]
    try:
        generator_mod.generate_ffn_records(manifest)
    except ValueError as exc:
        assert "activation" in str(exc)
    else:
        raise AssertionError("Expected missing activation rejection")

    manifest = generator_mod.get_tiny_ffn_manifest()
    manifest["ffn_hidden_size"] = 0
    try:
        generator_mod.generate_ffn_records(manifest)
    except ValueError as exc:
        assert "ffn_hidden_size" in str(exc)
        assert "positive" in str(exc)
    else:
        raise AssertionError("Expected invalid ffn_hidden_size rejection")


def test_ffn_lowering_only_lowers_projection_records_to_native_compute():
    semantic = generator_mod.generate_ffn_records()
    concrete = lowering_mod.lower_semantic_records_to_concrete(semantic)
    concrete_schema_mod.validate_sequence(concrete)
    lowered_kinds = {record["provenance"]["semantic_source"]["kind"] for record in concrete}
    assert lowered_kinds == {"FFNProjection", "PIMDataMove"}
    assert "PIMOperandResidency" not in lowered_kinds
    assert "PIMOperandReuse" not in lowered_kinds
    assert {record["opcode"] for record in concrete} <= concrete_schema_mod.CONCRETE_OPCODES
    assert "PIM_MAC" in {record["opcode"] for record in concrete}
    assert "PIM_BCAST" in {record["opcode"] for record in concrete}
    assert not {"PIM_SFM", "PIM_MV_BA", "PIM_WR_GB"} & {record["opcode"] for record in concrete}


def test_ffn_lowering_materializes_activation_setup_but_not_residency_or_reuse():
    semantic = generator_mod.generate_ffn_records()
    concrete = lowering_mod.lower_semantic_records_to_concrete(semantic)
    concrete_schema_mod.validate_sequence(concrete)

    lowered_kinds = {record["provenance"]["semantic_source"]["kind"] for record in concrete}
    assert "PIMDataMove" in lowered_kinds
    assert "FFNProjection" in lowered_kinds
    assert "PIMOperandResidency" not in lowered_kinds
    assert "PIMOperandReuse" not in lowered_kinds
    assert "PIM_BCAST" in {record["opcode"] for record in concrete}


def test_ffn_weight_moves_lower_to_pim_bcast_when_materialized_while_reuse_stays_semantic_only():
    semantic = generator_mod.generate_ffn_records()
    concrete = lowering_mod.lower_semantic_records_to_concrete(semantic, materialize_weights=True)
    concrete_schema_mod.validate_sequence(concrete)

    lowered_ops = [record["provenance"]["semantic_source"]["op"] for record in concrete]
    for op in ["ffn_up_weight_residency", "ffn_gate_weight_residency", "ffn_down_weight_residency"]:
        assert op in lowered_ops

    assert "ffn_hidden_reuse_for_up_gate" not in lowered_ops
    assert "ffn_intermediate_bank_local_residency" not in lowered_ops
    assert "ffn_gate_activation_accounting" not in lowered_ops
    assert "ffn_gated_multiply_accounting" not in lowered_ops

    weight_bcasts = [
        record
        for record in concrete
        if record["opcode"] == "PIM_BCAST"
        and record["provenance"]["semantic_source"]["op"]
        in {"ffn_up_weight_residency", "ffn_gate_weight_residency", "ffn_down_weight_residency"}
    ]
    assert len(weight_bcasts) == 3


def test_ffn_default_steady_state_lowering_skips_weight_data_moves():
    semantic = generator_mod.generate_ffn_records()
    concrete = lowering_mod.lower_semantic_records_to_concrete(semantic)
    concrete_schema_mod.validate_sequence(concrete)

    lowered_ops = [record["provenance"]["semantic_source"]["op"] for record in concrete]
    for op in ["ffn_up_weight_residency", "ffn_gate_weight_residency", "ffn_down_weight_residency"]:
        assert op not in lowered_ops

    assert "ffn_hidden_activation_tile_setup" in lowered_ops
    assert "FFNProjection" in {record["provenance"]["semantic_source"]["kind"] for record in concrete}
    assert "PIM_MAC" in {record["opcode"] for record in concrete}


def test_ffn_mac_counts_change_with_shape_and_datatype():
    manifest_a = generator_mod.get_tiny_ffn_manifest()
    manifest_a.update({"hidden_size": 32, "ffn_hidden_size": 64, "datatype": "int8"})
    manifest_b = generator_mod.get_tiny_ffn_manifest()
    manifest_b.update({"hidden_size": 64, "ffn_hidden_size": 128, "datatype": "fp16"})
    macs_a = sum(record["num_requests"] for record in generator_mod.generate_ffn_records(manifest_a) if record["kind"] == "FFNProjection")
    macs_b = sum(record["num_requests"] for record in generator_mod.generate_ffn_records(manifest_b) if record["kind"] == "FFNProjection")
    assert macs_b > macs_a


def test_ffn_lowered_concrete_trace_replays(tmp_path: Path):
    dram = create_dram(LPDDR5_PIM_CONFIG)
    manifest = generator_mod.get_tiny_ffn_manifest()
    manifest.update({"hidden_size": 32, "ffn_hidden_size": 64, "datatype": "int8"})
    semantic = generator_mod.generate_ffn_records(manifest)
    concrete = lowering_mod.lower_semantic_records_to_concrete(semantic, manifest_name=manifest["manifest_name"])
    trace_path = tmp_path / "ffn_concrete_opcode_trace.jsonl"
    concrete_schema_mod.write_jsonl(concrete, trace_path)

    sim = ramulator.Simulation(_frontend(trace_path, dram), _make_mem(dram))
    sim.run()
    stats = sim.stats["frontend"]
    assert stats["opcode_requests_completed"] == stats["opcode_requests_sent"]
    assert sim.stats["memory_system"]["controller"]["num_pim_reqs_served"] >= 3


def test_ffn_generator_single_tile_when_tile_size_equals_hidden_size():
    """Backward compat: ffn_activation_tile_size == hidden_size → 1 tile."""
    manifest = generator_mod.get_tiny_ffn_manifest()
    manifest.update({"hidden_size": 32, "ffn_activation_tile_size": 32})
    records = generator_mod.generate_ffn_records(manifest)
    tile_setups = [r for r in records if r["op"] == "ffn_hidden_activation_tile_setup"]
    assert len(tile_setups) == 1
    assert tile_setups[0]["movement_policy"]["tile_index"] == 0
    assert tile_setups[0]["movement_policy"]["tile_elements"] == 32
    assert tile_setups[0]["movement_policy"]["tile_start"] == 0


def test_ffn_generator_multi_tile_produces_distribution_metadata():
    """Each per-tile PIMDataMove carries tile_index, tile_elements, tile_start, distribution_scope."""
    manifest = generator_mod.get_tiny_ffn_manifest()
    manifest.update({"hidden_size": 64, "ffn_activation_tile_size": 16})
    records = generator_mod.generate_ffn_records(manifest)
    tile_setups = [r for r in records if r["op"] == "ffn_hidden_activation_tile_setup"]
    assert len(tile_setups) == 4  # 64 / 16
    for i, ts in enumerate(tile_setups):
        mp = ts["movement_policy"]
        assert mp["tile_index"] == i
        assert mp["tile_elements"] == 16
        assert mp["tile_start"] == i * 16
        assert mp["distribution_scope"] == "broadcast"
        assert mp["materialized"] is True
        assert mp["operand_role"] == "activation_input"
        oc = ts["operator_context"]
        assert oc["tile_id"] == i
        assert oc["tile_elements"] == 16
        assert oc["tile_start"] == i * 16
    # All tile setup IDs are in the PIMOperandReuse dependencies
    reuse = [r for r in records if r["op"] == "ffn_hidden_reuse_for_up_gate"][0]
    for ts in tile_setups:
        assert ts["record_id"] in reuse["logical_dependencies"]


def test_ffn_generator_partial_last_tile():
    """When hidden_size is not evenly divisible by tile_size, the last tile is smaller."""
    manifest = generator_mod.get_tiny_ffn_manifest()
    manifest.update({"hidden_size": 40, "ffn_activation_tile_size": 16})
    records = generator_mod.generate_ffn_records(manifest)
    tile_setups = [r for r in records if r["op"] == "ffn_hidden_activation_tile_setup"]
    assert len(tile_setups) == 3  # ceil(40/16)
    # Last tile should be 8 elements
    assert tile_setups[2]["movement_policy"]["tile_elements"] == 8
    assert tile_setups[2]["movement_policy"]["tile_start"] == 32


def test_ffn_manifest_rejects_invalid_tile_size():
    manifest = generator_mod.get_tiny_ffn_manifest()
    manifest["ffn_activation_tile_size"] = 0
    try:
        generator_mod.generate_ffn_records(manifest)
    except ValueError as exc:
        assert "ffn_activation_tile_size" in str(exc)
        assert "positive" in str(exc)
    else:
        raise AssertionError("Expected tile_size=0 rejection")

    manifest = generator_mod.get_tiny_ffn_manifest()
    manifest["ffn_activation_tile_size"] = 100  # > hidden_size=32
    try:
        generator_mod.generate_ffn_records(manifest)
    except ValueError as exc:
        assert "ffn_activation_tile_size" in str(exc)
        assert "hidden_size" in str(exc)
    else:
        raise AssertionError("Expected tile_size > hidden_size rejection")


def test_ffn_manifest_rejects_invalid_distribution_policy():
    manifest = generator_mod.get_tiny_ffn_manifest()
    manifest["activation_distribution_policy"] = "unicast"
    try:
        generator_mod.generate_ffn_records(manifest)
    except ValueError as exc:
        assert "activation_distribution_policy" in str(exc)
        assert "unicast" in str(exc)
    else:
        raise AssertionError("Expected invalid distribution_policy rejection")


def test_ffn_policies_vary_by_tile_index():
    """Different tile indices produce different dependency/row assignments."""
    manifest = generator_mod.get_tiny_ffn_manifest()
    p0 = generator_mod._ffn_policies(manifest, layer_index=0, stage_index=0, tile_index=0)
    p1 = generator_mod._ffn_policies(manifest, layer_index=0, stage_index=0, tile_index=1)
    assert p0["dependency_context"]["dependency_id"] != p1["dependency_context"]["dependency_id"]
    assert p0["dependency_context"]["tile_index"] == 0
    assert p1["dependency_context"]["tile_index"] == 1


def test_moe_generator_emits_router_topk_dispatch_selected_experts_and_combine():
    manifest = generator_mod.get_tiny_moe_manifest()
    manifest.update({"num_experts": 4, "top_k": 2, "selected_experts": [1, 3]})
    records = generator_mod.generate_moe_records(manifest)
    assert [record["op"] for record in records] == [
        "moe_router_input_setup",
        "moe_router_weight_residency",
        "moe_router_projection",
        "moe_topk_select_accounting",
        "moe_expert_dispatch_accounting",
        "moe_token_dispatch_materialized",
        "moe_expert_1_weight_residency",
        "moe_expert_1_ffn",
        "moe_expert_3_weight_residency",
        "moe_expert_3_ffn",
        "moe_expert_combine_accounting",
        "moe_expert_output_combine_materialized",
    ]
    by_op = {record["op"]: record for record in records}
    assert by_op["moe_router_projection"]["record_id"] in by_op["moe_topk_select_accounting"]["logical_dependencies"]
    assert by_op["moe_topk_select_accounting"]["record_id"] in by_op["moe_expert_dispatch_accounting"]["logical_dependencies"]
    assert by_op["moe_expert_dispatch_accounting"]["record_id"] in by_op["moe_token_dispatch_materialized"]["logical_dependencies"]
    assert by_op["moe_token_dispatch_materialized"]["record_id"] in by_op["moe_expert_1_ffn"]["logical_dependencies"]
    assert by_op["moe_token_dispatch_materialized"]["record_id"] in by_op["moe_expert_3_ffn"]["logical_dependencies"]
    assert by_op["moe_expert_combine_accounting"]["logical_dependencies"] == [
        by_op["moe_expert_1_ffn"]["record_id"],
        by_op["moe_expert_3_ffn"]["record_id"],
    ]
    assert by_op["moe_expert_combine_accounting"]["record_id"] in by_op["moe_expert_output_combine_materialized"]["logical_dependencies"]


def test_moe_generator_emits_operand_setup_dispatch_combine_and_residency_semantics():
    manifest = generator_mod.get_tiny_moe_manifest()
    assert manifest["operand_movement_policy"] == {
        "weights": "preloaded_stationary",
        "router_input_setup": "materialized",
        "token_dispatch": "materialized",
        "expert_output_combine": "materialized",
    }
    manifest.update({"num_experts": 4, "top_k": 2, "selected_experts": [1, 3]})
    records = generator_mod.generate_moe_records(manifest)
    by_op = {record["op"]: record for record in records}

    assert [record["op"] for record in records] == [
        "moe_router_input_setup",
        "moe_router_weight_residency",
        "moe_router_projection",
        "moe_topk_select_accounting",
        "moe_expert_dispatch_accounting",
        "moe_token_dispatch_materialized",
        "moe_expert_1_weight_residency",
        "moe_expert_1_ffn",
        "moe_expert_3_weight_residency",
        "moe_expert_3_ffn",
        "moe_expert_combine_accounting",
        "moe_expert_output_combine_materialized",
    ]
    assert by_op["moe_router_input_setup"]["kind"] == "PIMDataMove"
    assert by_op["moe_router_input_setup"]["movement_policy"]["operand_role"] == "activation_input"
    assert by_op["moe_router_weight_residency"]["kind"] == "PIMDataMove"
    assert by_op["moe_router_weight_residency"]["movement_policy"]["residency"] == "preloaded_stationary"
    assert by_op["moe_router_weight_residency"]["movement_policy"]["materialized"] is True
    assert by_op["moe_router_input_setup"]["record_id"] in by_op["moe_router_projection"]["logical_dependencies"]
    assert by_op["moe_router_weight_residency"]["record_id"] in by_op["moe_router_projection"]["logical_dependencies"]
    assert by_op["moe_token_dispatch_materialized"]["kind"] == "PIMDataMove"
    assert by_op["moe_expert_dispatch_accounting"]["record_id"] in by_op["moe_token_dispatch_materialized"]["logical_dependencies"]
    assert by_op["moe_token_dispatch_materialized"]["record_id"] in by_op["moe_expert_1_ffn"]["logical_dependencies"]
    assert by_op["moe_expert_1_weight_residency"]["record_id"] in by_op["moe_expert_1_ffn"]["logical_dependencies"]
    assert by_op["moe_expert_output_combine_materialized"]["kind"] == "PIMDataMove"
    assert by_op["moe_expert_combine_accounting"]["record_id"] in by_op["moe_expert_output_combine_materialized"]["logical_dependencies"]


def test_moe_weight_residency_is_materialized_data_move():
    manifest = generator_mod.get_tiny_moe_manifest()
    manifest.update({"num_experts": 4, "top_k": 2, "selected_experts": [1, 3]})
    records = generator_mod.generate_moe_records(manifest)
    by_op = {record["op"]: record for record in records}
    expected_router = generator_mod._num_requests(
        int(manifest["hidden_size"]) * int(manifest["num_experts"]),
        manifest["datatype"],
    )
    expected_expert = generator_mod._num_requests(
        int(manifest["hidden_size"]) * int(manifest["expert_hidden_size"]),
        manifest["datatype"],
    )

    for op, compute_op in [
        ("moe_router_weight_residency", "moe_router_projection"),
        ("moe_expert_1_weight_residency", "moe_expert_1_ffn"),
        ("moe_expert_3_weight_residency", "moe_expert_3_ffn"),
    ]:
        weight_move = by_op[op]
        assert weight_move["kind"] == "PIMDataMove"
        expected = expected_router if op == "moe_router_weight_residency" else expected_expert
        assert weight_move["num_requests"] == expected
        assert weight_move["movement_policy"]["operand_role"] == "weight"
        assert weight_move["movement_policy"]["residency"] == "preloaded_stationary"
        assert weight_move["movement_policy"]["materialized"] is True
        assert weight_move["datatype_metadata"]["behavior_claim"] == "semantic_movement_volume_proportional_not_tiled_or_silicon_faithful"
        assert weight_move["record_id"] in by_op[compute_op]["logical_dependencies"]
        assert weight_move["tensor_io"]["outputs"][0] in by_op[compute_op]["tensor_io"]["inputs"]


def test_moe_only_selected_experts_emit_compute_and_topk_is_sensitive():
    manifest_a = generator_mod.get_tiny_moe_manifest()
    manifest_a.update({"num_experts": 4, "top_k": 1, "selected_experts": [2]})
    records_a = generator_mod.generate_moe_records(manifest_a)
    manifest_b = generator_mod.get_tiny_moe_manifest()
    manifest_b.update({"num_experts": 4, "top_k": 3, "selected_experts": [0, 2, 3]})
    records_b = generator_mod.generate_moe_records(manifest_b)
    experts_a = [record for record in records_a if record["kind"] == "MoEExpertFFN"]
    experts_b = [record for record in records_b if record["kind"] == "MoEExpertFFN"]
    assert [record["operator_context"]["expert_id"] for record in experts_a] == [2]
    assert [record["operator_context"]["expert_id"] for record in experts_b] == [0, 2, 3]
    assert sum(record["num_requests"] for record in experts_b) > sum(record["num_requests"] for record in experts_a)


def test_moe_manifest_validation_rejects_invalid_topk_and_expert_selection():
    manifest = generator_mod.get_tiny_moe_manifest()
    manifest["top_k"] = 0
    try:
        generator_mod.generate_moe_records(manifest)
    except ValueError as exc:
        assert "top_k" in str(exc)
        assert "positive" in str(exc)
    else:
        raise AssertionError("Expected invalid top_k rejection")

    manifest = generator_mod.get_tiny_moe_manifest()
    manifest.update({"num_experts": 2, "top_k": 2, "selected_experts": [0, 3]})
    try:
        generator_mod.generate_moe_records(manifest)
    except ValueError as exc:
        assert "selected_experts" in str(exc)
    else:
        raise AssertionError("Expected out-of-range selected_experts rejection")


def test_moe_compute_only_lowering_lowers_router_and_selected_expert_compute():
    manifest = generator_mod.get_tiny_moe_manifest()
    manifest.update({"num_experts": 4, "top_k": 2, "selected_experts": [0, 2]})
    semantic = generator_mod.generate_moe_records(manifest)
    concrete = lowering_mod.lower_semantic_records_to_concrete(semantic, manifest_name=manifest["manifest_name"])
    concrete_schema_mod.validate_sequence(concrete)
    lowered_kinds = [record["provenance"]["semantic_source"]["kind"] for record in concrete if record["opcode"] == "PIM_MAC"]
    assert set(lowered_kinds) == {"MoERouter", "MoEExpertFFN"}
    assert "MoETopK" not in lowered_kinds
    assert "MoEDispatch" not in lowered_kinds
    assert "MoECombine" not in lowered_kinds
    assert not {"PIM_SFM", "PIM_MV_BA", "PIM_WR_GB"} & {record["opcode"] for record in concrete}


def test_moe_lowering_materializes_dispatch_and_combine_but_not_residency():
    manifest = generator_mod.get_tiny_moe_manifest()
    manifest.update({"num_experts": 4, "top_k": 2, "selected_experts": [0, 2]})
    semantic = generator_mod.generate_moe_records(manifest)
    concrete = lowering_mod.lower_semantic_records_to_concrete(semantic, manifest_name=manifest["manifest_name"])
    concrete_schema_mod.validate_sequence(concrete)

    lowered_kinds = {record["provenance"]["semantic_source"]["kind"] for record in concrete}
    assert "PIMDataMove" in lowered_kinds
    assert "MoERouter" in lowered_kinds
    assert "MoEExpertFFN" in lowered_kinds
    assert "PIMOperandResidency" not in lowered_kinds
    assert "PIMOperandReuse" not in lowered_kinds
    assert "MoEDispatch" not in lowered_kinds
    assert "MoECombine" not in lowered_kinds
    assert "PIM_BCAST" in {record["opcode"] for record in concrete}


def test_moe_weight_moves_lower_to_pim_bcast_when_materialized_while_routing_accounting_stays_semantic_only():
    manifest = generator_mod.get_tiny_moe_manifest()
    manifest.update({"num_experts": 4, "top_k": 2, "selected_experts": [0, 2]})
    semantic = generator_mod.generate_moe_records(manifest)
    concrete = lowering_mod.lower_semantic_records_to_concrete(
        semantic,
        manifest_name=manifest["manifest_name"],
        materialize_weights=True,
    )
    concrete_schema_mod.validate_sequence(concrete)

    lowered_ops = [record["provenance"]["semantic_source"]["op"] for record in concrete]
    for op in ["moe_router_weight_residency", "moe_expert_0_weight_residency", "moe_expert_2_weight_residency"]:
        assert op in lowered_ops

    assert "moe_topk_select_accounting" not in lowered_ops
    assert "moe_expert_dispatch_accounting" not in lowered_ops
    assert "moe_expert_combine_accounting" not in lowered_ops

    weight_bcasts = [
        record
        for record in concrete
        if record["opcode"] == "PIM_BCAST"
        and record["provenance"]["semantic_source"]["op"]
        in {"moe_router_weight_residency", "moe_expert_0_weight_residency", "moe_expert_2_weight_residency"}
    ]
    assert len(weight_bcasts) == 3


def test_moe_default_steady_state_lowering_skips_weight_data_moves():
    manifest = generator_mod.get_tiny_moe_manifest()
    manifest.update({"num_experts": 4, "top_k": 2, "selected_experts": [0, 2]})
    semantic = generator_mod.generate_moe_records(manifest)
    concrete = lowering_mod.lower_semantic_records_to_concrete(semantic, manifest_name=manifest["manifest_name"])
    concrete_schema_mod.validate_sequence(concrete)

    lowered_ops = [record["provenance"]["semantic_source"]["op"] for record in concrete]
    for op in ["moe_router_weight_residency", "moe_expert_0_weight_residency", "moe_expert_2_weight_residency"]:
        assert op not in lowered_ops

    for op in [
        "moe_router_input_setup",
        "moe_token_dispatch_materialized",
        "moe_expert_output_combine_materialized",
    ]:
        assert op in lowered_ops

    lowered_kinds = {record["provenance"]["semantic_source"]["kind"] for record in concrete}
    assert "MoERouter" in lowered_kinds
    assert "MoEExpertFFN" in lowered_kinds
    assert "PIM_MAC" in {record["opcode"] for record in concrete}


def test_moe_lowered_concrete_trace_replays(tmp_path: Path):
    dram = create_dram(LPDDR5_PIM_CONFIG)
    manifest = generator_mod.get_tiny_moe_manifest()
    manifest.update({"num_experts": 4, "top_k": 2, "selected_experts": [1, 3], "hidden_size": 32, "expert_hidden_size": 64})
    semantic = generator_mod.generate_moe_records(manifest)
    concrete = lowering_mod.lower_semantic_records_to_concrete(semantic, manifest_name=manifest["manifest_name"])
    trace_path = tmp_path / "moe_concrete_opcode_trace.jsonl"
    concrete_schema_mod.write_jsonl(concrete, trace_path)

    sim = ramulator.Simulation(_frontend(trace_path, dram), _make_mem(dram))
    sim.run()
    stats = sim.stats["frontend"]
    assert stats["opcode_requests_completed"] == stats["opcode_requests_sent"]
    assert sim.stats["memory_system"]["controller"]["num_pim_reqs_served"] >= 3


def _assert_dependencies_precede_consumers(records: list[dict]) -> None:
    positions = {record["record_id"]: index for index, record in enumerate(records)}
    assert len(positions) == len(records)
    for index, record in enumerate(records):
        for dependency in record.get("logical_dependencies", []):
            assert positions[dependency] < index


def test_attention_overlapped_head_schedule_differs_and_remains_legal():
    serialized = generator_mod.get_tiny_attention_manifest()
    serialized.update({"num_heads": 2, "past_len": 32, "score_tile_tokens": 32, "schedule_policy": "serialized"})
    overlapped = generator_mod.get_tiny_attention_manifest()
    overlapped.update({"num_heads": 2, "past_len": 32, "score_tile_tokens": 32, "schedule_policy": "overlap_independent_heads"})

    serialized_records = generator_mod.generate_attention_records(serialized)
    overlapped_records = generator_mod.generate_attention_records(overlapped)
    assert [record["op"] for record in serialized_records] != [record["op"] for record in overlapped_records]
    first_overlapped_softmax = next(index for index, record in enumerate(overlapped_records) if record["kind"] == "AttentionSoftmax")
    assert all(index < first_overlapped_softmax for index, record in enumerate(overlapped_records) if record["kind"] == "AttentionScore")
    assert all(record["provenance"].get("schedule_policy") == "overlap_independent_heads" for record in overlapped_records)
    assert all(
        record["operator_context"].get("dag_hint") == "heads_are_independent_safe_to_parallelize"
        for record in overlapped_records
    )
    _assert_dependencies_precede_consumers(overlapped_records)


def test_attention_overlap_requires_multiple_heads():
    manifest = generator_mod.get_tiny_attention_manifest()
    manifest.update({"num_heads": 1, "schedule_policy": "overlap_independent_heads"})

    with pytest.raises(ValueError, match="overlap_independent_heads requires num_heads >= 2"):
        generator_mod.generate_attention_records(manifest)


def test_attention_overlapped_schedule_lowers_and_replays(tmp_path: Path):
    dram = create_dram(LPDDR5_PIM_CONFIG)
    manifest = generator_mod.get_tiny_attention_manifest()
    manifest.update({"num_heads": 2, "past_len": 32, "score_tile_tokens": 32, "schedule_policy": "overlap_independent_heads"})
    semantic = generator_mod.generate_attention_records(manifest)
    concrete = lowering_mod.lower_semantic_records_to_concrete(semantic, manifest_name=manifest["manifest_name"])
    trace_path = tmp_path / "attention_overlap_concrete_opcode_trace.jsonl"
    concrete_schema_mod.write_jsonl(concrete, trace_path)

    sim = ramulator.Simulation(_frontend(trace_path, dram), _make_mem(dram))
    sim.run()
    assert sim.stats["frontend"]["opcode_requests_completed"] == sim.stats["frontend"]["opcode_requests_sent"]


def test_moe_overlapped_expert_schedule_differs_and_remains_legal():
    serialized = generator_mod.get_tiny_moe_manifest()
    serialized.update({"top_k": 3, "selected_experts": [0, 1, 2], "schedule_policy": "serialized"})
    overlapped = generator_mod.get_tiny_moe_manifest()
    overlapped.update({"top_k": 3, "selected_experts": [0, 1, 2], "schedule_policy": "overlap_selected_experts"})

    serialized_records = generator_mod.generate_moe_records(serialized)
    overlapped_records = generator_mod.generate_moe_records(overlapped)
    assert [record["op"] for record in serialized_records] != [record["op"] for record in overlapped_records]
    assert [record["operator_context"].get("expert_id") for record in overlapped_records if record["kind"] == "MoEExpertFFN"] == [0, 2, 1]
    assert all(record["provenance"].get("schedule_policy") == "overlap_selected_experts" for record in overlapped_records)
    assert all(
        record["operator_context"].get("dag_hint") == "selected_experts_are_independent_safe_to_parallelize"
        for record in overlapped_records
    )
    _assert_dependencies_precede_consumers(overlapped_records)


def test_moe_overlap_requires_multiple_selected_experts():
    manifest = generator_mod.get_tiny_moe_manifest()
    manifest.update({"top_k": 1, "selected_experts": [0], "schedule_policy": "overlap_selected_experts"})

    with pytest.raises(ValueError, match="overlap_selected_experts requires top_k >= 2"):
        generator_mod.generate_moe_records(manifest)


def test_combined_lowering_does_not_emit_duplicate_mode_commands():
    semantic = generator_mod.generate_full_transformer_layer_records()
    concrete = lowering_mod.lower_semantic_records_to_concrete(semantic, manifest_name="combined_mode_test")
    opcodes = [record["opcode"] for record in concrete]
    assert not any(lhs == rhs and lhs in {"SB", "HAB", "HAB_PIM"} for lhs, rhs in zip(opcodes, opcodes[1:]))
    assert not any(
        prev_opcode == "PIM_MAC" and opcode == "SB" and next_opcode == "PIM_MAC"
        for prev_opcode, opcode, next_opcode in zip(opcodes, opcodes[1:], opcodes[2:])
    )


def test_full_transformer_layer_combined_attention_ffn_moe_replays(tmp_path: Path):
    dram = create_dram(LPDDR5_PIM_CONFIG)
    attention = generator_mod.get_tiny_attention_manifest()
    attention.update({"num_heads": 1, "past_len": 32, "score_tile_tokens": 32, "head_dim": 32})
    ffn = generator_mod.get_tiny_ffn_manifest()
    ffn.update({"hidden_size": 32, "ffn_hidden_size": 64})
    moe = generator_mod.get_tiny_moe_manifest()
    moe.update({"hidden_size": 32, "expert_hidden_size": 64, "num_experts": 4, "top_k": 2, "selected_experts": [1, 3]})

    semantic = generator_mod.generate_full_transformer_layer_records(attention_manifest=attention, ffn_manifest=ffn, moe_manifest=moe)
    assert len({record["record_id"] for record in semantic}) == len(semantic)
    _assert_dependencies_precede_consumers(semantic)
    assert {"AttentionScore", "FFNProjection", "MoERouter", "MoEExpertFFN"} <= {record["kind"] for record in semantic}

    concrete = lowering_mod.lower_semantic_records_to_concrete(semantic, manifest_name="combined_p4_layer")
    assert {record["opcode"] for record in concrete} <= concrete_schema_mod.CONCRETE_OPCODES
    trace_path = tmp_path / "combined_p4_concrete_opcode_trace.jsonl"
    concrete_schema_mod.write_jsonl(concrete, trace_path)
    sim = ramulator.Simulation(_frontend(trace_path, dram), _make_mem(dram))
    sim.run()
    assert sim.stats["frontend"]["opcode_requests_completed"] == sim.stats["frontend"]["opcode_requests_sent"]


def test_build_full_transformer_provenance_summary_covers_attention_ffn_moe():
    semantic = generator_mod.generate_full_transformer_layer_records()
    summary = generator_mod.build_full_transformer_provenance_summary(semantic, manifest_name="combined_default")
    assert summary["record_counts_by_kind"]["AttentionScore"] >= 1
    assert summary["record_counts_by_kind"]["FFNProjection"] == 3
    assert summary["record_counts_by_kind"]["MoERouter"] == 1
    assert summary["record_counts_by_kind"]["MoEExpertFFN"] == 2
    assert "not_runtime_replay" in summary["non_claims"]


def test_attention_provenance_summary_accepts_manifest_name_keyword():
    semantic = generator_mod.generate_attention_records()

    summary = generator_mod.build_provenance_summary(semantic, manifest_name="attention_keyword")

    assert summary["manifest_name"] == "attention_keyword"
    assert summary["record_counts_by_kind"]["AttentionScore"] >= 1


def test_attention_provenance_summary_defaults_unknown_manifest_name():
    semantic = generator_mod.generate_attention_records()

    summary = generator_mod.build_provenance_summary(semantic)

    assert summary["manifest_name"] == "unknown"
    assert "semantic DAG summary" in summary["notes"]


def test_moe_router_mapping_is_distinct_from_expert_zero_when_stage_matches():
    manifest = generator_mod.get_tiny_moe_manifest()
    router_policy = generator_mod._moe_policies(manifest, layer_index=0, stage_index=0)
    expert_policy = generator_mod._moe_policies(manifest, layer_index=0, stage_index=0, expert_id=0)

    assert router_policy["dependency_context"]["expert_id"] is None
    assert expert_policy["dependency_context"]["expert_id"] == 0
    assert router_policy["dependency_context"]["dependency_id"] != expert_policy["dependency_context"]["dependency_id"]
    assert router_policy["row_policy"]["resolved_row"] != expert_policy["row_policy"]["resolved_row"]


def test_attention_cli_writes_bounded_artifacts(tmp_path: Path):
    output_dir = tmp_path / "attention"
    result = subprocess.run(
        [sys.executable, "-m", "ramulator.workload_surrogate.generate_full_transformer", "--output-dir", str(output_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    trace_path = output_dir / "structured_trace.jsonl"
    summary_path = output_dir / "provenance_summary.json"
    assert trace_path.exists()
    assert summary_path.exists()
    assert f"Generated: {trace_path}" in result.stdout

    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["record_counts_by_kind"]["AttentionScore"] == 4
    assert summary["record_counts_by_kind"]["AttentionContext"] == 4
    assert all("not_raw_attacc_schema" in record["provenance"]["non_claims"] for record in records)


def test_attention_lowered_concrete_trace_replays(tmp_path: Path):
    dram = create_dram(LPDDR5_PIM_CONFIG)
    manifest = generator_mod.get_tiny_attention_manifest()
    manifest.update({"num_heads": 1, "past_len": 32, "score_tile_tokens": 32, "head_dim": 32})
    semantic = generator_mod.generate_attention_records(manifest)
    concrete = lowering_mod.lower_semantic_records_to_concrete(semantic, manifest_name=manifest["manifest_name"])
    trace_path = tmp_path / "attention_concrete_opcode_trace.jsonl"
    concrete_schema_mod.write_jsonl(concrete, trace_path)

    sim = ramulator.Simulation(_frontend(trace_path, dram), _make_mem(dram))
    sim.run()
    stats = sim.stats["frontend"]
    assert stats["opcode_requests_completed"] == stats["opcode_requests_sent"]
    assert sim.stats["memory_system"]["controller"]["num_pim_reqs_served"] >= 2
