"""Tests for the deterministic decode-only workload-surrogate generator."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

ramulator = importlib.import_module("ramulator")

manifest_mod = importlib.import_module("ramulator.workload_surrogate.decode_only_manifest")
generator_mod = importlib.import_module("ramulator.workload_surrogate.generate_decode_only")
trace_mod = importlib.import_module("ramulator.workload_surrogate.structured_trace")

get_decode_only_manifest = manifest_mod.get_decode_only_manifest
generate_decode_only_records = generator_mod.generate_decode_only_records
generate_decode_only_artifacts = generator_mod.generate_decode_only_artifacts
build_provenance_summary = generator_mod.build_provenance_summary
DEFAULT_OUTPUT_DIR = generator_mod.DEFAULT_OUTPUT_DIR
REQUIRED_CLAIM_BOUNDARY = trace_mod.REQUIRED_CLAIM_BOUNDARY
REQUIRED_NON_CLAIMS = trace_mod.REQUIRED_NON_CLAIMS
SCHEMA_VERSION = trace_mod.SCHEMA_VERSION


def test_decode_only_manifest_matches_frozen_contract():
    manifest = get_decode_only_manifest()
    assert manifest["manifest_version"] == "v0.1"
    assert manifest["phase"] == "decode"
    assert manifest["batch"] == 1
    assert manifest["generated_tokens"] == 1
    assert manifest["datatype"] == "int8"
    assert manifest["model_family"] == "LLaMA-like decoder-only transformer"
    assert "pim_compute_operator_classes" in manifest
    assert "host_support_record_classes" in manifest
    assert "host_only_provenance_classes" in manifest
    assert "scaffolding_notes" in manifest
    assert manifest["scaffolding_notes"]["deterministic_mvp_scaffolding"] is True


def test_kv_cache_reads_are_host_support_not_pim_compute():
    manifest = get_decode_only_manifest()
    pim_ops = set(manifest["pim_compute_operator_classes"])
    support_ops = set(manifest["host_support_record_classes"])
    kv_reads = {"attention_k_cache_read", "attention_v_cache_read"}
    assert pim_ops & kv_reads == set(), "KV cache reads must not be in pim_compute_operator_classes"
    assert support_ops >= kv_reads, "host_support_record_classes must include KV cache reads"


def test_pim_compute_operator_classes_only_contain_gemv():
    manifest = get_decode_only_manifest()
    for op in manifest["pim_compute_operator_classes"]:
        assert op.endswith("_gemv"), f"pim_compute_operator_classes must only contain GEMV ops, found: {op}"


def test_host_only_provenance_includes_kv_cache_append():
    manifest = get_decode_only_manifest()
    assert "kv_cache_append_accounting" in manifest["host_only_provenance_classes"]


def test_operator_class_buckets_are_disjoint():
    manifest = get_decode_only_manifest()
    pim = set(manifest["pim_compute_operator_classes"])
    support = set(manifest["host_support_record_classes"])
    prov = set(manifest["host_only_provenance_classes"])
    assert pim & support == set()
    assert pim & prov == set()
    assert support & prov == set()


def test_provenance_summary_contains_renamed_buckets(tmp_path: Path):
    _, summary_path = generate_decode_only_artifacts(tmp_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "pim_compute_operator_classes" in summary
    assert "host_support_record_classes" in summary
    assert "host_only_provenance_classes" in summary
    assert "pim_eligible_operator_classes" not in summary
    assert "host_only_operator_classes" not in summary


def test_provenance_summary_includes_scaffolding_notes(tmp_path: Path):
    _, summary_path = generate_decode_only_artifacts(tmp_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "scaffolding_notes" in summary
    assert summary["scaffolding_notes"]["deterministic_mvp_scaffolding"] is True
    assert summary["scaffolding_notes"]["bank_count"] == 4


def test_generator_rejects_pim_compute_containing_kv_reads():
    manifest = get_decode_only_manifest()
    manifest["pim_compute_operator_classes"].append("attention_k_cache_read")
    try:
        generate_decode_only_records(manifest)
    except ValueError as exc:
        assert "overlap between pim_compute and host_support" in str(exc)
    else:
        raise AssertionError("Expected rejection of PIM compute containing KV reads")


def test_generator_rejects_missing_host_support_kv_reads():
    manifest = get_decode_only_manifest()
    manifest["host_support_record_classes"] = []
    try:
        generate_decode_only_records(manifest)
    except ValueError as exc:
        assert "host_support_record_classes must include KV cache reads" in str(exc)
    else:
        raise AssertionError("Expected rejection of manifest missing KV reads in host_support")


def test_generator_emits_only_phase2_mvp_record_kinds():
    records = generate_decode_only_records()
    assert {record["kind"] for record in records} == {"HostRead", "HostWrite", "PIMCompute", "Barrier", "Drain"}


def test_generator_keeps_all_bank_records_off_by_default():
    records = generate_decode_only_records()
    assert not {record["kind"] for record in records} & {"PIMLoadAll", "PIMComputeAll"}


def test_generated_records_have_required_common_fields():
    records = generate_decode_only_records()
    for record in records:
        assert record["schema_version"] == SCHEMA_VERSION
        for key in ("record_id", "kind", "phase", "layer", "op", "repeat", "provenance", "mapping_policy"):
            assert key in record


def test_pimcompute_records_have_required_fields():
    records = generate_decode_only_records()
    pim_records = [record for record in records if record["kind"] == "PIMCompute"]
    assert pim_records
    for record in pim_records:
        for key in ("num_requests", "bank_sequence", "dependency_context", "row_policy", "column_policy", "datatype_metadata"):
            assert key in record
        assert record["bank_sequence"] == [0, 1, 2, 3]


def test_generator_uses_manifest_driven_ramulator_defaults_and_pim_ops():
    manifest = get_decode_only_manifest()
    manifest["num_layers"] = 3
    manifest["hidden_size"] = 64
    manifest["ffn_hidden_size"] = 128
    manifest["pim_compute_operator_classes"] = ["q_projection_gemv", "ffn_up_projection_gemv"]
    manifest["pim_operator_request_widths"] = {
        "q_projection_gemv": "hidden_size",
        "ffn_up_projection_gemv": "ffn_hidden_size",
    }
    manifest["ramulator_visible_defaults"].update(
        {
            "bank_sequence": [3, 1],
            "bank_sequence_order": "frontend",
            "pim_banks_per_mpu": 1,
            "burst_length": 4,
            "row_start": 7,
            "row_count": 3,
            "dependency_count": 3,
            "tx_bytes": 128,
            "column_start": 5,
        }
    )

    records = generate_decode_only_records(manifest)
    pim_records = [record for record in records if record["kind"] == "PIMCompute"]
    assert [record["op"] for record in pim_records[:2]] == ["q_projection_gemv", "ffn_up_projection_gemv"]
    assert all(record["bank_sequence"] == [3, 1] for record in pim_records)
    assert all(record["burst_length"] == 4 for record in pim_records)
    assert [record["row_policy"]["resolved_row"] for record in pim_records[::2]] == [7, 8, 9]
    assert [record["dependency_context"]["dependency_id"] for record in pim_records[::2]] == [0, 1, 2]
    assert [record["column_policy"]["resolved_column"] for record in pim_records[::2]] == [5, 6, 7]
    assert pim_records[0]["num_requests"] == 2
    assert pim_records[1]["num_requests"] == 4
    host_records = [record for record in records if record["kind"] in {"HostRead", "HostWrite"}]
    assert all(record["address_policy"]["stride_bytes"] == 128 for record in host_records)
    assert all(record["mapping_policy"]["controller_bank_order"] == "frontend" for record in records)


def test_generator_rejects_unsupported_manifest_pim_operator():
    manifest = get_decode_only_manifest()
    manifest["pim_compute_operator_classes"] = ["unsupported_projection_gemv"]
    try:
        generate_decode_only_records(manifest)
    except ValueError as exc:
        assert "Unsupported PIM op" in str(exc)
    else:
        raise AssertionError("Expected unsupported PIM op rejection")


def test_host_records_have_required_fields():
    records = generate_decode_only_records()
    for record in records:
        if record["kind"] in {"HostRead", "HostWrite"}:
            assert record["bytes"] > 0
            assert record["address_policy"]["kind"] == "bounded_sequential_host_requests"


def test_barrier_and_drain_records_have_required_fields():
    records = generate_decode_only_records()
    barriers = [record for record in records if record["kind"] == "Barrier"]
    drains = [record for record in records if record["kind"] == "Drain"]
    assert len(barriers) == 32
    assert len(drains) == 1
    assert all(record["barrier_scope"] == "layer" for record in barriers)
    assert drains[0]["drain_scope"] == "trace"


def test_generation_is_byte_deterministic(tmp_path: Path):
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    trace_a, summary_a = generate_decode_only_artifacts(out_a)
    trace_b, summary_b = generate_decode_only_artifacts(out_b)
    assert trace_a.read_text(encoding="utf-8") == trace_b.read_text(encoding="utf-8")
    assert summary_a.read_text(encoding="utf-8") == summary_b.read_text(encoding="utf-8")


def test_generator_cli_writes_bounded_artifacts(tmp_path: Path):
    output_dir = tmp_path / "cli_decode_only"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ramulator.workload_surrogate.generate_decode_only",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    trace_path = output_dir / "structured_trace.jsonl"
    summary_path = output_dir / "provenance_summary.json"
    assert trace_path.exists()
    assert summary_path.exists()
    assert f"Generated: {trace_path}" in result.stdout
    assert f"Generated: {summary_path}" in result.stdout

    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert {record["kind"] for record in records} == {"HostRead", "HostWrite", "PIMCompute", "Barrier", "Drain"}
    assert summary["claim_boundary"] == REQUIRED_CLAIM_BOUNDARY
    assert summary["non_claims"] == REQUIRED_NON_CLAIMS


def test_provenance_summary_matches_trace_counts(tmp_path: Path):
    trace_path, summary_path = generate_decode_only_artifacts(tmp_path)
    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["total_logical_records"] == len(records)
    assert summary["total_expanded_records"] == len(records)
    assert summary["record_counts_by_kind"]["Drain"] == 1
    assert summary["record_counts_by_kind"]["Barrier"] == 32


def test_generator_rejects_mixed_phase_manifest_without_approval():
    manifest = get_decode_only_manifest()
    manifest["mixed_phase_extension"] = True
    manifest["phase"] = "mixed_prefill_decode"
    try:
        generate_decode_only_records(manifest)
    except ValueError as exc:
        assert "mixed_phase_approval='P2.9c'" in str(exc)
    else:
        raise AssertionError("Expected mixed-phase manifest rejection")


def test_generator_emits_mixed_prefill_decode_records_with_approval():
    manifest = get_decode_only_manifest()
    manifest.update(
        {
            "phase": "mixed_prefill_decode",
            "mixed_phase_extension": True,
            "mixed_phase_approval": "P2.9c",
            "num_layers": 1,
            "hidden_size": 64,
            "past_len": 4,
            "prefill_tokens": 8,
            "pim_compute_operator_classes": ["q_projection_gemv"],
            "pim_operator_request_widths": {"q_projection_gemv": "hidden_size"},
            "prefill_pim_compute_operator_classes": ["prefill_qkv_projection_gemv"],
            "prefill_pim_operator_request_widths": {"prefill_qkv_projection_gemv": "hidden_size"},
        }
    )

    records = generate_decode_only_records(manifest)
    phases = [record["phase"] for record in records]
    assert "prefill" in phases
    assert "decode" in phases
    assert records[0]["op"] == "prefill_prompt_activation_read"
    text = json.dumps(records, sort_keys=True)
    assert "not_runtime_replay" in text
    assert "not_vllm_replay" in text
    assert "mixed_prefill_decode_synthetic_only" in text
    assert "not_mixed_prefill_decode" not in text


def test_generator_rejects_all_bank_records_without_approval():
    manifest = get_decode_only_manifest()
    manifest["enable_all_bank_records"] = True
    try:
        generate_decode_only_records(manifest)
    except ValueError as exc:
        assert "all_bank_records_approval='P2.9b'" in str(exc)
    else:
        raise AssertionError("Expected all-bank generation rejection")


def test_generator_emits_all_bank_records_with_approval():
    manifest = get_decode_only_manifest()
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

    records = generate_decode_only_records(manifest)
    kinds = [record["kind"] for record in records]
    assert "PIMLoadAll" in kinds
    assert "PIMComputeAll" in kinds
    assert "PIM_BCAST" not in json.dumps(records, sort_keys=True)
    assert "PIM_MAC_AB" not in json.dumps(records, sort_keys=True)
    all_bank = [record for record in records if record["kind"] in {"PIMLoadAll", "PIMComputeAll"}]
    assert all(record["all_bank_scope"] == "rank" for record in all_bank)


def test_datatype_metadata_is_guardrail_only():
    records = generate_decode_only_records()
    summary = build_provenance_summary(records, get_decode_only_manifest())
    text = json.dumps({"records": records, "summary": summary}, sort_keys=True)
    assert "latency_scale" not in text
    assert "throughput_scale" not in text
    assert "energy_scale" not in text
    assert "bitwidth_ratio" not in text
    assert summary["claim_boundary"] == REQUIRED_CLAIM_BOUNDARY
    assert summary["non_claims"] == REQUIRED_NON_CLAIMS


def test_generator_rejects_datatype_scaling_fields_in_manifest():
    manifest = get_decode_only_manifest()
    manifest["datatype_metadata"] = {"latency_scale": 2}
    try:
        generate_decode_only_records(manifest)
    except ValueError as exc:
        assert "Datatype scaling fields are not supported" in str(exc)
    else:
        raise AssertionError("Expected datatype scaling field rejection")


def test_default_output_directory_is_bounded_test_data_path():
    assert DEFAULT_OUTPUT_DIR == Path("ramulator2/tests/data/structured_workload_surrogate/decode_only_v0_1")
