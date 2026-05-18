"""Helpers for structured workload-surrogate trace generation."""

from __future__ import annotations

import json
from pathlib import Path


SCHEMA_VERSION = "v0.1"
GENERATOR_VERSION = "v0.1"
BASE_SEMANTIC_RECORD_KINDS = {"HostRead", "HostWrite", "PIMCompute", "Barrier", "Drain"}
ALL_BANK_RECORD_KINDS = {"PIMLoadAll", "PIMComputeAll"}
TRANSFORMER_DATAFLOW_RECORD_KINDS = {
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
}
VALID_PIM_DATA_MOVE_MOVEMENT_KINDS = {
    "broadcast_or_accounted_tile_load",
    "preloaded_stationary_weight_residency",
    "dynamic_activation_tile",
    "bank_local_tile_activation",
    "cross_bank_operand_shuffle_accounting",
}
SEMANTIC_ONLY_PIM_DATA_MOVE_MOVEMENT_KINDS = {
    "dynamic_activation_tile",
    "bank_local_tile_activation",
    "cross_bank_operand_shuffle_accounting",
}
TRANSFORMER_COMPUTE_RECORD_KINDS = {
    "AttentionScore",
    "AttentionContext",
    "FFNProjection",
    "MoERouter",
    "MoEExpertFFN",
}
TRANSFORMER_ACCOUNTING_RECORD_KINDS = {
    "AttentionSoftmax",
    "PIMElementwise",
    "MoETopK",
    "MoEDispatch",
    "MoECombine",
    "PIMOperandResidency",
    "PIMOperandReuse",
}
REQUIRED_P4_RECORD_FAMILY = "p4_offline_transformer_dataflow_ir"
REQUIRED_P4_CLAIM_BOUNDARY = [
    "structured transformer dataflow surrogate",
    "simulator-diagnostic",
    "non-silicon-calibrated",
    "operator-internal-dataflow-first",
]
REQUIRED_P4_NON_CLAIMS = [
    "not_runtime_replay",
    "not_vllm_replay",
    "not_numerical_correctness",
    "not_silicon_faithful_softmax_or_data_movement",
    "not_raw_attacc_schema",
]
SEMANTIC_RECORD_KINDS = BASE_SEMANTIC_RECORD_KINDS | ALL_BANK_RECORD_KINDS | TRANSFORMER_DATAFLOW_RECORD_KINDS
MVP_RECORD_KINDS = BASE_SEMANTIC_RECORD_KINDS
FORBIDDEN_DATATYPE_SCALE_FIELDS = {"latency_scale", "throughput_scale", "energy_scale", "bitwidth_ratio"}
REQUIRED_CLAIM_BOUNDARY = [
    "structured workload-surrogate",
    "simulator-diagnostic",
    "non-silicon-calibrated",
    "decode-only-first",
]
REQUIRED_NON_CLAIMS = [
    "not_runtime_replay",
    "not_vllm_replay",
    "not_mixed_prefill_decode",
    "not_all_bank_fidelity",
]
MIXED_PHASE_NON_CLAIMS = [
    "not_runtime_replay",
    "not_vllm_replay",
    "not_application_replay",
    "mixed_prefill_decode_synthetic_only",
    "not_all_bank_fidelity",
]


def stable_json_dumps(data: object) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def stable_json_pretty(data: object) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


KV_CACHE_READ_OPS = {"attention_k_cache_read", "attention_v_cache_read"}
PIM_COMPUTE_OP_SUFFIX = "_gemv"


def _find_forbidden_datatype_scale_fields(value: object, path: str = "manifest") -> list[str]:
    matches: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in FORBIDDEN_DATATYPE_SCALE_FIELDS:
                matches.append(child_path)
            matches.extend(_find_forbidden_datatype_scale_fields(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            matches.extend(_find_forbidden_datatype_scale_fields(child, f"{path}[{index}]"))
    return matches


def validate_manifest(manifest: dict) -> None:
    required = {
        "manifest_version",
        "workload_class",
        "phase",
        "batch",
        "generated_tokens",
        "past_len",
        "hidden_size",
        "num_layers",
        "num_heads",
        "head_dim",
        "ffn_hidden_size",
        "datatype",
        "pim_compute_operator_classes",
        "host_support_record_classes",
        "host_only_provenance_classes",
        "mapping_policy",
        "literature_anchors",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"Manifest missing required fields: {missing}")
    mixed_phase = bool(manifest.get("mixed_phase_extension"))
    if mixed_phase and manifest.get("mixed_phase_approval") != "P2.9c":
        raise ValueError("Mixed prefill/decode generation requires mixed_phase_approval='P2.9c'")

    allowed_phases = {"decode", "mixed_prefill_decode"}
    if manifest["phase"] not in allowed_phases:
        raise ValueError(f"Phase 2 semantic generator supports phases {sorted(allowed_phases)}, got {manifest['phase']!r}")
    if manifest["phase"] == "mixed_prefill_decode" and not mixed_phase:
        raise ValueError("phase='mixed_prefill_decode' requires mixed_phase_extension=True")
    if mixed_phase and manifest["phase"] != "mixed_prefill_decode":
        raise ValueError("mixed_phase_extension=True requires phase='mixed_prefill_decode'")
    if manifest["batch"] != 1:
        raise ValueError("Phase 2 semantic generator only supports batch=1")
    if manifest["generated_tokens"] <= 0:
        raise ValueError("generated_tokens must be positive")
    if not mixed_phase and manifest["generated_tokens"] != 1:
        raise ValueError("Decode-only generator only supports generated_tokens=1 unless mixed_phase_extension=True")
    if manifest.get("enable_all_bank_records") and manifest.get("all_bank_records_approval") != "P2.9b":
        raise ValueError("All-bank record generation requires all_bank_records_approval='P2.9b'")
    forbidden_scale_fields = _find_forbidden_datatype_scale_fields(manifest)
    if forbidden_scale_fields:
        raise ValueError(f"Datatype scaling fields are not supported: {forbidden_scale_fields}")

    pim_ops = set(manifest["pim_compute_operator_classes"])
    host_support_ops = set(manifest["host_support_record_classes"])
    overlap = pim_ops & host_support_ops
    if overlap:
        raise ValueError(f"Operator classes overlap between pim_compute and host_support: {overlap}")
    overlap_prov = pim_ops & set(manifest["host_only_provenance_classes"])
    if overlap_prov:
        raise ValueError(f"Operator classes overlap between pim_compute and host_only_provenance: {overlap_prov}")
    overlap_prov2 = host_support_ops & set(manifest["host_only_provenance_classes"])
    if overlap_prov2:
        raise ValueError(f"Operator classes overlap between host_support and host_only_provenance: {overlap_prov2}")

    if not host_support_ops >= KV_CACHE_READ_OPS:
        raise ValueError(
            f"host_support_record_classes must include KV cache reads (missing: {sorted(KV_CACHE_READ_OPS - host_support_ops)})"
        )

    pim_kv_infiltration = {op for op in pim_ops if "cache_read" in op}
    if pim_kv_infiltration:
        raise ValueError(
            f"KV cache reads must be in host_support_record_classes, not pim_compute_operator_classes (found: {sorted(pim_kv_infiltration)})"
        )

    pim_non_gemv = {op for op in pim_ops if not op.endswith(PIM_COMPUTE_OP_SUFFIX)}
    if pim_non_gemv:
        raise ValueError(
            f"pim_compute_operator_classes must only contain GEMV operators (non-GEMV: {sorted(pim_non_gemv)})"
        )


def validate_record(record: dict) -> None:
    required_common = {
        "schema_version",
        "record_id",
        "kind",
        "phase",
        "layer",
        "op",
        "repeat",
        "provenance",
        "mapping_policy",
    }
    missing = sorted(required_common - set(record))
    if missing:
        raise ValueError(f"Record missing required common fields: {missing}")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version: {record['schema_version']}")
    if record["kind"] not in SEMANTIC_RECORD_KINDS:
        raise ValueError(f"Unsupported Phase 2 semantic record kind: {record['kind']}")
    if record["repeat"] <= 0:
        raise ValueError(f"Repeat must be positive, got {record['repeat']}")

    kind = record["kind"]
    if kind in {"HostRead", "HostWrite"}:
        for field in ("bytes", "address_policy"):
            if field not in record:
                raise ValueError(f"{kind} missing required field '{field}'")
    elif kind == "PIMCompute":
        for field in (
            "num_requests",
            "bank_sequence",
            "dependency_context",
            "row_policy",
            "column_policy",
            "datatype_metadata",
        ):
            if field not in record:
                raise ValueError(f"PIMCompute missing required field '{field}'")
    elif kind in ALL_BANK_RECORD_KINDS:
        for field in (
            "num_requests",
            "all_bank_scope",
            "dependency_context",
            "row_policy",
            "column_policy",
            "datatype_metadata",
        ):
            if field not in record:
                raise ValueError(f"{kind} missing required field '{field}'")
    elif kind == "Barrier":
        if "barrier_scope" not in record:
            raise ValueError("Barrier missing required field 'barrier_scope'")
    elif kind == "Drain":
        if "drain_scope" not in record:
            raise ValueError("Drain missing required field 'drain_scope'")
    elif kind in TRANSFORMER_DATAFLOW_RECORD_KINDS:
        _validate_transformer_dataflow_record(record)


def _validate_transformer_metadata(record: dict) -> None:
    for field in ("tensor_io", "logical_dependencies", "operator_context", "residency"):
        if field not in record:
            raise ValueError(f"{record['kind']} missing required transformer dataflow field '{field}'")

    tensor_io = record["tensor_io"]
    if not isinstance(tensor_io, dict):
        raise ValueError(f"{record['kind']} tensor_io must be a map")
    for field in ("inputs", "outputs"):
        if field not in tensor_io or not isinstance(tensor_io[field], list):
            raise ValueError(f"{record['kind']} tensor_io.{field} must be a list")

    dependencies = record["logical_dependencies"]
    if not isinstance(dependencies, list):
        raise ValueError(f"{record['kind']} logical_dependencies must be a list")
    if any(not isinstance(dependency, str) for dependency in dependencies):
        raise ValueError(f"{record['kind']} logical_dependencies entries must be strings")

    context = record["operator_context"]
    if not isinstance(context, dict):
        raise ValueError(f"{record['kind']} operator_context must be a map")
    for field in ("operator_family", "stage", "record_family"):
        if field not in context:
            raise ValueError(f"{record['kind']} operator_context missing required field '{field}'")
    if context["record_family"] != REQUIRED_P4_RECORD_FAMILY:
        raise ValueError(
            f"{record['kind']} operator_context.record_family must equal {REQUIRED_P4_RECORD_FAMILY!r}"
        )

    provenance = record["provenance"]
    if not isinstance(provenance, dict):
        raise ValueError(f"{record['kind']} provenance must be a map")
    for claim in REQUIRED_P4_CLAIM_BOUNDARY:
        if claim not in provenance.get("claim_boundary", []):
            raise ValueError(f"{record['kind']} provenance.claim_boundary missing {claim!r}")
    for non_claim in REQUIRED_P4_NON_CLAIMS:
        if non_claim not in provenance.get("non_claims", []):
            raise ValueError(f"{record['kind']} provenance.non_claims missing {non_claim!r}")

    residency = record["residency"]
    if not isinstance(residency, dict):
        raise ValueError(f"{record['kind']} residency must be a map")
    for tensor in tensor_io["inputs"] + tensor_io["outputs"]:
        if not isinstance(tensor, str):
            raise ValueError(f"{record['kind']} tensor_io entries must be strings")


def _validate_transformer_compute_record(record: dict) -> None:
    for field in (
        "num_requests",
        "bank_sequence",
        "dependency_context",
        "row_policy",
        "column_policy",
        "datatype_metadata",
        "compute_shape",
        "burst_length",
    ):
        if field not in record:
            raise ValueError(f"{record['kind']} missing required compute field '{field}'")
    if int(record["num_requests"]) <= 0:
        raise ValueError(f"{record['kind']} num_requests must be positive")
    if not isinstance(record["bank_sequence"], list) or not record["bank_sequence"]:
        raise ValueError(f"{record['kind']} bank_sequence must be a non-empty list")
    if int(record["burst_length"]) <= 0:
        raise ValueError(f"{record['kind']} burst_length must be positive")
    compute_shape = record["compute_shape"]
    if not isinstance(compute_shape, dict):
        raise ValueError(f"{record['kind']} compute_shape must be a map")
    for field in ("m", "n", "k", "output_elements", "datatype"):
        if field not in compute_shape:
            raise ValueError(f"{record['kind']} compute_shape missing required field '{field}'")


def _validate_transformer_data_move_record(record: dict) -> None:
    for field in ("movement_policy", "num_requests", "dependency_context", "row_policy", "column_policy", "datatype_metadata"):
        if field not in record:
            raise ValueError(f"PIMDataMove missing required movement field '{field}'")
    if int(record["num_requests"]) <= 0:
        raise ValueError("PIMDataMove num_requests must be positive")
    movement_policy = record["movement_policy"]
    if not isinstance(movement_policy, dict):
        raise ValueError("PIMDataMove movement_policy must be a map")
    if "movement_kind" not in movement_policy:
        raise ValueError("PIMDataMove movement_policy missing required field 'movement_kind'")
    movement_kind = movement_policy["movement_kind"]
    if movement_kind not in VALID_PIM_DATA_MOVE_MOVEMENT_KINDS:
        raise ValueError(
            f"PIMDataMove movement_policy.movement_kind {movement_kind!r} "
            f"is not a recognized movement kind; must be one of {sorted(VALID_PIM_DATA_MOVE_MOVEMENT_KINDS)}"
        )


def _validate_transformer_accounting_record(record: dict) -> None:
    if "accounting_metadata" not in record:
        raise ValueError(f"{record['kind']} missing required field 'accounting_metadata'")
    if not isinstance(record["accounting_metadata"], dict):
        raise ValueError(f"{record['kind']} accounting_metadata must be a map")


def _validate_transformer_dataflow_record(record: dict) -> None:
    _validate_transformer_metadata(record)
    kind = record["kind"]
    if kind in TRANSFORMER_COMPUTE_RECORD_KINDS:
        _validate_transformer_compute_record(record)
    elif kind == "PIMDataMove":
        _validate_transformer_data_move_record(record)
    elif kind in TRANSFORMER_ACCOUNTING_RECORD_KINDS:
        _validate_transformer_accounting_record(record)


def expanded_record_count(records: list[dict]) -> int:
    return sum(int(record["repeat"]) for record in records)


def write_jsonl(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            validate_record(record)
            handle.write(stable_json_dumps(record) + "\n")


def write_json(data: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(stable_json_pretty(data), encoding="utf-8")
