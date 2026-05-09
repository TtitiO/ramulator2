"""Tiny Phase 4 full-transformer semantic dataflow generator.

This module intentionally keeps workload compilation offline in Python.  It
emits native Ramulator2-ext semantic tensor-DAG records, not raw AttAcc traces
or HBM3 opcodes.  The first implemented slice is attention score -> softmax
accounting -> context with explicit tensor IO, dependencies, residency, and
tile/head metadata.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ramulator.workload_surrogate.structured_trace import (
    GENERATOR_VERSION,
    REQUIRED_P4_RECORD_FAMILY,
    SCHEMA_VERSION,
    expanded_record_count,
    validate_record,
    write_json,
    write_jsonl,
)


FULL_TRANSFORMER_GENERATOR_VERSION = f"{GENERATOR_VERSION}-p4-full-transformer"
DEFAULT_OUTPUT_DIR = Path("ramulator2/tests/data/structured_workload_surrogate/full_transformer_attention_v0_1")
REQUIRED_P4_NON_CLAIMS = {
    "not_runtime_replay",
    "not_vllm_replay",
    "not_numerical_correctness",
    "not_silicon_faithful_softmax_or_data_movement",
    "not_raw_attacc_schema",
}
SUPPORTED_ATTENTION_DATATYPES = {"int8", "fp16", "bf16"}
_VALID_DISTRIBUTION_POLICIES = {"broadcast", "bank_sharded", "replicated"}


def get_tiny_attention_manifest() -> dict:
    return {
        "manifest_version": "p4-attention-v0.1",
        "manifest_name": "tiny_attention_p4_v0_1",
        "workload_class": "structured_transformer_attention_surrogate",
        "phase": "decode",
        "model_family": "tiny decoder-only transformer attention slice",
        "num_layers": 1,
        "num_heads": 2,
        "head_dim": 32,
        "past_len": 64,
        "seq_len": 1,
        "datatype": "int8",
        "score_tile_tokens": 32,
        "context_tile_tokens": 32,
        "head_group_size": 1,
        "schedule_policy": "serialized",
        "operand_movement_policy": {
            "weights": "preloaded_stationary",
            "dynamic_activation_setup": "materialized",
            "ffn_intermediate": "bank_local_capacity_controlled",
        },
        "ramulator_visible_defaults": {
            "bank_sequence": [0, 1, 2, 3],
            "bank_sequence_order": "frontend",
            "pim_banks_per_mpu": 1,
            "burst_length": 1,
            "row_start": 0,
            "row_count": 16,
            "dependency_count": 8,
            "column_start": 0,
        },
        "mapping_policy": {
            "host_policy": "semantic_tensor_io_only",
            "pim_policy": "native_lpddr5_pim_attention_tiles",
            "bank_sequence_policy": "manifest_order",
            "mpu_grouping_policy": "manifest_pim_banks_per_mpu",
        },
        "literature_anchors": ["AttAcc workflow ideology only", "LPDDR5-PIM native opcode surface"],
        "non_claims": [
            "not_runtime_replay",
            "not_vllm_replay",
            "not_numerical_correctness",
            "not_silicon_faithful_softmax_or_data_movement",
            "not_raw_attacc_schema",
        ],
    }


def get_tiny_ffn_manifest() -> dict:
    return {
        "manifest_version": "p4-ffn-v0.1",
        "manifest_name": "tiny_ffn_swiglu_p4_v0_1",
        "workload_class": "structured_transformer_ffn_swiglu_surrogate",
        "phase": "decode",
        "model_family": "tiny decoder-only transformer FFN/SwiGLU slice",
        "num_layers": 1,
        "seq_len": 1,
        "hidden_size": 32,
        "ffn_hidden_size": 64,
        "ffn_activation_tile_size": 16,
        "activation_distribution_policy": "broadcast",
        "activation": "silu",
        "datatype": "int8",
        "schedule_policy": "serialized",
        "operand_movement_policy": {
            "weights": "preloaded_stationary",
            "dynamic_activation_setup": "materialized",
            "ffn_intermediate": "bank_local_capacity_controlled",
        },
        "ramulator_visible_defaults": {
            "bank_sequence": [0, 1, 2, 3],
            "bank_sequence_order": "frontend",
            "pim_banks_per_mpu": 1,
            "burst_length": 1,
            "row_start": 0,
            "row_count": 16,
            "dependency_count": 8,
            "column_start": 0,
        },
        "mapping_policy": {
            "host_policy": "semantic_tensor_io_only",
            "pim_policy": "native_lpddr5_pim_ffn_tiles",
            "bank_sequence_policy": "manifest_order",
            "mpu_grouping_policy": "manifest_pim_banks_per_mpu",
        },
        "literature_anchors": ["LPDDR5-PIM native opcode surface"],
        "non_claims": [
            "not_runtime_replay",
            "not_vllm_replay",
            "not_numerical_correctness",
            "not_silicon_faithful_softmax_or_data_movement",
            "not_raw_attacc_schema",
        ],
    }


def get_tiny_moe_manifest() -> dict:
    return {
        "manifest_version": "p4-moe-v0.1",
        "manifest_name": "tiny_moe_p4_v0_1",
        "workload_class": "structured_transformer_moe_surrogate",
        "phase": "decode",
        "model_family": "tiny decoder-only transformer MoE slice",
        "num_layers": 1,
        "seq_len": 1,
        "hidden_size": 32,
        "expert_hidden_size": 64,
        "num_experts": 4,
        "top_k": 2,
        "selected_experts": [0, 1],
        "datatype": "int8",
        "schedule_policy": "serialized",
        "operand_movement_policy": {
            "weights": "preloaded_stationary",
            "router_input_setup": "materialized",
            "token_dispatch": "materialized",
            "expert_output_combine": "materialized",
        },
        "ramulator_visible_defaults": {
            "bank_sequence": [0, 1, 2, 3],
            "bank_sequence_order": "frontend",
            "pim_banks_per_mpu": 1,
            "burst_length": 1,
            "row_start": 0,
            "row_count": 16,
            "dependency_count": 8,
            "column_start": 0,
        },
        "mapping_policy": {
            "host_policy": "semantic_tensor_io_only",
            "pim_policy": "native_lpddr5_pim_moe_tiles",
            "bank_sequence_policy": "manifest_order",
            "mpu_grouping_policy": "manifest_pim_banks_per_mpu",
        },
        "literature_anchors": ["LPDDR5-PIM native opcode surface"],
        "non_claims": [
            "not_runtime_replay",
            "not_vllm_replay",
            "not_numerical_correctness",
            "not_silicon_faithful_softmax_or_data_movement",
            "not_raw_attacc_schema",
        ],
    }


def _validate_ramulator_defaults(manifest: dict, *, manifest_kind: str) -> None:
    defaults = manifest["ramulator_visible_defaults"]
    for field in ("bank_sequence", "burst_length", "row_start", "row_count", "dependency_count", "column_start"):
        if field not in defaults:
            raise ValueError(f"{manifest_kind} manifest ramulator_visible_defaults missing {field}")
    if not defaults["bank_sequence"]:
        raise ValueError(f"{manifest_kind} manifest bank_sequence must be non-empty")
    for field in ("burst_length", "row_count", "dependency_count"):
        if int(defaults[field]) <= 0:
            raise ValueError(f"{manifest_kind} manifest ramulator_visible_defaults.{field} must be positive")


def _validate_attention_manifest(manifest: dict) -> None:
    required = {
        "manifest_version",
        "manifest_name",
        "workload_class",
        "phase",
        "model_family",
        "num_layers",
        "num_heads",
        "head_dim",
        "past_len",
        "seq_len",
        "datatype",
        "score_tile_tokens",
        "context_tile_tokens",
        "head_group_size",
        "schedule_policy",
        "ramulator_visible_defaults",
        "mapping_policy",
        "literature_anchors",
        "non_claims",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"Attention manifest missing required fields: {missing}")
    if manifest["phase"] != "decode":
        raise ValueError("Tiny attention generator currently supports phase='decode' only")
    if manifest["schedule_policy"] not in {"serialized", "overlap_independent_heads"}:
        raise ValueError("Attention schedule_policy must be 'serialized' or 'overlap_independent_heads'")
    if manifest["datatype"] not in SUPPORTED_ATTENTION_DATATYPES:
        raise ValueError(
            f"Unsupported attention datatype {manifest['datatype']!r}; supported datatypes are {sorted(SUPPORTED_ATTENTION_DATATYPES)}"
        )
    missing_non_claims = sorted(REQUIRED_P4_NON_CLAIMS - set(manifest["non_claims"]))
    if missing_non_claims:
        raise ValueError(f"Attention manifest non_claims missing required entries: {missing_non_claims}")
    for field in ("num_layers", "num_heads", "head_dim", "past_len", "seq_len", "score_tile_tokens", "context_tile_tokens", "head_group_size"):
        if int(manifest[field]) <= 0:
            raise ValueError(f"Attention manifest {field} must be positive")
    if manifest["schedule_policy"] == "overlap_independent_heads" and int(manifest["num_heads"]) < 2:
        raise ValueError("overlap_independent_heads requires num_heads >= 2")
    if int(manifest["context_tile_tokens"]) != int(manifest["score_tile_tokens"]):
        raise ValueError("P4.2 attention first slice requires context_tile_tokens == score_tile_tokens")
    _validate_ramulator_defaults(manifest, manifest_kind="Attention")


def _validate_ffn_manifest(manifest: dict) -> None:
    required = {
        "manifest_version",
        "manifest_name",
        "workload_class",
        "phase",
        "model_family",
        "num_layers",
        "seq_len",
        "hidden_size",
        "ffn_hidden_size",
        "ffn_activation_tile_size",
        "activation_distribution_policy",
        "activation",
        "datatype",
        "schedule_policy",
        "ramulator_visible_defaults",
        "mapping_policy",
        "literature_anchors",
        "non_claims",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"FFN manifest missing required fields: {missing}")
    if manifest["phase"] != "decode":
        raise ValueError("Tiny FFN generator currently supports phase='decode' only")
    if manifest["schedule_policy"] != "serialized":
        raise ValueError("P4.3 implements serialized FFN scheduling first")
    if manifest["datatype"] not in SUPPORTED_ATTENTION_DATATYPES:
        raise ValueError(f"Unsupported FFN datatype {manifest['datatype']!r}; supported datatypes are {sorted(SUPPORTED_ATTENTION_DATATYPES)}")
    missing_non_claims = sorted(REQUIRED_P4_NON_CLAIMS - set(manifest["non_claims"]))
    if missing_non_claims:
        raise ValueError(f"FFN manifest non_claims missing required entries: {missing_non_claims}")
    for field in ("num_layers", "seq_len", "hidden_size", "ffn_hidden_size", "ffn_activation_tile_size"):
        if int(manifest[field]) <= 0:
            raise ValueError(f"FFN manifest {field} must be positive")
    if int(manifest["ffn_activation_tile_size"]) > int(manifest["hidden_size"]):
        raise ValueError("FFN manifest ffn_activation_tile_size must be <= hidden_size")
    if manifest["activation_distribution_policy"] not in _VALID_DISTRIBUTION_POLICIES:
        raise ValueError(
            f"FFN manifest activation_distribution_policy {manifest['activation_distribution_policy']!r} "
            f"not in {sorted(_VALID_DISTRIBUTION_POLICIES)}"
        )
    _validate_ramulator_defaults(manifest, manifest_kind="FFN")


def _validate_moe_manifest(manifest: dict) -> None:
    required = {
        "manifest_version",
        "manifest_name",
        "workload_class",
        "phase",
        "model_family",
        "num_layers",
        "seq_len",
        "hidden_size",
        "expert_hidden_size",
        "num_experts",
        "top_k",
        "selected_experts",
        "datatype",
        "schedule_policy",
        "ramulator_visible_defaults",
        "mapping_policy",
        "literature_anchors",
        "non_claims",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"MoE manifest missing required fields: {missing}")
    if manifest["phase"] != "decode":
        raise ValueError("Tiny MoE generator currently supports phase='decode' only")
    if manifest["schedule_policy"] not in {"serialized", "overlap_selected_experts"}:
        raise ValueError("MoE schedule_policy must be 'serialized' or 'overlap_selected_experts'")
    if manifest["datatype"] not in SUPPORTED_ATTENTION_DATATYPES:
        raise ValueError(f"Unsupported MoE datatype {manifest['datatype']!r}; supported datatypes are {sorted(SUPPORTED_ATTENTION_DATATYPES)}")
    missing_non_claims = sorted(REQUIRED_P4_NON_CLAIMS - set(manifest["non_claims"]))
    if missing_non_claims:
        raise ValueError(f"MoE manifest non_claims missing required entries: {missing_non_claims}")
    for field in ("num_layers", "seq_len", "hidden_size", "expert_hidden_size", "num_experts", "top_k"):
        if int(manifest[field]) <= 0:
            raise ValueError(f"MoE manifest {field} must be positive")
    if manifest["schedule_policy"] == "overlap_selected_experts" and int(manifest["top_k"]) < 2:
        raise ValueError("overlap_selected_experts requires top_k >= 2")
    if int(manifest["top_k"]) > int(manifest["num_experts"]):
        raise ValueError("MoE manifest top_k must be <= num_experts")
    selected = list(manifest["selected_experts"])
    if len(selected) != int(manifest["top_k"]):
        raise ValueError("MoE manifest selected_experts length must equal top_k")
    if len(set(selected)) != len(selected) or any(int(expert) < 0 or int(expert) >= int(manifest["num_experts"]) for expert in selected):
        raise ValueError("MoE manifest selected_experts must be unique ids in [0, num_experts)")
    _validate_ramulator_defaults(manifest, manifest_kind="MoE")


def _lanes(datatype: str) -> int:
    if datatype not in SUPPORTED_ATTENTION_DATATYPES:
        raise ValueError(f"Unsupported attention datatype {datatype!r}; supported datatypes are {sorted(SUPPORTED_ATTENTION_DATATYPES)}")
    return 32 if datatype == "int8" else 16


def _p4_context(operator_family: str, stage: str, **fields: object) -> dict:
    return {
        "record_family": REQUIRED_P4_RECORD_FAMILY,
        "operator_family": operator_family,
        "stage": stage,
        **fields,
    }


def _num_requests(elements: int, datatype: str) -> int:
    """Return MAC requests needed to cover scalar elements at datatype lane width."""
    lanes = _lanes(datatype)
    return max(1, (int(elements) + lanes - 1) // lanes)


def _mapping_policy(manifest: dict) -> dict:
    defaults = manifest["ramulator_visible_defaults"]
    policy = dict(manifest["mapping_policy"])
    policy.update(
        {
            "controller_bank_order": defaults.get("bank_sequence_order", "frontend"),
            "bank_sequence": list(defaults["bank_sequence"]),
            "pim_banks_per_mpu": int(defaults.get("pim_banks_per_mpu", 1)),
            "burst_length": int(defaults["burst_length"]),
            "row_start": int(defaults["row_start"]),
            "row_count": int(defaults["row_count"]),
        }
    )
    return policy


def _provenance(manifest: dict, op: str) -> dict:
    return {
        "source_kind": "generated",
        "tuple_manifest": manifest["manifest_name"],
        "literature_anchor": list(manifest["literature_anchors"]),
        "generator_version": FULL_TRANSFORMER_GENERATOR_VERSION,
        "schedule_policy": manifest.get("schedule_policy", "serialized"),
        "claim_boundary": [
            "structured transformer dataflow surrogate",
            "simulator-diagnostic",
            "non-silicon-calibrated",
            "operator-internal-dataflow-first",
        ],
        "non_claims": list(manifest["non_claims"]),
        "selected_model_family": manifest["model_family"],
        "selected_workload_regime": manifest["workload_class"],
        "notes": f"bounded Phase 4 semantic tensor-DAG record for {op}",
    }


def _base_record(record_id: str, kind: str, layer_index: int, op: str, manifest: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "record_id": record_id,
        "kind": kind,
        "phase": manifest["phase"],
        "layer": f"layer_{layer_index:02d}",
        "op": op,
        "repeat": 1,
        "provenance": _provenance(manifest, op),
        "mapping_policy": _mapping_policy(manifest),
    }


def _residency(*tensors: str) -> dict:
    return {tensor: "logical_pim_resident_or_accounted" for tensor in tensors}


def _policies(manifest: dict, *, layer_index: int, head_index: int, tile_index: int) -> dict:
    defaults = manifest["ramulator_visible_defaults"]
    row_count = int(defaults["row_count"])
    dependency_count = int(defaults["dependency_count"])
    row_start = int(defaults["row_start"])
    column_start = int(defaults["column_start"])
    dependency_id = (layer_index * int(manifest["num_heads"]) + head_index + tile_index) % dependency_count
    resolved_row = row_start + ((layer_index + head_index + tile_index) % row_count)
    resolved_column = column_start + dependency_id
    return {
        "dependency_context": {
            "kind": "attention_tile_dependency",
            "dependency_count": dependency_count,
            "dependency_id": dependency_id,
            "head_id": head_index,
            "tile_id": tile_index,
        },
        "row_policy": {
            "kind": "bounded_attention_tile_rows",
            "row_start": row_start,
            "row_count": row_count,
            "resolved_row": resolved_row,
        },
        "column_policy": {
            "kind": "attention_dependency_column_round_robin",
            "column_start": column_start,
            "resolved_column": resolved_column,
        },
    }


def _attention_compute_record(
    record_id: str,
    kind: str,
    layer_index: int,
    op: str,
    manifest: dict,
    *,
    head_index: int,
    tile_index: int,
    tile_tokens: int,
    inputs: list[str],
    outputs: list[str],
    dependencies: list[str],
    tile_start: int = 0,
) -> dict:
    """Build an attention score/context compute record with attention metadata."""
    record = _base_record(record_id, kind, layer_index, op, manifest)
    policies = _policies(manifest, layer_index=layer_index, head_index=head_index, tile_index=tile_index)
    record.update(
        {
            "tensor_io": {"inputs": inputs, "outputs": outputs},
            "logical_dependencies": dependencies,
            "operator_context": _p4_context(
                "attention",
                "score" if kind == "AttentionScore" else "context",
                layer_id=layer_index,
                head_id=head_index,
                head_group_id=head_index // int(manifest["head_group_size"]),
                tile_id=tile_index,
                tile_tokens=tile_tokens,
                tile_start=tile_start,
            ),
            "residency": _residency(*(inputs + outputs)),
            "compute_shape": {
                "m": int(manifest["seq_len"]),
                "n": tile_tokens if kind == "AttentionScore" else int(manifest["head_dim"]),
                "k": int(manifest["head_dim"]) if kind == "AttentionScore" else tile_tokens,
                "output_elements": max(
                    1,
                    int(manifest["seq_len"]) * (tile_tokens if kind == "AttentionScore" else int(manifest["head_dim"])),
                ),
                "datatype": manifest["datatype"],
            },
            "num_requests": _num_requests(tile_tokens * int(manifest["head_dim"]), manifest["datatype"]),
            "bank_sequence": list(manifest["ramulator_visible_defaults"]["bank_sequence"]),
            "datatype_metadata": {
                "datatype": manifest["datatype"],
                "role": "phase4_attention_tensor_dag_metadata",
                "behavior_claim": "explicit_resource_rows_only",
            },
            "burst_length": int(manifest["ramulator_visible_defaults"]["burst_length"]),
        }
    )
    record.update(policies)
    return record


def _softmax_record(
    record_id: str,
    layer_index: int,
    manifest: dict,
    *,
    head_index: int,
    tile_index: int,
    tile_tokens: int,
    score_record_ids: list[str],
    score_tensors: list[str] | None = None,
    probability_tensors: list[str] | None = None,
    tile_start: int = 0,
) -> dict:
    score_tensor = f"L{layer_index}.H{head_index}.T{tile_index}.score"
    prob_tensor = f"L{layer_index}.H{head_index}.T{tile_index}.probability"
    score_tensors = [score_tensor] if score_tensors is None else score_tensors
    probability_tensors = [prob_tensor] if probability_tensors is None else probability_tensors
    record = _base_record(record_id, "AttentionSoftmax", layer_index, "attention_softmax_accounting", manifest)
    record.update(
        {
            "tensor_io": {"inputs": list(score_tensors), "outputs": list(probability_tensors)},
            "logical_dependencies": list(score_record_ids),
            "operator_context": _p4_context(
                "attention",
                "global_softmax_accounting",
                layer_id=layer_index,
                head_id=head_index,
                head_group_id=head_index // int(manifest["head_group_size"]),
                tile_id=tile_index,
                tile_tokens=tile_tokens,
                tile_start=tile_start,
            ),
            "residency": _residency(*(list(score_tensors) + list(probability_tensors))),
            "accounting_metadata": {
                "kind": "semantic_only_softmax",
                "scope": "head_global_across_score_tiles",
                "elements": max(1, int(manifest["seq_len"]) * tile_tokens),
                "lowering": "not_lowered_to_native_opcode_in_p4_2",
            },
        }
    )
    return record


def _data_move_record(
    record_id: str,
    layer_index: int,
    op: str,
    manifest: dict,
    *,
    head_index: int,
    tile_index: int,
    tile_tokens: int,
    inputs: list[str],
    outputs: list[str],
    dependencies: list[str],
    tile_start: int = 0,
) -> dict:
    record = _base_record(record_id, "PIMDataMove", layer_index, op, manifest)
    policies = _policies(manifest, layer_index=layer_index, head_index=head_index, tile_index=tile_index)
    record.update(
        {
            "tensor_io": {"inputs": inputs, "outputs": outputs},
            "logical_dependencies": dependencies,
            "operator_context": _p4_context(
                "attention",
                "data_movement",
                layer_id=layer_index,
                head_id=head_index,
                head_group_id=head_index // int(manifest["head_group_size"]),
                tile_id=tile_index,
                tile_tokens=tile_tokens,
                tile_start=tile_start,
            ),
            "residency": _residency(*(inputs + outputs)),
            "movement_policy": {
                "movement_kind": "broadcast_or_accounted_tile_load",
                "lowering_preference": "native_pim_bcast_when_supported",
                "tile_tokens": tile_tokens,
            },
            "num_requests": 1,
            "datatype_metadata": {
                "datatype": manifest["datatype"],
                "role": "phase4_attention_data_movement_metadata",
                "behavior_claim": "semantic_movement_not_silicon_faithful",
            },
        }
    )
    record.update(policies)
    return record


def _semantic_operand_record(
    record_id: str,
    kind: str,
    layer_index: int,
    op: str,
    manifest: dict,
    *,
    operator_family: str,
    stage: str,
    inputs: list[str],
    outputs: list[str],
    dependencies: list[str],
    operand_role: str,
    residency: str,
    materialized: bool,
    reuse_scope: str,
    lowering: str,
    context_fields: dict | None = None,
) -> dict:
    if kind not in {"PIMOperandResidency", "PIMOperandReuse"}:
        raise ValueError(f"Unsupported semantic operand record kind: {kind}")
    context_fields = {} if context_fields is None else dict(context_fields)
    record = _base_record(record_id, kind, layer_index, op, manifest)
    record.update(
        {
            "tensor_io": {"inputs": inputs, "outputs": outputs},
            "logical_dependencies": dependencies,
            "operator_context": _p4_context(
                operator_family,
                stage,
                layer_id=layer_index,
                **context_fields,
            ),
            "residency": _residency(*(inputs + outputs)),
            "accounting_metadata": {
                "kind": "semantic_only_operand_residency" if kind == "PIMOperandResidency" else "semantic_only_operand_reuse",
                "operand_role": operand_role,
                "residency": residency,
                "materialized": materialized,
                "reuse_scope": reuse_scope,
                "lowering": lowering,
            },
        }
    )
    return record


def _ffn_data_move_record(
    record_id: str,
    layer_index: int,
    manifest: dict,
    *,
    op: str,
    stage_index: int,
    inputs: list[str],
    outputs: list[str],
    dependencies: list[str],
    operand_role: str,
    residency: str,
    reuse_scope: str,
    tile_index: int = 0,
    tile_elements: int = 0,
    tile_start: int = 0,
    movement_elements: int = 0,
) -> dict:
    record = _base_record(record_id, "PIMDataMove", layer_index, op, manifest)
    distribution_policy = manifest.get("activation_distribution_policy", "broadcast")
    movement_num_requests = _num_requests(movement_elements, manifest["datatype"]) if movement_elements > 0 else 1
    movement_behavior_claim = (
        "semantic_movement_volume_proportional_not_tiled_or_silicon_faithful"
        if operand_role == "weight" and movement_elements > 0
        else "semantic_movement_not_silicon_faithful"
    )
    record.update(
        {
            "tensor_io": {"inputs": inputs, "outputs": outputs},
            "logical_dependencies": dependencies,
            "operator_context": _p4_context(
                "ffn_swiglu",
                "data_movement",
                layer_id=layer_index,
                stage_index=stage_index,
                tile_id=tile_index,
                tile_elements=tile_elements,
                tile_start=tile_start,
            ),
            "residency": _residency(*(inputs + outputs)),
            "movement_policy": {
                "movement_kind": "broadcast_or_accounted_tile_load",
                "lowering_preference": "native_pim_bcast_when_supported",
                "operand_role": operand_role,
                "residency": residency,
                "materialized": True,
                "reuse_scope": reuse_scope,
                "lowering": "native_pim_bcast_when_supported",
                "tile_index": tile_index,
                "tile_elements": tile_elements,
                "tile_start": tile_start,
                "distribution_scope": distribution_policy,
            },
            "num_requests": movement_num_requests,
            "datatype_metadata": {
                "datatype": manifest["datatype"],
                "role": "phase4_ffn_operand_movement_metadata",
                "behavior_claim": movement_behavior_claim,
            },
        }
    )
    record.update(_ffn_policies(manifest, layer_index=layer_index, stage_index=stage_index, tile_index=tile_index))
    return record


def _moe_data_move_record(
    record_id: str,
    layer_index: int,
    manifest: dict,
    *,
    op: str,
    stage_index: int,
    inputs: list[str],
    outputs: list[str],
    dependencies: list[str],
    operand_role: str,
    residency: str,
    reuse_scope: str,
    expert_id: int | None = None,
    movement_elements: int = 0,
) -> dict:
    record = _base_record(record_id, "PIMDataMove", layer_index, op, manifest)
    movement_num_requests = _num_requests(movement_elements, manifest["datatype"]) if movement_elements > 0 else 1
    movement_behavior_claim = (
        "semantic_movement_volume_proportional_not_tiled_or_silicon_faithful"
        if operand_role == "weight" and movement_elements > 0
        else "semantic_movement_not_silicon_faithful"
    )
    record.update(
        {
            "tensor_io": {"inputs": inputs, "outputs": outputs},
            "logical_dependencies": dependencies,
            "operator_context": _p4_context(
                "moe",
                "data_movement",
                layer_id=layer_index,
                stage_index=stage_index,
                expert_id=expert_id,
            ),
            "residency": _residency(*(inputs + outputs)),
            "movement_policy": {
                "movement_kind": "broadcast_or_accounted_tile_load",
                "lowering_preference": "native_pim_bcast_when_supported",
                "operand_role": operand_role,
                "residency": residency,
                "materialized": True,
                "reuse_scope": reuse_scope,
                "lowering": "native_pim_bcast_when_supported",
            },
            "num_requests": movement_num_requests,
            "datatype_metadata": {
                "datatype": manifest["datatype"],
                "role": "phase4_moe_operand_movement_metadata",
                "behavior_claim": movement_behavior_claim,
            },
        }
    )
    record.update(_moe_policies(manifest, layer_index=layer_index, stage_index=stage_index, expert_id=expert_id))
    return record


def _context_reduction_record(
    record_id: str,
    layer_index: int,
    manifest: dict,
    *,
    head_index: int,
    context_record_ids: list[str],
    context_tensors: list[str],
) -> dict:
    output_tensor = f"L{layer_index}.H{head_index}.context_reduced"
    record = _base_record(record_id, "PIMElementwise", layer_index, "attention_context_reduction_accounting", manifest)
    record.update(
        {
            "tensor_io": {"inputs": list(context_tensors), "outputs": [output_tensor]},
            "logical_dependencies": list(context_record_ids),
            "operator_context": _p4_context(
                "attention",
                "context_reduction_accounting",
                layer_id=layer_index,
                head_id=head_index,
                head_group_id=head_index // int(manifest["head_group_size"]),
                tile_count=len(context_tensors),
            ),
            "residency": _residency(*(list(context_tensors) + [output_tensor])),
            "accounting_metadata": {
                "kind": "semantic_only_context_tile_reduction",
                "scope": "head_global_across_context_tiles",
                "elements": max(1, int(manifest["seq_len"]) * int(manifest["head_dim"])),
                "lowering": "not_lowered_to_native_opcode_in_p4_2",
            },
        }
    )
    return record


def _ffn_policies(manifest: dict, *, layer_index: int, stage_index: int, tile_index: int = 0) -> dict:
    defaults = manifest["ramulator_visible_defaults"]
    row_count = int(defaults["row_count"])
    dependency_count = int(defaults["dependency_count"])
    row_start = int(defaults["row_start"])
    column_start = int(defaults["column_start"])
    dependency_id = (layer_index + stage_index + tile_index) % dependency_count
    resolved_row = row_start + ((layer_index + stage_index + tile_index) % row_count)
    resolved_column = column_start + dependency_id
    return {
        "dependency_context": {
            "kind": "ffn_stage_dependency",
            "dependency_count": dependency_count,
            "dependency_id": dependency_id,
            "stage_index": stage_index,
            "tile_index": tile_index,
        },
        "row_policy": {
            "kind": "bounded_ffn_stage_rows",
            "row_start": row_start,
            "row_count": row_count,
            "resolved_row": resolved_row,
        },
        "column_policy": {
            "kind": "ffn_dependency_column_round_robin",
            "column_start": column_start,
            "resolved_column": resolved_column,
        },
    }


def _ffn_projection_record(
    record_id: str,
    layer_index: int,
    manifest: dict,
    *,
    op: str,
    stage_index: int,
    inputs: list[str],
    outputs: list[str],
    dependencies: list[str],
) -> dict:
    hidden_size = int(manifest["hidden_size"])
    ffn_hidden_size = int(manifest["ffn_hidden_size"])
    seq_len = int(manifest["seq_len"])
    if op in {"ffn_up_projection", "ffn_gate_projection"}:
        n = ffn_hidden_size
        k = hidden_size
    elif op == "ffn_down_projection":
        n = hidden_size
        k = ffn_hidden_size
    else:
        raise ValueError(f"Unsupported FFN projection op: {op}")

    record = _base_record(record_id, "FFNProjection", layer_index, op, manifest)
    record.update(
        {
            "tensor_io": {"inputs": inputs, "outputs": outputs},
            "logical_dependencies": dependencies,
            "operator_context": _p4_context(
                "ffn_swiglu",
                op,
                layer_id=layer_index,
                stage_index=stage_index,
            ),
            "residency": _residency(*(inputs + outputs)),
            "compute_shape": {
                "m": seq_len,
                "n": n,
                "k": k,
                "output_elements": max(1, seq_len * n),
                "datatype": manifest["datatype"],
            },
            "num_requests": _num_requests(seq_len * n * k, manifest["datatype"]),
            "bank_sequence": list(manifest["ramulator_visible_defaults"]["bank_sequence"]),
            "datatype_metadata": {
                "datatype": manifest["datatype"],
                "role": "phase4_ffn_tensor_dag_metadata",
                "behavior_claim": "explicit_resource_rows_only",
            },
            "burst_length": int(manifest["ramulator_visible_defaults"]["burst_length"]),
        }
    )
    record.update(_ffn_policies(manifest, layer_index=layer_index, stage_index=stage_index))
    return record


def _ffn_elementwise_record(
    record_id: str,
    layer_index: int,
    manifest: dict,
    *,
    op: str,
    stage_index: int,
    inputs: list[str],
    outputs: list[str],
    dependencies: list[str],
    elementwise_kind: str,
) -> dict:
    record = _base_record(record_id, "PIMElementwise", layer_index, op, manifest)
    record.update(
        {
            "tensor_io": {"inputs": inputs, "outputs": outputs},
            "logical_dependencies": dependencies,
            "operator_context": _p4_context(
                "ffn_swiglu",
                op,
                layer_id=layer_index,
                stage_index=stage_index,
            ),
            "residency": _residency(*(inputs + outputs)),
            "accounting_metadata": {
                "kind": elementwise_kind,
                "activation": manifest["activation"] if "activation" in elementwise_kind else None,
                "elements": max(1, int(manifest["seq_len"]) * int(manifest["ffn_hidden_size"])),
                "lowering": "not_lowered_to_native_opcode_in_p4_3",
            },
        }
    )
    return record


def _moe_policies(manifest: dict, *, layer_index: int, stage_index: int, expert_id: int | None = None) -> dict:
    assert expert_id is None or expert_id >= 0
    defaults = manifest["ramulator_visible_defaults"]
    row_count = int(defaults["row_count"])
    dependency_count = int(defaults["dependency_count"])
    row_start = int(defaults["row_start"])
    column_start = int(defaults["column_start"])
    expert_offset = int(manifest["num_experts"]) if expert_id is None else expert_id
    dependency_id = (layer_index + stage_index + expert_offset) % dependency_count
    resolved_row = row_start + ((layer_index + stage_index + expert_offset) % row_count)
    resolved_column = column_start + dependency_id
    return {
        "dependency_context": {
            "kind": "moe_stage_dependency",
            "dependency_count": dependency_count,
            "dependency_id": dependency_id,
            "stage_index": stage_index,
            "expert_id": expert_id,
            "moe_role": "router" if expert_id is None else "expert",
        },
        "row_policy": {
            "kind": "bounded_moe_stage_rows",
            "row_start": row_start,
            "row_count": row_count,
            "resolved_row": resolved_row,
        },
        "column_policy": {
            "kind": "moe_dependency_column_round_robin",
            "column_start": column_start,
            "resolved_column": resolved_column,
        },
    }


def _moe_compute_record(
    record_id: str,
    kind: str,
    layer_index: int,
    manifest: dict,
    *,
    op: str,
    stage_index: int,
    inputs: list[str],
    outputs: list[str],
    dependencies: list[str],
    expert_id: int | None = None,
) -> dict:
    hidden_size = int(manifest["hidden_size"])
    expert_hidden_size = int(manifest["expert_hidden_size"])
    seq_len = int(manifest["seq_len"])
    if kind == "MoERouter":
        n = int(manifest["num_experts"])
        k = hidden_size
    elif kind == "MoEExpertFFN":
        n = expert_hidden_size
        k = hidden_size
    else:
        raise ValueError(f"Unsupported MoE compute kind: {kind}")

    record = _base_record(record_id, kind, layer_index, op, manifest)
    record.update(
        {
            "tensor_io": {"inputs": inputs, "outputs": outputs},
            "logical_dependencies": dependencies,
            "operator_context": _p4_context(
                "moe",
                op,
                layer_id=layer_index,
                stage_index=stage_index,
                expert_id=expert_id,
            ),
            "residency": _residency(*(inputs + outputs)),
            "compute_shape": {
                "m": seq_len,
                "n": n,
                "k": k,
                "output_elements": max(1, seq_len * n),
                "datatype": manifest["datatype"],
            },
            "num_requests": _num_requests(seq_len * n * k, manifest["datatype"]),
            "bank_sequence": list(manifest["ramulator_visible_defaults"]["bank_sequence"]),
            "datatype_metadata": {
                "datatype": manifest["datatype"],
                "role": "phase4_moe_tensor_dag_metadata",
                "behavior_claim": "explicit_resource_rows_only",
            },
            "burst_length": int(manifest["ramulator_visible_defaults"]["burst_length"]),
        }
    )
    record.update(_moe_policies(manifest, layer_index=layer_index, stage_index=stage_index, expert_id=expert_id))
    return record


def _moe_accounting_record(
    record_id: str,
    kind: str,
    layer_index: int,
    manifest: dict,
    *,
    op: str,
    stage_index: int,
    inputs: list[str],
    outputs: list[str],
    dependencies: list[str],
    accounting_kind: str,
) -> dict:
    record = _base_record(record_id, kind, layer_index, op, manifest)
    selected_experts = [int(expert) for expert in manifest["selected_experts"]]
    record.update(
        {
            "tensor_io": {"inputs": inputs, "outputs": outputs},
            "logical_dependencies": dependencies,
            "operator_context": _p4_context(
                "moe",
                op,
                layer_id=layer_index,
                stage_index=stage_index,
                selected_experts=selected_experts,
                top_k=int(manifest["top_k"]),
            ),
            "residency": _residency(*(inputs + outputs)),
            "accounting_metadata": {
                "kind": accounting_kind,
                "selected_experts": selected_experts,
                "top_k": int(manifest["top_k"]),
                "elements": max(1, int(manifest["seq_len"]) * max(1, len(outputs))),
                "lowering": "not_lowered_to_native_opcode_in_p4_4",
            },
        }
    )
    return record


def _tile_ranges(total_tokens: int, tile_tokens: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < total_tokens:
        size = min(tile_tokens, total_tokens - start)
        ranges.append((start, size))
        start += size
    return ranges


def _add_dag_hint(records: list[dict], hint: str) -> None:
    for record in records:
        record["operator_context"]["dag_hint"] = hint


def _append_attention_score_records(
    records: list[dict],
    next_id: int,
    manifest: dict,
    *,
    layer_index: int,
    head_index: int,
    head_tiles: list[tuple[int, int]],
    score_record_ids: list[str],
) -> int:
    for tile_index, (tile_start, tile_tokens) in enumerate(head_tiles):
        q_tensor = f"L{layer_index}.H{head_index}.Q"
        k_tensor = f"L{layer_index}.H{head_index}.T{tile_index}.K"
        score_tensor = f"L{layer_index}.H{head_index}.T{tile_index}.score"
        k_load_id = f"rec_{next_id:04d}"
        records.append(
            _data_move_record(
                k_load_id,
                layer_index,
                "attention_k_tile_load",
                manifest,
                head_index=head_index,
                tile_index=tile_index,
                tile_tokens=tile_tokens,
                inputs=[k_tensor],
                outputs=[f"{k_tensor}.resident"],
                dependencies=[],
                tile_start=tile_start,
            )
        )
        next_id += 1

        score_id = f"rec_{next_id:04d}"
        records.append(
            _attention_compute_record(
                score_id,
                "AttentionScore",
                layer_index,
                "attention_score_gemv",
                manifest,
                head_index=head_index,
                tile_index=tile_index,
                tile_tokens=tile_tokens,
                inputs=[q_tensor, f"{k_tensor}.resident"],
                outputs=[score_tensor],
                dependencies=[k_load_id],
                tile_start=tile_start,
            )
        )
        score_record_ids.append(score_id)
        next_id += 1
    return next_id


def _append_attention_softmax_context_records(
    records: list[dict],
    next_id: int,
    manifest: dict,
    *,
    layer_index: int,
    head_index: int,
    head_tiles: list[tuple[int, int]],
    score_record_ids: list[str],
) -> int:
    softmax_id = f"rec_{next_id:04d}"
    score_tensors = [f"L{layer_index}.H{head_index}.T{tile_index}.score" for tile_index, _ in enumerate(head_tiles)]
    probability_tensors = [f"L{layer_index}.H{head_index}.T{tile_index}.probability" for tile_index, _ in enumerate(head_tiles)]
    records.append(
        _softmax_record(
            softmax_id,
            layer_index,
            manifest,
            head_index=head_index,
            # Softmax is head-global: tile_index=0 and tile_tokens=past_len encode the full head span.
            tile_index=0,
            tile_tokens=int(manifest["past_len"]),
            score_record_ids=score_record_ids,
            score_tensors=score_tensors,
            probability_tensors=probability_tensors,
            tile_start=0,
        )
    )
    next_id += 1

    context_record_ids: list[str] = []
    context_tensors: list[str] = []
    for tile_index, (tile_start, tile_tokens) in enumerate(head_tiles):
        prob_tensor = f"L{layer_index}.H{head_index}.T{tile_index}.probability"
        v_tensor = f"L{layer_index}.H{head_index}.T{tile_index}.V"
        context_tensor = f"L{layer_index}.H{head_index}.T{tile_index}.context"
        v_load_id = f"rec_{next_id:04d}"
        records.append(
            _data_move_record(
                v_load_id,
                layer_index,
                "attention_v_tile_load",
                manifest,
                head_index=head_index,
                tile_index=tile_index,
                tile_tokens=tile_tokens,
                inputs=[v_tensor],
                outputs=[f"{v_tensor}.resident"],
                dependencies=[softmax_id],
                tile_start=tile_start,
            )
        )
        next_id += 1

        context_id = f"rec_{next_id:04d}"
        records.append(
            _attention_compute_record(
                context_id,
                "AttentionContext",
                layer_index,
                "attention_context_gemv",
                manifest,
                head_index=head_index,
                tile_index=tile_index,
                tile_tokens=tile_tokens,
                inputs=[prob_tensor, f"{v_tensor}.resident"],
                outputs=[context_tensor],
                dependencies=[softmax_id, v_load_id],
                tile_start=tile_start,
            )
        )
        context_record_ids.append(context_id)
        context_tensors.append(context_tensor)
        next_id += 1

    if len(context_record_ids) > 1:
        records.append(
            _context_reduction_record(
                f"rec_{next_id:04d}",
                layer_index,
                manifest,
                head_index=head_index,
                context_record_ids=context_record_ids,
                context_tensors=context_tensors,
            )
        )
        next_id += 1
    return next_id


def _generate_attention_records_overlapped_heads(manifest: dict) -> list[dict]:
    records: list[dict] = []
    next_id = 0
    for layer_index in range(int(manifest["num_layers"])):
        head_tiles = list(_tile_ranges(int(manifest["past_len"]), int(manifest["score_tile_tokens"])))
        score_ids_by_head: dict[int, list[str]] = {}
        for head_index in range(int(manifest["num_heads"])):
            score_ids_by_head[head_index] = []
            next_id = _append_attention_score_records(
                records,
                next_id,
                manifest,
                layer_index=layer_index,
                head_index=head_index,
                head_tiles=head_tiles,
                score_record_ids=score_ids_by_head[head_index],
            )

        for head_index in range(int(manifest["num_heads"])):
            next_id = _append_attention_softmax_context_records(
                records,
                next_id,
                manifest,
                layer_index=layer_index,
                head_index=head_index,
                head_tiles=head_tiles,
                score_record_ids=score_ids_by_head[head_index],
            )
    _add_dag_hint(records, "heads_are_independent_safe_to_parallelize")
    return records


def generate_attention_records(manifest: dict | None = None) -> list[dict]:
    """Generate attention records.

    ``overlap_independent_heads`` reorders the flat offline IR so independent
    head score phases are adjacent before head-local softmax/context phases.
    It does not add runtime parallelism constructs to the sequential concrete
    opcode stream; ``operator_context.dag_hint`` marks future-safe DAG parallelism.
    """
    manifest = get_tiny_attention_manifest() if manifest is None else manifest
    _validate_attention_manifest(manifest)

    if manifest["schedule_policy"] == "overlap_independent_heads":
        records = _generate_attention_records_overlapped_heads(manifest)
        for record in records:
            validate_record(record)
        return records

    records: list[dict] = []
    next_id = 0
    for layer_index in range(int(manifest["num_layers"])):
        for head_index in range(int(manifest["num_heads"])):
            head_tiles = list(_tile_ranges(int(manifest["past_len"]), int(manifest["score_tile_tokens"])))
            score_record_ids: list[str] = []
            next_id = _append_attention_score_records(
                records,
                next_id,
                manifest,
                layer_index=layer_index,
                head_index=head_index,
                head_tiles=head_tiles,
                score_record_ids=score_record_ids,
            )
            next_id = _append_attention_softmax_context_records(
                records,
                next_id,
                manifest,
                layer_index=layer_index,
                head_index=head_index,
                head_tiles=head_tiles,
                score_record_ids=score_record_ids,
            )

    for record in records:
        validate_record(record)
    return records


def generate_ffn_records(manifest: dict | None = None) -> list[dict]:
    manifest = get_tiny_ffn_manifest() if manifest is None else manifest
    _validate_ffn_manifest(manifest)

    records: list[dict] = []
    next_id = 0
    ffn_activation_tile_size = int(manifest["ffn_activation_tile_size"])
    hidden_size = int(manifest["hidden_size"])
    for layer_index in range(int(manifest["num_layers"])):
        hidden = f"L{layer_index}.hidden"
        hidden_resident = f"L{layer_index}.hidden.resident"
        hidden_reused = f"L{layer_index}.hidden.reused_for_up_gate"
        up_weight = f"L{layer_index}.ffn.up_weight"
        gate_weight = f"L{layer_index}.ffn.gate_weight"
        down_weight = f"L{layer_index}.ffn.down_weight"
        up_intermediate = f"L{layer_index}.ffn.up_intermediate"
        gate_intermediate = f"L{layer_index}.ffn.gate_intermediate"
        activated_gate = f"L{layer_index}.ffn.activated_gate"
        gated_product = f"L{layer_index}.ffn.gated_multiply"
        gated_product_resident = f"L{layer_index}.ffn.gated_multiply.bank_local"
        hidden_out = f"L{layer_index}.hidden_out"

        # Per-tile hidden activation setup: emit one PIMDataMove per tile
        tile_ranges = list(_tile_ranges(hidden_size, ffn_activation_tile_size))
        hidden_setup_ids: list[str] = []
        for tile_index, (tile_start, tile_elements) in enumerate(tile_ranges):
            tile_hidden = f"L{layer_index}.T{tile_index}.hidden"
            tile_hidden_resident = f"L{layer_index}.T{tile_index}.hidden.resident"
            setup_id = f"rec_{next_id:04d}"
            records.append(
                _ffn_data_move_record(
                    setup_id,
                    layer_index,
                    manifest,
                    op="ffn_hidden_activation_tile_setup",
                    stage_index=0,
                    inputs=[tile_hidden],
                    outputs=[tile_hidden_resident],
                    dependencies=[],
                    operand_role="activation_input",
                    residency="dynamic_activation_tile",
                    reuse_scope="ffn_layer",
                    tile_index=tile_index,
                    tile_elements=tile_elements,
                    tile_start=tile_start,
                )
            )
            hidden_setup_ids.append(setup_id)
            next_id += 1

        weight_ids: dict[str, str] = {}
        for offset, (weight_name, op_name, output_name) in enumerate(
            [
                ("up", "ffn_up_weight_residency", up_weight),
                ("gate", "ffn_gate_weight_residency", gate_weight),
                ("down", "ffn_down_weight_residency", down_weight),
            ]
        ):
            weight_record_id = f"rec_{next_id:04d}"
            records.append(
                _ffn_data_move_record(
                    weight_record_id,
                    layer_index,
                    manifest,
                    op=op_name,
                    stage_index=offset,
                    inputs=[output_name],
                    outputs=[f"{output_name}.resident"],
                    dependencies=[],
                    operand_role="weight",
                    residency="preloaded_stationary",
                    reuse_scope=f"ffn_{weight_name}_projection",
                    movement_elements=hidden_size * int(manifest["ffn_hidden_size"]),
                )
            )
            weight_ids[weight_name] = weight_record_id
            next_id += 1

        hidden_reuse_id = f"rec_{next_id:04d}"
        records.append(
            _semantic_operand_record(
                hidden_reuse_id,
                "PIMOperandReuse",
                layer_index,
                "ffn_hidden_reuse_for_up_gate",
                manifest,
                operator_family="ffn_swiglu",
                stage="operand_reuse",
                inputs=[hidden_resident],
                outputs=[hidden_reused],
                dependencies=hidden_setup_ids,
                operand_role="activation_input",
                residency="dynamic_activation_tile",
                materialized=False,
                reuse_scope="ffn_up_gate_pair",
                lowering="semantic_only_operand_reuse",
                context_fields={"stage_index": 0},
            )
        )
        next_id += 1

        up_id = f"rec_{next_id:04d}"
        records.append(
            _ffn_projection_record(
                up_id,
                layer_index,
                manifest,
                op="ffn_up_projection",
                stage_index=0,
                inputs=[hidden_reused, f"{up_weight}.resident"],
                outputs=[up_intermediate],
                dependencies=[hidden_reuse_id, weight_ids["up"]],
            )
        )
        next_id += 1

        gate_id = f"rec_{next_id:04d}"
        records.append(
            _ffn_projection_record(
                gate_id,
                layer_index,
                manifest,
                op="ffn_gate_projection",
                stage_index=1,
                inputs=[hidden_reused, f"{gate_weight}.resident"],
                outputs=[gate_intermediate],
                dependencies=[hidden_reuse_id, weight_ids["gate"]],
            )
        )
        next_id += 1

        activation_id = f"rec_{next_id:04d}"
        records.append(
            _ffn_elementwise_record(
                activation_id,
                layer_index,
                manifest,
                op="ffn_gate_activation_accounting",
                stage_index=2,
                inputs=[gate_intermediate],
                outputs=[activated_gate],
                dependencies=[gate_id],
                elementwise_kind="semantic_only_activation",
            )
        )
        next_id += 1

        gated_id = f"rec_{next_id:04d}"
        records.append(
            _ffn_elementwise_record(
                gated_id,
                layer_index,
                manifest,
                op="ffn_gated_multiply_accounting",
                stage_index=3,
                inputs=[activated_gate, up_intermediate],
                outputs=[gated_product],
                dependencies=[activation_id, up_id],
                elementwise_kind="semantic_only_gated_multiply",
            )
        )
        next_id += 1

        intermediate_residency_id = f"rec_{next_id:04d}"
        records.append(
            _semantic_operand_record(
                intermediate_residency_id,
                "PIMOperandResidency",
                layer_index,
                "ffn_intermediate_bank_local_residency",
                manifest,
                operator_family="ffn_swiglu",
                stage="operand_residency",
                inputs=[gated_product],
                outputs=[gated_product_resident],
                dependencies=[gated_id],
                operand_role="activation_intermediate",
                residency="bank_local_capacity_controlled",
                materialized=False,
                reuse_scope="ffn_down_projection",
                lowering="semantic_only_bank_local_capacity_controlled_operand",
                context_fields={"stage_index": 4},
            )
        )
        next_id += 1

        records.append(
            _ffn_projection_record(
                f"rec_{next_id:04d}",
                layer_index,
                manifest,
                op="ffn_down_projection",
                stage_index=4,
                inputs=[gated_product_resident, f"{down_weight}.resident"],
                outputs=[hidden_out],
                dependencies=[intermediate_residency_id, weight_ids["down"]],
            )
        )
        next_id += 1

    for record in records:
        validate_record(record)
    return records


def generate_moe_records(manifest: dict | None = None) -> list[dict]:
    """Generate MoE records.

    ``overlap_selected_experts`` reorders independent selected expert groups in
    the flat offline IR. Concrete replay remains a sequential opcode stream;
    ``operator_context.dag_hint`` marks future-safe expert-level parallelism.
    """
    manifest = get_tiny_moe_manifest() if manifest is None else manifest
    _validate_moe_manifest(manifest)

    records: list[dict] = []
    next_id = 0
    for layer_index in range(int(manifest["num_layers"])):
        hidden = f"L{layer_index}.hidden"
        router_input = f"L{layer_index}.moe.router_input.resident"
        router_weight = f"L{layer_index}.moe.router_weight"
        router_logits = f"L{layer_index}.moe.router_logits"
        topk_tensor = f"L{layer_index}.moe.topk_experts"
        dispatch_accounted_tensor = f"L{layer_index}.moe.dispatch_accounted_tokens"
        dispatch_tensor = f"L{layer_index}.moe.dispatched_tokens"

        router_setup_id = f"rec_{next_id:04d}"
        records.append(
            _moe_data_move_record(
                router_setup_id,
                layer_index,
                manifest,
                op="moe_router_input_setup",
                stage_index=0,
                inputs=[hidden],
                outputs=[router_input],
                dependencies=[],
                operand_role="activation_input",
                residency="dynamic_activation_tile",
                reuse_scope="moe_router",
            )
        )
        next_id += 1

        router_weight_id = f"rec_{next_id:04d}"
        records.append(
            _moe_data_move_record(
                router_weight_id,
                layer_index,
                manifest,
                op="moe_router_weight_residency",
                stage_index=0,
                inputs=[router_weight],
                outputs=[f"{router_weight}.resident"],
                dependencies=[],
                operand_role="weight",
                residency="preloaded_stationary",
                reuse_scope="moe_router",
                expert_id=None,
                movement_elements=int(manifest["hidden_size"]) * int(manifest["num_experts"]),
            )
        )
        next_id += 1

        router_id = f"rec_{next_id:04d}"
        records.append(
            _moe_compute_record(
                router_id,
                "MoERouter",
                layer_index,
                manifest,
                op="moe_router_projection",
                stage_index=0,
                inputs=[router_input, f"{router_weight}.resident"],
                outputs=[router_logits],
                dependencies=[router_setup_id, router_weight_id],
            )
        )
        next_id += 1

        topk_id = f"rec_{next_id:04d}"
        records.append(
            _moe_accounting_record(
                topk_id,
                "MoETopK",
                layer_index,
                manifest,
                op="moe_topk_select_accounting",
                stage_index=1,
                inputs=[router_logits],
                outputs=[topk_tensor],
                dependencies=[router_id],
                accounting_kind="semantic_only_topk_select",
            )
        )
        next_id += 1

        dispatch_id = f"rec_{next_id:04d}"
        records.append(
            _moe_accounting_record(
                dispatch_id,
                "MoEDispatch",
                layer_index,
                manifest,
                op="moe_expert_dispatch_accounting",
                stage_index=2,
                inputs=[hidden, topk_tensor],
                outputs=[dispatch_accounted_tensor],
                dependencies=[topk_id],
                accounting_kind="semantic_only_expert_dispatch",
            )
        )
        next_id += 1

        dispatch_move_id = f"rec_{next_id:04d}"
        records.append(
            _moe_data_move_record(
                dispatch_move_id,
                layer_index,
                manifest,
                op="moe_token_dispatch_materialized",
                stage_index=2,
                inputs=[dispatch_accounted_tensor],
                outputs=[dispatch_tensor],
                dependencies=[dispatch_id],
                operand_role="token_dispatch",
                residency="dynamic_dispatched_activation_tile",
                reuse_scope="moe_selected_experts",
            )
        )
        next_id += 1

        expert_ids: list[str] = []
        expert_outputs: list[str] = []
        selected_experts = [int(expert) for expert in manifest["selected_experts"]]
        if manifest["schedule_policy"] == "overlap_selected_experts":
            selected_experts = selected_experts[::2] + selected_experts[1::2]
        for expert_id in selected_experts:
            expert_weight_id = f"rec_{next_id:04d}"
            expert_weight = f"L{layer_index}.moe.expert_{expert_id}.weight"
            records.append(
                _moe_data_move_record(
                    expert_weight_id,
                    layer_index,
                    manifest,
                    op=f"moe_expert_{expert_id}_weight_residency",
                    stage_index=3,
                    inputs=[expert_weight],
                    outputs=[f"{expert_weight}.resident"],
                    dependencies=[],
                    operand_role="weight",
                    residency="preloaded_stationary",
                    reuse_scope=f"moe_expert_{expert_id}",
                    expert_id=expert_id,
                    movement_elements=int(manifest["hidden_size"]) * int(manifest["expert_hidden_size"]),
                )
            )
            next_id += 1

            expert_record_id = f"rec_{next_id:04d}"
            expert_output = f"L{layer_index}.moe.expert_{expert_id}.output"
            records.append(
                _moe_compute_record(
                    expert_record_id,
                    "MoEExpertFFN",
                    layer_index,
                    manifest,
                    op=f"moe_expert_{expert_id}_ffn",
                    stage_index=3,
                    inputs=[dispatch_tensor, f"{expert_weight}.resident"],
                    outputs=[expert_output],
                    dependencies=[dispatch_move_id, expert_weight_id],
                    expert_id=expert_id,
                )
            )
            expert_ids.append(expert_record_id)
            expert_outputs.append(expert_output)
            next_id += 1

        combine_id = f"rec_{next_id:04d}"
        combined_output = f"L{layer_index}.moe.combined_output"
        records.append(
            _moe_accounting_record(
                combine_id,
                "MoECombine",
                layer_index,
                manifest,
                op="moe_expert_combine_accounting",
                stage_index=4,
                inputs=expert_outputs,
                outputs=[combined_output],
                dependencies=expert_ids,
                accounting_kind="semantic_only_expert_combine",
            )
        )
        next_id += 1

        records.append(
            _moe_data_move_record(
                f"rec_{next_id:04d}",
                layer_index,
                manifest,
                op="moe_expert_output_combine_materialized",
                stage_index=4,
                inputs=[combined_output],
                outputs=[f"{combined_output}.materialized"],
                dependencies=[combine_id],
                operand_role="expert_output_combine",
                residency="dynamic_combined_expert_output",
                reuse_scope="moe_layer_output",
            )
        )
        next_id += 1

    for record in records:
        if manifest["schedule_policy"] == "overlap_selected_experts":
            record["operator_context"]["dag_hint"] = "selected_experts_are_independent_safe_to_parallelize"
        validate_record(record)
    return records


def _renumber_records(records: list[dict], start_id: int) -> tuple[list[dict], int]:
    id_map = {record["record_id"]: f"rec_{start_id + index:04d}" for index, record in enumerate(records)}
    renumbered: list[dict] = []
    for record in records:
        updated = dict(record)
        updated["record_id"] = id_map[record["record_id"]]
        updated["logical_dependencies"] = [id_map.get(dependency, dependency) for dependency in record.get("logical_dependencies", [])]
        renumbered.append(updated)
    return renumbered, start_id + len(records)


def generate_full_transformer_layer_records(
    *,
    attention_manifest: dict | None = None,
    ffn_manifest: dict | None = None,
    moe_manifest: dict | None = None,
) -> list[dict]:
    components = [
        generate_attention_records(get_tiny_attention_manifest() if attention_manifest is None else attention_manifest),
        generate_ffn_records(get_tiny_ffn_manifest() if ffn_manifest is None else ffn_manifest),
        generate_moe_records(get_tiny_moe_manifest() if moe_manifest is None else moe_manifest),
    ]
    combined: list[dict] = []
    next_id = 0
    for component in components:
        renumbered, next_id = _renumber_records(component, next_id)
        combined.extend(renumbered)
    for record in combined:
        validate_record(record)
    return combined


def _summary_manifest_fields(manifest: dict | None = None, manifest_name: str | None = None) -> dict:
    if manifest is not None:
        return {
            "manifest_version": manifest.get("manifest_version"),
            "manifest_name": manifest.get("manifest_name"),
            "phase": manifest.get("phase"),
            "datatype": manifest.get("datatype"),
            "non_claims": list(manifest.get("non_claims", [])),
            "mapping_policy": _mapping_policy(manifest),
        }
    return {
        "manifest_name": manifest_name or "unknown",
        "non_claims": [
            "not_runtime_replay",
            "not_vllm_replay",
            "not_numerical_correctness",
            "not_silicon_faithful_softmax_or_data_movement",
            "not_raw_attacc_schema",
        ],
    }


def _build_provenance_summary(
    records: list[dict],
    manifest_or_name: dict | str | None = None,
    *,
    manifest_name: str | None = None,
    notes: str,
    compute_kinds: set[str],
) -> dict:
    counts: dict[str, int] = {}
    for record in records:
        counts[record["kind"]] = counts.get(record["kind"], 0) + 1
    manifest_fields = _summary_manifest_fields(
        manifest_or_name if isinstance(manifest_or_name, dict) else None,
        manifest_name if manifest_name is not None else (manifest_or_name if isinstance(manifest_or_name, str) else None),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": FULL_TRANSFORMER_GENERATOR_VERSION,
        **manifest_fields,
        "record_counts_by_kind": counts,
        "total_logical_records": len(records),
        "total_expanded_records": expanded_record_count(records),
        "estimated_pim_compute_requests": sum(
            int(record.get("num_requests", 0))
            for record in records
            if record["kind"] in compute_kinds
        ),
        "notes": notes,
    }


def build_full_transformer_provenance_summary(records: list[dict], manifest_or_name: dict | str | None = None, *, manifest_name: str | None = None) -> dict:
    return _build_provenance_summary(
        records,
        manifest_or_name,
        manifest_name=manifest_name,
        notes="combined P4 attention+FFN+MoE offline semantic DAG summary; concrete opcode trace is the replay path",
        compute_kinds={"AttentionScore", "AttentionContext", "FFNProjection", "MoERouter", "MoEExpertFFN"},
    )


def build_provenance_summary(records: list[dict], manifest_or_name: dict | str | None = None, *, manifest_name: str | None = None) -> dict:
    return _build_provenance_summary(
        records,
        manifest_or_name,
        manifest_name=manifest_name,
        notes="P4 semantic DAG summary; softmax remains semantic/accounting-only",
        compute_kinds={"AttentionScore", "AttentionContext"},
    )


def generate_attention_artifacts(output_dir: Path | str = DEFAULT_OUTPUT_DIR, manifest: dict | None = None) -> tuple[Path, Path]:
    manifest = get_tiny_attention_manifest() if manifest is None else manifest
    _validate_attention_manifest(manifest)
    records = generate_attention_records(manifest)
    summary = build_provenance_summary(records, manifest)
    output_dir = Path(output_dir)
    trace_path = output_dir / "structured_trace.jsonl"
    summary_path = output_dir / "provenance_summary.json"
    write_jsonl(records, trace_path)
    write_json(summary, summary_path)
    return trace_path, summary_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate tiny P4 attention semantic tensor-DAG artifacts")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> int:
    opts = _build_arg_parser().parse_args()
    trace_path, summary_path = generate_attention_artifacts(output_dir=opts.output_dir)
    print(f"Generated: {trace_path}")
    print(f"Generated: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
