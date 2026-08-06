"""Legacy deterministic Phase-2 decode-only structured workload-surrogate generator.

Deprecated: this module preserves the frozen P2 MVP replay contract and does
not implement ``paper/algorithms/llama2_decode_trace_algorithm.tex``. Use
``ramulator.workload_surrogate.generate_full_transformer.generate_llama2_7b_dense_decoder_records``
for the current decode-block v2 Llama2 dense-decoder surrogate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ramulator.workload_surrogate.decode_only_manifest import get_decode_only_manifest
from ramulator.workload_surrogate.structured_trace import (
    GENERATOR_VERSION,
    MIXED_PHASE_NON_CLAIMS,
    REQUIRED_CLAIM_BOUNDARY,
    REQUIRED_NON_CLAIMS,
    SCHEMA_VERSION,
    expanded_record_count,
    validate_manifest,
    validate_record,
    write_json,
    write_jsonl,
)


DEFAULT_OUTPUT_DIR = Path("ramulator2/tests/data/structured_workload_surrogate/decode_only_v0_1")
DECODE_ONLY_GENERATOR_STATUS = "legacy_deprecated_p2_mvp"
DECODE_ONLY_REPLACEMENT_GENERATOR = "ramulator.workload_surrogate.generate_full_transformer.generate_llama2_7b_dense_decoder_records"
DECODE_ONLY_DEPRECATION_NOTE = (
    "Legacy P2 MVP replay fixture; does not implement "
    "paper/algorithms/llama2_decode_trace_algorithm.tex. "
    f"Use {DECODE_ONLY_REPLACEMENT_GENERATOR} for decode-block v2."
)


def _ramulator_defaults(manifest: dict) -> dict:
    defaults = dict(manifest.get("ramulator_visible_defaults", {}))
    required = {
        "bank_sequence",
        "bank_sequence_order",
        "pim_banks_per_mpu",
        "burst_length",
        "row_start",
        "row_count",
        "dependency_count",
        "tx_bytes",
        "column_start",
    }
    missing = sorted(required - set(defaults))
    if missing:
        raise ValueError(f"Manifest missing ramulator_visible_defaults fields: {missing}")
    if not defaults["bank_sequence"]:
        raise ValueError("Manifest ramulator_visible_defaults.bank_sequence must be non-empty")
    if int(defaults["row_count"]) <= 0:
        raise ValueError("Manifest ramulator_visible_defaults.row_count must be positive")
    if int(defaults["dependency_count"]) <= 0:
        raise ValueError("Manifest ramulator_visible_defaults.dependency_count must be positive")
    if int(defaults["burst_length"]) <= 0:
        raise ValueError("Manifest ramulator_visible_defaults.burst_length must be positive")
    if int(defaults["tx_bytes"]) <= 0:
        raise ValueError("Manifest ramulator_visible_defaults.tx_bytes must be positive")
    return defaults


def _pim_operator_classes(manifest: dict) -> list[str]:
    return list(manifest["pim_compute_operator_classes"])


def _pim_operator_width_field(op: str, manifest: dict) -> str:
    widths = dict(manifest.get("pim_operator_request_widths", {}))
    if op not in widths:
        raise ValueError(f"Unsupported PIM op: {op}")
    width_field = widths[op]
    if width_field not in manifest:
        raise ValueError(f"Unsupported PIM op width field for {op}: {width_field}")
    return width_field


def _claim_boundary() -> list[str]:
    return list(REQUIRED_CLAIM_BOUNDARY)


def _non_claims(manifest: dict | None = None) -> list[str]:
    if manifest is not None and manifest.get("mixed_phase_extension"):
        return list(MIXED_PHASE_NON_CLAIMS)
    return list(REQUIRED_NON_CLAIMS)


def _mapping_policy(manifest: dict) -> dict:
    defaults = _ramulator_defaults(manifest)
    policy = dict(manifest["mapping_policy"])
    policy.update(
        {
            "controller_bank_order": defaults["bank_sequence_order"],
            "bank_sequence": list(defaults["bank_sequence"]),
            "pim_banks_per_mpu": int(defaults["pim_banks_per_mpu"]),
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
        "generator_version": GENERATOR_VERSION,
        "claim_boundary": _claim_boundary(),
        "non_claims": _non_claims(manifest),
        "selected_model_family": manifest["model_family"],
        "selected_workload_regime": manifest["workload_class"],
        "notes": f"bounded deterministic semantic generator record for {op}",
    }


def _host_read_bytes(manifest: dict) -> int:
    return manifest["past_len"] * manifest["hidden_size"]


def _prefill_host_bytes(manifest: dict) -> int:
    prefill_tokens = int(manifest.get("prefill_tokens", manifest.get("prompt_len", manifest["past_len"])))
    return prefill_tokens * manifest["hidden_size"]


def _host_write_bytes(manifest: dict) -> int:
    return 2 * manifest["hidden_size"]


def _prefill_kv_write_bytes(manifest: dict) -> int:
    prefill_tokens = int(manifest.get("prefill_tokens", manifest.get("prompt_len", manifest["past_len"])))
    return max(1, 2 * prefill_tokens * manifest["hidden_size"])


def _pim_num_requests(op: str, manifest: dict) -> int:
    lanes = 32 if manifest["datatype"] == "int8" else 16
    width = manifest[_pim_operator_width_field(op, manifest)]
    return max(1, width // lanes)


def _base_record(record_id: str, kind: str, layer: str, op: str, manifest: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "record_id": record_id,
        "kind": kind,
        "phase": manifest["phase"],
        "layer": layer,
        "op": op,
        "repeat": 1,
        "provenance": _provenance(manifest, op),
        "mapping_policy": _mapping_policy(manifest),
    }


def _with_phase(record: dict, phase: str) -> dict:
    record["phase"] = phase
    return record


def _host_record(record_id: str, kind: str, layer_index: int, op: str, manifest: dict, *, base_byte: int, size_bytes: int) -> dict:
    defaults = _ramulator_defaults(manifest)
    tx_bytes = int(defaults["tx_bytes"])
    layer = f"layer_{layer_index:02d}"
    record = _base_record(record_id, kind, layer, op, manifest)
    record.update(
        {
            "bytes": size_bytes,
            "address_policy": {
                "kind": "bounded_sequential_host_requests",
                "base_byte": base_byte,
                "stride_bytes": tx_bytes,
                "count": max(1, size_bytes // tx_bytes),
            },
        }
    )
    return record


def _pim_record(record_id: str, layer_index: int, op: str, manifest: dict) -> dict:
    defaults = _ramulator_defaults(manifest)
    row_start = int(defaults["row_start"])
    row_count = int(defaults["row_count"])
    dependency_count = int(defaults["dependency_count"])
    layer = f"layer_{layer_index:02d}"
    row_offset = layer_index % row_count
    record = _base_record(record_id, "PIMCompute", layer, op, manifest)
    record.update(
        {
            "num_requests": _pim_num_requests(op, manifest),
            "bank_sequence": list(defaults["bank_sequence"]),
            "dependency_context": {
                "kind": "column_dependency_count",
                "dependency_count": dependency_count,
                "dependency_id": layer_index % dependency_count,
            },
            "row_policy": {
                "kind": "bounded_row_window_decode_only",
                "row_start": row_start,
                "row_count": row_count,
                "resolved_row": row_start + row_offset,
            },
            "column_policy": {
                "kind": "dependency_column_round_robin",
                "column_start": int(defaults["column_start"]),
                "resolved_column": int(defaults["column_start"]) + (layer_index % dependency_count),
            },
            "datatype_metadata": {
                "datatype": manifest["datatype"],
                "role": "homogeneous_run_level_metadata",
                "behavior_claim": "explicit_resource_rows_only",
            },
            "burst_length": int(defaults["burst_length"]),
        }
    )
    return record


def _all_bank_record(record_id: str, kind: str, layer_index: int, op: str, manifest: dict) -> dict:
    defaults = _ramulator_defaults(manifest)
    row_start = int(defaults["row_start"])
    row_count = int(defaults["row_count"])
    dependency_count = int(defaults["dependency_count"])
    layer = f"layer_{layer_index:02d}"
    row_offset = layer_index % row_count
    record = _base_record(record_id, kind, layer, op, manifest)
    record.update(
        {
            "num_requests": 1,
            "all_bank_scope": "rank",
            "dependency_context": {
                "kind": "column_dependency_count",
                "dependency_count": dependency_count,
                "dependency_id": layer_index % dependency_count,
            },
            "row_policy": {
                "kind": "bounded_row_window_all_bank_semantic",
                "row_start": row_start,
                "row_count": row_count,
                "resolved_row": row_start + row_offset,
            },
            "column_policy": {
                "kind": "dependency_column_round_robin",
                "column_start": int(defaults["column_start"]),
                "resolved_column": int(defaults["column_start"]) + (layer_index % dependency_count),
            },
            "datatype_metadata": {
                "datatype": manifest["datatype"],
                "role": "homogeneous_run_level_metadata",
                "behavior_claim": "explicit_resource_rows_only",
            },
        }
    )
    return record


def _barrier_record(record_id: str, layer_index: int, manifest: dict) -> dict:
    layer = f"layer_{layer_index:02d}"
    record = _base_record(record_id, "Barrier", layer, "layer_transition_barrier", manifest)
    record["barrier_scope"] = "layer"
    return record


def _drain_record(record_id: str, manifest: dict) -> dict:
    record = _base_record(record_id, "Drain", "decode_tail", "final_drain", manifest)
    record["drain_scope"] = "trace"
    return record


def _emit_decode_layer(records: list[dict], next_id: int, layer_index: int, manifest: dict, *, phase: str = "decode") -> int:
    host_read_bytes = _host_read_bytes(manifest)
    host_write_bytes = _host_write_bytes(manifest)
    pim_ops = _pim_operator_classes(manifest)
    layer_base = layer_index * 0x100000

    records.append(
        _with_phase(
            _host_record(
                f"rec_{next_id:04d}",
                "HostRead",
                layer_index,
                "attention_k_cache_read",
                manifest,
                base_byte=layer_base,
                size_bytes=host_read_bytes,
            ),
            phase,
        )
    )
    next_id += 1
    records.append(
        _with_phase(
            _host_record(
                f"rec_{next_id:04d}",
                "HostRead",
                layer_index,
                "attention_v_cache_read",
                manifest,
                base_byte=layer_base + host_read_bytes,
                size_bytes=host_read_bytes,
            ),
            phase,
        )
    )
    next_id += 1

    for op in pim_ops:
        if manifest.get("enable_all_bank_records"):
            records.append(_with_phase(_all_bank_record(f"rec_{next_id:04d}", "PIMLoadAll", layer_index, f"{op}_all_bank_load", manifest), phase))
            next_id += 1
            records.append(_with_phase(_all_bank_record(f"rec_{next_id:04d}", "PIMComputeAll", layer_index, f"{op}_all_bank_compute", manifest), phase))
            next_id += 1
        else:
            records.append(_with_phase(_pim_record(f"rec_{next_id:04d}", layer_index, op, manifest), phase))
            next_id += 1

    records.append(
        _with_phase(
            _host_record(
                f"rec_{next_id:04d}",
                "HostWrite",
                layer_index,
                "kv_cache_append_accounting",
                manifest,
                base_byte=layer_base + 2 * host_read_bytes,
                size_bytes=host_write_bytes,
            ),
            phase,
        )
    )
    next_id += 1
    records.append(_with_phase(_barrier_record(f"rec_{next_id:04d}", layer_index, manifest), phase))
    next_id += 1
    return next_id


def _emit_prefill_layer(records: list[dict], next_id: int, layer_index: int, manifest: dict) -> int:
    prefill_bytes = _prefill_host_bytes(manifest)
    kv_write_bytes = _prefill_kv_write_bytes(manifest)
    layer_base = 0x80000000 + layer_index * 0x100000

    records.append(
        _with_phase(
            _host_record(
                f"rec_{next_id:04d}",
                "HostRead",
                layer_index,
                "prefill_prompt_activation_read",
                manifest,
                base_byte=layer_base,
                size_bytes=prefill_bytes,
            ),
            "prefill",
        )
    )
    next_id += 1

    for op in manifest.get("prefill_pim_compute_operator_classes", ["prefill_qkv_projection_gemm"]):
        width_field = manifest.get("prefill_pim_operator_request_widths", {}).get(op, "hidden_size")
        if width_field not in manifest:
            raise ValueError(f"Unsupported prefill PIM op width field for {op}: {width_field}")
        manifest.setdefault("pim_operator_request_widths", {})[op] = width_field
        records.append(_with_phase(_pim_record(f"rec_{next_id:04d}", layer_index, op, manifest), "prefill"))
        next_id += 1

    records.append(
        _with_phase(
            _host_record(
                f"rec_{next_id:04d}",
                "HostWrite",
                layer_index,
                "prefill_kv_cache_population_accounting",
                manifest,
                base_byte=layer_base + prefill_bytes,
                size_bytes=kv_write_bytes,
            ),
            "prefill",
        )
    )
    next_id += 1
    records.append(_with_phase(_barrier_record(f"rec_{next_id:04d}", layer_index, manifest), "prefill"))
    next_id += 1
    return next_id


def generate_decode_only_records(manifest: dict | None = None) -> list[dict]:
    manifest = get_decode_only_manifest() if manifest is None else manifest
    validate_manifest(manifest)

    records: list[dict] = []
    next_id = 0
    if manifest.get("mixed_phase_extension"):
        for layer_index in range(int(manifest.get("prefill_layers", min(1, manifest["num_layers"])))):
            next_id = _emit_prefill_layer(records, next_id, layer_index, manifest)

    for layer_index in range(manifest["num_layers"]):
        next_id = _emit_decode_layer(records, next_id, layer_index, manifest)

    records.append(_drain_record(f"rec_{next_id:04d}", manifest))

    for record in records:
        validate_record(record)
    return records


def build_provenance_summary(records: list[dict], manifest: dict) -> dict:
    counts_by_kind: dict[str, int] = {}
    estimated_host_bytes = 0
    estimated_pim_requests = 0
    barrier_count = 0
    drain_count = 0

    for record in records:
        counts_by_kind[record["kind"]] = counts_by_kind.get(record["kind"], 0) + 1
        if record["kind"] in {"HostRead", "HostWrite"}:
            estimated_host_bytes += int(record["bytes"])
        elif record["kind"] in {"PIMCompute", "PIMLoadAll", "PIMComputeAll"}:
            estimated_pim_requests += int(record["num_requests"])
        elif record["kind"] == "Barrier":
            barrier_count += 1
        elif record["kind"] == "Drain":
            drain_count += 1

    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "manifest_version": manifest["manifest_version"],
        "model_family": manifest["model_family"],
        "phase": manifest["phase"],
        "mixed_phase_extension": bool(manifest.get("mixed_phase_extension", False)),
        "mixed_phase_approval": manifest.get("mixed_phase_approval"),
        "enable_all_bank_records": bool(manifest.get("enable_all_bank_records", False)),
        "all_bank_records_approval": manifest.get("all_bank_records_approval"),
        "datatype": manifest["datatype"],
        "record_counts_by_kind": counts_by_kind,
        "total_logical_records": len(records),
        "total_expanded_records": expanded_record_count(records),
        "pim_compute_operator_classes": list(manifest["pim_compute_operator_classes"]),
        "host_support_record_classes": list(manifest["host_support_record_classes"]),
        "host_only_provenance_classes": list(manifest["host_only_provenance_classes"]),
        "lifecycle_status": manifest.get("lifecycle_status", DECODE_ONLY_GENERATOR_STATUS),
        "replacement_generator": manifest.get("replacement_generator", DECODE_ONLY_REPLACEMENT_GENERATOR),
        "deprecation_note": manifest.get("deprecation_note", DECODE_ONLY_DEPRECATION_NOTE),
        "ramulator_visible_defaults": dict(manifest["ramulator_visible_defaults"]),
        "pim_operator_request_widths": dict(manifest["pim_operator_request_widths"]),
        "scaffolding_notes": dict(manifest.get("scaffolding_notes", {})),
        "mapping_policy": _mapping_policy(manifest),
        "literature_anchors": list(manifest["literature_anchors"]),
        "claim_boundary": _claim_boundary(),
        "non_claims": _non_claims(manifest),
        "num_layers_emitted": manifest["num_layers"],
        "estimated_host_bytes": estimated_host_bytes,
        "estimated_pim_requests": estimated_pim_requests,
        "barrier_count": barrier_count,
        "drain_count": drain_count,
        "notes": "deterministic decode-only-first structured workload-surrogate generator output, P2.4-pre frozen contract",
    }


def generate_decode_only_artifacts(output_dir: Path | str = DEFAULT_OUTPUT_DIR, manifest: dict | None = None) -> tuple[Path, Path]:
    manifest = get_decode_only_manifest() if manifest is None else manifest
    validate_manifest(manifest)
    output_dir = Path(output_dir)
    records = generate_decode_only_records(manifest)
    summary = build_provenance_summary(records, manifest)
    trace_path = output_dir / "structured_trace.jsonl"
    summary_path = output_dir / "provenance_summary.json"
    write_jsonl(records, trace_path)
    write_json(summary, summary_path)
    return trace_path, summary_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate deterministic decode-only workload-surrogate artifacts")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for structured_trace.jsonl and provenance_summary.json",
    )
    return parser


def main() -> int:
    opts = _build_arg_parser().parse_args()
    trace_path, summary_path = generate_decode_only_artifacts(output_dir=opts.output_dir)
    print(f"Generated: {trace_path}")
    print(f"Generated: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
