"""Machine-readable validation for researcher traces and simulator results."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ramulator.pimscope.compat import canonicalize_legacy_result
from ramulator.pimscope.config import MANIFEST_SCHEMA_VERSION, resolve_experiment_manifest
from ramulator.workload_surrogate.lpddr5_pim_concrete_trace import (
    CONCRETE_SCHEMA_VERSIONS,
    expanded_record_count,
    read_jsonl,
)

RESULT_SCHEMA_VERSION = 1
RESULT_SCHEMA_NAME = "pimscope-result-v1"
AGGREGATE_SCHEMA_VERSION = 1
AGGREGATE_SCHEMA_NAMES = {
    "decode_cycles": "pimscope-decode-aggregate-v1",
    "prefill_cycles": "pimscope-prefill-aggregate-v1",
    "pim_sharing_comparison": "pimscope-sharing-aggregate-v1",
}
_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")


def _require(mapping: Mapping[str, Any], field: str, path: str) -> Any:
    if field not in mapping:
        raise ValueError(f"{path}: missing required field {field!r}")
    return mapping[field]


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path}: must be an object, got {type(value).__name__}")
    return value


def _nonnegative_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{path}: must be a non-negative integer, got {value!r}")
    return value


def validate_result(
    payload: Mapping[str, Any], *, expected_manifest_fingerprint: str | None = None
) -> dict[str, Any]:
    """Validate and normalize one ``pimscope run`` result object.

    The returned object is canonicalized for the shared-block terminology
    migration.  Legacy names are accepted only through the explicit
    compatibility layer, and conflicting old/new fields fail closed.
    """
    result = canonicalize_legacy_result(payload)
    if result.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError(
            f"result.schema_version: must be {RESULT_SCHEMA_VERSION}, "
            f"got {result.get('schema_version')!r}"
        )

    required = (
        "experiment",
        "status",
        "manifest_fingerprint",
        "resolved_manifest",
        "resolved_hardware",
        "workload_summary",
        "simulation",
        "provenance",
    )
    for field in required:
        _require(result, field, "result")
    if not isinstance(result["experiment"], str) or not result["experiment"].strip():
        raise ValueError("result.experiment: must be a non-empty string")
    if result["status"] not in {"PASS", "FAIL"}:
        raise ValueError("result.status: must be 'PASS' or 'FAIL'")

    fingerprint = result["manifest_fingerprint"]
    if not isinstance(fingerprint, str) or not _FINGERPRINT_RE.fullmatch(fingerprint):
        raise ValueError("result.manifest_fingerprint: must be a lowercase SHA-256 hex string")
    if expected_manifest_fingerprint is not None and fingerprint != expected_manifest_fingerprint:
        raise ValueError(
            "result.manifest_fingerprint: does not match the supplied configuration"
        )

    manifest = _object(result["resolved_manifest"], "result.resolved_manifest")
    resolved = resolve_experiment_manifest(manifest, source="result.resolved_manifest")
    if resolved.fingerprint != fingerprint:
        raise ValueError(
            "result.resolved_manifest: fingerprint does not match manifest_fingerprint"
        )
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("result.resolved_manifest.schema_version: unsupported manifest version")

    hardware = _object(result["resolved_hardware"], "result.resolved_hardware")
    address_layout = _object(
        _require(hardware, "address_layout", "result.resolved_hardware"),
        "result.resolved_hardware.address_layout",
    )
    for field in ("mapping_version", "internal_prefetch_size", "tx_bytes", "capacity_bytes"):
        _nonnegative_int(
            _require(address_layout, field, "result.resolved_hardware.address_layout"),
            f"result.resolved_hardware.address_layout.{field}",
        )
    dram_class = _require(address_layout, "dram_class", "result.resolved_hardware.address_layout")
    if dram_class not in {"LPDDR5PIM", "LPDDR6PIM"}:
        raise ValueError(
            "result.resolved_hardware.address_layout.dram_class: unsupported PIM backend"
        )
    if dram_class != manifest["hardware"]["dram_class"]:
        raise ValueError(
            "result.resolved_hardware.address_layout.dram_class: does not match manifest"
        )
    for field in ("level_names", "level_sizes", "address_level_sizes"):
        value = _require(address_layout, field, "result.resolved_hardware.address_layout")
        if not isinstance(value, list) or not value:
            raise ValueError(
                f"result.resolved_hardware.address_layout.{field}: must be a non-empty list"
            )
    if len(address_layout["level_names"]) != len(address_layout["level_sizes"]):
        raise ValueError(
            "result.resolved_hardware.address_layout: level_names and level_sizes must align"
        )

    workload = _object(result["workload_summary"], "result.workload_summary")
    for field in ("model", "phase", "semantic_records", "concrete_records"):
        _require(workload, field, "result.workload_summary")
    _nonnegative_int(workload["semantic_records"], "result.workload_summary.semantic_records")
    _nonnegative_int(workload["concrete_records"], "result.workload_summary.concrete_records")

    simulation = _object(result["simulation"], "result.simulation")
    replay_ok = _require(simulation, "replay_ok", "result.simulation")
    if not isinstance(replay_ok, bool):
        raise ValueError("result.simulation.replay_ok: must be a boolean")
    simulation_dram_class = _require(simulation, "dram_class", "result.simulation")
    if simulation_dram_class != dram_class:
        raise ValueError("result.simulation.dram_class: does not match resolved address layout")
    trace_schema = _require(simulation, "trace_schema", "result.simulation")
    expected_trace_schema = CONCRETE_SCHEMA_VERSIONS[dram_class]
    if trace_schema != expected_trace_schema:
        raise ValueError(
            f"result.simulation.trace_schema: must be {expected_trace_schema!r} for {dram_class}"
        )
    cycles = _nonnegative_int(
        _require(simulation, "cycles", "result.simulation"), "result.simulation.cycles"
    )
    if result["status"] == "PASS" and (not replay_ok or cycles <= 0):
        raise ValueError("result: PASS requires replay_ok=true and positive simulation cycles")
    if result["status"] == "FAIL" and replay_ok:
        raise ValueError("result: FAIL requires replay_ok=false")

    _object(result["provenance"], "result.provenance")
    return result


def validate_trace_file(
    path: str | Path,
    *,
    address_layout: Mapping[str, Any],
    max_expanded_records: int | None = None,
) -> dict[str, Any]:
    """Validate a concrete opcode JSONL file against a resolved DRAM layout."""
    trace_path = Path(path)
    header, records = read_jsonl(
        trace_path,
        address_layout=address_layout,
        max_expanded_records=max_expanded_records,
        expected_dram_class=address_layout.get("dram_class"),
    )
    return {
        "valid": True,
        "path": str(trace_path),
        "schema_version": header["schema_version"],
        "dram_class": header.get("dram_class", "LPDDR5PIM"),
        "records": len(records),
        "expanded_records": expanded_record_count(records),
        "mapping_version": address_layout.get("mapping_version"),
        "capacity_bytes": address_layout.get("capacity_bytes"),
        "max_expanded_records": max_expanded_records,
    }


def validate_aggregate(
    payload: Mapping[str, Any], *, kind: str
) -> dict[str, Any]:
    """Validate one paper-artifact aggregate envelope and its row basics."""
    if kind not in AGGREGATE_SCHEMA_NAMES:
        raise ValueError(f"unknown aggregate kind: {kind}")
    aggregate = _object(payload, "aggregate")
    if aggregate.get("schema_version") != AGGREGATE_SCHEMA_VERSION:
        raise ValueError(
            f"aggregate.schema_version: must be {AGGREGATE_SCHEMA_VERSION}, "
            f"got {aggregate.get('schema_version')!r}"
        )
    if aggregate.get("schema_name") != AGGREGATE_SCHEMA_NAMES[kind]:
        raise ValueError(
            f"aggregate.schema_name: must be {AGGREGATE_SCHEMA_NAMES[kind]!r}"
        )
    if kind == "decode_cycles" and aggregate.get("phase") != "decode":
        raise ValueError("aggregate.phase: decode_cycles must have phase='decode'")
    if kind == "prefill_cycles" and aggregate.get("phase") != "prefill":
        raise ValueError("aggregate.phase: prefill_cycles must have phase='prefill'")
    rows = aggregate.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("aggregate.rows: must be a non-empty list")
    for index, row in enumerate(rows):
        row = _object(row, f"aggregate.rows[{index}]")
        for field in ("cycles", "runtime_ns"):
            if field in row:
                value = row[field]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                    raise ValueError(f"aggregate.rows[{index}].{field}: must be non-negative")
        if "replay_status" in row and row["replay_status"] not in {"PASS", "FAIL"}:
            raise ValueError(f"aggregate.rows[{index}].replay_status: invalid status")
    return aggregate


def load_json_object(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    try:
        value = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{file_path}: invalid JSON: {exc.msg}") from exc
    return _object(value, str(file_path))


__all__ = [
    "AGGREGATE_SCHEMA_NAMES",
    "AGGREGATE_SCHEMA_VERSION",
    "CONCRETE_SCHEMA_VERSIONS",
    "RESULT_SCHEMA_NAME",
    "RESULT_SCHEMA_VERSION",
    "load_json_object",
    "validate_aggregate",
    "validate_result",
    "validate_trace_file",
]
