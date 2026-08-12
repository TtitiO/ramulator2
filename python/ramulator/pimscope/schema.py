"""Validate PIMScope traces, results, and aggregates."""

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

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
    """Validate one ``pimscope run`` result object."""
    result = dict(payload)
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
        raise ValueError("result.manifest_fingerprint: does not match the supplied configuration")

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
    preflight = _object(
        _require(hardware, "preflight", "result.resolved_hardware"),
        "result.resolved_hardware.preflight",
    )
    for field in ("records", "expanded_records", "trace_bytes"):
        _nonnegative_int(
            _require(preflight, field, "result.resolved_hardware.preflight"),
            f"result.resolved_hardware.preflight.{field}",
        )
    for field in ("max_expanded_records", "max_trace_bytes"):
        value = _nonnegative_int(
            _require(preflight, field, "result.resolved_hardware.preflight"),
            f"result.resolved_hardware.preflight.{field}",
        )
        if value <= 0:
            raise ValueError(f"result.resolved_hardware.preflight.{field}: must be positive")
    if preflight["expanded_records"] < preflight["records"]:
        raise ValueError("result.resolved_hardware.preflight: expanded records must cover records")
    if preflight["expanded_records"] > preflight["max_expanded_records"]:
        raise ValueError("result.resolved_hardware.preflight: expanded record limit exceeded")
    if preflight["trace_bytes"] > preflight["max_trace_bytes"]:
        raise ValueError("result.resolved_hardware.preflight: trace byte limit exceeded")
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
    subchannel_model = address_layout.get("subchannel_model")
    if dram_class == "LPDDR6PIM":
        subchannel_model = _object(
            subchannel_model,
            "result.resolved_hardware.address_layout.subchannel_model",
        )
        expected_subchannel_model = {
            "status": "single_subchannel_only",
            "modeled_subchannels_per_channel": 1,
            "refresh_density_reference_subchannels": 2,
            "independent_subchannel_scheduling": False,
        }
        if subchannel_model != expected_subchannel_model:
            raise ValueError(
                "result.resolved_hardware.address_layout.subchannel_model: "
                "unsupported LPDDR6 sub-channel interpretation"
            )
    elif subchannel_model is not None:
        raise ValueError(
            "result.resolved_hardware.address_layout.subchannel_model: must be null for LPDDR5PIM"
        )

    workload = _object(result["workload_summary"], "result.workload_summary")
    for field in ("workload_type", "model", "phase", "semantic_records", "concrete_records"):
        _require(workload, field, "result.workload_summary")
    if workload["workload_type"] != manifest["workload"]["workload_type"]:
        raise ValueError("result.workload_summary.workload_type: does not match manifest")
    _nonnegative_int(workload["semantic_records"], "result.workload_summary.semantic_records")
    _nonnegative_int(workload["concrete_records"], "result.workload_summary.concrete_records")
    if workload["concrete_records"] != preflight["records"]:
        raise ValueError(
            "result.workload_summary.concrete_records: does not match preflight records"
        )
    surrogate_scope = _object(
        _require(workload, "surrogate_scope", "result.workload_summary"),
        "result.workload_summary.surrogate_scope",
    )
    for field in (
        "architecture_dimensions",
        "weights",
        "host_operator_model",
    ):
        if not isinstance(
            _require(surrogate_scope, field, "result.workload_summary.surrogate_scope"), str
        ):
            raise ValueError(f"result.workload_summary.surrogate_scope.{field}: must be a string")
    for field in ("runtime_trace", "numerical_execution", "silicon_calibration"):
        if not isinstance(
            _require(surrogate_scope, field, "result.workload_summary.surrogate_scope"), bool
        ):
            raise ValueError(f"result.workload_summary.surrogate_scope.{field}: must be a boolean")
    if not isinstance(_require(workload, "claim_boundary", "result.workload_summary"), str):
        raise ValueError("result.workload_summary.claim_boundary: must be a string")
    materialization = _object(
        _require(workload, "weight_materialization", "result.workload_summary"),
        "result.workload_summary.weight_materialization",
    )
    if materialization.get("policy") not in {"resident", "full_preload"}:
        raise ValueError(
            "result.workload_summary.weight_materialization.policy: unsupported policy"
        )
    for field in ("write_records", "write_transactions", "generated_bytes"):
        _nonnegative_int(
            _require(materialization, field, "result.workload_summary.weight_materialization"),
            f"result.workload_summary.weight_materialization.{field}",
        )
    if materialization.get("synthetic") is not True:
        raise ValueError("result.workload_summary.weight_materialization.synthetic: must be true")
    if materialization["policy"] == "resident" and materialization["write_transactions"] != 0:
        raise ValueError("resident weight policy must not report preload writes")
    if materialization["policy"] == "full_preload" and materialization["write_transactions"] == 0:
        raise ValueError("full_preload weight policy must report preload writes")

    simulation = _object(result["simulation"], "result.simulation")
    replay_ok = _require(simulation, "replay_ok", "result.simulation")
    if not isinstance(replay_ok, bool):
        raise ValueError("result.simulation.replay_ok: must be a boolean")
    simulation_dram_class = _require(simulation, "dram_class", "result.simulation")
    if simulation_dram_class != dram_class:
        raise ValueError("result.simulation.dram_class: does not match resolved address layout")
    trace_schema = _require(simulation, "trace_schema", "result.simulation")
    refresh_manager = _require(simulation, "refresh_manager", "result.simulation")
    if refresh_manager != manifest["hardware"]["controller"]["refresh_manager"]:
        raise ValueError("result.simulation.refresh_manager: does not match manifest")
    execution_model = _require(simulation, "pim_mac_execution_model", "result.simulation")
    if execution_model != manifest["hardware"]["pim"]["pim_mac_execution_model"]:
        raise ValueError("result.simulation.pim_mac_execution_model: does not match manifest")
    expected_model_status = (
        "experimental" if execution_model == "subbank_overlap_experimental" else "supported"
    )
    if simulation.get("pim_mac_execution_model_status") != expected_model_status:
        raise ValueError("result.simulation.pim_mac_execution_model_status: invalid status")
    opcode_counts = _object(
        _require(simulation, "trace_opcode_counts", "result.simulation"),
        "result.simulation.trace_opcode_counts",
    )
    for opcode, count in opcode_counts.items():
        if not isinstance(opcode, str) or not opcode:
            raise ValueError("result.simulation.trace_opcode_counts: opcode names must be strings")
        _nonnegative_int(count, f"result.simulation.trace_opcode_counts.{opcode}")
    if simulation.get("pim_bcast_issued") != opcode_counts.get("PIM_BCAST", 0):
        raise ValueError("result.simulation.pim_bcast_issued: does not match trace opcode count")
    power_accounting = _object(
        _require(simulation, "power_accounting", "result.simulation"),
        "result.simulation.power_accounting",
    )
    if power_accounting.get("energy_units") != "pJ":
        raise ValueError("result.simulation.power_accounting.energy_units: must be 'pJ'")
    if not isinstance(power_accounting.get("standard_background_command_energy_available"), bool):
        raise ValueError(
            "result.simulation.power_accounting.standard_background_command_energy_available: "
            "must be a boolean"
        )
    if not isinstance(power_accounting.get("pim_event_coefficients_available"), bool):
        raise ValueError(
            "result.simulation.power_accounting.pim_event_coefficients_available: must be a boolean"
        )
    if dram_class == "LPDDR5PIM":
        if power_accounting.get("status") != "paper_two_layer":
            raise ValueError(
                "result.simulation.power_accounting.status: LPDDR5PIM must use "
                "the paper two-layer accounting"
            )
        if power_accounting.get("equation") != "E = E_LPDDR + E_PIM":
            raise ValueError(
                "result.simulation.power_accounting.equation: unsupported LPDDR5PIM equation"
            )
        units = _object(
            power_accounting.get("standard_energy_units"),
            "result.simulation.power_accounting.standard_energy_units",
        )
        if units != {"current": "mA", "voltage": "V", "time": "ns", "energy": "pJ"}:
            raise ValueError(
                "result.simulation.power_accounting.standard_energy_units: unsupported units"
            )
        source = _object(
            power_accounting.get("standard_power_source"),
            "result.simulation.power_accounting.standard_power_source",
        )
        for field, expected in {
            "profile_scope": "camera_ready_lpddr5_6400_analysis",
            "calibration": "not_device_calibrated",
            "dimensional_equation": "V[V] * I[mA] * t[ns] = E[pJ]",
            "legacy_conversion": "multiply_by_1e-3_after_V_mA_ns_product",
            "legacy_conversion_scale": 1e-3,
        }.items():
            if source.get(field) != expected:
                raise ValueError(
                    f"result.simulation.power_accounting.standard_power_source.{field}: "
                    "unsupported LPDDR5 provenance"
                )
    if dram_class == "LPDDR6PIM":
        if power_accounting.get("status") != "experimental_drampower_reference":
            raise ValueError(
                "result.simulation.power_accounting.status: LPDDR6PIM must identify "
                "experimental DRAMPower reference accounting"
            )
        if power_accounting.get("power_profile") != "DRAMPOWER_V620_LPDDR6_TEST_PROFILE":
            raise ValueError(
                "result.simulation.power_accounting.power_profile: unsupported LPDDR6 profile"
            )
        if power_accounting.get("standard_power_calibrated_to_device") is not False:
            raise ValueError(
                "result.simulation.power_accounting: LPDDR6 reference power must not "
                "claim device calibration"
            )
        standard_power_source = _object(
            power_accounting.get("standard_power_source"),
            "result.simulation.power_accounting.standard_power_source",
        )
        if (
            standard_power_source.get("version") != "v6.2.0"
            or standard_power_source.get("calibration") != "test_fixture_not_device_datasheet"
        ):
            raise ValueError(
                "result.simulation.power_accounting.standard_power_source: "
                "unsupported LPDDR6 reference provenance"
            )
    for field in (
        "total_standard_energy_pJ",
        "total_pim_event_energy_pJ",
        "total_energy_pJ",
    ):
        value = power_accounting.get(field)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0
        ):
            raise ValueError(f"result.simulation.power_accounting.{field}: must be non-negative")
    standard_energy = power_accounting.get("total_standard_energy_pJ")
    pim_energy = power_accounting.get("total_pim_event_energy_pJ")
    total_energy = power_accounting.get("total_energy_pJ")
    if standard_energy is not None and pim_energy is not None and total_energy is not None:
        if abs(total_energy - (standard_energy + pim_energy)) > max(1e-6, abs(total_energy) * 1e-9):
            raise ValueError(
                "result.simulation.power_accounting.total_energy_pJ: must equal "
                "standard plus PIM event energy"
            )
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

    provenance = _object(result["provenance"], "result.provenance")
    packages = _object(
        _require(provenance, "python_packages", "result.provenance"),
        "result.provenance.python_packages",
    )
    for package in ("ramulator", "pimscope", "numpy", "matplotlib", "PyYAML"):
        value = _require(packages, package, "result.provenance.python_packages")
        if not isinstance(value, str) or not value:
            raise ValueError(f"result.provenance.python_packages.{package}: must be a string")
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


def validate_aggregate(payload: Mapping[str, Any], *, kind: str) -> dict[str, Any]:
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
        raise ValueError(f"aggregate.schema_name: must be {AGGREGATE_SCHEMA_NAMES[kind]!r}")
    if kind == "decode_cycles" and aggregate.get("phase") != "decode":
        raise ValueError("aggregate.phase: decode_cycles must have phase='decode'")
    if kind == "prefill_cycles" and aggregate.get("phase") != "prefill":
        raise ValueError("aggregate.phase: prefill_cycles must have phase='prefill'")
    if kind == "pim_sharing_comparison" and "phase" in aggregate:
        raise ValueError("aggregate.phase: sharing aggregate must not declare a phase")
    rows = aggregate.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("aggregate.rows: must be a non-empty list")

    required = {
        "decode_cycles": ("model_name", "mode", "cycles", "runtime_ns", "replay_status"),
        "prefill_cycles": (
            "model_name",
            "model_key",
            "mode",
            "cycles",
            "runtime_ns",
            "replay_status",
        ),
        "pim_sharing_comparison": (
            "workload",
            "cycles_k1",
            "cycles_k2",
            "replay_ok_k1",
            "replay_ok_k2",
        ),
    }[kind]
    keys: set[tuple[Any, ...]] = set()
    for index, row in enumerate(rows):
        row = _object(row, f"aggregate.rows[{index}]")
        for field in required:
            _require(row, field, f"aggregate.rows[{index}]")
        for field in ("cycles", "runtime_ns"):
            if field in row:
                value = row[field]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                    raise ValueError(f"aggregate.rows[{index}].{field}: must be non-negative")
        if "replay_status" in row and row["replay_status"] != "PASS":
            raise ValueError(f"aggregate.rows[{index}].replay_status: must be PASS")
        for field in ("replay_ok_k1", "replay_ok_k2"):
            if field in row and row[field] is not True:
                raise ValueError(f"aggregate.rows[{index}].{field}: must be true")
        if kind == "pim_sharing_comparison":
            for field in ("cycles_k1", "cycles_k2"):
                value = row[field]
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(f"aggregate.rows[{index}].{field}: must be positive integer")
            key = (row["workload"],)
        else:
            key = (row["model_name"], row["mode"])
        if key in keys:
            raise ValueError(f"aggregate.rows[{index}]: duplicate row key {key!r}")
        keys.add(key)
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
