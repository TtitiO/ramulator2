"""Validated public experiment manifests for PIMScope.

This module is deliberately independent of the compiled Ramulator extension so
researchers can validate and inspect manifests before building or simulating.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ramulator.pimscope.compat import canonicalize_legacy_pim_config

MANIFEST_SCHEMA_VERSION = 1

DEFAULT_HARDWARE = {
    "dram_class": "LPDDR5PIM",
    "org_preset": "LPDDR5_8Gb_x16",
    "timing_preset": "LPDDR5_6400",
    "org_overrides": {},
    "timing_overrides": {},
    "pim": {
        "pim_datatype": "int8",
        "pim_banks_per_block": 2,
        "pim_mac_execution_model": "shared_block_serial",
    },
    "controller": {
        "scheduler": "FRFCFS",
        "refresh_manager": "NoRefresh",
        "row_policy": "Open",
        "addr_mapper": "PassThroughAddrMapper",
    },
    "memory_system": {
        "clock_ratio": 1,
        "channel_mapper": "CacheLineInterleave",
    },
    "topology": {
        "controllers": 1,
        "channels": 1,
    },
    "frontend_clock_ratio": 4,
}

DEFAULT_WORKLOAD = {
    "model": "opt-125m",
    "datatype": "int8",
    "phase": "decode",
    "past_len": 32,
    "prompt_len": 12,
    "schedule_policy": "serialized",
    "weight_residency": "resident",
    "mac_mode": "per_kind",
    "seed": 12345,
    "max_inflight_requests": 16,
    "interleave_depth": 4,
}

SUPPORTED_TOP_LEVEL_FIELDS = {"schema_version", "experiment", "hardware", "workload", "output"}
SUPPORTED_HARDWARE_FIELDS = {
    "dram_class",
    "org_preset",
    "timing_preset",
    "org_overrides",
    "timing_overrides",
    "pim",
    "controller",
    "memory_system",
    "topology",
    "frontend_clock_ratio",
}
SUPPORTED_WORKLOAD_FIELDS = set(DEFAULT_WORKLOAD) | {"datatype"}
SUPPORTED_OUTPUT_FIELDS = {"path"}
SUPPORTED_MODEL_SPEC_FIELDS = {
    "name",
    "num_layers",
    "hidden_size",
    "num_heads",
    "num_kv_heads",
    "head_dim",
    "ffn_hidden_size",
    "ffn_variant",
    "activation",
    "citation",
    "paper_anchor",
}
REQUIRED_MODEL_SPEC_FIELDS = {
    "name",
    "num_layers",
    "hidden_size",
    "num_heads",
    "head_dim",
    "ffn_hidden_size",
}
SUPPORTED_CONTROLLER_FIELDS = {"scheduler", "refresh_manager", "row_policy", "addr_mapper"}
SUPPORTED_MEMORY_SYSTEM_FIELDS = {"clock_ratio", "channel_mapper"}
SUPPORTED_TOPOLOGY_FIELDS = {"controllers", "channels"}
SUPPORTED_SCHEDULERS = {"FRFCFS", "FRFCFSRowHit"}
SUPPORTED_REFRESH_MANAGERS = {"NoRefresh", "AllBank", "PerBank"}
SUPPORTED_ROW_POLICIES = {"Open", "ClosedCAP"}
SUPPORTED_ADDR_MAPPERS = {
    "PassThroughAddrMapper",
    "ChRaBaRoCo",
    "RoBaRaCoCh",
    "MOP4CLXOR",
    "RITAddrMapper",
}
SUPPORTED_CHANNEL_MAPPERS = {"CacheLineInterleave", "PassThroughChannelMapper"}
SUPPORTED_PHASES = {"decode", "prefill"}
SUPPORTED_WEIGHT_RESIDENCY = {"resident", "full_preload"}
SUPPORTED_MAC_MODES = {"per_kind", "per_bank", "all_bank"}
SUPPORTED_SCHEDULE_POLICIES = {"serialized", "overlap_independent_heads"}
SUPPORTED_PIM_DATATYPES = {"int8", "fp16", "int16", "bf16"}
SUPPORTED_WORKLOAD_DATATYPES = {"int8", "fp16", "bf16"}
SUPPORTED_FFN_VARIANTS = {"swiglu_3proj", "geglu_3proj", "relu_2proj"}
SUPPORTED_PIM_EXECUTION_MODELS = {"shared_block_serial", "subbank_overlap_experimental"}

# Public manifest fields accepted by ramulator.dram.LPDDR5PIM. Compatibility
# aliases/deprecated scale parameters are intentionally excluded.
SUPPORTED_PIM_FIELDS = {
    "pim_blocks_per_bank",
    "pim_banks_per_block",
    "pim_mac_execution_model",
    "pim_datatype",
    "pim_datatype_class",
    "pim_datatype_behavior_enabled",
    "pim_datatype_bits",
    "pim_simd_width_bits",
    "pim_lanes",
    "pim_ops_per_mac",
    "pim_ops_per_block_issue",
    "pim_ops_per_request",
    "pim_mac_issue_interval_cycles",
    "pim_mac_pipeline_latency_cycles",
    "pim_movement_cycles",
    "pim_writeback_cycles",
    "pim_slots_per_request",
    "pim_compute_energy_pJ_per_mac",
    "pim_array_local_energy_pJ",
    "pim_cell_to_pim_energy_pJ_per_256b",
    "pim_vrf_access_energy_pJ",
    "pim_srf_access_energy_pJ",
    "pim_mode_switch_energy_pJ",
}


@dataclass(frozen=True)
class ResolvedExperiment:
    manifest: dict[str, Any]
    source: str

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _fail(path: str, message: str) -> None:
    raise ValueError(f"{path}: {message}")


def _expect_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(path, f"must be an object, got {type(value).__name__}")
    return value


def _reject_unknown(mapping: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        _fail(path, f"unknown field(s): {', '.join(unknown)}")


def _positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(path, f"must be a positive integer, got {value!r}")
    return value


def _nonnegative_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(path, f"must be a non-negative integer, got {value!r}")
    return value


def _nonnegative_number(value: Any, path: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        _fail(path, f"must be a non-negative number, got {value!r}")
    return value


def _choice(value: Any, choices: set[str], path: str) -> str:
    if not isinstance(value, str) or value not in choices:
        _fail(path, f"must be one of {sorted(choices)}, got {value!r}")
    return value


def _load_text_manifest(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        data = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "YAML manifests require PyYAML; install the Ramulator Python package"
            ) from exc
        data = yaml.safe_load(text)
    else:
        raise ValueError(f"{path}: manifest extension must be .json, .yaml, or .yml")
    return _expect_mapping(data, str(path))


def _resolve_hardware(raw: Any) -> dict[str, Any]:
    hardware = copy.deepcopy(DEFAULT_HARDWARE)
    supplied = copy.deepcopy(_expect_mapping(raw or {}, "hardware"))
    if isinstance(supplied.get("pim"), dict):
        supplied["pim"] = canonicalize_legacy_pim_config(supplied["pim"])
    _reject_unknown(supplied, SUPPORTED_HARDWARE_FIELDS, "hardware")

    for field in ("dram_class", "org_preset", "timing_preset", "frontend_clock_ratio"):
        if field in supplied:
            hardware[field] = copy.deepcopy(supplied[field])
    for field in (
        "org_overrides",
        "timing_overrides",
        "pim",
        "controller",
        "memory_system",
        "topology",
    ):
        if field in supplied:
            section = _expect_mapping(supplied[field], f"hardware.{field}")
            if field in {"org_overrides", "timing_overrides"}:
                hardware[field] = copy.deepcopy(section)
            else:
                hardware[field].update(copy.deepcopy(section))

    _reject_unknown(hardware["topology"], SUPPORTED_TOPOLOGY_FIELDS, "hardware.topology")
    for field in SUPPORTED_TOPOLOGY_FIELDS:
        _positive_int(hardware["topology"][field], f"hardware.topology.{field}")
    if hardware["topology"]["controllers"] != 1:
        _fail(
            "hardware.topology.controllers",
            "unsupported by the public runner; only one controller is currently validated",
        )
    if hardware["topology"]["channels"] != 1:
        _fail(
            "hardware.topology.channels",
            "unsupported by the public runner; only one channel is currently validated",
        )

    if hardware["dram_class"] != "LPDDR5PIM":
        _fail(
            "hardware.dram_class", "the public experiment runner currently supports only LPDDR5PIM"
        )
    for field in ("org_preset", "timing_preset"):
        if not isinstance(hardware[field], str) or not hardware[field]:
            _fail(f"hardware.{field}", "must be a non-empty string")
    _positive_int(hardware["frontend_clock_ratio"], "hardware.frontend_clock_ratio")

    _reject_unknown(hardware["pim"], SUPPORTED_PIM_FIELDS, "hardware.pim")
    _reject_unknown(hardware["controller"], SUPPORTED_CONTROLLER_FIELDS, "hardware.controller")
    _reject_unknown(
        hardware["memory_system"], SUPPORTED_MEMORY_SYSTEM_FIELDS, "hardware.memory_system"
    )
    _choice(
        hardware["controller"]["scheduler"], SUPPORTED_SCHEDULERS, "hardware.controller.scheduler"
    )
    _choice(
        hardware["controller"]["refresh_manager"],
        SUPPORTED_REFRESH_MANAGERS,
        "hardware.controller.refresh_manager",
    )
    _choice(
        hardware["controller"]["row_policy"],
        SUPPORTED_ROW_POLICIES,
        "hardware.controller.row_policy",
    )
    _choice(
        hardware["controller"]["addr_mapper"],
        SUPPORTED_ADDR_MAPPERS,
        "hardware.controller.addr_mapper",
    )
    _choice(
        hardware["memory_system"]["channel_mapper"],
        SUPPORTED_CHANNEL_MAPPERS,
        "hardware.memory_system.channel_mapper",
    )
    _positive_int(hardware["memory_system"]["clock_ratio"], "hardware.memory_system.clock_ratio")

    pim = hardware["pim"]
    _choice(pim["pim_datatype"], SUPPORTED_PIM_DATATYPES, "hardware.pim.pim_datatype")
    if "pim_datatype_class" in pim:
        _choice(
            pim["pim_datatype_class"],
            SUPPORTED_PIM_DATATYPES,
            "hardware.pim.pim_datatype_class",
        )
        if pim["pim_datatype_class"] != pim["pim_datatype"]:
            _fail("hardware.pim.pim_datatype_class", "must equal pim_datatype")
    _choice(
        pim["pim_mac_execution_model"],
        SUPPORTED_PIM_EXECUTION_MODELS,
        "hardware.pim.pim_mac_execution_model",
    )
    if "pim_datatype_behavior_enabled" in pim and not isinstance(
        pim["pim_datatype_behavior_enabled"], bool
    ):
        _fail("hardware.pim.pim_datatype_behavior_enabled", "must be a boolean")

    for field in (
        "pim_blocks_per_bank",
        "pim_banks_per_block",
        "pim_datatype_bits",
        "pim_simd_width_bits",
        "pim_lanes",
        "pim_mac_issue_interval_cycles",
        "pim_mac_pipeline_latency_cycles",
        "pim_slots_per_request",
    ):
        if field in pim:
            _positive_int(pim[field], f"hardware.pim.{field}")
    for field in ("pim_movement_cycles", "pim_writeback_cycles"):
        if field in pim:
            value = _nonnegative_number(pim[field], f"hardware.pim.{field}")
            if not isinstance(value, int):
                _fail(f"hardware.pim.{field}", "must be an integer number of cycles")
    for field in (
        "pim_ops_per_mac",
        "pim_ops_per_block_issue",
        "pim_ops_per_request",
        "pim_compute_energy_pJ_per_mac",
        "pim_array_local_energy_pJ",
        "pim_cell_to_pim_energy_pJ_per_256b",
        "pim_vrf_access_energy_pJ",
        "pim_srf_access_energy_pJ",
        "pim_mode_switch_energy_pJ",
    ):
        if field in pim:
            _nonnegative_number(pim[field], f"hardware.pim.{field}")
    for field, value in hardware["org_overrides"].items():
        _positive_int(value, f"hardware.org_overrides.{field}")
    for field, value in hardware["timing_overrides"].items():
        _nonnegative_number(value, f"hardware.timing_overrides.{field}")
    return hardware


def _resolve_model(value: Any) -> str | dict[str, Any]:
    if isinstance(value, str):
        if not value:
            _fail("workload.model", "must be a non-empty built-in model key")
        return value
    model = copy.deepcopy(_expect_mapping(value, "workload.model"))
    _reject_unknown(model, SUPPORTED_MODEL_SPEC_FIELDS, "workload.model")
    missing = sorted(REQUIRED_MODEL_SPEC_FIELDS - set(model))
    if missing:
        _fail("workload.model", f"missing required field(s): {', '.join(missing)}")
    if not isinstance(model["name"], str) or not model["name"].strip():
        _fail("workload.model.name", "must be a non-empty string")
    for field in ("num_layers", "hidden_size", "num_heads", "head_dim", "ffn_hidden_size"):
        _positive_int(model[field], f"workload.model.{field}")
    if "num_kv_heads" in model:
        _positive_int(model["num_kv_heads"], "workload.model.num_kv_heads")
        if model["num_kv_heads"] > model["num_heads"]:
            _fail("workload.model.num_kv_heads", "must not exceed num_heads")
        if model["num_heads"] % model["num_kv_heads"] != 0:
            _fail("workload.model.num_kv_heads", "must divide num_heads")
    if "ffn_variant" in model:
        _choice(model["ffn_variant"], SUPPORTED_FFN_VARIANTS, "workload.model.ffn_variant")
    for field in ("activation", "citation", "paper_anchor"):
        if field in model and (not isinstance(model[field], str) or not model[field].strip()):
            _fail(f"workload.model.{field}", "must be a non-empty string")
    return model


def _resolve_workload(raw: Any) -> dict[str, Any]:
    workload = copy.deepcopy(DEFAULT_WORKLOAD)
    supplied = _expect_mapping(raw or {}, "workload")
    _reject_unknown(supplied, SUPPORTED_WORKLOAD_FIELDS, "workload")
    workload.update(copy.deepcopy(supplied))
    workload["model"] = _resolve_model(workload["model"])
    workload["datatype"] = workload.get("datatype", "int8")

    _choice(workload["datatype"], SUPPORTED_WORKLOAD_DATATYPES, "workload.datatype")
    _choice(workload["phase"], SUPPORTED_PHASES, "workload.phase")
    _choice(workload["weight_residency"], SUPPORTED_WEIGHT_RESIDENCY, "workload.weight_residency")
    _choice(workload["mac_mode"], SUPPORTED_MAC_MODES, "workload.mac_mode")
    _choice(
        workload["schedule_policy"],
        SUPPORTED_SCHEDULE_POLICIES,
        "workload.schedule_policy",
    )
    _positive_int(workload["past_len"], "workload.past_len")
    _positive_int(workload["prompt_len"], "workload.prompt_len")
    _nonnegative_int(workload["seed"], "workload.seed")
    _positive_int(workload["max_inflight_requests"], "workload.max_inflight_requests")
    _positive_int(workload["interleave_depth"], "workload.interleave_depth")
    if workload["phase"] == "prefill" and workload["schedule_policy"] != "serialized":
        _fail("workload.schedule_policy", "prefill currently supports only 'serialized'")
    if workload["model"] == "mixtral-8x7b" and workload["phase"] != "decode":
        _fail("workload.phase", "Mixtral-8x7B currently supports decode only")
    if isinstance(workload["model"], dict) and workload["schedule_policy"] != "serialized":
        _fail(
            "workload.schedule_policy",
            "custom dense models currently support only 'serialized'",
        )
    return workload


def resolve_experiment_manifest(
    raw: dict[str, Any], *, source: str = "<memory>"
) -> ResolvedExperiment:
    raw = _expect_mapping(raw, source)
    _reject_unknown(raw, SUPPORTED_TOP_LEVEL_FIELDS, source)
    schema_version = raw.get("schema_version", MANIFEST_SCHEMA_VERSION)
    if schema_version != MANIFEST_SCHEMA_VERSION:
        _fail("schema_version", f"must be {MANIFEST_SCHEMA_VERSION}, got {schema_version!r}")
    experiment = raw.get("experiment", "pimscope-experiment")
    if not isinstance(experiment, str) or not experiment.strip():
        _fail("experiment", "must be a non-empty string")

    output = _expect_mapping(raw.get("output", {"path": "results/custom/result.json"}), "output")
    _reject_unknown(output, SUPPORTED_OUTPUT_FIELDS, "output")
    output_path = output.get("path", "results/custom/result.json")
    if not isinstance(output_path, str) or not output_path.strip():
        _fail("output.path", "must be a non-empty path string")

    hardware = _resolve_hardware(raw.get("hardware", {}))
    workload = _resolve_workload(raw.get("workload", {}))
    if hardware["pim"]["pim_datatype"] != workload["datatype"]:
        _fail(
            "workload.datatype",
            "must equal hardware.pim.pim_datatype so workload lowering and PIM resources agree",
        )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment": experiment,
        "hardware": hardware,
        "workload": workload,
        "output": {"path": output_path},
    }
    return ResolvedExperiment(manifest=manifest, source=source)


def load_experiment_manifest(path: str | Path) -> ResolvedExperiment:
    manifest_path = Path(path).expanduser().resolve()
    return resolve_experiment_manifest(
        _load_text_manifest(manifest_path), source=str(manifest_path)
    )


def apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    updated = copy.deepcopy(raw)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override must use dotted.path=value syntax, got {override!r}")
        dotted, encoded = override.split("=", 1)
        keys = [key for key in dotted.split(".") if key]
        if not keys:
            raise ValueError(f"override path is empty in {override!r}")
        try:
            value = json.loads(encoded)
        except json.JSONDecodeError:
            value = encoded
        cursor = updated
        for key in keys[:-1]:
            existing = cursor.get(key)
            if existing is None:
                cursor[key] = {}
            elif not isinstance(existing, dict):
                raise ValueError(f"override path {dotted!r} crosses non-object field {key!r}")
            cursor = cursor[key]
        cursor[keys[-1]] = value
    return updated


def load_raw_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    return _load_text_manifest(manifest_path)
