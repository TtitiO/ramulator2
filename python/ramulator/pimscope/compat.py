"""Compatibility helpers for the shared-block terminology migration.

Canonical public names use ``PIM block`` terminology.  The aliases in this
module are intentionally narrow, emit deprecation warnings, and are removed
from normalized manifests/results so fingerprints and downstream consumers see
one vocabulary.  This is a one-release migration aid, not a second public
architecture model.
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Mapping
from typing import Any

LEGACY_PIM_FIELD_ALIASES = {
    "pim_banks_per_mpu": "pim_banks_per_block",
}
LEGACY_EXECUTION_MODEL_ALIASES = {
    "shared_mpu_serial": "shared_block_serial",
}
LEGACY_RESULT_FIELD_ALIASES = {
    "pim_mpu_group_stalls": "pim_shared_block_stalls",
    "pim_banks_per_mpu": "pim_banks_per_block",
    "pim_mpu_group_count": "pim_shared_block_count",
    "effective_mpu_groups": "effective_shared_blocks",
    "num_mpu_group_busy_blocked_cycles": "num_shared_block_busy_blocked_cycles",
    "mpu_grouping_policy": "shared_block_grouping_policy",
}


def _warn(old: str, new: str) -> None:
    warnings.warn(
        f"legacy PIMScope name {old!r} is deprecated; use {new!r}",
        DeprecationWarning,
        stacklevel=3,
    )


def _coalesce(mapping: dict[str, Any], old: str, new: str, *, path: str) -> None:
    if old not in mapping:
        return
    if new in mapping and mapping[new] != mapping[old]:
        raise ValueError(
            f"{path}: legacy field {old!r} conflicts with canonical field {new!r}"
        )
    _warn(old, new)
    mapping[new] = mapping.pop(old)


def canonicalize_legacy_pim_config(
    value: Mapping[str, Any], *, path: str = "hardware.pim"
) -> dict[str, Any]:
    """Normalize legacy PIM manifest fields and execution-model values."""
    result = copy.deepcopy(dict(value))
    for old, new in LEGACY_PIM_FIELD_ALIASES.items():
        _coalesce(result, old, new, path=path)
    model = result.get("pim_mac_execution_model")
    if model in LEGACY_EXECUTION_MODEL_ALIASES:
        new = LEGACY_EXECUTION_MODEL_ALIASES[model]
        _warn(model, new)
        result["pim_mac_execution_model"] = new
    return result


def _canonicalize_result_object(value: Any, *, path: str) -> Any:
    if isinstance(value, list):
        return [
            _canonicalize_result_object(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if not isinstance(value, dict):
        return value

    result = dict(value)
    for old, new in LEGACY_RESULT_FIELD_ALIASES.items():
        _coalesce(result, old, new, path=path)
    if result.get("pim_mac_execution_model") in LEGACY_EXECUTION_MODEL_ALIASES:
        old = result["pim_mac_execution_model"]
        new = LEGACY_EXECUTION_MODEL_ALIASES[old]
        _warn(old, new)
        result["pim_mac_execution_model"] = new
    return {
        key: _canonicalize_result_object(item, path=f"{path}.{key}")
        for key, item in result.items()
    }


def canonicalize_legacy_result(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a normalized result/artifact object with canonical field names.

    Legacy and canonical fields may coexist only when their values agree.  A
    conflicting pair fails closed rather than allowing a result to silently
    mix two terminology generations.
    """
    if not isinstance(value, Mapping):
        raise ValueError(f"result must be an object, got {type(value).__name__}")
    return _canonicalize_result_object(copy.deepcopy(dict(value)), path="result")
