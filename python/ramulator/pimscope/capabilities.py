"""Declared capabilities and adaptation boundaries of public PIMScope backends."""

from __future__ import annotations

from typing import Any

PIM_BACKEND_CAPABILITIES: dict[str, dict[str, Any]] = {
    "LPDDR5PIM": {
        "status": "supported",
        "dram_class": "LPDDR5PIM",
        "controller": "LPDDR5PIM",
        "frontend": "LPDDR5PIMConcreteTrace",
        "trace_schema": "lpddr5-pim-opcode-v0.2",
        "paper_artifact_backend": True,
        "notes": [
            "Validated for the public one-controller/channel topology.",
            "Paper artifact reproduction remains pinned to this backend.",
        ],
    },
    "LPDDR6PIM": {
        "status": "planned",
        "base_dram_class": "LPDDR6",
        "available_base_dram_model": True,
        "controller": None,
        "frontend": None,
        "trace_schema": None,
        "paper_artifact_backend": False,
        "notes": [
            "Generic LPDDR6 timing support is not LPDDR6-PIM support.",
            "Requires standard-specific PIM commands, scheduling, trace, hierarchy, "
            "refresh, and power semantics.",
        ],
    },
}


def pim_backend_capabilities() -> dict[str, dict[str, Any]]:
    """Return a copy of the public backend capability declaration."""
    return {
        name: {
            **capability,
            "notes": list(capability.get("notes", [])),
        }
        for name, capability in PIM_BACKEND_CAPABILITIES.items()
    }


def require_supported_pim_backend(dram_class: str) -> dict[str, Any]:
    """Return the supported backend or fail with an explicit adaptation boundary."""
    if dram_class == "LPDDR6":
        raise ValueError(
            "hardware.dram_class: generic LPDDR6 is available in Ramulator, but "
            "LPDDR6-PIM is not implemented; use LPDDR5PIM or follow the P1-27 adaptation plan"
        )
    capability = PIM_BACKEND_CAPABILITIES.get(dram_class)
    if capability is None:
        supported = [
            name
            for name, item in PIM_BACKEND_CAPABILITIES.items()
            if item["status"] == "supported"
        ]
        raise ValueError(
            f"hardware.dram_class: unknown PIM backend {dram_class!r}; "
            f"supported backends: {supported}"
        )
    if capability["status"] != "supported":
        base = capability.get("base_dram_class")
        base_note = f"; base DRAM standard {base} is available" if base else ""
        raise ValueError(
            f"hardware.dram_class: PIM backend {dram_class!r} is {capability['status']}{base_note}"
        )
    return capability


__all__ = [
    "PIM_BACKEND_CAPABILITIES",
    "pim_backend_capabilities",
    "require_supported_pim_backend",
]
