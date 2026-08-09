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
        "status": "experimental",
        "base_dram_class": "LPDDR6",
        "available_base_dram_model": True,
        "dram_class": "LPDDR6PIM",
        "controller": "LPDDR6PIM",
        "frontend": "LPDDR6PIMConcreteTrace",
        "trace_schema": "lpddr6-pim-opcode-v0.1",
        "paper_artifact_backend": False,
        "notes": [
            "Uses LPDDR6 CAS and short/long host access timing vocabulary.",
            "PIM event energy is reported separately; standard LPDDR6 power is unavailable.",
            "Paper artifact reproduction remains pinned to LPDDR5PIM.",
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
            "hardware.dram_class: generic LPDDR6 is available in Ramulator; "
            "select LPDDR6PIM explicitly for the experimental PIM backend"
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
    if capability["status"] not in {"supported", "experimental"}:
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
