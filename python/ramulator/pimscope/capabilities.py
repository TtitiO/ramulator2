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
        "validated_rank_counts": [1, 2],
        "validated_topology": {"controllers": 1, "channels": 1},
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
        "validated_rank_counts": [1, 2],
        "validated_topology": {"controllers": 1, "channels": 1},
        "subchannel_model": {
            "status": "single_subchannel_only",
            "modeled_subchannels_per_channel": 1,
            "refresh_density_reference_subchannels": 2,
            "independent_subchannel_scheduling": False,
        },
        "notes": [
            "Uses LPDDR6 CAS and short/long host access timing vocabulary.",
            (
                "One modeled Channel represents one 12-bit sub-channel; the paired "
                "physical sub-channel is not instantiated."
            ),
            (
                "Rank-local modes, refresh, addressing, and all-bank lowering are "
                "validated for one and two ranks."
            ),
            (
                "Standard energy uses the DRAMPower v6.2 LPDDR6 test fixture and is "
                "not calibrated to a production device."
            ),
            "PIM event energy follows the LPDDR5PIM coefficient method.",
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
            "validated_rank_counts": list(capability.get("validated_rank_counts", [])),
            "validated_topology": dict(capability.get("validated_topology", {})),
            "subchannel_model": dict(capability.get("subchannel_model", {})),
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
