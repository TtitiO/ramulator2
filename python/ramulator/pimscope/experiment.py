"""Public researcher experiment execution API for LPDDR5-PIM."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from ramulator.dram.addressing import extract_dram_layout
from ramulator.pimscope.backend import (
    count_concrete_opcodes,
    create_dram,
    hardware_config_from_manifest,
    replay_concrete_trace,
)
from ramulator.pimscope.compat import canonicalize_legacy_result
from ramulator.pimscope.config import ResolvedExperiment
from ramulator.pimscope.schema import RESULT_SCHEMA_NAME, RESULT_SCHEMA_VERSION

RAMULATOR_ROOT = Path(__file__).resolve().parents[3]


def _git_revision(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def validate_backend(resolved: ResolvedExperiment) -> dict[str, Any]:
    """Resolve the configured DRAM and return its canonical public layout."""
    cfg = hardware_config_from_manifest(resolved)
    dram = create_dram(cfg)
    organization, timing = dram.resolve()
    layout = extract_dram_layout(dram)
    return {
        "organization": organization,
        "timing": timing,
        "address_layout": {
            "mapping_version": layout["mapping_version"],
            "level_names": layout["level_names"],
            "level_sizes": layout["level_sizes"],
            "address_level_sizes": layout["address_level_sizes"],
            "internal_prefetch_size": layout["internal_prefetch_size"],
            "tx_bytes": layout["tx_bytes"],
            "capacity_bytes": layout["capacity_bytes"],
        },
        "dram_config": dram.to_config(),
    }


def _custom_model_spec(model: dict[str, Any], datatype: str):
    from ramulator.workload_surrogate.generate_full_transformer import ModelSpec

    return ModelSpec(
        name=model["name"],
        num_layers=model["num_layers"],
        hidden_size=model["hidden_size"],
        num_heads=model["num_heads"],
        num_kv_heads=model.get("num_kv_heads"),
        head_dim=model["head_dim"],
        ffn_hidden_size=model["ffn_hidden_size"],
        datatype=datatype,
        citation=model.get("citation"),
        paper_anchor=model.get("paper_anchor"),
        ffn_variant=model.get("ffn_variant", "swiglu_3proj"),
        activation=model.get("activation", "silu"),
    )


def retarget_semantic_banks(records: list[dict], total_bank_units: int) -> None:
    """Retarget generated bank sequences to the resolved device topology."""
    if total_bank_units <= 0:
        raise ValueError("resolved hardware must expose at least one bank unit")
    for record in records:
        bank_sequence = record.get("bank_sequence")
        if bank_sequence:
            record["bank_sequence"] = list(range(total_bank_units))
        mapping = record.get("mapping_policy")
        if isinstance(mapping, dict) and "bank_sequence_policy" in mapping:
            mapping["bank_sequence_policy"] = "resolved_hardware_round_robin"


def _generate_semantic(workload: dict[str, Any]) -> tuple[list[dict], str]:
    from ramulator.workload_surrogate.generate_full_transformer import (
        generate_dense_decoder_records_for_model,
        generate_dense_decoder_records_from_spec,
        generate_dense_prefill_records_for_model,
        generate_dense_prefill_records_from_spec,
        generate_mixtral_8x7b_decoder_records,
        get_mixtral_8x7b_moe_decoder_manifests,
    )

    phase = workload["phase"]
    model = workload["model"]
    schedule = workload["schedule_policy"]
    if model == "mixtral-8x7b":
        attention, moe = get_mixtral_8x7b_moe_decoder_manifests(
            past_len=workload["past_len"], schedule_policy=schedule
        )
        return (
            generate_mixtral_8x7b_decoder_records(attention_manifest=attention, moe_manifest=moe),
            model,
        )
    if isinstance(model, dict):
        spec = _custom_model_spec(model, workload["datatype"])
        if phase == "decode":
            records = generate_dense_decoder_records_from_spec(
                spec, past_len=workload["past_len"], schedule_policy=schedule
            )
        else:
            records = generate_dense_prefill_records_from_spec(
                spec, prompt_len=workload["prompt_len"], schedule_policy=schedule
            )
        return records, model["name"]
    if phase == "decode":
        records = generate_dense_decoder_records_for_model(
            model, past_len=workload["past_len"], schedule_policy=schedule
        )
    else:
        records = generate_dense_prefill_records_for_model(
            model, prompt_len=workload["prompt_len"], schedule_policy=schedule
        )
    return records, model


def run_experiment(
    resolved: ResolvedExperiment,
    *,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate, lower, validate, and replay one researcher manifest."""
    from ramulator.workload_surrogate.generate_lpddr5_pim_concrete import (
        lower_semantic_records_to_concrete,
    )

    manifest = resolved.manifest
    workload = manifest["workload"]
    backend_cfg = hardware_config_from_manifest(resolved)
    dram = create_dram(backend_cfg)
    organization, timing = dram.resolve()
    semantic, model_name = _generate_semantic(workload)
    layout = extract_dram_layout(dram)
    retarget_semantic_banks(semantic, layout["total_bank_units"])

    interleave_banks = workload["max_inflight_requests"] > 1
    lower_kwargs: dict[str, Any] = {
        "materialize_weights": workload["weight_residency"] == "full_preload",
        "interleave_banks": interleave_banks,
        "mac_mode": workload["mac_mode"],
        "address_layout": layout,
        "synthetic_address_policy": "bounded_surrogate_v1",
        "addr_vec_size": layout["addr_vec_size"],
        "bank_positions": layout["bank_positions"],
        "bank_counts": layout["bank_counts"],
        "row_level": layout["row_pos"],
        "col_level": layout["col_pos"],
        "max_expanded_records": 100_000_000_000,
    }
    if interleave_banks:
        lower_kwargs["interleave_depth"] = workload["interleave_depth"]
    concrete = lower_semantic_records_to_concrete(semantic, **lower_kwargs)

    effective_inflight = workload["max_inflight_requests"]
    if interleave_banks and workload["mac_mode"] in {"per_kind", "per_bank"}:
        max_span = max(
            (len(record["bank_sequence"]) for record in semantic if record.get("bank_sequence")),
            default=1,
        )
        effective_inflight = max(effective_inflight, max_span * workload["interleave_depth"])
    elif workload["mac_mode"] == "all_bank":
        effective_inflight = 1

    replay = replay_concrete_trace(
        concrete,
        max_inflight_requests=effective_inflight,
        backend_cfg=backend_cfg,
    )
    provenance_payload: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "ramulator2_commit": _git_revision(RAMULATOR_ROOT),
        # Keep provenance portable when the manifest came from an absolute path.
        "config_source": Path(resolved.source).name
        if resolved.source != "<memory>"
        else resolved.source,
    }
    if provenance:
        provenance_payload.update(dict(provenance))

    return canonicalize_legacy_result({
        "schema_version": RESULT_SCHEMA_VERSION,
        "schema_name": RESULT_SCHEMA_NAME,
        "experiment": manifest["experiment"],
        "status": "PASS" if replay["replay_ok"] else "FAIL",
        "manifest_fingerprint": resolved.fingerprint,
        "resolved_manifest": manifest,
        "resolved_hardware": {
            "organization": organization,
            "timing": timing,
            "address_layout": {
                "mapping_version": layout["mapping_version"],
                "level_names": layout["level_names"],
                "level_sizes": layout["level_sizes"],
                "address_level_sizes": layout["address_level_sizes"],
                "internal_prefetch_size": layout["internal_prefetch_size"],
                "tx_bytes": layout["tx_bytes"],
                "capacity_bytes": layout["capacity_bytes"],
            },
            "effective_max_inflight_requests": effective_inflight,
        },
        "workload_summary": {
            "model": model_name,
            "phase": workload["phase"],
            "semantic_records": len(semantic),
            "concrete_records": len(concrete),
            "concrete_opcode_counts": count_concrete_opcodes(concrete),
            "host_address_policy": "bounded_surrogate_v1",
            "surrogate_scope": {
                "architecture_dimensions": "published_or_user_supplied",
                "runtime_trace": False,
                "numerical_execution": False,
                "silicon_calibration": False,
            },
        },
        "simulation": replay,
        "provenance": provenance_payload,
    })
