"""Data collection helpers for P4 full-transformer paper figures.

Collects command counts, replay stats, and parameter sensitivity data
from the P4 semantic generators and concrete lowering pipeline.
"""

from __future__ import annotations

import json
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from ramulator.workload_surrogate.generate_full_transformer import (
    generate_attention_records,
    generate_ffn_records,
    generate_moe_records,
    generate_full_transformer_layer_records,
    get_tiny_attention_manifest,
    get_tiny_ffn_manifest,
    get_tiny_moe_manifest,
    build_full_transformer_provenance_summary,
    build_provenance_summary,
    get_model_spec,
)
from ramulator.workload_surrogate.generate_lpddr5_pim_concrete import (
    lower_semantic_records_to_concrete,
)
from ramulator.workload_surrogate.lpddr5_pim_concrete_trace import (
    CONCRETE_OPCODES,
    MODE_OPCODES,
    REQUEST_OPCODES,
)
from ramulator.dram.lpddr5_pim import PIM_DATATYPE_RESOURCES

# Paper manifests (realistic model configurations)
try:
    from tests.analysis.figures.p4_paper_manifests import (
        DEFAULT_PAPER_ATTENTION,
        DEFAULT_PAPER_FFN,
        DEFAULT_PAPER_MOE,
        PAPER_ATTENTION_CONFIGS,
        PAPER_FFN_CONFIGS,
        PAPER_MOE_CONFIGS,
    )
except ImportError:
    PAPER_ATTENTION_CONFIGS = {}
    PAPER_FFN_CONFIGS = {}
    PAPER_MOE_CONFIGS = {}
    DEFAULT_PAPER_ATTENTION = ""
    DEFAULT_PAPER_FFN = ""
    DEFAULT_PAPER_MOE = ""


# ─── Helpers ───────────────────────────────────────────────────────────


def _count_concrete_opcodes(concrete_records: list[dict]) -> dict[str, int]:
    """Count concrete opcodes by type (SB, HAB, PIM_MAC, etc.)."""
    counts: dict[str, int] = {}
    for rec in concrete_records:
        opcode = rec["opcode"]
        repeat = rec.get("repeat", 1)
        counts[opcode] = counts.get(opcode, 0) + repeat
    return counts


def _ceil_div(numerator: int, denominator: int) -> int:
    """Return ceil(numerator / denominator) for positive integer denominators."""
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return (int(numerator) + int(denominator) - 1) // int(denominator)


def _moe_expert_pim_mac_per_layer(
    *,
    hidden_size: int,
    expert_hidden_size: int,
    top_k: int,
    lanes: int,
    projections: int,
) -> int:
    """Return per-layer expert PIM_MAC repeats for active MoE experts.

    ``projections=1`` is the current traceable fused-expert surrogate.  Real
    Mixtral-style SwiGLU experts require three GEMV projections (gate/up/down),
    so ``projections=3`` is reported as an analytical diagnostic unless the
    trace generator emits those three expert records explicitly.
    """
    return int(top_k) * int(projections) * _ceil_div(
        int(hidden_size) * int(expert_hidden_size), int(lanes)
    )


def _count_semantic_kinds(semantic_records: list[dict]) -> dict[str, int]:
    """Count semantic record kinds (AttentionScore, FFNProjection, etc.)."""
    counts: dict[str, int] = {}
    for rec in semantic_records:
        kind = rec["kind"]
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def _host_access_request_count(record: dict) -> int:
    """Count repeat-expanded host transactions represented by a HostRead/HostWrite record."""
    address_policy = dict(record.get("address_policy", {}))
    repeat = int(record.get("repeat", 1))
    return max(1, int(address_policy.get("count", 1))) * repeat


def _pim_data_move_request_count(record: dict) -> int:
    """Count recurring semantic PIM setup moves without re-labeling them as native BCASTs."""
    movement_policy = dict(record.get("movement_policy", {}))
    if movement_policy.get("operand_role") == "weight":
        # Paper figures use steady-state decode semantics: weights are resident.
        return 0
    return max(1, int(record.get("num_requests", 1))) * int(record.get("repeat", 1))


def _pim_data_move_display_op(record: dict) -> str:
    """Map semantic setup/materialization moves onto paper-facing READ/WRITE buckets."""
    operand_role = dict(record.get("movement_policy", {})).get("operand_role")
    if operand_role == "expert_output_combine":
        return "WRITE"
    return "READ"


def _operator_decomposition_counts(semantic_records: list[dict], concrete_counts: dict[str, int]) -> dict[str, int]:
    """Return the current paper-facing op decomposition.

    The old figures displayed backend setup/control commands directly. Decode-block
    v2 models ordinary KV-cache traffic as semantic HostRead/HostWrite, which now
    lower to native READ/WRITE concrete requests, while compute tiles remain native
    PIM_MAC replay operations. Paper-facing figures show those replay-visible
    operations instead of stale setup opcodes.
    """
    counts: dict[str, int] = {}
    for record in semantic_records:
        kind = record["kind"]
        if kind == "HostRead":
            counts["READ"] = counts.get("READ", 0) + _host_access_request_count(record)
        elif kind == "HostWrite":
            counts["WRITE"] = counts.get("WRITE", 0) + _host_access_request_count(record)
        elif kind == "PIMDataMove":
            move_count = _pim_data_move_request_count(record)
            if move_count:
                op = _pim_data_move_display_op(record)
                counts[op] = counts.get(op, 0) + move_count
    if concrete_counts.get("PIM_MAC", 0):
        counts["PIM_MAC"] = int(concrete_counts["PIM_MAC"])
    return counts


def _pim_bcast_by_materialization_mode(semantic_records: list[dict], *, manifest_name: str) -> dict[str, int]:
    """Return PIM_BCAST repeats for steady-state and cold-start lowering."""
    steady = lower_semantic_records_to_concrete(
        semantic_records,
        manifest_name=f"{manifest_name}_steady_state",
        materialize_weights=False,
    )
    cold = lower_semantic_records_to_concrete(
        semantic_records,
        manifest_name=f"{manifest_name}_cold_start",
        materialize_weights=True,
    )
    return {
        "steady_state": _count_concrete_opcodes(steady).get("PIM_BCAST", 0),
        "cold_start": _count_concrete_opcodes(cold).get("PIM_BCAST", 0),
    }


# ─── Attention data ────────────────────────────────────────────────────


def collect_attention_decomposition(
    manifest: dict | None = None,
) -> dict[str, Any]:
    """Generate attention semantic records and lower to concrete opcodes.

    Returns a dict with:
      - semantic_counts: dict of record kind -> count
      - num_semantic_records: total semantic records
      - concrete_counts: dict of opcode -> effective count (including repeat)
      - num_concrete_records: total concrete record entries
      - provenance: the attention provenance summary
    """
    if manifest is None:
        manifest = get_tiny_attention_manifest()

    semantic = generate_attention_records(manifest)
    concrete = lower_semantic_records_to_concrete(
        semantic, manifest_name=manifest.get("manifest_name", "attention")
    )
    provenance = build_provenance_summary(semantic, manifest)

    concrete_counts = _count_concrete_opcodes(concrete)
    return {
        "manifest_name": manifest.get("manifest_name", "tiny_attention"),
        "num_heads": int(manifest["num_heads"]),
        "head_dim": int(manifest["head_dim"]),
        "past_len": int(manifest["past_len"]),
        "datatype": manifest["datatype"],
        "schedule_policy": manifest.get("schedule_policy", "serialized"),
        "semantic_counts": _count_semantic_kinds(semantic),
        "num_semantic_records": len(semantic),
        "concrete_counts": concrete_counts,
        "operator_counts": _operator_decomposition_counts(semantic, concrete_counts),
        "pim_bcast_by_mode": _pim_bcast_by_materialization_mode(
            semantic,
            manifest_name=manifest.get("manifest_name", "attention"),
        ),
        "num_concrete_records": len(concrete),
        "provenance": provenance,
    }


def collect_attention_sweep(
    num_heads_list: list[int] | None = None,
    past_len_list: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Run parameter sweep over num_heads and past_len.

    Returns list of dicts, each with the decomposition for one config.
    """
    if num_heads_list is None:
        num_heads_list = [1, 2, 4, 8]
    if past_len_list is None:
        past_len_list = [32, 64, 128, 256]

    results = []
    for num_heads in num_heads_list:
        for past_len in past_len_list:
            manifest = get_tiny_attention_manifest()
            manifest["num_heads"] = num_heads
            manifest["past_len"] = past_len
            # Ensure tile_tokens divides past_len
            tile_tokens = min(manifest["score_tile_tokens"], past_len)
            manifest["score_tile_tokens"] = tile_tokens
            manifest["context_tile_tokens"] = tile_tokens

            result = collect_attention_decomposition(manifest)
            result["_num_heads"] = num_heads
            result["_past_len"] = past_len
            results.append(result)

    return results


# ─── FFN data ──────────────────────────────────────────────────────────


def collect_ffn_decomposition(
    manifest: dict | None = None,
) -> dict[str, Any]:
    """Generate FFN semantic records and lower to concrete opcodes."""
    if manifest is None:
        manifest = get_tiny_ffn_manifest()

    semantic = generate_ffn_records(manifest)
    concrete = lower_semantic_records_to_concrete(
        semantic, manifest_name=manifest.get("manifest_name", "ffn")
    )

    concrete_counts = _count_concrete_opcodes(concrete)
    return {
        "manifest_name": manifest.get("manifest_name", "tiny_ffn"),
        "hidden_size": int(manifest["hidden_size"]),
        "ffn_hidden_size": int(manifest["ffn_hidden_size"]),
        "datatype": manifest["datatype"],
        "activation": manifest.get("activation", "silu"),
        "semantic_counts": _count_semantic_kinds(semantic),
        "num_semantic_records": len(semantic),
        "concrete_counts": concrete_counts,
        "operator_counts": _operator_decomposition_counts(semantic, concrete_counts),
        "pim_bcast_by_mode": _pim_bcast_by_materialization_mode(
            semantic,
            manifest_name=manifest.get("manifest_name", "ffn"),
        ),
        "num_concrete_records": len(concrete),
    }


# ─── MoE data ──────────────────────────────────────────────────────────


def collect_moe_expert_sensitivity_sweep(
    top_k_values: list[int] | None = None,
    num_experts_values: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Sweep top_k and num_experts for generator-level MoE decomposition."""
    if top_k_values is None:
        top_k_values = [1, 2, 3, 4]
    if num_experts_values is None:
        num_experts_values = [4, 8, 16]

    results: list[dict[str, Any]] = []
    for num_experts in num_experts_values:
        for top_k in top_k_values:
            if top_k > num_experts:
                continue
            manifest = get_tiny_moe_manifest()
            manifest["num_experts"] = num_experts
            manifest["top_k"] = top_k
            manifest["selected_experts"] = list(range(top_k))
            result = collect_moe_decomposition(manifest)
            result["_num_experts"] = num_experts
            result["_top_k"] = top_k
            results.append(result)

    return results


def collect_moe_decomposition(
    manifest: dict | None = None,
) -> dict[str, Any]:
    """Generate MoE semantic records and lower to concrete opcodes."""
    if manifest is None:
        manifest = get_tiny_moe_manifest()

    semantic = generate_moe_records(manifest)
    concrete = lower_semantic_records_to_concrete(
        semantic, manifest_name=manifest.get("manifest_name", "moe")
    )

    concrete_counts = _count_concrete_opcodes(concrete)
    return {
        "manifest_name": manifest.get("manifest_name", "tiny_moe"),
        "hidden_size": int(manifest["hidden_size"]),
        "expert_hidden_size": int(manifest["expert_hidden_size"]),
        "num_experts": int(manifest["num_experts"]),
        "top_k": int(manifest["top_k"]),
        "selected_experts": list(manifest["selected_experts"]),
        "datatype": manifest["datatype"],
        "schedule_policy": manifest.get("schedule_policy", "serialized"),
        "semantic_counts": _count_semantic_kinds(semantic),
        "num_semantic_records": len(semantic),
        "concrete_counts": concrete_counts,
        "operator_counts": _operator_decomposition_counts(semantic, concrete_counts),
        "pim_bcast_by_mode": _pim_bcast_by_materialization_mode(
            semantic,
            manifest_name=manifest.get("manifest_name", "moe"),
        ),
        "num_concrete_records": len(concrete),
    }


# ─── Combined layer data ──────────────────────────────────────────────


def collect_combined_layer_data() -> dict[str, Any]:
    """Generate combined attention+FFN+MoE layer and lower to concrete."""
    semantic = generate_full_transformer_layer_records()
    concrete = lower_semantic_records_to_concrete(
        semantic, manifest_name="combined_layer"
    )
    provenance = build_full_transformer_provenance_summary(semantic)

    return {
        "manifest_name": "combined_tiny_layer",
        "semantic_counts": _count_semantic_kinds(semantic),
        "num_semantic_records": len(semantic),
        "concrete_counts": _count_concrete_opcodes(concrete),
        "num_concrete_records": len(concrete),
        "estimated_pim_compute_requests": provenance.get(
            "estimated_pim_compute_requests"
        ),
        "record_counts_by_kind": provenance.get("record_counts_by_kind", {}),
    }


# ─── Paper-scale data collection (realistic model configurations) ─────


def _manifest_for(manifest_arg, tiny_fn, paper_configs, default_key):
    """Resolve a manifest argument: None→paper default, 'tiny'→tiny, str→paper key."""
    if manifest_arg is None:
        if paper_configs and default_key in paper_configs:
            return paper_configs[default_key]
        return tiny_fn()
    if manifest_arg == "tiny":
        return tiny_fn()
    if isinstance(manifest_arg, dict):
        return manifest_arg
    if isinstance(manifest_arg, str) and manifest_arg in paper_configs:
        return paper_configs[manifest_arg]
    raise ValueError(
        f"Unknown manifest: {manifest_arg!r}. "
        f"Use None for paper default, 'tiny' for test-scale, "
        f"or one of {list(paper_configs.keys())}"
    )


def collect_attention_paper(
    manifest: dict | str | None = None,
) -> dict[str, Any]:
    """Generate attention decomposition using a paper-scale manifest."""
    manifest = _manifest_for(manifest, get_tiny_attention_manifest,
                             PAPER_ATTENTION_CONFIGS, DEFAULT_PAPER_ATTENTION)
    return collect_attention_decomposition(manifest)


def collect_attention_sweep_paper(
    num_heads_list: list[int] | None = None,
    past_len_list: list[int] | None = None,
    base_config: str | dict = "llama_7b",
) -> list[dict[str, Any]]:
    """Run parameter sweep using LLaMA-style base configuration."""
    if num_heads_list is None:
        num_heads_list = [8, 16, 24, 32]
    if past_len_list is None:
        past_len_list = [128, 256, 512, 1024]

    results = []
    for num_heads in num_heads_list:
        for past_len in past_len_list:
            if isinstance(base_config, str):
                manifest = dict(PAPER_ATTENTION_CONFIGS[base_config])
            else:
                manifest = dict(base_config)
            manifest["num_heads"] = num_heads
            manifest["past_len"] = past_len
            tile_tokens = min(manifest["score_tile_tokens"], past_len)
            manifest["score_tile_tokens"] = tile_tokens
            manifest["context_tile_tokens"] = tile_tokens

            result = collect_attention_decomposition(manifest)
            result["_num_heads"] = num_heads
            result["_past_len"] = past_len
            results.append(result)
    return results


def collect_ffn_paper(
    manifest: dict | str | None = None,
) -> dict[str, Any]:
    """Generate FFN decomposition using a paper-scale manifest."""
    manifest = _manifest_for(manifest, get_tiny_ffn_manifest,
                             PAPER_FFN_CONFIGS, DEFAULT_PAPER_FFN)
    return collect_ffn_decomposition(manifest)


def collect_moe_paper(
    manifest: dict | str | None = None,
) -> dict[str, Any]:
    """Generate MoE decomposition using a paper-scale manifest."""
    manifest = _manifest_for(manifest, get_tiny_moe_manifest,
                             PAPER_MOE_CONFIGS, DEFAULT_PAPER_MOE)
    return collect_moe_decomposition(manifest)


# ─── Replay stats ─────────────────────────────────────────────────────


def _collect_replay_for_trace(
    trace_name: str,
    semantic: list[dict],
    manifest_name: str,
) -> dict[str, Any]:
    """Run one concrete trace through the LPDDR5-PIM simulator and extract stats.

    Uses the same replay pattern as test_full_transformer_generator.py.
    """
    import ramulator

    from tests.analysis.testcases.lpddr5_pim import CONFIG as LPDDR5_PIM_CONFIG
    from tests.utils.dram import create_dram
    from tests.analysis.figures._sim_helpers import _make_mem, _frontend
    from ramulator.workload_surrogate.lpddr5_pim_concrete_trace import write_jsonl

    concrete = lower_semantic_records_to_concrete(
        semantic, manifest_name=manifest_name
    )
    command_counts = _count_concrete_opcodes(concrete)

    with tempfile.TemporaryDirectory() as tmpdir:
        trace_path = Path(tmpdir) / f"{trace_name}_trace.jsonl"
        write_jsonl(concrete, trace_path)

        dram = create_dram(LPDDR5_PIM_CONFIG)
        sim = ramulator.Simulation(_frontend(trace_path, dram), _make_mem(dram))
        sim.run()

        frontend_stats = dict(sim.stats["frontend"])
        ctrl = sim.stats["memory_system"]["controller"]

        cycles = int(ctrl.get("cycles", 0))
        # LPDDR5_6400: tCK = 0.625 ns (1600 MHz clock)
        tCK_ns = 0.625
        runtime_ns = cycles * tCK_ns

        return {
            "trace_name": trace_name,
            "semantic_records": len(semantic),
            "concrete_records": len(concrete),
            "replay_status": (
                "PASS"
                if frontend_stats.get("opcode_requests_completed")
                == frontend_stats.get("opcode_requests_sent")
                else "FAIL"
            ),
            "opcodes_sent": frontend_stats.get("opcode_requests_sent", 0),
            "opcodes_completed": frontend_stats.get("opcode_requests_completed", 0),
            "pim_mac_issued": ctrl.get("num_pim_reqs_served", 0),
            "pim_bcast_issued": command_counts.get("PIM_BCAST", 0),
            "command_counts": command_counts,
            "cycles": cycles,
            "runtime_ns": runtime_ns,
        }


def collect_replay_stats() -> list[dict[str, Any]]:
    """Collect replay stats for all trace types."""
    results = []

    # Attention — serialized
    attn_manifest = get_tiny_attention_manifest()
    attn_manifest["schedule_policy"] = "serialized"
    attn_manifest["past_len"] = 32  # keep replay fast
    attn_semantic = generate_attention_records(attn_manifest)
    results.append(
        _collect_replay_for_trace("attention_serialized", attn_semantic, "attn_serialized")
    )

    # Attention — overlapped (requires num_heads >= 2)
    attn_overlap = dict(attn_manifest)
    attn_overlap["schedule_policy"] = "overlap_independent_heads"
    attn_overlap["num_heads"] = 2
    attn_overlap_semantic = generate_attention_records(attn_overlap)
    results.append(
        _collect_replay_for_trace(
            "attention_overlapped", attn_overlap_semantic, "attn_overlapped"
        )
    )

    # FFN/SwiGLU
    ffn_manifest = get_tiny_ffn_manifest()
    ffn_semantic = generate_ffn_records(ffn_manifest)
    results.append(
        _collect_replay_for_trace("ffn_swiglu", ffn_semantic, "ffn_swiglu")
    )

    # MoE
    moe_manifest = get_tiny_moe_manifest()
    moe_semantic = generate_moe_records(moe_manifest)
    results.append(
        _collect_replay_for_trace("moe_top2", moe_semantic, "moe_top2")
    )

    # Combined layer
    combined_semantic = generate_full_transformer_layer_records()
    results.append(
        _collect_replay_for_trace(
            "combined_layer", combined_semantic, "combined_layer"
        )
    )

    return results


# ─── Llama2 dense decoder data ─────────────────────────────────────────


def _collect_llama2_dense_decoder_data(
    model_key: str,
    manifest_name: str,
    replay_stats_fn,
    *,
    past_len: int = 1024,
) -> dict[str, Any]:
    from ramulator.workload_surrogate.generate_full_transformer import (
        generate_llama2_7b_dense_decoder_records,
        generate_llama2_13b_dense_decoder_records,
        get_llama2_7b_dense_decoder_manifests,
        get_llama2_13b_dense_decoder_manifests,
    )

    spec = get_model_spec(model_key)
    if past_len <= 0:
        raise ValueError("past_len must be positive")
    if model_key == "llama2-13b":
        attention_manifest, ffn_manifest = get_llama2_13b_dense_decoder_manifests(past_len=past_len)
        semantic = generate_llama2_13b_dense_decoder_records(
            attention_manifest=attention_manifest,
            ffn_manifest=ffn_manifest,
        )
    else:
        attention_manifest, ffn_manifest = get_llama2_7b_dense_decoder_manifests(past_len=past_len)
        semantic = generate_llama2_7b_dense_decoder_records(
            attention_manifest=attention_manifest,
            ffn_manifest=ffn_manifest,
        )
    concrete = lower_semantic_records_to_concrete(
        semantic, manifest_name=manifest_name
    )
    concrete_counts = _count_concrete_opcodes(concrete)
    per_layer = {k: v // spec.num_layers for k, v in concrete_counts.items()} if concrete_counts else {}

    datatype = spec.datatype
    datatype_resources = PIM_DATATYPE_RESOURCES[datatype]
    pim_mac_lanes = int(datatype_resources["pim_lanes"])
    primitive_ops_per_mac = int(datatype_resources["pim_ops_per_mac"])
    qkvo_projection_per_layer = _ceil_div(spec.hidden_size * spec.hidden_size, pim_mac_lanes) * 4
    attention_per_layer = _ceil_div(past_len * spec.num_heads * spec.head_dim, pim_mac_lanes) * 2
    ffn_per_layer = _ceil_div(spec.hidden_size * spec.ffn_hidden_size, pim_mac_lanes) * 3
    expected_pim_mac_repeats = spec.num_layers * (qkvo_projection_per_layer + attention_per_layer + ffn_per_layer)
    actual_pim_mac_repeats = concrete_counts.get("PIM_MAC", 0)
    if actual_pim_mac_repeats != expected_pim_mac_repeats:
        raise ValueError(
            f"{spec.name} dense-decoder PIM_MAC repeats diverged from the formula-derived expectation: "
            f"actual={actual_pim_mac_repeats}, expected={expected_pim_mac_repeats}"
        )

    return {
        "manifest_name": manifest_name,
        "model_name": spec.name,
        "num_layers": spec.num_layers,
        "hidden_size": spec.hidden_size,
        "ffn_hidden_size": spec.ffn_hidden_size,
        "num_heads": spec.num_heads,
        "head_dim": spec.head_dim,
        "past_len": past_len,
        "datatype": datatype,
        "pim_mac_lanes": pim_mac_lanes,
        "primitive_ops_per_mac": primitive_ops_per_mac,
        "attention_scope": "Decode-block v2 includes Q/K/V/O projections plus QK^T + PV attention core; excludes RoPE, norm, residual, and softmax hardware cost",
        "qkvo_projection_pim_mac_per_layer": qkvo_projection_per_layer,
        "attention_pim_mac_per_layer": attention_per_layer,
        "ffn_pim_mac_per_layer": ffn_per_layer,
        "expected_pim_mac_repeats": expected_pim_mac_repeats,
        "semantic_counts": _count_semantic_kinds(semantic),
        "num_semantic_records": len(semantic),
        "concrete_counts": concrete_counts,
        "concrete_per_layer": per_layer,
        "num_concrete_records": len(concrete),
        "total_pim_mac_repeats": actual_pim_mac_repeats,
        "total_pim_bcast_repeats": concrete_counts.get("PIM_BCAST", 0),
        "backend_replay_stats": replay_stats_fn(),
    }


def _backend_replay_row(trace_name: str, stats: dict[str, Any]) -> dict[str, Any]:
    """Convert backend replay stats to replay-validation row format."""
    frontend_stats = stats.get("frontend_stats", {})
    return {
        "trace_name": trace_name,
        "semantic_records": stats.get("semantic_records", 0),
        "concrete_records": stats.get("concrete_records", 0),
        "replay_status": "PASS" if stats.get("replay_ok", False) else "FAIL",
        "opcodes_sent": frontend_stats.get("opcode_requests_sent", 0),
        "opcodes_completed": frontend_stats.get("opcode_requests_completed", 0),
        "pim_mac_issued": stats.get("pim_mac_issued", 0),
        "pim_bcast_issued": stats.get("command_counts", {}).get("PIM_BCAST", 0),
        "cycles": stats.get("cycles", 0),
        "runtime_ns": stats.get("runtime_ns", 0),
        "command_counts": stats.get("command_counts", {}),
        "data_source": "real_backend_simulation",
    }


@lru_cache(maxsize=1)
def _collect_llama2_7b_backend_stats_cached() -> dict[str, dict[str, Any]]:
    """Run and cache full 32-layer Llama2-7B backend replay stats in-process."""
    from tests.analysis.figures.p4_backend_data import (
        collect_all_backend_stats_llama2_7b,
    )

    return collect_all_backend_stats_llama2_7b()


def collect_llama2_7b_replay_stats() -> list[dict[str, Any]]:
    """Collect backend replay stats for 32-layer Llama2-7B dense decoder."""
    backend_stats = _collect_llama2_7b_backend_stats_cached()
    trace_names = [
        "llama2_7b_32_layer_steady_state",
        "llama2_7b_32_layer_cold_start",
    ]
    return [
        _backend_replay_row(trace_name, backend_stats[trace_name])
        for trace_name in trace_names
    ]


@lru_cache(maxsize=1)
def _collect_llama2_13b_backend_stats_cached() -> dict[str, dict[str, Any]]:
    """Run and cache full 40-layer Llama2-13B backend replay stats in-process."""
    from tests.analysis.figures.p4_backend_data import (
        collect_all_backend_stats_llama2_13b,
    )

    return collect_all_backend_stats_llama2_13b()


def collect_llama2_13b_replay_stats() -> list[dict[str, Any]]:
    """Collect backend replay stats for 40-layer Llama2-13B dense decoder."""
    backend_stats = _collect_llama2_13b_backend_stats_cached()
    trace_names = [
        "llama2_13b_40_layer_steady_state",
        "llama2_13b_40_layer_cold_start",
    ]
    return [
        _backend_replay_row(trace_name, backend_stats[trace_name])
        for trace_name in trace_names
    ]


def collect_llama2_7b_dense_decoder_data(
    *,
    past_len: int = 1024,
    replay_stats_fn=None,
) -> dict[str, Any]:
    """Generate full 32-layer Llama2-7B dense decoder semantic + concrete data."""
    if replay_stats_fn is None:
        replay_stats_fn = collect_llama2_7b_replay_stats
    return _collect_llama2_dense_decoder_data(
        "llama2-7b",
        f"llama2_7b_32_layer_past_len_{past_len}_dense_decoder",
        replay_stats_fn,
        past_len=past_len,
    )


def collect_llama2_13b_dense_decoder_data(
    *,
    past_len: int = 1024,
    replay_stats_fn=None,
) -> dict[str, Any]:
    """Generate full 40-layer Llama2-13B dense decoder semantic + concrete data."""
    if replay_stats_fn is None:
        replay_stats_fn = collect_llama2_13b_replay_stats
    return _collect_llama2_dense_decoder_data(
        "llama2-13b",
        f"llama2_13b_40_layer_past_len_{past_len}_dense_decoder",
        replay_stats_fn,
        past_len=past_len,
    )


def collect_llama2_7b_decode_context_length_sweep(
    past_len_values: list[int] | None = None,
    modes: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Collect backend-backed Llama2-7B decode context-length sweep data."""
    from ramulator.workload_surrogate.generate_full_transformer import (
        FULL_TRANSFORMER_GENERATOR_VERSION,
        get_llama2_7b_dense_decoder_manifests,
    )
    from tests.analysis.figures.p4_backend_data import (
        collect_all_backend_stats_llama2_7b_past_len,
    )

    if past_len_values is None:
        past_len_values = [128, 256, 512, 1024]
    if modes is None:
        modes = ("steady_state",)

    rows: list[dict[str, Any]] = []
    for past_len in past_len_values:
        attention_manifest, _ = get_llama2_7b_dense_decoder_manifests(past_len=past_len)
        data = collect_llama2_7b_dense_decoder_data(past_len=past_len, replay_stats_fn=lambda: [])
        backend_stats = collect_all_backend_stats_llama2_7b_past_len(
            past_len,
            modes=modes,
        )
        for mode in modes:
            materialize_weights = mode == "cold_start"
            trace_name = f"llama2_7b_32_layer_past_len_{past_len}_{mode}"
            stats = backend_stats[trace_name]
            cycles = int(stats.get("cycles", 0) or 0)
            concrete_counts = dict(stats.get("command_counts", {}))
            pim_mac = int(concrete_counts.get("PIM_MAC", 0))
            rows.append(
                {
                    "model": "Llama2-7B",
                    "phase": "decode",
                    "seq_len": 1,
                    "past_len": int(past_len),
                    "num_layers": int(data["num_layers"]),
                    "datatype": data["datatype"],
                    "mode": mode,
                    "materialize_weights": materialize_weights,
                    "status": "PASS" if stats.get("replay_ok", False) else "FAIL",
                    "runtime_ns": stats.get("runtime_ns"),
                    "cycles": cycles,
                    "pim_mac": pim_mac,
                    "pim_bcast": int(concrete_counts.get("PIM_BCAST", 0)),
                    "pim_mac_density": (pim_mac / cycles) if cycles > 0 else None,
                    "controller_pim_mac_issued": stats.get("pim_mac_issued", 0),
                    "avg_pim_latency_cycles": stats.get("avg_pim_latency_cycles", 0),
                    "per_layer_pim_mac_buckets": {
                        "qkvo_projection": data["qkvo_projection_pim_mac_per_layer"],
                        "attention": data["attention_pim_mac_per_layer"],
                        "ffn": data["ffn_pim_mac_per_layer"],
                    },
                    "concrete_opcode_counts_replay_input": concrete_counts,
                }
            )

    return {
        "schema_version": 1,
        "figure_id": "fig11_llama2_7b_decode_context_length_sweep",
        "description": "Decode-only context-length backend sweep for bounded surrogate replay",
        "phase": "decode",
        "seq_len": 1,
        "model": "Llama2-7B",
        "provenance": {
            "generator_version": FULL_TRANSFORMER_GENERATOR_VERSION,
            "replay_mode": "backend",
        },
        "sweep": {
            "past_len_values": [int(value) for value in past_len_values],
            "score_tile_tokens": int(attention_manifest.get("score_tile_tokens", 256)),
            "context_tile_tokens": int(attention_manifest.get("context_tile_tokens", 256)),
            "effective_tile_policy": "min(tile_tokens, past_len)",
        },
        "rows": rows,
        "caveats": [
            "decode_only_seq_len_1",
            "bounded_surrogate_not_serving",
            "non_silicon_calibrated",
        ],
    }


def collect_llama2_7b_generated_token_sweep(
    *,
    initial_past_len: int = 1024,
    num_generated_tokens: int = 4,
    modes: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Collect per-token backend replays for growing decode context length.

    Each row is an independent single-token replay at
    ``past_len = initial_past_len + generated_token_index``. This does not model
    one continuous multi-token backend trace.
    """
    from ramulator.workload_surrogate.generate_full_transformer import (
        FULL_TRANSFORMER_GENERATOR_VERSION,
    )
    from tests.analysis.figures.p4_backend_data import (
        collect_all_backend_stats_llama2_7b_past_len,
    )

    if initial_past_len <= 0:
        raise ValueError("initial_past_len must be positive")
    if num_generated_tokens <= 0:
        raise ValueError("num_generated_tokens must be positive")
    if modes is None:
        modes = ("steady_state",)

    cumulative_runtime_by_mode = {mode: 0.0 for mode in modes}
    rows: list[dict[str, Any]] = []
    for token_index in range(num_generated_tokens):
        past_len = initial_past_len + token_index
        data = collect_llama2_7b_dense_decoder_data(past_len=past_len, replay_stats_fn=lambda: [])
        backend_stats = collect_all_backend_stats_llama2_7b_past_len(
            past_len,
            modes=modes,
        )
        for mode in modes:
            trace_name = f"llama2_7b_32_layer_past_len_{past_len}_{mode}"
            stats = backend_stats[trace_name]
            cycles = int(stats.get("cycles", 0) or 0)
            runtime_ns = float(stats.get("runtime_ns", 0) or 0)
            cumulative_runtime_by_mode[mode] += runtime_ns
            concrete_counts = dict(stats.get("command_counts", {}))
            pim_mac = int(concrete_counts.get("PIM_MAC", 0))
            rows.append(
                {
                    "model": "Llama2-7B",
                    "phase": "decode",
                    "seq_len_per_step": 1,
                    "initial_past_len": int(initial_past_len),
                    "generated_token_index": int(token_index),
                    "generated_tokens_total": int(token_index + 1),
                    "past_len": int(past_len),
                    "num_layers": int(data["num_layers"]),
                    "datatype": data["datatype"],
                    "mode": mode,
                    "materialize_weights": mode == "cold_start",
                    "status": "PASS" if stats.get("replay_ok", False) else "FAIL",
                    "runtime_ns": runtime_ns,
                    "cumulative_runtime_ns": cumulative_runtime_by_mode[mode],
                    "cycles": cycles,
                    "pim_mac": pim_mac,
                    "pim_bcast": int(concrete_counts.get("PIM_BCAST", 0)),
                    "pim_mac_density": (pim_mac / cycles) if cycles > 0 else None,
                    "controller_pim_mac_issued": stats.get("pim_mac_issued", 0),
                    "avg_pim_latency_cycles": stats.get("avg_pim_latency_cycles", 0),
                    "per_layer_pim_mac_buckets": {
                        "qkvo_projection": data["qkvo_projection_pim_mac_per_layer"],
                        "attention": data["attention_pim_mac_per_layer"],
                        "ffn": data["ffn_pim_mac_per_layer"],
                    },
                    "concrete_opcode_counts_replay_input": concrete_counts,
                }
            )

    return {
        "schema_version": 1,
        "figure_id": "fig12_llama2_7b_generated_token_sweep",
        "description": "Decode-only generated-token sweep using independent single-token backend replays",
        "phase": "decode",
        "seq_len_per_step": 1,
        "initial_past_len": int(initial_past_len),
        "num_generated_tokens": int(num_generated_tokens),
        "model": "Llama2-7B",
        "provenance": {
            "generator_version": FULL_TRANSFORMER_GENERATOR_VERSION,
            "replay_mode": "backend_independent_single_token_replays",
        },
        "rows": rows,
        "caveats": [
            "independent_single_token_replays",
            "decode_only_seq_len_1",
            "not_one_continuous_multitoken_trace",
            "bounded_surrogate_not_serving",
            "non_silicon_calibrated",
        ],
    }


# ─── Mixtral-8x7B MoE decoder data ──────────────────────────────────────


def collect_mixtral_8x7b_moe_decoder_data(
    *,
    past_len: int = 1024,
    replay_stats_fn=None,
) -> dict[str, Any]:
    """Generate full 32-layer Mixtral-8x7B MoE decoder semantic + concrete data."""
    from ramulator.workload_surrogate.generate_full_transformer import (
        generate_mixtral_8x7b_decoder_records,
        get_mixtral_8x7b_moe_decoder_manifests,
    )

    attention_manifest, moe_manifest = get_mixtral_8x7b_moe_decoder_manifests(past_len=past_len)
    semantic = generate_mixtral_8x7b_decoder_records(
        attention_manifest=attention_manifest, moe_manifest=moe_manifest
    )
    manifest_name = "mixtral_8x7b_32_layer_moe_decoder"
    concrete = lower_semantic_records_to_concrete(semantic, manifest_name=manifest_name)
    concrete_counts = _count_concrete_opcodes(concrete)

    num_layers = int(moe_manifest["num_layers"])
    hidden = int(moe_manifest["hidden_size"])
    expert_hidden = int(moe_manifest["expert_hidden_size"])
    num_experts = int(moe_manifest["num_experts"])
    top_k = int(moe_manifest["top_k"])
    real_dims = moe_manifest.get("real_model_dimensions", {})
    real_hidden = int(real_dims.get("hidden_size", hidden)) if isinstance(real_dims, dict) else hidden
    real_expert_hidden = int(real_dims.get("expert_hidden_size", expert_hidden)) if isinstance(real_dims, dict) else expert_hidden
    num_heads = int(attention_manifest["num_heads"])
    num_kv_heads = int(attention_manifest["num_kv_heads"])
    head_dim = int(attention_manifest["head_dim"])
    datatype = moe_manifest["datatype"]
    datatype_resources = PIM_DATATYPE_RESOURCES[datatype]
    lanes = int(datatype_resources["pim_lanes"])

    q_proj = _ceil_div(hidden * hidden, lanes)
    kv_output_dim = num_kv_heads * head_dim
    k_proj = _ceil_div(hidden * kv_output_dim, lanes)
    v_proj = _ceil_div(hidden * kv_output_dim, lanes)
    o_proj = _ceil_div(hidden * hidden, lanes)
    qkvo_per_layer = q_proj + k_proj + v_proj + o_proj

    attention_per_layer = _ceil_div(past_len * num_heads * head_dim, lanes) * 2

    router_per_layer = _ceil_div(1 * num_experts * hidden, lanes)

    expert_fused_per_layer = _moe_expert_pim_mac_per_layer(
        hidden_size=hidden,
        expert_hidden_size=expert_hidden,
        top_k=top_k,
        lanes=lanes,
        projections=1,
    )
    expert_real_per_layer = _moe_expert_pim_mac_per_layer(
        hidden_size=hidden,
        expert_hidden_size=expert_hidden,
        top_k=top_k,
        lanes=lanes,
        projections=3,
    )
    expert_full_dim_real_per_layer = _moe_expert_pim_mac_per_layer(
        hidden_size=real_hidden,
        expert_hidden_size=real_expert_hidden,
        top_k=top_k,
        lanes=lanes,
        projections=3,
    )

    expected_pim_mac = num_layers * (qkvo_per_layer + attention_per_layer + router_per_layer + expert_fused_per_layer)
    actual_pim_mac = concrete_counts.get("PIM_MAC", 0)
    if actual_pim_mac != expected_pim_mac:
        raise ValueError(
            "Mixtral-8x7B PIM_MAC accounting mismatch: "
            f"actual={actual_pim_mac}, expected={expected_pim_mac}"
        )

    per_layer = {k: v // num_layers for k, v in concrete_counts.items()} if concrete_counts else {}

    result: dict[str, Any] = {
        "manifest_name": manifest_name,
        "model_name": "Mixtral-8x7B (scaled)",
        "num_layers": num_layers,
        "hidden_size": hidden,
        "expert_hidden_size": expert_hidden,
        "real_hidden_size": real_hidden,
        "real_expert_hidden_size": real_expert_hidden,
        "num_experts": num_experts,
        "top_k": top_k,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "past_len": past_len,
        "datatype": datatype,
        "pim_mac_lanes": lanes,
        "attention_scope": "GQA no-reuse (4:1 Q:KV ratio); Mixtral-style MoE decode surrogate",
        "qkvo_projection_pim_mac_per_layer": qkvo_per_layer,
        "attention_pim_mac_per_layer": attention_per_layer,
        "router_pim_mac_per_layer": router_per_layer,
        "expert_ffn_pim_mac_fused_per_layer": expert_fused_per_layer,
        "expert_ffn_pim_mac_real_per_layer": expert_real_per_layer,
        "expert_ffn_pim_mac_full_dim_real_per_layer": expert_full_dim_real_per_layer,
        "expected_pim_mac_repeats": expected_pim_mac,
        "semantic_counts": _count_semantic_kinds(semantic),
        "num_semantic_records": len(semantic),
        "concrete_counts": concrete_counts,
        "concrete_per_layer": per_layer,
        "num_concrete_records": len(concrete),
        "total_pim_mac_repeats": actual_pim_mac,
        "total_pim_bcast_repeats": concrete_counts.get("PIM_BCAST", 0),
    }
    if replay_stats_fn is not None:
        result["backend_replay_stats"] = replay_stats_fn()
    return result


def collect_mixtral_8x7b_replay_stats() -> list[dict[str, Any]]:
    """Collect backend replay stats for 32-layer Mixtral-8x7B MoE decoder."""
    from tests.analysis.figures.p4_backend_data import (
        collect_all_backend_stats_mixtral_8x7b,
    )

    backend_stats = collect_all_backend_stats_mixtral_8x7b()
    trace_names = [
        "mixtral_8x7b_32_layer_steady_state",
        "mixtral_8x7b_32_layer_cold_start",
    ]
    return [
        _backend_replay_row(trace_name, backend_stats[trace_name])
        for trace_name in trace_names
    ]
