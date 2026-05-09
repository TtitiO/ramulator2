"""Data collection helpers for P4 full-transformer paper figures.

Collects command counts, replay stats, and parameter sensitivity data
from the P4 semantic generators and concrete lowering pipeline.
"""

from __future__ import annotations

import json
import tempfile
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
)
from ramulator.workload_surrogate.generate_lpddr5_pim_concrete import (
    lower_semantic_records_to_concrete,
)
from ramulator.workload_surrogate.lpddr5_pim_concrete_trace import (
    CONCRETE_OPCODES,
    MODE_OPCODES,
    REQUEST_OPCODES,
)

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


def _count_semantic_kinds(semantic_records: list[dict]) -> dict[str, int]:
    """Count semantic record kinds (AttentionScore, FFNProjection, etc.)."""
    counts: dict[str, int] = {}
    for rec in semantic_records:
        kind = rec["kind"]
        counts[kind] = counts.get(kind, 0) + 1
    return counts


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

    return {
        "manifest_name": manifest.get("manifest_name", "tiny_attention"),
        "num_heads": int(manifest["num_heads"]),
        "head_dim": int(manifest["head_dim"]),
        "past_len": int(manifest["past_len"]),
        "datatype": manifest["datatype"],
        "schedule_policy": manifest.get("schedule_policy", "serialized"),
        "semantic_counts": _count_semantic_kinds(semantic),
        "num_semantic_records": len(semantic),
        "concrete_counts": _count_concrete_opcodes(concrete),
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

    return {
        "manifest_name": manifest.get("manifest_name", "tiny_ffn"),
        "hidden_size": int(manifest["hidden_size"]),
        "ffn_hidden_size": int(manifest["ffn_hidden_size"]),
        "datatype": manifest["datatype"],
        "activation": manifest.get("activation", "silu"),
        "semantic_counts": _count_semantic_kinds(semantic),
        "num_semantic_records": len(semantic),
        "concrete_counts": _count_concrete_opcodes(concrete),
        "num_concrete_records": len(concrete),
    }


# ─── MoE data ──────────────────────────────────────────────────────────


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
        "concrete_counts": _count_concrete_opcodes(concrete),
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
    base_config: str | dict = "llama_style",
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
