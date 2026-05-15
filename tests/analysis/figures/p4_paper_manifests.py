"""Realistic LLM model manifests for paper evaluation.

Provides production-grade model configurations drawn from published
LLM architectures. Every manifest records `model_total_layers` (the
full model depth) alongside `num_layers=1` (single-layer simulation).
Tiny manifests remain available via `get_tiny_*_manifest()` for unit testing only.

Published source dimensions:
  OPT-125M:     12 layers, 12 attn heads, hidden=768,  FFN=3072,  head_dim=64
  OPT-350M:     24 layers, 16 attn heads, hidden=1024, FFN=4096,  head_dim=64
  OPT-1.3B:     24 layers, 32 attn heads, hidden=2048, FFN=8192,  head_dim=64
  LLaMA-7B:     32 layers, 32 attn heads, hidden=4096, FFN=11008, head_dim=128
  LLaMA-13B:    40 layers, 40 attn heads, hidden=5120, FFN=13824, head_dim=128
  Mixtral-8x7B: 32 layers, 32 Q / 8 KV heads, hidden=4096,
                expert FFN=14336, 8 experts, top-2, head_dim=128
"""

from __future__ import annotations

import copy
from typing import Any


# ─── Shared Ramulator-visible defaults ────────────────────────────────

_RAMULATOR_DEFAULTS: dict[str, Any] = {
    "bank_sequence": [0, 1, 2, 3],
    "bank_sequence_order": "frontend",
    "pim_banks_per_mpu": 1,
    "burst_length": 1,
    "row_start": 0,
    "row_count": 16,
    "dependency_count": 8,
    "column_start": 0,
}

_MAPPING_POLICY: dict[str, str] = {
    "host_policy": "semantic_tensor_io_only",
    "pim_policy": "native_lpddr5_pim_operator_tiles",
    "bank_sequence_policy": "manifest_order",
    "mpu_grouping_policy": "manifest_pim_banks_per_mpu",
}

_COMMON_NON_CLAIMS: list[str] = [
    "not_runtime_replay",
    "not_vllm_replay",
    "not_numerical_correctness",
    "not_silicon_faithful_softmax_or_data_movement",
    "not_raw_attacc_schema",
]


# ═══════════════════════════════════════════════════════════════════════
# Attention manifests  (single-layer decode, published head/dim counts)
# ═══════════════════════════════════════════════════════════════════════

def _attn_base(
    model_family: str,
    model_citation: str,
    model_total_layers: int,
    num_heads: int,
    head_dim: int,
    hidden_size: int,
    *,
    manifest_name: str,
    kv_heads: int | None = None,
    past_len: int = 512,
    schedule_policy: str = "serialized",
    literature_anchors: list[str] | None = None,
) -> dict[str, Any]:
    """Build an attention manifest dict from published dimensions."""
    tile_tokens = min(256, past_len)
    out: dict[str, Any] = {
        "manifest_version": "paper-attention-v1.1",
        "manifest_name": manifest_name,
        "provenance_class": "representative-model-scale",
        "model_citation": model_citation,
        "model_total_layers": model_total_layers,
        "data_movement_assumption": "semantic_kv_cache_host_read_per_tile",
        "data_movement_note": (
            "Decode-block v2 models ordinary K/V-cache tile access as HostRead semantic accounting, "
            "not PIMDataMove/PIM_BCAST. PIM_BCAST is reserved for true PIM-local setup/broadcast."
        ),
        "workload_class": "structured_transformer_attention_surrogate",
        "phase": "decode",
        "model_family": model_family,
        "num_layers": 1,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "hidden_size": hidden_size,
        "past_len": past_len,
        "seq_len": 1,
        "datatype": "int8",
        "score_tile_tokens": tile_tokens,
        "context_tile_tokens": tile_tokens,
        "head_group_size": 1,
        "schedule_policy": schedule_policy,
        "ramulator_visible_defaults": dict(_RAMULATOR_DEFAULTS),
        "mapping_policy": dict(_MAPPING_POLICY),
        "literature_anchors": literature_anchors or [
            model_citation.split(":")[0].strip(),
            "LPDDR5-PIM native opcode surface",
        ],
        "non_claims": list(_COMMON_NON_CLAIMS),
    }
    if kv_heads is not None:
        out["kv_heads"] = kv_heads
        out["attention_variant"] = "grouped_query_attention"
    return out


def get_opt_125m_attention_manifest(
    *, past_len: int = 512, schedule_policy: str = "serialized"
) -> dict[str, Any]:
    """OPT-125M: 12 layers, 12 heads × 64-dim, hidden=768."""
    return _attn_base(
        "OPT-125M decoder-only transformer attention slice",
        "OPT (Zhang et al., arXiv:2205.01068)",
        model_total_layers=12,
        num_heads=12,
        head_dim=64,
        hidden_size=768,
        manifest_name="opt_125m_attention_decode",
        past_len=past_len,
        schedule_policy=schedule_policy,
    )


def get_opt_350m_attention_manifest(
    *, past_len: int = 512, schedule_policy: str = "serialized"
) -> dict[str, Any]:
    """OPT-350M: 24 layers, 16 heads × 64-dim, hidden=1024."""
    return _attn_base(
        "OPT-350M decoder-only transformer attention slice",
        "OPT (Zhang et al., arXiv:2205.01068)",
        model_total_layers=24,
        num_heads=16,
        head_dim=64,
        hidden_size=1024,
        manifest_name="opt_350m_attention_decode",
        past_len=past_len,
        schedule_policy=schedule_policy,
    )


def get_opt_1_3b_attention_manifest(
    *, past_len: int = 512, schedule_policy: str = "serialized"
) -> dict[str, Any]:
    """OPT-1.3B: 24 layers, 32 heads × 64-dim, hidden=2048."""
    return _attn_base(
        "OPT-1.3B decoder-only transformer attention slice",
        "OPT (Zhang et al., arXiv:2205.01068)",
        model_total_layers=24,
        num_heads=32,
        head_dim=64,
        hidden_size=2048,
        manifest_name="opt_1_3b_attention_decode",
        past_len=past_len,
        schedule_policy=schedule_policy,
    )


def get_llama_7b_attention_manifest(
    *, past_len: int = 1024, schedule_policy: str = "serialized"
) -> dict[str, Any]:
    """LLaMA-7B: 32 layers, 32 heads × 128-dim, hidden=4096."""
    return _attn_base(
        "LLaMA-7B decoder-only transformer attention slice",
        "LLaMA (Touvron et al., arXiv:2302.13971)",
        model_total_layers=32,
        num_heads=32,
        head_dim=128,
        hidden_size=4096,
        manifest_name="llama_7b_attention_decode",
        past_len=past_len,
        schedule_policy=schedule_policy,
    )


def get_llama_13b_attention_manifest(
    *, past_len: int = 1024, schedule_policy: str = "serialized"
) -> dict[str, Any]:
    """LLaMA-13B: 40 layers, 40 heads × 128-dim, hidden=5120."""
    return _attn_base(
        "LLaMA-13B decoder-only transformer attention slice",
        "LLaMA (Touvron et al., arXiv:2302.13971)",
        model_total_layers=40,
        num_heads=40,
        head_dim=128,
        hidden_size=5120,
        manifest_name="llama_13b_attention_decode",
        past_len=past_len,
        schedule_policy=schedule_policy,
    )


def get_mixtral_attention_manifest(
    *, past_len: int = 1024, schedule_policy: str = "serialized"
) -> dict[str, Any]:
    """Mixtral-8x7B attention: 32 layers, 32 Q heads / 8 KV heads (GQA), head_dim=128, hidden=4096."""
    return _attn_base(
        "Mixtral-8x7B decoder-only transformer attention slice (GQA)",
        "Mixtral of Experts (Jiang et al., arXiv:2401.04088)",
        model_total_layers=32,
        num_heads=32,
        head_dim=128,
        hidden_size=4096,
        kv_heads=8,
        manifest_name="mixtral_8x7b_attention_decode",
        past_len=past_len,
        schedule_policy=schedule_policy,
        literature_anchors=[
            "Mixtral-8x7B (Jiang et al., 2024)",
            "LPDDR5-PIM native opcode surface",
        ],
    )


# Backward-compatible llama_style alias (LLaMA-7B defaults)
def get_llama_style_attention_manifest(
    *,
    num_heads: int = 32,
    head_dim: int = 128,
    hidden_size: int = 4096,
    past_len: int = 1024,
    schedule_policy: str = "serialized",
) -> dict[str, Any]:
    return get_llama_7b_attention_manifest(past_len=past_len, schedule_policy=schedule_policy)


# ═══════════════════════════════════════════════════════════════════════
# FFN manifests  (single-layer decode, published hidden/FFN dimensions)
# ═══════════════════════════════════════════════════════════════════════

def _ffn_base(
    model_family: str,
    model_citation: str,
    model_total_layers: int,
    hidden_size: int,
    ffn_hidden_size: int,
    *,
    manifest_name: str,
    schedule_policy: str = "serialized",
) -> dict[str, Any]:
    return {
        "manifest_version": "paper-ffn-v1.1",
        "manifest_name": manifest_name,
        "provenance_class": "representative-model-scale",
        "model_citation": model_citation,
        "model_total_layers": model_total_layers,
        "workload_class": "structured_transformer_ffn_swiglu_surrogate",
        "phase": "decode",
        "model_family": model_family,
        "num_layers": 1,
        "seq_len": 1,
        "hidden_size": hidden_size,
        "ffn_hidden_size": ffn_hidden_size,
        "ffn_activation_tile_size": hidden_size,
        "activation_distribution_policy": "broadcast",
        "activation": "silu",
        "datatype": "int8",
        "schedule_policy": schedule_policy,
        "operand_movement_policy": {
            "weights": "preloaded_stationary",
            "dynamic_activation_setup": "materialized",
            "ffn_intermediate": "bank_local_capacity_controlled",
        },
        "ramulator_visible_defaults": dict(_RAMULATOR_DEFAULTS),
        "mapping_policy": dict(_MAPPING_POLICY),
        "literature_anchors": [
            model_citation.split(":")[0].strip(),
            "LPDDR5-PIM native opcode surface",
        ],
        "non_claims": list(_COMMON_NON_CLAIMS),
    }


def get_opt_125m_ffn_manifest(
    *, schedule_policy: str = "serialized"
) -> dict[str, Any]:
    """OPT-125M FFN: 12 layers, hidden=768, ffn_hidden=3072."""
    return _ffn_base(
        "OPT-125M decoder-only transformer FFN/SwiGLU slice",
        "OPT (Zhang et al., arXiv:2205.01068)",
        model_total_layers=12,
        hidden_size=768,
        ffn_hidden_size=3072,
        manifest_name="opt_125m_ffn_swiglu_decode",
        schedule_policy=schedule_policy,
    )


def get_opt_350m_ffn_manifest(
    *, schedule_policy: str = "serialized"
) -> dict[str, Any]:
    """OPT-350M FFN: 24 layers, hidden=1024, ffn_hidden=4096."""
    return _ffn_base(
        "OPT-350M decoder-only transformer FFN/SwiGLU slice",
        "OPT (Zhang et al., arXiv:2205.01068)",
        model_total_layers=24,
        hidden_size=1024,
        ffn_hidden_size=4096,
        manifest_name="opt_350m_ffn_swiglu_decode",
        schedule_policy=schedule_policy,
    )


def get_opt_1_3b_ffn_manifest(
    *, schedule_policy: str = "serialized"
) -> dict[str, Any]:
    """OPT-1.3B FFN: 24 layers, hidden=2048, ffn_hidden=8192."""
    return _ffn_base(
        "OPT-1.3B decoder-only transformer FFN/SwiGLU slice",
        "OPT (Zhang et al., arXiv:2205.01068)",
        model_total_layers=24,
        hidden_size=2048,
        ffn_hidden_size=8192,
        manifest_name="opt_1_3b_ffn_swiglu_decode",
        schedule_policy=schedule_policy,
    )


def get_llama_7b_ffn_manifest(
    *, schedule_policy: str = "serialized"
) -> dict[str, Any]:
    """LLaMA-7B FFN: 32 layers, hidden=4096, ffn_hidden=11008."""
    return _ffn_base(
        "LLaMA-7B decoder-only transformer FFN/SwiGLU slice",
        "LLaMA (Touvron et al., arXiv:2302.13971)",
        model_total_layers=32,
        hidden_size=4096,
        ffn_hidden_size=11008,
        manifest_name="llama_7b_ffn_swiglu_decode",
        schedule_policy=schedule_policy,
    )


def get_llama_13b_ffn_manifest(
    *, schedule_policy: str = "serialized"
) -> dict[str, Any]:
    """LLaMA-13B FFN: 40 layers, hidden=5120, ffn_hidden=13824."""
    return _ffn_base(
        "LLaMA-13B decoder-only transformer FFN/SwiGLU slice",
        "LLaMA (Touvron et al., arXiv:2302.13971)",
        model_total_layers=40,
        hidden_size=5120,
        ffn_hidden_size=13824,
        manifest_name="llama_13b_ffn_swiglu_decode",
        schedule_policy=schedule_policy,
    )


# ═══════════════════════════════════════════════════════════════════════
# MoE manifests  (single-layer decode, published expert dimensions)
# ═══════════════════════════════════════════════════════════════════════

def get_mixtral_moe_manifest(
    *,
    schedule_policy: str = "serialized",
    selected_experts: list[int] | None = None,
) -> dict[str, Any]:
    """Mixtral-8x7B MoE: 32 layers, 8 experts, top-2.

    Real dimensions: hidden=4096, expert_ffn=14336, head_dim=128.
    Per-token compute is limited to top-2 active experts.

    .. note::
       Full-dimension Mixtral MoE generates ~11 M PIM_MAC per token
       (2 experts, each 58.7 M MAC elements), producing traces > 1 GB.
       For paper figures, the hidden dimension is scaled to 512
       (expert FFN 2048) to keep replay fast; scaling factors are
       documented in the manifest provenance.
    """
    if selected_experts is None:
        selected_experts = [0, 1]

    return {
        "manifest_version": "paper-moe-v1.1",
        "manifest_name": "mixtral_8x7b_moe_decode",
        "provenance_class": "representative-model-scale",
        "model_citation": "Mixtral of Experts (Jiang et al., arXiv:2401.04088)",
        "model_total_layers": 32,
        "real_model_dimensions": {
            "hidden_size": 4096,
            "expert_hidden_size": 14336,
            "num_experts": 8,
            "top_k": 2,
            "num_query_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
        },
        "simulation_scaling": "hidden=512, expert_ffn=2048 for trace-size feasibility; per-token MAC pattern preserved",
        "workload_class": "structured_transformer_moe_surrogate",
        "phase": "decode",
        "model_family": "Mixtral-8x7B decoder-only transformer MoE slice",
        "num_layers": 1,
        "seq_len": 1,
        "hidden_size": 512,
        "expert_hidden_size": 2048,
        "num_experts": 8,
        "top_k": 2,
        "selected_experts": selected_experts,
        "datatype": "int8",
        "schedule_policy": schedule_policy,
        "ramulator_visible_defaults": dict(_RAMULATOR_DEFAULTS),
        "mapping_policy": dict(_MAPPING_POLICY),
        "literature_anchors": [
            "Mixtral-8x7B (Jiang et al., 2024)",
            "LPDDR5-PIM native opcode surface",
        ],
        "non_claims": list(_COMMON_NON_CLAIMS),
    }


# Backward-compatible alias
get_mixtral_style_moe_manifest = get_mixtral_moe_manifest


# ═══════════════════════════════════════════════════════════════════════
# Named configurations and defaults
# ═══════════════════════════════════════════════════════════════════════

PAPER_ATTENTION_CONFIGS: dict[str, dict[str, Any]] = {
    "opt_125m":  get_opt_125m_attention_manifest(),
    "opt_350m":  get_opt_350m_attention_manifest(),
    "opt_1_3b":  get_opt_1_3b_attention_manifest(),
    "llama_7b":  get_llama_7b_attention_manifest(),
    "llama_13b": get_llama_13b_attention_manifest(),
    "mixtral":   get_mixtral_attention_manifest(),
}

PAPER_FFN_CONFIGS: dict[str, dict[str, Any]] = {
    "opt_125m":  get_opt_125m_ffn_manifest(),
    "opt_350m":  get_opt_350m_ffn_manifest(),
    "opt_1_3b":  get_opt_1_3b_ffn_manifest(),
    "llama_7b":  get_llama_7b_ffn_manifest(),
    "llama_13b": get_llama_13b_ffn_manifest(),
}

PAPER_MOE_CONFIGS: dict[str, dict[str, Any]] = {
    "mixtral": get_mixtral_moe_manifest(),
}

# Paper defaults ← used when no explicit config key is given
DEFAULT_PAPER_ATTENTION = "opt_125m"
DEFAULT_PAPER_FFN = "opt_125m"
DEFAULT_PAPER_MOE = "mixtral"

# ═══════════════════════════════════════════════════════════════════════
# Model cross-reference cheat-sheet  (for paper evaluation text)
# ═══════════════════════════════════════════════════════════════════════
#
#  ┌─────────────┬────────┬───────┬──────────┬──────────┬──────────┐
#  │ Model       │ Layers │ Heads │ h_dim    │ hidden   │ FFN      │
#  ├─────────────┼────────┼───────┼──────────┼──────────┼──────────┤
#  │ OPT-125M    │     12 │    12 │      64  │      768 │    3072  │
#  │ OPT-350M    │     24 │    16 │      64  │     1024 │    4096  │
#  │ OPT-1.3B    │     24 │    32 │      64  │     2048 │    8192  │
#  │ LLaMA-7B    │     32 │    32 │     128  │     4096 │   11008  │
#  │ LLaMA-13B   │     40 │    40 │     128  │     5120 │   13824  │
#  │ Mixtral-8x7B│     32 │ 32Q/8K│     128  │     4096 │   14336* │
#  └─────────────┴────────┴───────┴──────────┴──────────┴──────────┘
#  * Mixtral-8x7B FFN = per-expert FFN dim; 2 of 8 experts active per token.
