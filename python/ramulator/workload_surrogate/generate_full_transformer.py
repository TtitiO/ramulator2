"""Tiny Phase 4 full-transformer semantic dataflow generator.

This module intentionally keeps workload compilation offline in Python.  It
emits native Ramulator2-ext semantic tensor-DAG records, not raw AttAcc traces
or HBM3 opcodes.  The first implemented slice is attention score -> softmax
accounting -> context with explicit tensor IO, dependencies, residency, and
tile/head metadata.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from ramulator.workload_surrogate.structured_trace import (
    GENERATOR_VERSION,
    REQUIRED_P4_RECORD_FAMILY,
    REQUIRED_P4_NON_CLAIMS as STRUCTURED_REQUIRED_P4_NON_CLAIMS,
    SCHEMA_VERSION,
    expanded_record_count,
    validate_record,
    write_json,
    write_jsonl,
)


FULL_TRANSFORMER_GENERATOR_VERSION = f"{GENERATOR_VERSION}-p4-full-transformer"
DEFAULT_OUTPUT_DIR = Path("ramulator2/tests/data/structured_workload_surrogate/full_transformer_attention_v0_1")
REQUIRED_P4_NON_CLAIMS = set(STRUCTURED_REQUIRED_P4_NON_CLAIMS)
REQUIRED_P4_PREFILL_NON_CLAIMS = REQUIRED_P4_NON_CLAIMS | {
    "not_flashattention_equivalence",
    "not_chunked_prefill_runtime",
    "not_prefix_cache_or_page_table_model",
    "not_serving_goodput",
    "prefill_only_structured_surrogate",
}
SUPPORTED_ATTENTION_DATATYPES = {"int8", "fp16", "bf16"}
LLAMA2_7B_NUM_LAYERS = 32
LLAMA2_7B_NUM_HEADS = 32
LLAMA2_7B_HEAD_DIM = 128
LLAMA2_7B_HIDDEN_SIZE = 4096
LLAMA2_7B_FFN_HIDDEN_SIZE = 11008
LLAMA2_7B_DEFAULT_PAST_LEN = 1024
# The 8Gb LPDDR5-PIM preset exposes 16 bank units (one rank). Dense transformer
# manifests round-robin PIM_MAC work across all 16 banks so the per-bank (k=1)
# vs shared-MPU (k=2) comparison matches the F4/F5/F6 device configuration.
DENSE_PIM_BANK_SEQUENCE = list(range(16))
LLAMA2_13B_NUM_LAYERS = 40
LLAMA2_13B_NUM_HEADS = 40
LLAMA2_13B_HEAD_DIM = 128
LLAMA2_13B_HIDDEN_SIZE = 5120
LLAMA2_13B_FFN_HIDDEN_SIZE = 13824
LLAMA2_13B_DEFAULT_PAST_LEN = 1024
LLAMA2_70B_NUM_LAYERS = 80
LLAMA2_70B_NUM_HEADS = 64
LLAMA2_70B_NUM_KV_HEADS = 8
LLAMA2_70B_HEAD_DIM = 128
LLAMA2_70B_HIDDEN_SIZE = 8192
LLAMA2_70B_FFN_HIDDEN_SIZE = 28672
LLAMA2_70B_DEFAULT_PAST_LEN = 1024
_VALID_DISTRIBUTION_POLICIES = {"broadcast", "bank_sharded", "replicated"}
FFN_VARIANT_SWIGLU_3PROJ = "swiglu_3proj"
FFN_VARIANT_GEGLU_3PROJ = "geglu_3proj"
FFN_VARIANT_RELU_2PROJ = "relu_2proj"
SUPPORTED_FFN_VARIANTS = {FFN_VARIANT_SWIGLU_3PROJ, FFN_VARIANT_GEGLU_3PROJ, FFN_VARIANT_RELU_2PROJ}
FFN_VARIANT_PROJECTION_COUNTS = {
    FFN_VARIANT_SWIGLU_3PROJ: 3,
    FFN_VARIANT_GEGLU_3PROJ: 3,
    FFN_VARIANT_RELU_2PROJ: 2,
}

# Mixtral-8x7B (GQA MoE decoder)
MIXTRAL_8X7B_NUM_LAYERS = 32
MIXTRAL_8X7B_HIDDEN_SIZE = 4096
MIXTRAL_8X7B_NUM_HEADS = 32
MIXTRAL_8X7B_NUM_KV_HEADS = 8
MIXTRAL_8X7B_HEAD_DIM = 128
MIXTRAL_8X7B_NUM_EXPERTS = 8
MIXTRAL_8X7B_TOP_K = 2
MIXTRAL_8X7B_EXPERT_HIDDEN_SIZE = 14336
MIXTRAL_8X7B_DEFAULT_PAST_LEN = 1024
# Legacy scaled dimensions kept only for explicitly requested fast/debug traces.
MIXTRAL_8X7B_SCALED_HIDDEN = 512
MIXTRAL_8X7B_SCALED_EXPERT_HIDDEN = 2048

# OPT dense decoder (MHA, ReLU 2-projection)
OPT_125M_NUM_LAYERS = 12
OPT_125M_HIDDEN_SIZE = 768
OPT_125M_NUM_HEADS = 12
OPT_125M_HEAD_DIM = 64
OPT_125M_FFN_HIDDEN_SIZE = 3072
OPT_125M_DEFAULT_PAST_LEN = 512

OPT_350M_NUM_LAYERS = 24
OPT_350M_HIDDEN_SIZE = 1024
OPT_350M_NUM_HEADS = 16
OPT_350M_HEAD_DIM = 64
OPT_350M_FFN_HIDDEN_SIZE = 4096
OPT_350M_DEFAULT_PAST_LEN = 512

OPT_1_3B_NUM_LAYERS = 24
OPT_1_3B_HIDDEN_SIZE = 2048
OPT_1_3B_NUM_HEADS = 32
OPT_1_3B_HEAD_DIM = 64
OPT_1_3B_FFN_HIDDEN_SIZE = 8192
OPT_1_3B_DEFAULT_PAST_LEN = 512

# Qwen2/Qwen2.5 dense decoder (GQA, SwiGLU)
QWEN25_7B_NUM_LAYERS = 28
QWEN25_7B_HIDDEN_SIZE = 3584
QWEN25_7B_NUM_HEADS = 28
QWEN25_7B_NUM_KV_HEADS = 4
QWEN25_7B_HEAD_DIM = 128
QWEN25_7B_FFN_HIDDEN_SIZE = 18944
QWEN25_7B_DEFAULT_PAST_LEN = 1024

QWEN25_14B_NUM_LAYERS = 48
QWEN25_14B_HIDDEN_SIZE = 5120
QWEN25_14B_NUM_HEADS = 40
QWEN25_14B_NUM_KV_HEADS = 8
QWEN25_14B_HEAD_DIM = 128
QWEN25_14B_FFN_HIDDEN_SIZE = 13824
QWEN25_14B_DEFAULT_PAST_LEN = 1024

QWEN25_32B_NUM_LAYERS = 64
QWEN25_32B_HIDDEN_SIZE = 5120
QWEN25_32B_NUM_HEADS = 40
QWEN25_32B_NUM_KV_HEADS = 8
QWEN25_32B_HEAD_DIM = 128
QWEN25_32B_FFN_HIDDEN_SIZE = 27648
QWEN25_32B_DEFAULT_PAST_LEN = 1024

QWEN25_72B_NUM_LAYERS = 80
QWEN25_72B_HIDDEN_SIZE = 8192
QWEN25_72B_NUM_HEADS = 64
QWEN25_72B_NUM_KV_HEADS = 8
QWEN25_72B_HEAD_DIM = 128
QWEN25_72B_FFN_HIDDEN_SIZE = 29568
QWEN25_72B_DEFAULT_PAST_LEN = 1024

# Gemma/Gemma2 dense decoder (GQA/MQA, GeGLU)
GEMMA_2B_NUM_LAYERS = 18
GEMMA_2B_HIDDEN_SIZE = 2048
GEMMA_2B_NUM_HEADS = 8
GEMMA_2B_NUM_KV_HEADS = 1  # MQA
GEMMA_2B_HEAD_DIM = 256
GEMMA_2B_FFN_HIDDEN_SIZE = 16384
GEMMA_2B_DEFAULT_PAST_LEN = 1024

GEMMA_7B_NUM_LAYERS = 28
GEMMA_7B_HIDDEN_SIZE = 3072
GEMMA_7B_NUM_HEADS = 16
GEMMA_7B_NUM_KV_HEADS = 16  # MHA
GEMMA_7B_HEAD_DIM = 256
GEMMA_7B_FFN_HIDDEN_SIZE = 24576
GEMMA_7B_DEFAULT_PAST_LEN = 1024

GEMMA2_9B_NUM_LAYERS = 42
GEMMA2_9B_HIDDEN_SIZE = 3584
GEMMA2_9B_NUM_HEADS = 16
GEMMA2_9B_NUM_KV_HEADS = 8
GEMMA2_9B_HEAD_DIM = 256
GEMMA2_9B_FFN_HIDDEN_SIZE = 14336
GEMMA2_9B_DEFAULT_PAST_LEN = 1024

GEMMA2_27B_NUM_LAYERS = 46
GEMMA2_27B_HIDDEN_SIZE = 4608
GEMMA2_27B_NUM_HEADS = 32
GEMMA2_27B_NUM_KV_HEADS = 16
GEMMA2_27B_HEAD_DIM = 128
GEMMA2_27B_FFN_HIDDEN_SIZE = 73728
GEMMA2_27B_DEFAULT_PAST_LEN = 1024


@dataclass(frozen=True)
class ModelSpec:
    name: str
    num_layers: int
    hidden_size: int
    num_heads: int
    head_dim: int
    ffn_hidden_size: int
    datatype: str = "int8"
    num_kv_heads: int | None = None
    citation: str | None = None
    paper_anchor: str | None = None
    ffn_variant: str = FFN_VARIANT_SWIGLU_3PROJ
    activation: str = "silu"

    def __post_init__(self):
        if self.num_kv_heads is None:
            object.__setattr__(self, "num_kv_heads", self.num_heads)
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads}) for grouped-query attention"
            )
        if self.ffn_variant not in SUPPORTED_FFN_VARIANTS:
            raise ValueError(
                f"Unsupported ffn_variant {self.ffn_variant!r}; "
                f"supported variants are {sorted(SUPPORTED_FFN_VARIANTS)}"
            )
        if self.citation is None:
            object.__setattr__(self, "citation", f"{self.name} (auto-generated workload surrogate)")
        if self.paper_anchor is None:
            object.__setattr__(self, "paper_anchor", self.citation)


LLAMA2_7B_MODEL_SPEC = ModelSpec(
    name="Llama2-7B",
    num_layers=LLAMA2_7B_NUM_LAYERS,
    hidden_size=LLAMA2_7B_HIDDEN_SIZE,
    num_heads=LLAMA2_7B_NUM_HEADS,
    head_dim=LLAMA2_7B_HEAD_DIM,
    ffn_hidden_size=LLAMA2_7B_FFN_HIDDEN_SIZE,
    citation="Llama 2 (Touvron et al., arXiv:2307.09288)",
    paper_anchor="Llama 2 (Touvron et al., 2023)",
)
LLAMA2_13B_MODEL_SPEC = ModelSpec(
    name="Llama2-13B",
    num_layers=LLAMA2_13B_NUM_LAYERS,
    hidden_size=LLAMA2_13B_HIDDEN_SIZE,
    num_heads=LLAMA2_13B_NUM_HEADS,
    head_dim=LLAMA2_13B_HEAD_DIM,
    ffn_hidden_size=LLAMA2_13B_FFN_HIDDEN_SIZE,
    citation="Llama 2 (Touvron et al., arXiv:2307.09288)",
    paper_anchor="Llama 2 (Touvron et al., 2023)",
)
LLAMA2_70B_MODEL_SPEC = ModelSpec(
    name="Llama2-70B",
    num_layers=LLAMA2_70B_NUM_LAYERS,
    hidden_size=LLAMA2_70B_HIDDEN_SIZE,
    num_heads=LLAMA2_70B_NUM_HEADS,
    head_dim=LLAMA2_70B_HEAD_DIM,
    ffn_hidden_size=LLAMA2_70B_FFN_HIDDEN_SIZE,
    num_kv_heads=LLAMA2_70B_NUM_KV_HEADS,
    citation="Llama 2 (Touvron et al., arXiv:2307.09288)",
    paper_anchor="Llama 2 (Touvron et al., 2023)",
)

# Qwen2/Qwen2.5 dense decoder model specs (GQA, SwiGLU)
QWEN25_7B_MODEL_SPEC = ModelSpec(
    name="Qwen2.5-7B",
    num_layers=QWEN25_7B_NUM_LAYERS,
    hidden_size=QWEN25_7B_HIDDEN_SIZE,
    num_heads=QWEN25_7B_NUM_HEADS,
    head_dim=QWEN25_7B_HEAD_DIM,
    ffn_hidden_size=QWEN25_7B_FFN_HIDDEN_SIZE,
    num_kv_heads=QWEN25_7B_NUM_KV_HEADS,
    citation="Qwen2.5 (Yang et al., arXiv:2412.15115)",
    paper_anchor="Qwen2.5 (Yang et al., 2024)",
)
QWEN25_14B_MODEL_SPEC = ModelSpec(
    name="Qwen2.5-14B",
    num_layers=QWEN25_14B_NUM_LAYERS,
    hidden_size=QWEN25_14B_HIDDEN_SIZE,
    num_heads=QWEN25_14B_NUM_HEADS,
    head_dim=QWEN25_14B_HEAD_DIM,
    ffn_hidden_size=QWEN25_14B_FFN_HIDDEN_SIZE,
    num_kv_heads=QWEN25_14B_NUM_KV_HEADS,
    citation="Qwen2.5 (Yang et al., arXiv:2412.15115)",
    paper_anchor="Qwen2.5 (Yang et al., 2024)",
)
QWEN25_32B_MODEL_SPEC = ModelSpec(
    name="Qwen2.5-32B",
    num_layers=QWEN25_32B_NUM_LAYERS,
    hidden_size=QWEN25_32B_HIDDEN_SIZE,
    num_heads=QWEN25_32B_NUM_HEADS,
    head_dim=QWEN25_32B_HEAD_DIM,
    ffn_hidden_size=QWEN25_32B_FFN_HIDDEN_SIZE,
    num_kv_heads=QWEN25_32B_NUM_KV_HEADS,
    citation="Qwen2.5 (Yang et al., arXiv:2412.15115)",
    paper_anchor="Qwen2.5 (Yang et al., 2024)",
)
QWEN25_72B_MODEL_SPEC = ModelSpec(
    name="Qwen2.5-72B",
    num_layers=QWEN25_72B_NUM_LAYERS,
    hidden_size=QWEN25_72B_HIDDEN_SIZE,
    num_heads=QWEN25_72B_NUM_HEADS,
    head_dim=QWEN25_72B_HEAD_DIM,
    ffn_hidden_size=QWEN25_72B_FFN_HIDDEN_SIZE,
    num_kv_heads=QWEN25_72B_NUM_KV_HEADS,
    citation="Qwen2.5 (Yang et al., arXiv:2412.15115)",
    paper_anchor="Qwen2.5 (Yang et al., 2024)",
)

# Gemma/Gemma2 dense decoder model specs (MQA/GQA/MHA, GeGLU)
GEMMA_2B_MODEL_SPEC = ModelSpec(
    name="Gemma-2B",
    num_layers=GEMMA_2B_NUM_LAYERS,
    hidden_size=GEMMA_2B_HIDDEN_SIZE,
    num_heads=GEMMA_2B_NUM_HEADS,
    head_dim=GEMMA_2B_HEAD_DIM,
    ffn_hidden_size=GEMMA_2B_FFN_HIDDEN_SIZE,
    num_kv_heads=GEMMA_2B_NUM_KV_HEADS,
    ffn_variant=FFN_VARIANT_GEGLU_3PROJ,
    activation="gelu_pytorch_tanh",
    citation="Gemma (Gemma Team, arXiv:2403.08295)",
    paper_anchor="Gemma (Gemma Team, 2024)",
)
GEMMA_7B_MODEL_SPEC = ModelSpec(
    name="Gemma-7B",
    num_layers=GEMMA_7B_NUM_LAYERS,
    hidden_size=GEMMA_7B_HIDDEN_SIZE,
    num_heads=GEMMA_7B_NUM_HEADS,
    head_dim=GEMMA_7B_HEAD_DIM,
    ffn_hidden_size=GEMMA_7B_FFN_HIDDEN_SIZE,
    num_kv_heads=GEMMA_7B_NUM_KV_HEADS,
    ffn_variant=FFN_VARIANT_GEGLU_3PROJ,
    activation="gelu_pytorch_tanh",
    citation="Gemma (Gemma Team, arXiv:2403.08295)",
    paper_anchor="Gemma (Gemma Team, 2024)",
)
GEMMA2_9B_MODEL_SPEC = ModelSpec(
    name="Gemma-2-9B",
    num_layers=GEMMA2_9B_NUM_LAYERS,
    hidden_size=GEMMA2_9B_HIDDEN_SIZE,
    num_heads=GEMMA2_9B_NUM_HEADS,
    head_dim=GEMMA2_9B_HEAD_DIM,
    ffn_hidden_size=GEMMA2_9B_FFN_HIDDEN_SIZE,
    num_kv_heads=GEMMA2_9B_NUM_KV_HEADS,
    ffn_variant=FFN_VARIANT_GEGLU_3PROJ,
    activation="gelu_pytorch_tanh",
    citation="Gemma 2 (Gemma Team, arXiv:2408.00118)",
    paper_anchor="Gemma 2 (Gemma Team, 2024)",
)
GEMMA2_27B_MODEL_SPEC = ModelSpec(
    name="Gemma-2-27B",
    num_layers=GEMMA2_27B_NUM_LAYERS,
    hidden_size=GEMMA2_27B_HIDDEN_SIZE,
    num_heads=GEMMA2_27B_NUM_HEADS,
    head_dim=GEMMA2_27B_HEAD_DIM,
    ffn_hidden_size=GEMMA2_27B_FFN_HIDDEN_SIZE,
    num_kv_heads=GEMMA2_27B_NUM_KV_HEADS,
    ffn_variant=FFN_VARIANT_GEGLU_3PROJ,
    activation="gelu_pytorch_tanh",
    citation="Gemma 2 (Gemma Team, arXiv:2408.00118)",
    paper_anchor="Gemma 2 (Gemma Team, 2024)",
)

# OPT dense decoder model specs (MHA, ReLU 2-projection)
OPT_125M_MODEL_SPEC = ModelSpec(
    name="OPT-125M",
    num_layers=OPT_125M_NUM_LAYERS,
    hidden_size=OPT_125M_HIDDEN_SIZE,
    num_heads=OPT_125M_NUM_HEADS,
    head_dim=OPT_125M_HEAD_DIM,
    ffn_hidden_size=OPT_125M_FFN_HIDDEN_SIZE,
    ffn_variant=FFN_VARIANT_RELU_2PROJ,
    activation="relu",
    citation="OPT (Zhang et al., arXiv:2205.01068)",
    paper_anchor="OPT (Zhang et al., 2022)",
)
OPT_350M_MODEL_SPEC = ModelSpec(
    name="OPT-350M",
    num_layers=OPT_350M_NUM_LAYERS,
    hidden_size=OPT_350M_HIDDEN_SIZE,
    num_heads=OPT_350M_NUM_HEADS,
    head_dim=OPT_350M_HEAD_DIM,
    ffn_hidden_size=OPT_350M_FFN_HIDDEN_SIZE,
    ffn_variant=FFN_VARIANT_RELU_2PROJ,
    activation="relu",
    citation="OPT (Zhang et al., arXiv:2205.01068)",
    paper_anchor="OPT (Zhang et al., 2022)",
)
OPT_1_3B_MODEL_SPEC = ModelSpec(
    name="OPT-1.3B",
    num_layers=OPT_1_3B_NUM_LAYERS,
    hidden_size=OPT_1_3B_HIDDEN_SIZE,
    num_heads=OPT_1_3B_NUM_HEADS,
    head_dim=OPT_1_3B_HEAD_DIM,
    ffn_hidden_size=OPT_1_3B_FFN_HIDDEN_SIZE,
    ffn_variant=FFN_VARIANT_RELU_2PROJ,
    activation="relu",
    citation="OPT (Zhang et al., arXiv:2205.01068)",
    paper_anchor="OPT (Zhang et al., 2022)",
)

MODEL_REGISTRY = {
    "llama2-7b": LLAMA2_7B_MODEL_SPEC,
    "llama2_7b": LLAMA2_7B_MODEL_SPEC,
    "Llama2-7B": LLAMA2_7B_MODEL_SPEC,
    "llama2-13b": LLAMA2_13B_MODEL_SPEC,
    "llama2_13b": LLAMA2_13B_MODEL_SPEC,
    "Llama2-13B": LLAMA2_13B_MODEL_SPEC,
    "llama2-70b": LLAMA2_70B_MODEL_SPEC,
    "llama2_70b": LLAMA2_70B_MODEL_SPEC,
    "Llama2-70B": LLAMA2_70B_MODEL_SPEC,
    "qwen2.5-7b": QWEN25_7B_MODEL_SPEC,
    "qwen25-7b": QWEN25_7B_MODEL_SPEC,
    "qwen2.5-14b": QWEN25_14B_MODEL_SPEC,
    "qwen25-14b": QWEN25_14B_MODEL_SPEC,
    "qwen2.5-32b": QWEN25_32B_MODEL_SPEC,
    "qwen25-32b": QWEN25_32B_MODEL_SPEC,
    "qwen2.5-72b": QWEN25_72B_MODEL_SPEC,
    "qwen25-72b": QWEN25_72B_MODEL_SPEC,
    "gemma-2b": GEMMA_2B_MODEL_SPEC,
    "gemma-7b": GEMMA_7B_MODEL_SPEC,
    "gemma-2-9b": GEMMA2_9B_MODEL_SPEC,
    "gemma2-9b": GEMMA2_9B_MODEL_SPEC,
    "gemma-2-27b": GEMMA2_27B_MODEL_SPEC,
    "gemma2-27b": GEMMA2_27B_MODEL_SPEC,
    "opt-125m": OPT_125M_MODEL_SPEC,
    "opt_125m": OPT_125M_MODEL_SPEC,
    "opt-350m": OPT_350M_MODEL_SPEC,
    "opt_350m": OPT_350M_MODEL_SPEC,
    "opt-1.3b": OPT_1_3B_MODEL_SPEC,
    "opt_1_3b": OPT_1_3B_MODEL_SPEC,
}

SUPPORTED_MODEL_FAMILIES = {"llama2", "qwen", "gemma", "mixtral"}


def get_model_spec(name: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported model spec: {name!r}. "
            f"Registered model names: {sorted(MODEL_REGISTRY)}."
        ) from exc


def _attention_num_kv_heads(manifest: dict) -> int:
    """Return validated effective KV-head count for MHA/GQA/MQA manifests."""
    num_heads = int(manifest["num_heads"])
    if "kv_heads" in manifest and "num_kv_heads" in manifest and int(manifest["kv_heads"]) != int(manifest["num_kv_heads"]):
        raise ValueError("kv_heads and num_kv_heads aliases must match")
    num_kv_heads = int(manifest.get("kv_heads", manifest.get("num_kv_heads", num_heads)))
    if num_kv_heads > num_heads:
        raise ValueError("num_kv_heads must not exceed num_heads")
    if num_kv_heads <= 0:
        raise ValueError("num_kv_heads must be positive")
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads}) "
            f"for grouped-query attention support"
        )
    return num_kv_heads


def _attention_head_group_size(manifest: dict) -> int:
    """Return the Q-heads-per-KV-head group size and reject stale metadata."""
    num_heads = int(manifest["num_heads"])
    num_kv_heads = _attention_num_kv_heads(manifest)
    derived = num_heads // num_kv_heads
    if "head_group_size" in manifest and int(manifest["head_group_size"]) != derived:
        raise ValueError(
            f"head_group_size ({manifest['head_group_size']}) must equal "
            f"num_heads // num_kv_heads ({derived})"
        )
    return derived


def ffn_variant(manifest: dict) -> str:
    """Return the FFN topology variant, defaulting legacy manifests to SwiGLU."""
    return str(manifest.get("ffn_variant", FFN_VARIANT_SWIGLU_3PROJ))


def ffn_projection_count(manifest: dict) -> int:
    """Return the number of GEMV/GEMM projections implied by an FFN manifest."""
    variant = ffn_variant(manifest)
    try:
        return FFN_VARIANT_PROJECTION_COUNTS[variant]
    except KeyError as exc:
        raise ValueError(f"Unsupported FFN variant {variant!r}; supported variants are {sorted(SUPPORTED_FFN_VARIANTS)}") from exc


def _ffn_operator_family(manifest: dict) -> str:
    variant = ffn_variant(manifest)
    if variant == FFN_VARIANT_RELU_2PROJ:
        return "ffn_relu_2proj"
    if variant == FFN_VARIANT_SWIGLU_3PROJ:
        return "ffn_swiglu"
    if variant == FFN_VARIANT_GEGLU_3PROJ:
        return "ffn_geglu"
    raise ValueError(f"Unsupported FFN variant {variant!r}; supported variants are {sorted(SUPPORTED_FFN_VARIANTS)}")


def _ffn_family_label(spec: "ModelSpec") -> str:
    """Return a human-readable FFN family label for a ModelSpec."""
    variant = spec.ffn_variant
    if variant == FFN_VARIANT_SWIGLU_3PROJ:
        return "SwiGLU"
    if variant == FFN_VARIANT_GEGLU_3PROJ:
        return "GeGLU"
    if variant == FFN_VARIANT_RELU_2PROJ:
        return "ReLU 2-proj"
    return variant.replace("_", " ").title()


def get_tiny_attention_manifest() -> dict:
    return {
        "manifest_version": "p4-attention-v0.1",
        "manifest_name": "tiny_attention_p4_v0_1",
        "workload_class": "structured_transformer_attention_surrogate",
        "phase": "decode",
        "model_family": "tiny decoder-only transformer attention slice",
        "num_layers": 1,
        "num_heads": 2,
        "head_dim": 32,
        "past_len": 64,
        "seq_len": 1,
        "datatype": "int8",
        "score_tile_tokens": 32,
        "context_tile_tokens": 32,
        "head_group_size": 1,
        "schedule_policy": "serialized",
        "operand_movement_policy": {
            "weights": "preloaded_stationary",
            "dynamic_activation_setup": "materialized",
            "ffn_intermediate": "bank_local_capacity_controlled",
        },
        "ramulator_visible_defaults": {
            "bank_sequence": [0, 1, 2, 3],
            "bank_sequence_order": "frontend",
            "pim_banks_per_mpu": 1,
            "burst_length": 1,
            "row_start": 0,
            "row_count": 16,
            "dependency_count": 8,
            "column_start": 0,
        },
        "mapping_policy": {
            "host_policy": "semantic_tensor_io_only",
            "pim_policy": "native_lpddr5_pim_attention_tiles",
            "bank_sequence_policy": "manifest_order",
            "mpu_grouping_policy": "manifest_pim_banks_per_mpu",
        },
        "literature_anchors": ["AttAcc workflow ideology only", "LPDDR5-PIM native opcode surface"],
        "non_claims": [
            "not_runtime_replay",
            "not_vllm_replay",
            "not_numerical_correctness",
            "not_silicon_faithful_softmax_or_data_movement",
            "not_raw_attacc_schema",
        ],
    }


def get_tiny_ffn_manifest() -> dict:
    return {
        "manifest_version": "p4-ffn-v0.1",
        "manifest_name": "tiny_ffn_swiglu_p4_v0_1",
        "workload_class": "structured_transformer_ffn_swiglu_surrogate",
        "phase": "decode",
        "model_family": "tiny decoder-only transformer FFN/SwiGLU slice",
        "ffn_variant": FFN_VARIANT_SWIGLU_3PROJ,
        "num_layers": 1,
        "seq_len": 1,
        "hidden_size": 32,
        "ffn_hidden_size": 64,
        "ffn_activation_tile_size": 16,
        "activation_distribution_policy": "broadcast",
        "activation": "silu",
        "datatype": "int8",
        "schedule_policy": "serialized",
        "operand_movement_policy": {
            "weights": "preloaded_stationary",
            "dynamic_activation_setup": "materialized",
            "ffn_intermediate": "bank_local_capacity_controlled",
        },
        "ramulator_visible_defaults": {
            "bank_sequence": [0, 1, 2, 3],
            "bank_sequence_order": "frontend",
            "pim_banks_per_mpu": 1,
            "burst_length": 1,
            "row_start": 0,
            "row_count": 16,
            "dependency_count": 8,
            "column_start": 0,
        },
        "mapping_policy": {
            "host_policy": "semantic_tensor_io_only",
            "pim_policy": "native_lpddr5_pim_ffn_tiles",
            "bank_sequence_policy": "manifest_order",
            "mpu_grouping_policy": "manifest_pim_banks_per_mpu",
        },
        "literature_anchors": ["LPDDR5-PIM native opcode surface"],
        "non_claims": [
            "not_runtime_replay",
            "not_vllm_replay",
            "not_numerical_correctness",
            "not_silicon_faithful_softmax_or_data_movement",
            "not_raw_attacc_schema",
        ],
    }


def _llama2_common_non_claims() -> list[str]:
    return list(STRUCTURED_REQUIRED_P4_NON_CLAIMS)


def _llama2_prefill_non_claims() -> list[str]:
    return sorted(REQUIRED_P4_PREFILL_NON_CLAIMS)


def get_llama2_dense_decoder_attention_manifest(
    model: ModelSpec | str,
    *,
    past_len: int = LLAMA2_7B_DEFAULT_PAST_LEN,
    schedule_policy: str = "serialized",
) -> dict:
    spec = get_model_spec(model) if isinstance(model, str) else model
    tile_tokens = min(256, past_len)
    model_slug = spec.name.lower().replace("-", "_")
    name_dims = "" if spec == LLAMA2_7B_MODEL_SPEC else f"_hidden{spec.hidden_size}"
    return {
        "manifest_version": f"{spec.name.lower()}-attention-v0.1",
        "manifest_name": f"{model_slug}_{spec.num_layers}_layer{name_dims}_attention_decode",
        "workload_class": "structured_transformer_attention_surrogate",
        "phase": "decode",
        "model_family": f"{spec.name} dense decoder attention slice",
        "model_citation": spec.citation,
        "model_total_layers": spec.num_layers,
        "num_layers": spec.num_layers,
        "num_heads": spec.num_heads,
        "num_kv_heads": spec.num_kv_heads,
        "head_dim": spec.head_dim,
        "hidden_size": spec.hidden_size,
        "past_len": past_len,
        "seq_len": 1,
        "datatype": spec.datatype,
        "score_tile_tokens": tile_tokens,
        "context_tile_tokens": tile_tokens,
        "head_group_size": spec.num_heads // spec.num_kv_heads,
        "schedule_policy": schedule_policy,
        "operand_movement_policy": {
            "weights": "preloaded_stationary",
            "dynamic_activation_setup": "materialized",
            "ffn_intermediate": "bank_local_capacity_controlled",
        },
        "ramulator_visible_defaults": {
            "bank_sequence": list(DENSE_PIM_BANK_SEQUENCE),
            "bank_sequence_order": "frontend",
            "pim_banks_per_mpu": 1,
            "burst_length": 1,
            "row_start": 0,
            "row_count": 16,
            "dependency_count": 8,
            "column_start": 0,
        },
        "mapping_policy": {
            "host_policy": "semantic_tensor_io_only",
            "pim_policy": "native_lpddr5_pim_attention_tiles",
            "bank_sequence_policy": "manifest_order",
            "mpu_grouping_policy": "manifest_pim_banks_per_mpu",
        },
        "literature_anchors": [
            spec.paper_anchor,
            "LPDDR5-PIM native opcode surface",
        ],
        "non_claims": _llama2_common_non_claims(),
    }


def get_llama2_dense_decoder_ffn_manifest(
    model: ModelSpec | str,
    *,
    schedule_policy: str = "serialized",
) -> dict:
    spec = get_model_spec(model) if isinstance(model, str) else model
    model_slug = spec.name.lower().replace("-", "_")
    ffn_slug = spec.ffn_variant.replace("_3proj", "").replace("_2proj", "")
    variant_suffix = ffn_slug + "_decode"
    name_dims = "" if spec == LLAMA2_7B_MODEL_SPEC else f"_hidden{spec.hidden_size}_ffn{spec.ffn_hidden_size}"
    return {
        "manifest_version": f"{spec.name.lower()}-ffn-v0.1",
        "manifest_name": f"{model_slug}_{spec.num_layers}_layer{name_dims}_ffn_{variant_suffix}",
        "workload_class": f"structured_transformer_ffn_{ffn_slug}_surrogate",
        "phase": "decode",
        "model_family": f"{spec.name} dense decoder FFN/{_ffn_family_label(spec)} slice",
        "ffn_variant": spec.ffn_variant,
        "model_citation": spec.citation,
        "model_total_layers": spec.num_layers,
        "num_layers": spec.num_layers,
        "seq_len": 1,
        "hidden_size": spec.hidden_size,
        "ffn_hidden_size": spec.ffn_hidden_size,
        "ffn_activation_tile_size": spec.hidden_size,
        "activation_distribution_policy": "broadcast",
        "activation": spec.activation,
        "datatype": spec.datatype,
        "schedule_policy": schedule_policy,
        "operand_movement_policy": {
            "weights": "preloaded_stationary",
            "dynamic_activation_setup": "materialized",
            "ffn_intermediate": "bank_local_capacity_controlled",
        },
        "ramulator_visible_defaults": {
            "bank_sequence": list(DENSE_PIM_BANK_SEQUENCE),
            "bank_sequence_order": "frontend",
            "pim_banks_per_mpu": 1,
            "burst_length": 1,
            "row_start": 0,
            "row_count": 16,
            "dependency_count": 8,
            "column_start": 0,
        },
        "mapping_policy": {
            "host_policy": "semantic_tensor_io_only",
            "pim_policy": "native_lpddr5_pim_ffn_tiles",
            "bank_sequence_policy": "manifest_order",
            "mpu_grouping_policy": "manifest_pim_banks_per_mpu",
        },
        "literature_anchors": [
            spec.paper_anchor,
            "LPDDR5-PIM native opcode surface",
        ],
        "non_claims": _llama2_common_non_claims(),
    }


def get_llama2_dense_prefill_attention_manifest(
    model: ModelSpec | str,
    *,
    prompt_len: int,
    schedule_policy: str = "serialized",
    score_tile_tokens: int | None = None,
    context_tile_tokens: int | None = None,
) -> dict:
    spec = get_model_spec(model) if isinstance(model, str) else model
    if prompt_len <= 0:
        raise ValueError("prompt_len must be positive for prefill manifests")
    tile_tokens = min(256, prompt_len) if score_tile_tokens is None else min(int(score_tile_tokens), prompt_len)
    context_tokens = tile_tokens if context_tile_tokens is None else min(int(context_tile_tokens), prompt_len)
    model_slug = spec.name.lower().replace("-", "_")
    manifest = get_llama2_dense_decoder_attention_manifest(spec, past_len=prompt_len, schedule_policy=schedule_policy)
    manifest.update(
        {
            "manifest_name": f"{model_slug}_{spec.num_layers}_layer_attention_prefill_P{prompt_len}",
            "phase": "prefill",
            "model_family": f"{spec.name} dense prefill attention slice",
            "prompt_len": prompt_len,
            "past_len": 0,
            "seq_len": prompt_len,
            "score_tile_tokens": tile_tokens,
            "context_tile_tokens": context_tokens,
            "attention_mode": "causal_prefill",
            "prefill_only": True,
            "residency_policy": {
                "k_v_operand_source": "layer_local_projection_no_host_kv_read",
                "kv_cache_population": "HostWrite persists projected K/V for later decode",
                "prefill_attention_reads": "semantic PIMOperandResidency from same-layer Q/K/V projections; no HostRead KV-cache tiles",
            },
            "non_claims": _llama2_prefill_non_claims(),
        }
    )
    return manifest


def get_llama2_dense_prefill_ffn_manifest(
    model: ModelSpec | str,
    *,
    prompt_len: int,
    schedule_policy: str = "serialized",
) -> dict:
    spec = get_model_spec(model) if isinstance(model, str) else model
    if prompt_len <= 0:
        raise ValueError("prompt_len must be positive for prefill manifests")
    model_slug = spec.name.lower().replace("-", "_")
    manifest = get_llama2_dense_decoder_ffn_manifest(spec, schedule_policy=schedule_policy)
    manifest.update(
        {
            "manifest_name": f"{model_slug}_{spec.num_layers}_layer_ffn_prefill_P{prompt_len}",
            "phase": "prefill",
            "model_family": f"{spec.name} dense prefill FFN/{_ffn_family_label(spec)} slice",
            "prompt_len": prompt_len,
            "seq_len": prompt_len,
            "prefill_only": True,
            "non_claims": _llama2_prefill_non_claims(),
        }
    )
    return manifest


def get_dense_prefill_manifests(
    spec: ModelSpec,
    *,
    prompt_len: int,
    schedule_policy: str = "serialized",
) -> tuple[dict, dict]:
    return (
        get_llama2_dense_prefill_attention_manifest(spec, prompt_len=prompt_len, schedule_policy=schedule_policy),
        get_llama2_dense_prefill_ffn_manifest(spec, prompt_len=prompt_len, schedule_policy=schedule_policy),
    )


def get_llama2_7b_full_depth_attention_manifest(
    *, past_len: int = LLAMA2_7B_DEFAULT_PAST_LEN, schedule_policy: str = "serialized"
) -> dict:
    return get_llama2_dense_decoder_attention_manifest(LLAMA2_7B_MODEL_SPEC, past_len=past_len, schedule_policy=schedule_policy)


def get_llama2_7b_full_depth_ffn_manifest(*, schedule_policy: str = "serialized") -> dict:
    return get_llama2_dense_decoder_ffn_manifest(LLAMA2_7B_MODEL_SPEC, schedule_policy=schedule_policy)


def get_llama2_7b_dense_decoder_manifests(
    *, past_len: int = LLAMA2_7B_DEFAULT_PAST_LEN, schedule_policy: str = "serialized"
) -> tuple[dict, dict]:
    return (
        get_llama2_7b_full_depth_attention_manifest(past_len=past_len, schedule_policy=schedule_policy),
        get_llama2_7b_full_depth_ffn_manifest(schedule_policy=schedule_policy),
    )


def get_llama2_13b_full_depth_attention_manifest(
    *, past_len: int = LLAMA2_13B_DEFAULT_PAST_LEN, schedule_policy: str = "serialized"
) -> dict:
    return get_llama2_dense_decoder_attention_manifest(LLAMA2_13B_MODEL_SPEC, past_len=past_len, schedule_policy=schedule_policy)


def get_llama2_13b_full_depth_ffn_manifest(*, schedule_policy: str = "serialized") -> dict:
    return get_llama2_dense_decoder_ffn_manifest(LLAMA2_13B_MODEL_SPEC, schedule_policy=schedule_policy)


def get_llama2_13b_dense_decoder_manifests(
    *, past_len: int = LLAMA2_13B_DEFAULT_PAST_LEN, schedule_policy: str = "serialized"
) -> tuple[dict, dict]:
    return (
        get_llama2_13b_full_depth_attention_manifest(past_len=past_len, schedule_policy=schedule_policy),
        get_llama2_13b_full_depth_ffn_manifest(schedule_policy=schedule_policy),
    )


# -- Mixtral-8x7B MoE decoder manifests (full 32-layer, GQA attention) ---

def _mixtral_ramulator_defaults() -> dict:
    return {
        "bank_sequence": [0, 1, 2, 3],
        "bank_sequence_order": "frontend",
        "pim_banks_per_mpu": 1,
        "burst_length": 1,
        "row_start": 0,
        "row_count": 16,
        "dependency_count": 8,
        "column_start": 0,
    }


def _mixtral_mapping_policy() -> dict:
    return {
        "host_policy": "semantic_tensor_io_only",
        "pim_policy": "native_lpddr5_pim_operator_tiles",
        "bank_sequence_policy": "manifest_order",
        "mpu_grouping_policy": "manifest_pim_banks_per_mpu",
    }


def _mixtral_operand_movement_policy() -> dict:
    return {
        "weights": "preloaded_stationary",
        "dynamic_activation_setup": "materialized",
        "ffn_intermediate": "bank_local_capacity_controlled",
    }


def get_mixtral_8x7b_moe_decoder_attention_manifest(
    *,
    past_len: int = MIXTRAL_8X7B_DEFAULT_PAST_LEN,
    schedule_policy: str = "serialized",
    scaled: bool = False,
) -> dict:
    """Mixtral-8x7B GQA attention manifest for full-depth MoE decode pipeline.

    Defaults to real Mixtral-8x7B dimensions (hidden=4096, 32 Q heads, 8 KV
    heads).  Pass ``scaled=True`` only for fast/debug traces that intentionally
    preserve the 4:1 GQA ratio at hidden=512.
    """
    if scaled:
        hidden = MIXTRAL_8X7B_SCALED_HIDDEN
        num_heads = hidden // MIXTRAL_8X7B_HEAD_DIM
        num_kv_heads = max(1, num_heads // 4)
        manifest_suffix = "scaled"
        attention_variant = "grouped_query_attention_scaled"
        model_family = "Mixtral-8x7B decoder attention slice (scaled GQA)"
    else:
        hidden = MIXTRAL_8X7B_HIDDEN_SIZE
        num_heads = MIXTRAL_8X7B_NUM_HEADS
        num_kv_heads = MIXTRAL_8X7B_NUM_KV_HEADS
        manifest_suffix = "real"
        attention_variant = "grouped_query_attention"
        model_family = "Mixtral-8x7B decoder attention slice (real GQA)"
    tile_tokens = min(256, past_len)
    return {
        "manifest_version": "mixtral-8x7b-attention-v0.1",
        "manifest_name": f"mixtral_8x7b_{MIXTRAL_8X7B_NUM_LAYERS}_layer_attention_decode_{manifest_suffix}",
        "provenance_class": "representative-model-scale",
        "model_citation": "Mixtral of Experts (Jiang et al., arXiv:2401.04088)",
        "attention_variant": attention_variant,
        "data_movement_assumption": "semantic_kv_cache_host_read_per_tile",
        "workload_class": "structured_transformer_attention_surrogate",
        "phase": "decode",
        "model_family": model_family,
        "model_total_layers": MIXTRAL_8X7B_NUM_LAYERS,
        "num_layers": MIXTRAL_8X7B_NUM_LAYERS,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": MIXTRAL_8X7B_HEAD_DIM,
        "hidden_size": hidden,
        "past_len": past_len,
        "seq_len": 1,
        "datatype": "int8",
        "score_tile_tokens": tile_tokens,
        "context_tile_tokens": tile_tokens,
        "head_group_size": num_heads // num_kv_heads,
        "schedule_policy": schedule_policy,
        "operand_movement_policy": _mixtral_operand_movement_policy(),
        "ramulator_visible_defaults": _mixtral_ramulator_defaults(),
        "mapping_policy": _mixtral_mapping_policy(),
        "literature_anchors": [
            "Mixtral-8x7B (Jiang et al., 2024)",
            "LPDDR5-PIM native opcode surface",
        ],
        "non_claims": _llama2_common_non_claims(),
    }


def get_mixtral_8x7b_moe_decoder_moe_manifest(
    *,
    schedule_policy: str = "serialized",
    selected_experts: list[int] | None = None,
    scaled: bool = False,
) -> dict:
    """Mixtral-8x7B MoE manifest for full-depth MoE decode pipeline.

    Defaults to real Mixtral-8x7B dimensions (hidden=4096,
    expert_ffn=14336, 8 experts, top-2).  Pass ``scaled=True`` only for
    explicitly bounded fast/debug traces.
    """
    if selected_experts is None:
        selected_experts = [0, 1]
    hidden = MIXTRAL_8X7B_SCALED_HIDDEN if scaled else MIXTRAL_8X7B_HIDDEN_SIZE
    expert_hidden = MIXTRAL_8X7B_SCALED_EXPERT_HIDDEN if scaled else MIXTRAL_8X7B_EXPERT_HIDDEN_SIZE
    manifest_suffix = "scaled" if scaled else "real"
    return {
        "manifest_version": "mixtral-moe-decode-v0.1",
        "manifest_name": f"mixtral_8x7b_{MIXTRAL_8X7B_NUM_LAYERS}_layer_moe_decode_{manifest_suffix}",
        "provenance_class": "representative-model-scale",
        "model_citation": "Mixtral of Experts (Jiang et al., arXiv:2401.04088)",
        "real_model_dimensions": {
            "hidden_size": MIXTRAL_8X7B_HIDDEN_SIZE,
            "expert_hidden_size": MIXTRAL_8X7B_EXPERT_HIDDEN_SIZE,
            "num_experts": MIXTRAL_8X7B_NUM_EXPERTS,
            "top_k": MIXTRAL_8X7B_TOP_K,
            "num_query_heads": MIXTRAL_8X7B_NUM_HEADS,
            "num_kv_heads": MIXTRAL_8X7B_NUM_KV_HEADS,
            "head_dim": MIXTRAL_8X7B_HEAD_DIM,
        },
        "simulation_scaling": (
            f"hidden={hidden}, expert_ffn={expert_hidden}; "
            + ("scaled fast/debug trace" if scaled else "real backend trace dimensions")
        ),
        "workload_class": "structured_transformer_moe_surrogate",
        "phase": "decode",
        "model_family": "Mixtral-8x7B decoder-only transformer MoE slice (8 experts, top-2)",
        "model_total_layers": MIXTRAL_8X7B_NUM_LAYERS,
        "num_layers": MIXTRAL_8X7B_NUM_LAYERS,
        "seq_len": 1,
        "hidden_size": hidden,
        "expert_hidden_size": expert_hidden,
        "ffn_hidden_size": expert_hidden,  # for Q/K/V/O projection record compatibility
        "num_experts": MIXTRAL_8X7B_NUM_EXPERTS,
        "top_k": MIXTRAL_8X7B_TOP_K,
        "selected_experts": selected_experts,
        "datatype": "int8",
        "schedule_policy": schedule_policy,
        "operand_movement_policy": {
            "weights": "preloaded_stationary",
            "router_input_setup": "materialized",
            "token_dispatch": "materialized",
            "expert_output_combine": "materialized",
        },
        "ramulator_visible_defaults": _mixtral_ramulator_defaults(),
        "mapping_policy": _mixtral_mapping_policy(),
        "literature_anchors": [
            "Mixtral-8x7B (Jiang et al., 2024)",
            "LPDDR5-PIM native opcode surface",
        ],
        "non_claims": _llama2_common_non_claims(),
    }


def get_mixtral_8x7b_moe_decoder_manifests(
    *,
    past_len: int = MIXTRAL_8X7B_DEFAULT_PAST_LEN,
    schedule_policy: str = "serialized",
    scaled: bool = False,
) -> tuple[dict, dict]:
    return (
        get_mixtral_8x7b_moe_decoder_attention_manifest(past_len=past_len, schedule_policy=schedule_policy, scaled=scaled),
        get_mixtral_8x7b_moe_decoder_moe_manifest(schedule_policy=schedule_policy, scaled=scaled),
    )


def get_tiny_moe_manifest() -> dict:
    return {
        "manifest_version": "p4-moe-v0.1",
        "manifest_name": "tiny_moe_p4_v0_1",
        "workload_class": "structured_transformer_moe_surrogate",
        "phase": "decode",
        "model_family": "tiny decoder-only transformer MoE slice",
        "num_layers": 1,
        "seq_len": 1,
        "hidden_size": 32,
        "expert_hidden_size": 64,
        "num_experts": 4,
        "top_k": 2,
        "selected_experts": [0, 1],
        "datatype": "int8",
        "schedule_policy": "serialized",
        "operand_movement_policy": {
            "weights": "preloaded_stationary",
            "router_input_setup": "materialized",
            "token_dispatch": "materialized",
            "expert_output_combine": "materialized",
        },
        "ramulator_visible_defaults": {
            "bank_sequence": [0, 1, 2, 3],
            "bank_sequence_order": "frontend",
            "pim_banks_per_mpu": 1,
            "burst_length": 1,
            "row_start": 0,
            "row_count": 16,
            "dependency_count": 8,
            "column_start": 0,
        },
        "mapping_policy": {
            "host_policy": "semantic_tensor_io_only",
            "pim_policy": "native_lpddr5_pim_moe_tiles",
            "bank_sequence_policy": "manifest_order",
            "mpu_grouping_policy": "manifest_pim_banks_per_mpu",
        },
        "literature_anchors": ["LPDDR5-PIM native opcode surface"],
        "non_claims": [
            "not_runtime_replay",
            "not_vllm_replay",
            "not_numerical_correctness",
            "not_silicon_faithful_softmax_or_data_movement",
            "not_raw_attacc_schema",
        ],
    }


def _validate_ramulator_defaults(manifest: dict, *, manifest_kind: str) -> None:
    defaults = manifest["ramulator_visible_defaults"]
    for field in ("bank_sequence", "burst_length", "row_start", "row_count", "dependency_count", "column_start"):
        if field not in defaults:
            raise ValueError(f"{manifest_kind} manifest ramulator_visible_defaults missing {field}")
    if not defaults["bank_sequence"]:
        raise ValueError(f"{manifest_kind} manifest bank_sequence must be non-empty")
    for field in ("burst_length", "row_count", "dependency_count"):
        if int(defaults[field]) <= 0:
            raise ValueError(f"{manifest_kind} manifest ramulator_visible_defaults.{field} must be positive")


def _validate_attention_manifest(manifest: dict) -> None:
    required = {
        "manifest_version",
        "manifest_name",
        "workload_class",
        "phase",
        "model_family",
        "num_layers",
        "num_heads",
        "head_dim",
        "past_len",
        "seq_len",
        "datatype",
        "score_tile_tokens",
        "context_tile_tokens",
        "head_group_size",
        "schedule_policy",
        "ramulator_visible_defaults",
        "mapping_policy",
        "literature_anchors",
        "non_claims",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"Attention manifest missing required fields: {missing}")
    if manifest["phase"] not in {"decode", "prefill"}:
        raise ValueError("Attention generator supports phase='decode' or phase='prefill'")
    if manifest["schedule_policy"] not in {"serialized", "overlap_independent_heads"}:
        raise ValueError("Attention schedule_policy must be 'serialized' or 'overlap_independent_heads'")
    if manifest["phase"] == "prefill" and manifest["schedule_policy"] != "serialized":
        raise ValueError("Prefill attention generator currently supports schedule_policy='serialized' only")
    if manifest["datatype"] not in SUPPORTED_ATTENTION_DATATYPES:
        raise ValueError(
            f"Unsupported attention datatype {manifest['datatype']!r}; supported datatypes are {sorted(SUPPORTED_ATTENTION_DATATYPES)}"
        )
    required_non_claims = REQUIRED_P4_PREFILL_NON_CLAIMS if manifest["phase"] == "prefill" else REQUIRED_P4_NON_CLAIMS
    missing_non_claims = sorted(required_non_claims - set(manifest["non_claims"]))
    if missing_non_claims:
        raise ValueError(f"Attention manifest non_claims missing required entries: {missing_non_claims}")
    for field in ("num_layers", "num_heads", "head_dim", "seq_len", "score_tile_tokens", "context_tile_tokens", "head_group_size"):
        if int(manifest[field]) <= 0:
            raise ValueError(f"Attention manifest {field} must be positive")
    if int(manifest["past_len"]) < 0:
        raise ValueError("Attention manifest past_len must be non-negative")
    _attention_head_group_size(manifest)
    if manifest["schedule_policy"] == "overlap_independent_heads" and int(manifest["num_heads"]) < 2:
        raise ValueError("overlap_independent_heads requires num_heads >= 2")
    if int(manifest["context_tile_tokens"]) != int(manifest["score_tile_tokens"]):
        raise ValueError("P4.2 attention first slice requires context_tile_tokens == score_tile_tokens")
    if manifest["phase"] == "decode" and int(manifest["past_len"]) <= 0:
        raise ValueError("Decode attention manifest past_len must be positive")
    if manifest["phase"] == "prefill":
        if int(manifest.get("prompt_len", -1)) != int(manifest["seq_len"]):
            raise ValueError("Prefill attention manifest requires prompt_len == seq_len")
        if int(manifest["past_len"]) != 0:
            raise ValueError("Prefill attention manifest requires past_len == 0")
        if manifest.get("attention_mode") != "causal_prefill" or manifest.get("prefill_only") is not True:
            raise ValueError("Prefill attention manifest requires attention_mode='causal_prefill' and prefill_only=True")
        if int(manifest["score_tile_tokens"]) > int(manifest["seq_len"]):
            raise ValueError("Prefill attention score_tile_tokens must be <= prompt_len")
    _validate_ramulator_defaults(manifest, manifest_kind="Attention")


def _validate_ffn_manifest(manifest: dict) -> None:
    required = {
        "manifest_version",
        "manifest_name",
        "workload_class",
        "phase",
        "model_family",
        "num_layers",
        "seq_len",
        "hidden_size",
        "ffn_hidden_size",
        "ffn_activation_tile_size",
        "activation_distribution_policy",
        "activation",
        "datatype",
        "schedule_policy",
        "ramulator_visible_defaults",
        "mapping_policy",
        "literature_anchors",
        "non_claims",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"FFN manifest missing required fields: {missing}")
    if manifest["phase"] not in {"decode", "prefill"}:
        raise ValueError("FFN generator supports phase='decode' or phase='prefill'")
    if manifest["schedule_policy"] != "serialized":
        raise ValueError("P4.3 implements serialized FFN scheduling first")
    if manifest["datatype"] not in SUPPORTED_ATTENTION_DATATYPES:
        raise ValueError(f"Unsupported FFN datatype {manifest['datatype']!r}; supported datatypes are {sorted(SUPPORTED_ATTENTION_DATATYPES)}")
    required_non_claims = REQUIRED_P4_PREFILL_NON_CLAIMS if manifest["phase"] == "prefill" else REQUIRED_P4_NON_CLAIMS
    missing_non_claims = sorted(required_non_claims - set(manifest["non_claims"]))
    if missing_non_claims:
        raise ValueError(f"FFN manifest non_claims missing required entries: {missing_non_claims}")
    for field in ("num_layers", "seq_len", "hidden_size", "ffn_hidden_size", "ffn_activation_tile_size"):
        if int(manifest[field]) <= 0:
            raise ValueError(f"FFN manifest {field} must be positive")
    if int(manifest["ffn_activation_tile_size"]) > int(manifest["hidden_size"]):
        raise ValueError("FFN manifest ffn_activation_tile_size must be <= hidden_size")
    if manifest["phase"] == "prefill":
        if int(manifest.get("prompt_len", -1)) != int(manifest["seq_len"]):
            raise ValueError("Prefill FFN manifest requires prompt_len == seq_len")
        if manifest.get("prefill_only") is not True:
            raise ValueError("Prefill FFN manifest requires prefill_only=True")
    if manifest["activation_distribution_policy"] not in _VALID_DISTRIBUTION_POLICIES:
        raise ValueError(
            f"FFN manifest activation_distribution_policy {manifest['activation_distribution_policy']!r} "
            f"not in {sorted(_VALID_DISTRIBUTION_POLICIES)}"
        )
    variant = ffn_variant(manifest)
    if variant not in SUPPORTED_FFN_VARIANTS:
        raise ValueError(f"Unsupported FFN variant {variant!r}; supported variants are {sorted(SUPPORTED_FFN_VARIANTS)}")
    if variant == FFN_VARIANT_RELU_2PROJ and manifest["activation"] != "relu":
        raise ValueError("relu_2proj FFN manifests must use activation='relu'")
    if variant == FFN_VARIANT_SWIGLU_3PROJ and manifest["activation"] == "relu":
        raise ValueError("swiglu_3proj FFN manifests must not use activation='relu'")
    if variant == FFN_VARIANT_GEGLU_3PROJ and manifest["activation"] not in {"gelu", "gelu_pytorch_tanh"}:
        raise ValueError("geglu_3proj FFN manifests must use activation='gelu' or activation='gelu_pytorch_tanh'")
    _validate_ramulator_defaults(manifest, manifest_kind="FFN")


def _validate_moe_manifest(manifest: dict) -> None:
    required = {
        "manifest_version",
        "manifest_name",
        "workload_class",
        "phase",
        "model_family",
        "num_layers",
        "seq_len",
        "hidden_size",
        "expert_hidden_size",
        "num_experts",
        "top_k",
        "selected_experts",
        "datatype",
        "schedule_policy",
        "ramulator_visible_defaults",
        "mapping_policy",
        "literature_anchors",
        "non_claims",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"MoE manifest missing required fields: {missing}")
    if manifest["phase"] != "decode":
        raise ValueError("Tiny MoE generator currently supports phase='decode' only")
    if manifest["schedule_policy"] not in {"serialized", "overlap_selected_experts"}:
        raise ValueError("MoE schedule_policy must be 'serialized' or 'overlap_selected_experts'")
    if manifest["datatype"] not in SUPPORTED_ATTENTION_DATATYPES:
        raise ValueError(f"Unsupported MoE datatype {manifest['datatype']!r}; supported datatypes are {sorted(SUPPORTED_ATTENTION_DATATYPES)}")
    missing_non_claims = sorted(REQUIRED_P4_NON_CLAIMS - set(manifest["non_claims"]))
    if missing_non_claims:
        raise ValueError(f"MoE manifest non_claims missing required entries: {missing_non_claims}")
    for field in ("num_layers", "seq_len", "hidden_size", "expert_hidden_size", "num_experts", "top_k"):
        if int(manifest[field]) <= 0:
            raise ValueError(f"MoE manifest {field} must be positive")
    if manifest["schedule_policy"] == "overlap_selected_experts" and int(manifest["top_k"]) < 2:
        raise ValueError("overlap_selected_experts requires top_k >= 2")
    if int(manifest["top_k"]) > int(manifest["num_experts"]):
        raise ValueError("MoE manifest top_k must be <= num_experts")
    selected = list(manifest["selected_experts"])
    if len(selected) != int(manifest["top_k"]):
        raise ValueError("MoE manifest selected_experts length must equal top_k")
    if len(set(selected)) != len(selected) or any(int(expert) < 0 or int(expert) >= int(manifest["num_experts"]) for expert in selected):
        raise ValueError("MoE manifest selected_experts must be unique ids in [0, num_experts)")
    _validate_ramulator_defaults(manifest, manifest_kind="MoE")


def _lanes(datatype: str) -> int:
    if datatype not in SUPPORTED_ATTENTION_DATATYPES:
        raise ValueError(f"Unsupported attention datatype {datatype!r}; supported datatypes are {sorted(SUPPORTED_ATTENTION_DATATYPES)}")
    return 32 if datatype == "int8" else 16


def _p4_context(operator_family: str, stage: str, **fields: object) -> dict:
    return {
        "record_family": REQUIRED_P4_RECORD_FAMILY,
        "operator_family": operator_family,
        "stage": stage,
        **fields,
    }


def _num_requests(elements: int, datatype: str) -> int:
    """Return MAC requests needed to cover scalar elements at datatype lane width."""
    lanes = _lanes(datatype)
    return max(1, (int(elements) + lanes - 1) // lanes)


def _attention_q_projection_dim(manifest: dict) -> int:
    return int(manifest["num_heads"]) * int(manifest["head_dim"])


def _attention_kv_projection_dim(manifest: dict) -> int:
    return _attention_num_kv_heads(manifest) * int(manifest["head_dim"])


def _mapping_policy(manifest: dict) -> dict:
    defaults = manifest["ramulator_visible_defaults"]
    policy = dict(manifest["mapping_policy"])
    policy.update(
        {
            "controller_bank_order": defaults.get("bank_sequence_order", "frontend"),
            "bank_sequence": list(defaults["bank_sequence"]),
            "pim_banks_per_mpu": int(defaults.get("pim_banks_per_mpu", 1)),
            "burst_length": int(defaults["burst_length"]),
            "row_start": int(defaults["row_start"]),
            "row_count": int(defaults["row_count"]),
        }
    )
    return policy


def _provenance(manifest: dict, op: str) -> dict:
    return {
        "source_kind": "generated",
        "tuple_manifest": manifest["manifest_name"],
        "literature_anchor": list(manifest["literature_anchors"]),
        "generator_version": FULL_TRANSFORMER_GENERATOR_VERSION,
        "schedule_policy": manifest.get("schedule_policy", "serialized"),
        "claim_boundary": [
            "structured transformer dataflow surrogate",
            "simulator-diagnostic",
            "non-silicon-calibrated",
            "operator-internal-dataflow-first",
        ],
        "non_claims": list(manifest["non_claims"]),
        "selected_model_family": manifest["model_family"],
        "selected_workload_regime": manifest["workload_class"],
        "notes": f"bounded Phase 4 semantic tensor-DAG record for {op}",
    }


def _base_record(record_id: str, kind: str, layer_index: int, op: str, manifest: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "record_id": record_id,
        "kind": kind,
        "phase": manifest["phase"],
        "layer": f"layer_{layer_index:02d}",
        "op": op,
        "repeat": 1,
        "provenance": _provenance(manifest, op),
        "mapping_policy": _mapping_policy(manifest),
    }


def _residency(*tensors: str) -> dict:
    return {tensor: "logical_pim_resident_or_accounted" for tensor in tensors}


def _policies(manifest: dict, *, layer_index: int, head_index: int, tile_index: int) -> dict:
    defaults = manifest["ramulator_visible_defaults"]
    row_count = int(defaults["row_count"])
    dependency_count = int(defaults["dependency_count"])
    row_start = int(defaults["row_start"])
    column_start = int(defaults["column_start"])
    dependency_id = (layer_index * int(manifest["num_heads"]) + head_index + tile_index) % dependency_count
    resolved_row = row_start + ((layer_index + head_index + tile_index) % row_count)
    resolved_column = column_start + dependency_id
    return {
        "dependency_context": {
            "kind": "attention_tile_dependency",
            "dependency_count": dependency_count,
            "dependency_id": dependency_id,
            "head_id": head_index,
            "tile_id": tile_index,
        },
        "row_policy": {
            "kind": "bounded_attention_tile_rows",
            "row_start": row_start,
            "row_count": row_count,
            "resolved_row": resolved_row,
        },
        "column_policy": {
            "kind": "attention_dependency_column_round_robin",
            "column_start": column_start,
            "resolved_column": resolved_column,
        },
    }


def _attention_compute_record(
    record_id: str,
    kind: str,
    layer_index: int,
    op: str,
    manifest: dict,
    *,
    head_index: int,
    tile_index: int,
    tile_tokens: int,
    inputs: list[str],
    outputs: list[str],
    dependencies: list[str],
    tile_start: int = 0,
    query_tokens: int | None = None,
    effective_pairs: int | None = None,
    attention_mode: str | None = None,
    key_start: int | None = None,
    query_start: int | None = None,
) -> dict:
    """Build an attention score/context compute record with attention metadata."""
    head_group_size = _attention_head_group_size(manifest)
    record = _base_record(record_id, kind, layer_index, op, manifest)
    policies = _policies(manifest, layer_index=layer_index, head_index=head_index, tile_index=tile_index)
    m = int(manifest["seq_len"]) if query_tokens is None else int(query_tokens)
    n = tile_tokens if kind == "AttentionScore" else int(manifest["head_dim"])
    pairs = tile_tokens if effective_pairs is None else int(effective_pairs)
    context_fields = {
        "layer_id": layer_index,
        "head_id": head_index,
        "head_group_id": head_index // head_group_size,
        "tile_id": tile_index,
        "tile_tokens": tile_tokens,
        "tile_start": tile_start,
    }
    if attention_mode is not None:
        context_fields["attention_mode"] = attention_mode
    if query_start is not None:
        context_fields["query_start"] = query_start
    if key_start is not None:
        context_fields["key_start"] = key_start
    if effective_pairs is not None:
        context_fields["causal_pair_count"] = effective_pairs
    output_tensor_elements = max(1, m * n)
    issued_work_elements = max(1, pairs * int(manifest["head_dim"]))
    record.update(
        {
            "tensor_io": {"inputs": inputs, "outputs": outputs},
            "logical_dependencies": dependencies,
            "operator_context": _p4_context(
                "attention",
                "score" if kind == "AttentionScore" else "context",
                **context_fields,
            ),
            "residency": _residency(*(inputs + outputs)),
            "compute_shape": {
                "m": m,
                "n": n,
                "k": int(manifest["head_dim"]) if kind == "AttentionScore" else tile_tokens,
                "output_elements": max(1, m * n if kind == "AttentionContext" else pairs),
                "output_shape": {"rows": m, "cols": n},
                "output_tensor_elements": output_tensor_elements,
                "valid_attention_pairs": max(1, pairs),
                "issued_work_elements": issued_work_elements,
                "datatype": manifest["datatype"],
            },
            "num_requests": _num_requests(pairs * int(manifest["head_dim"]), manifest["datatype"]),
            "bank_sequence": list(manifest["ramulator_visible_defaults"]["bank_sequence"]),
            "datatype_metadata": {
                "datatype": manifest["datatype"],
                "role": "phase4_attention_tensor_dag_metadata",
                "behavior_claim": "explicit_resource_rows_only",
            },
            "burst_length": int(manifest["ramulator_visible_defaults"]["burst_length"]),
        }
    )
    record.update(policies)
    return record


def _softmax_record(
    record_id: str,
    layer_index: int,
    manifest: dict,
    *,
    head_index: int,
    tile_index: int,
    tile_tokens: int,
    score_record_ids: list[str],
    score_tensors: list[str] | None = None,
    probability_tensors: list[str] | None = None,
    tile_start: int = 0,
) -> dict:
    head_group_size = _attention_head_group_size(manifest)
    score_tensor = f"L{layer_index}.H{head_index}.T{tile_index}.score"
    prob_tensor = f"L{layer_index}.H{head_index}.T{tile_index}.probability"
    score_tensors = [score_tensor] if score_tensors is None else score_tensors
    probability_tensors = [prob_tensor] if probability_tensors is None else probability_tensors
    record = _base_record(record_id, "AttentionSoftmax", layer_index, "attention_softmax_accounting", manifest)
    record.update(
        {
            "tensor_io": {"inputs": list(score_tensors), "outputs": list(probability_tensors)},
            "logical_dependencies": list(score_record_ids),
            "operator_context": _p4_context(
                "attention",
                "global_softmax_accounting",
                layer_id=layer_index,
                head_id=head_index,
                head_group_id=head_index // head_group_size,
                tile_id=tile_index,
                tile_tokens=tile_tokens,
                tile_start=tile_start,
            ),
            "residency": _residency(*(list(score_tensors) + list(probability_tensors))),
            "accounting_metadata": {
                "kind": "semantic_only_softmax",
                "scope": "head_global_across_score_tiles",
                "elements": max(1, int(manifest["seq_len"]) * tile_tokens),
                "lowering": "not_lowered_to_native_opcode_in_p4_2",
            },
        }
    )
    return record


def _data_move_record(
    record_id: str,
    layer_index: int,
    op: str,
    manifest: dict,
    *,
    head_index: int,
    tile_index: int,
    tile_tokens: int,
    inputs: list[str],
    outputs: list[str],
    dependencies: list[str],
    tile_start: int = 0,
) -> dict:
    head_group_size = _attention_head_group_size(manifest)
    record = _base_record(record_id, "PIMDataMove", layer_index, op, manifest)
    policies = _policies(manifest, layer_index=layer_index, head_index=head_index, tile_index=tile_index)
    record.update(
        {
            "tensor_io": {"inputs": inputs, "outputs": outputs},
            "logical_dependencies": dependencies,
            "operator_context": _p4_context(
                "attention",
                "data_movement",
                layer_id=layer_index,
                head_id=head_index,
                head_group_id=head_index // head_group_size,
                tile_id=tile_index,
                tile_tokens=tile_tokens,
                tile_start=tile_start,
            ),
            "residency": _residency(*(inputs + outputs)),
            "movement_policy": {
                "movement_kind": "broadcast_or_accounted_tile_load",
                "lowering_preference": "native_pim_bcast_when_supported",
                "tile_tokens": tile_tokens,
            },
            "num_requests": 1,
            "datatype_metadata": {
                "datatype": manifest["datatype"],
                "role": "phase4_attention_data_movement_metadata",
                "behavior_claim": "semantic_movement_not_silicon_faithful",
            },
        }
    )
    record.update(policies)
    return record


def _host_access_record(
    record_id: str,
    kind: str,
    layer_index: int,
    op: str,
    manifest: dict,
    *,
    head_index: int | None = None,
    address_head_index: int | None = None,
    tile_index: int | None = None,
    tile_tokens: int | None = None,
    inputs: list[str],
    outputs: list[str],
    dependencies: list[str],
    byte_elements: int,
    address_scope: str,
    tile_start: int = 0,
) -> dict:
    if kind not in {"HostRead", "HostWrite"}:
        raise ValueError(f"Unsupported host access kind: {kind}")
    datatype = manifest["datatype"]
    bytes_per_element = 1 if datatype == "int8" else 2
    total_bytes = max(1, int(byte_elements) * bytes_per_element)
    tx_bytes = int(manifest["ramulator_visible_defaults"].get("tx_bytes", 64))
    if tx_bytes <= 0:
        raise ValueError("Host access ramulator_visible_defaults.tx_bytes must be positive")
    context_fields = {"layer_id": layer_index, "address_scope": address_scope}
    if head_index is not None:
        head_group_size = _attention_head_group_size(manifest)
        context_fields["head_id"] = head_index
        context_fields["head_group_id"] = head_index // head_group_size
    if tile_index is not None:
        context_fields["tile_id"] = tile_index
        context_fields["tile_start"] = tile_start
    if tile_tokens is not None:
        context_fields["tile_tokens"] = tile_tokens
    effective_head = address_head_index if address_head_index is not None else head_index
    record = _base_record(record_id, kind, layer_index, op, manifest)
    record.update(
        {
            "tensor_io": {"inputs": inputs, "outputs": outputs},
            "logical_dependencies": dependencies,
            "operator_context": _p4_context("host_kv_cache_accounting", op, **context_fields),
            "residency": _residency(*(inputs + outputs)),
            "bytes": total_bytes,
            "address_policy": {
                "kind": "structured_host_dram_request_stream",
                "scope": address_scope,
                "lowering": "structured_replay_regular_dram_request",
                "base_byte": _host_access_base_byte(
                    layer_index=layer_index,
                    head_index=effective_head,
                    tile_index=tile_index,
                    address_scope=address_scope,
                    tile_start=tile_start,
                    tx_bytes=tx_bytes,
                ),
                "stride_bytes": tx_bytes,
                "count": max(1, (total_bytes + tx_bytes - 1) // tx_bytes),
            },
        }
    )
    return record


def _host_access_base_byte(
    *,
    layer_index: int,
    head_index: int | None,
    tile_index: int | None,
    address_scope: str,
    tile_start: int,
    tx_bytes: int,
) -> int:
    scope_offsets = {
        "kv_cache_k_append": 0,
        "kv_cache_v_append": 1,
        "kv_cache_k_tile": 2,
        "kv_cache_v_tile": 3,
        "kv_cache_k_prefill_population": 4,
        "kv_cache_v_prefill_population": 5,
    }
    scope_offset = scope_offsets.get(address_scope, 6)
    head_offset = 0 if head_index is None else int(head_index)
    tile_offset = 0 if tile_index is None else int(tile_index)
    request_index = (
        int(layer_index) * 1_000_000
        + scope_offset * 100_000
        + head_offset * 1_000
        + tile_offset * 10
        + max(0, int(tile_start))
    )
    return request_index * tx_bytes


def _barrier_record(record_id: str, layer_index: int, manifest: dict, *, op: str = "layer_transition") -> dict:
    if op not in {"layer_start", "layer_transition"}:
        raise ValueError(f"Unsupported decode-block barrier op: {op}")
    record = _base_record(record_id, "Barrier", layer_index, op, manifest)
    record.update(
        {
            "barrier_scope": {
                "kind": op,
                "layer_id": layer_index,
                "lowering": "semantic_ordering_only",
            }
        }
    )
    return record


def _drain_record(record_id: str, layer_index: int, manifest: dict) -> dict:
    record = _base_record(record_id, "Drain", layer_index, "final_drain", manifest)
    record.update(
        {
            "drain_scope": {
                "kind": "final_drain",
                "layer_id": layer_index,
                "lowering": "semantic_ordering_only",
            }
        }
    )
    return record


def _semantic_operand_record(
    record_id: str,
    kind: str,
    layer_index: int,
    op: str,
    manifest: dict,
    *,
    operator_family: str,
    stage: str,
    inputs: list[str],
    outputs: list[str],
    dependencies: list[str],
    operand_role: str,
    residency: str,
    materialized: bool,
    reuse_scope: str,
    lowering: str,
    context_fields: dict | None = None,
) -> dict:
    if kind not in {"PIMOperandResidency", "PIMOperandReuse"}:
        raise ValueError(f"Unsupported semantic operand record kind: {kind}")
    context_fields = {} if context_fields is None else dict(context_fields)
    record = _base_record(record_id, kind, layer_index, op, manifest)
    record.update(
        {
            "tensor_io": {"inputs": inputs, "outputs": outputs},
            "logical_dependencies": dependencies,
            "operator_context": _p4_context(
                operator_family,
                stage,
                layer_id=layer_index,
                **context_fields,
            ),
            "residency": _residency(*(inputs + outputs)),
            "accounting_metadata": {
                "kind": "semantic_only_operand_residency" if kind == "PIMOperandResidency" else "semantic_only_operand_reuse",
                "operand_role": operand_role,
                "residency": residency,
                "materialized": materialized,
                "reuse_scope": reuse_scope,
                "lowering": lowering,
            },
        }
    )
    return record


def _ffn_data_move_record(
    record_id: str,
    layer_index: int,
    manifest: dict,
    *,
    op: str,
    stage_index: int,
    inputs: list[str],
    outputs: list[str],
    dependencies: list[str],
    operand_role: str,
    residency: str,
    reuse_scope: str,
    tile_index: int = 0,
    tile_elements: int = 0,
    tile_start: int = 0,
    movement_elements: int = 0,
) -> dict:
    record = _base_record(record_id, "PIMDataMove", layer_index, op, manifest)
    distribution_policy = manifest.get("activation_distribution_policy", "broadcast")
    movement_num_requests = _num_requests(movement_elements, manifest["datatype"]) if movement_elements > 0 else 1
    is_weight = operand_role == "weight"
    movement_behavior_claim = (
        "semantic_movement_volume_proportional_not_tiled_or_silicon_faithful"
        if is_weight and movement_elements > 0
        else "semantic_movement_not_silicon_faithful"
    )
    lowering = "semantic_only_steady_state_or_host_write_preload" if is_weight else "native_pim_bcast_when_supported"
    movement_policy = {
        "movement_kind": "preloaded_stationary_weight_residency" if is_weight else "broadcast_or_accounted_tile_load",
        "lowering_preference": lowering,
        "operand_role": operand_role,
        "residency": residency,
        "materialized": True,
        "reuse_scope": reuse_scope,
        "lowering": lowering,
        "tile_index": tile_index,
        "tile_elements": tile_elements,
        "tile_start": tile_start,
        "distribution_scope": "bank_local_preloaded" if is_weight else distribution_policy,
    }
    if movement_elements > 0:
        movement_policy["movement_elements"] = int(movement_elements)
    if is_weight:
        movement_policy["materialization_lowering"] = "regular_host_write_preload_when_materialized"
    record.update(
        {
            "tensor_io": {"inputs": inputs, "outputs": outputs},
            "logical_dependencies": dependencies,
            "operator_context": _p4_context(
                _ffn_operator_family(manifest),
                "data_movement",
                layer_id=layer_index,
                stage_index=stage_index,
                tile_id=tile_index,
                tile_elements=tile_elements,
                tile_start=tile_start,
            ),
            "residency": _residency(*(inputs + outputs)),
            "movement_policy": movement_policy,
            "num_requests": movement_num_requests,
            "datatype_metadata": {
                "datatype": manifest["datatype"],
                "role": "phase4_ffn_operand_movement_metadata",
                "behavior_claim": movement_behavior_claim,
            },
        }
    )
    record.update(_ffn_policies(manifest, layer_index=layer_index, stage_index=stage_index, tile_index=tile_index))
    return record


def _moe_data_move_record(
    record_id: str,
    layer_index: int,
    manifest: dict,
    *,
    op: str,
    stage_index: int,
    inputs: list[str],
    outputs: list[str],
    dependencies: list[str],
    operand_role: str,
    residency: str,
    reuse_scope: str,
    expert_id: int | None = None,
    movement_elements: int = 0,
) -> dict:
    record = _base_record(record_id, "PIMDataMove", layer_index, op, manifest)
    movement_num_requests = _num_requests(movement_elements, manifest["datatype"]) if movement_elements > 0 else 1
    is_weight = operand_role == "weight"
    movement_behavior_claim = (
        "semantic_movement_volume_proportional_not_tiled_or_silicon_faithful"
        if is_weight and movement_elements > 0
        else "semantic_movement_not_silicon_faithful"
    )
    lowering = "semantic_only_steady_state_or_host_write_preload" if is_weight else "native_pim_bcast_when_supported"
    movement_policy = {
        "movement_kind": "preloaded_stationary_weight_residency" if is_weight else "broadcast_or_accounted_tile_load",
        "lowering_preference": lowering,
        "operand_role": operand_role,
        "residency": residency,
        "materialized": True,
        "reuse_scope": reuse_scope,
        "lowering": lowering,
    }
    if movement_elements > 0:
        movement_policy["movement_elements"] = int(movement_elements)
    if is_weight:
        movement_policy["materialization_lowering"] = "regular_host_write_preload_when_materialized"
    record.update(
        {
            "tensor_io": {"inputs": inputs, "outputs": outputs},
            "logical_dependencies": dependencies,
            "operator_context": _p4_context(
                "moe",
                "data_movement",
                layer_id=layer_index,
                stage_index=stage_index,
                expert_id=expert_id,
            ),
            "residency": _residency(*(inputs + outputs)),
            "movement_policy": movement_policy,
            "num_requests": movement_num_requests,
            "datatype_metadata": {
                "datatype": manifest["datatype"],
                "role": "phase4_moe_operand_movement_metadata",
                "behavior_claim": movement_behavior_claim,
            },
        }
    )
    record.update(_moe_policies(manifest, layer_index=layer_index, stage_index=stage_index, expert_id=expert_id))
    return record


def _context_reduction_record(
    record_id: str,
    layer_index: int,
    manifest: dict,
    *,
    head_index: int,
    context_record_ids: list[str],
    context_tensors: list[str],
    output_tensor: str | None = None,
    context_fields: dict | None = None,
    query_tokens: int | None = None,
) -> dict:
    head_group_size = _attention_head_group_size(manifest)
    output_tensor = f"L{layer_index}.H{head_index}.context_reduced" if output_tensor is None else output_tensor
    extra_context = {} if context_fields is None else dict(context_fields)
    record = _base_record(record_id, "PIMElementwise", layer_index, "attention_context_reduction_accounting", manifest)
    record.update(
        {
            "tensor_io": {"inputs": list(context_tensors), "outputs": [output_tensor]},
            "logical_dependencies": list(context_record_ids),
            "operator_context": _p4_context(
                "attention",
                "context_reduction_accounting",
                layer_id=layer_index,
                head_id=head_index,
                head_group_id=head_index // head_group_size,
                tile_count=len(context_tensors),
                **extra_context,
            ),
            "residency": _residency(*(list(context_tensors) + [output_tensor])),
            "accounting_metadata": {
                "kind": "semantic_only_context_tile_reduction",
                "scope": "head_global_across_context_tiles",
                "elements": max(1, (query_tokens if query_tokens is not None else int(manifest["seq_len"])) * int(manifest["head_dim"])),
                "lowering": "not_lowered_to_native_opcode_in_p4_2",
            },
        }
    )
    return record


def _terminal_attention_record_ids(records: list[dict]) -> list[str]:
    """Return per-head/per-query-tile terminal attention records for O projection.

    Reduction records supersede the context records they consume.  Context
    records that are not inputs to any reduction remain terminal.  This keeps
    decode behavior unchanged while preserving every prefill query-tile output.
    """
    reduced_context_ids = {
        dependency
        for record in records
        if record.get("kind") == "PIMElementwise" and record.get("op") == "attention_context_reduction_accounting"
        for dependency in record.get("logical_dependencies", [])
    }
    terminals = [
        record["record_id"]
        for record in records
        if record.get("kind") == "PIMElementwise" and record.get("op") == "attention_context_reduction_accounting"
    ]
    terminals.extend(
        record["record_id"]
        for record in records
        if record.get("kind") == "AttentionContext" and record["record_id"] not in reduced_context_ids
    )
    return terminals


def _ffn_policies(manifest: dict, *, layer_index: int, stage_index: int, tile_index: int = 0) -> dict:
    defaults = manifest["ramulator_visible_defaults"]
    row_count = int(defaults["row_count"])
    dependency_count = int(defaults["dependency_count"])
    row_start = int(defaults["row_start"])
    column_start = int(defaults["column_start"])
    dependency_id = (layer_index + stage_index + tile_index) % dependency_count
    resolved_row = row_start + ((layer_index + stage_index + tile_index) % row_count)
    resolved_column = column_start + dependency_id
    return {
        "dependency_context": {
            "kind": "ffn_stage_dependency",
            "dependency_count": dependency_count,
            "dependency_id": dependency_id,
            "stage_index": stage_index,
            "tile_index": tile_index,
        },
        "row_policy": {
            "kind": "bounded_ffn_stage_rows",
            "row_start": row_start,
            "row_count": row_count,
            "resolved_row": resolved_row,
        },
        "column_policy": {
            "kind": "ffn_dependency_column_round_robin",
            "column_start": column_start,
            "resolved_column": resolved_column,
        },
    }


def _ffn_projection_record(
    record_id: str,
    layer_index: int,
    manifest: dict,
    *,
    op: str,
    stage_index: int,
    inputs: list[str],
    outputs: list[str],
    dependencies: list[str],
    projection_input_dim: int | None = None,
    projection_output_dim: int | None = None,
) -> dict:
    hidden_size = int(manifest["hidden_size"])
    ffn_hidden_size = int(manifest["ffn_hidden_size"])
    seq_len = int(manifest["seq_len"])
    if op in {"ffn_up_projection", "ffn_gate_projection"}:
        n = ffn_hidden_size
        k = hidden_size
    elif op == "ffn_down_projection":
        n = hidden_size
        k = ffn_hidden_size
    elif op == "q_projection":
        n = projection_output_dim if projection_output_dim is not None else hidden_size
        k = hidden_size
    elif op == "o_projection":
        n = hidden_size
        k = projection_input_dim if projection_input_dim is not None else hidden_size
    elif op in {"k_projection", "v_projection"}:
        n = projection_output_dim if projection_output_dim is not None else hidden_size
        k = hidden_size
    else:
        raise ValueError(f"Unsupported projection op: {op}")
    operator_family = "dense_qkvo_projection" if op in {"q_projection", "k_projection", "v_projection", "o_projection"} else _ffn_operator_family(manifest)

    record = _base_record(record_id, "FFNProjection", layer_index, op, manifest)
    record.update(
        {
            "tensor_io": {"inputs": inputs, "outputs": outputs},
            "logical_dependencies": dependencies,
            "operator_context": _p4_context(
                operator_family,
                op,
                layer_id=layer_index,
                stage_index=stage_index,
            ),
            "residency": _residency(*(inputs + outputs)),
            "compute_shape": {
                "m": seq_len,
                "n": n,
                "k": k,
                "output_elements": max(1, seq_len * n),
                "datatype": manifest["datatype"],
            },
            "num_requests": _num_requests(seq_len * n * k, manifest["datatype"]),
            "bank_sequence": list(manifest["ramulator_visible_defaults"]["bank_sequence"]),
            "datatype_metadata": {
                "datatype": manifest["datatype"],
                "role": "phase4_ffn_tensor_dag_metadata",
                "behavior_claim": "explicit_resource_rows_only",
            },
            "burst_length": int(manifest["ramulator_visible_defaults"]["burst_length"]),
        }
    )
    record.update(_ffn_policies(manifest, layer_index=layer_index, stage_index=stage_index))
    return record


def _ffn_elementwise_record(
    record_id: str,
    layer_index: int,
    manifest: dict,
    *,
    op: str,
    stage_index: int,
    inputs: list[str],
    outputs: list[str],
    dependencies: list[str],
    elementwise_kind: str,
) -> dict:
    if "expert_hidden_size" in manifest and "num_experts" in manifest:
        # MoE experts can coexist with dense-FFN compatibility fields; elementwise
        # accounting must follow the active expert width, not a dense FFN width.
        intermediate_size = int(manifest["expert_hidden_size"])
    else:
        intermediate_size = int(manifest["ffn_hidden_size"])
    record = _base_record(record_id, "PIMElementwise", layer_index, op, manifest)
    record.update(
        {
            "tensor_io": {"inputs": inputs, "outputs": outputs},
            "logical_dependencies": dependencies,
            "operator_context": _p4_context(
                _ffn_operator_family(manifest),
                op,
                layer_id=layer_index,
                stage_index=stage_index,
            ),
            "residency": _residency(*(inputs + outputs)),
            "accounting_metadata": {
                "kind": elementwise_kind,
                "activation": manifest.get("activation", "silu") if "activation" in elementwise_kind else None,
                "elements": max(
                    1,
                    int(manifest["seq_len"]) * intermediate_size,
                ),
                "lowering": "not_lowered_to_native_opcode_in_p4_3",
            },
        }
    )
    return record


def _moe_policies(manifest: dict, *, layer_index: int, stage_index: int, expert_id: int | None = None) -> dict:
    assert expert_id is None or expert_id >= 0
    defaults = manifest["ramulator_visible_defaults"]
    row_count = int(defaults["row_count"])
    dependency_count = int(defaults["dependency_count"])
    row_start = int(defaults["row_start"])
    column_start = int(defaults["column_start"])
    expert_offset = int(manifest["num_experts"]) if expert_id is None else expert_id
    dependency_id = (layer_index + stage_index + expert_offset) % dependency_count
    resolved_row = row_start + ((layer_index + stage_index + expert_offset) % row_count)
    resolved_column = column_start + dependency_id
    return {
        "dependency_context": {
            "kind": "moe_stage_dependency",
            "dependency_count": dependency_count,
            "dependency_id": dependency_id,
            "stage_index": stage_index,
            "expert_id": expert_id,
            "moe_role": "router" if expert_id is None else "expert",
        },
        "row_policy": {
            "kind": "bounded_moe_stage_rows",
            "row_start": row_start,
            "row_count": row_count,
            "resolved_row": resolved_row,
        },
        "column_policy": {
            "kind": "moe_dependency_column_round_robin",
            "column_start": column_start,
            "resolved_column": resolved_column,
        },
    }


def _moe_compute_record(
    record_id: str,
    kind: str,
    layer_index: int,
    manifest: dict,
    *,
    op: str,
    stage_index: int,
    inputs: list[str],
    outputs: list[str],
    dependencies: list[str],
    expert_id: int | None = None,
    projection: str | None = None,
) -> dict:
    hidden_size = int(manifest["hidden_size"])
    expert_hidden_size = int(manifest["expert_hidden_size"])
    seq_len = int(manifest["seq_len"])
    if kind == "MoERouter":
        n = int(manifest["num_experts"])
        k = hidden_size
    elif kind == "MoEExpertFFN":
        if projection == "down":
            n = hidden_size
            k = expert_hidden_size
        else:
            n = expert_hidden_size
            k = hidden_size
    else:
        raise ValueError(f"Unsupported MoE compute kind: {kind}")

    record = _base_record(record_id, kind, layer_index, op, manifest)
    record.update(
        {
            "tensor_io": {"inputs": inputs, "outputs": outputs},
            "logical_dependencies": dependencies,
            "operator_context": _p4_context(
                "moe",
                op,
                layer_id=layer_index,
                stage_index=stage_index,
                expert_id=expert_id,
                ffn_projection=projection,
            ),
            "residency": _residency(*(inputs + outputs)),
            "compute_shape": {
                "m": seq_len,
                "n": n,
                "k": k,
                "output_elements": max(1, seq_len * n),
                "datatype": manifest["datatype"],
            },
            "num_requests": _num_requests(seq_len * n * k, manifest["datatype"]),
            "bank_sequence": list(manifest["ramulator_visible_defaults"]["bank_sequence"]),
            "datatype_metadata": {
                "datatype": manifest["datatype"],
                "role": "phase4_moe_tensor_dag_metadata",
                "behavior_claim": "explicit_resource_rows_only",
            },
            "burst_length": int(manifest["ramulator_visible_defaults"]["burst_length"]),
        }
    )
    record.update(_moe_policies(manifest, layer_index=layer_index, stage_index=stage_index, expert_id=expert_id))
    return record


def _moe_accounting_record(
    record_id: str,
    kind: str,
    layer_index: int,
    manifest: dict,
    *,
    op: str,
    stage_index: int,
    inputs: list[str],
    outputs: list[str],
    dependencies: list[str],
    accounting_kind: str,
) -> dict:
    record = _base_record(record_id, kind, layer_index, op, manifest)
    selected_experts = [int(expert) for expert in manifest["selected_experts"]]
    record.update(
        {
            "tensor_io": {"inputs": inputs, "outputs": outputs},
            "logical_dependencies": dependencies,
            "operator_context": _p4_context(
                "moe",
                op,
                layer_id=layer_index,
                stage_index=stage_index,
                selected_experts=selected_experts,
                top_k=int(manifest["top_k"]),
            ),
            "residency": _residency(*(inputs + outputs)),
            "accounting_metadata": {
                "kind": accounting_kind,
                "selected_experts": selected_experts,
                "top_k": int(manifest["top_k"]),
                "elements": max(1, int(manifest["seq_len"]) * max(1, len(outputs))),
                "lowering": "not_lowered_to_native_opcode_in_p4_4",
            },
        }
    )
    return record


def _tile_ranges(total_tokens: int, tile_tokens: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < total_tokens:
        size = min(tile_tokens, total_tokens - start)
        ranges.append((start, size))
        start += size
    return ranges


def _causal_pair_count(query_start: int, query_tokens: int, key_start: int, key_tokens: int) -> int:
    """Count legal causal (q,k) pairs for tiled prefill with inclusive k <= q."""
    count = 0
    key_end = key_start + key_tokens
    for q in range(query_start, query_start + query_tokens):
        count += max(0, min(key_end, q + 1) - key_start)
    return count


def _add_dag_hint(records: list[dict], hint: str) -> None:
    for record in records:
        record["operator_context"]["dag_hint"] = hint


def _append_attention_score_records(
    records: list[dict],
    next_id: int,
    manifest: dict,
    *,
    layer_index: int,
    head_index: int,
    head_tiles: list[tuple[int, int]],
    score_record_ids: list[str],
) -> int:
    head_group_size = _attention_head_group_size(manifest)
    kv_head = head_index // head_group_size
    for tile_index, (tile_start, tile_tokens) in enumerate(head_tiles):
        q_tensor = f"L{layer_index}.H{head_index}.Q"
        k_tensor = f"L{layer_index}.H{head_index}.T{tile_index}.K"
        score_tensor = f"L{layer_index}.H{head_index}.T{tile_index}.score"
        k_load_id = f"rec_{next_id:04d}"
        records.append(
            _host_access_record(
                k_load_id,
                "HostRead",
                layer_index,
                "kv_cache_k_tile_read",
                manifest,
                head_index=head_index,
                address_head_index=kv_head,
                tile_index=tile_index,
                tile_tokens=tile_tokens,
                inputs=[k_tensor],
                outputs=[f"{k_tensor}.resident"],
                dependencies=[],
                byte_elements=tile_tokens * int(manifest["head_dim"]),
                address_scope="kv_cache_k_tile",
                tile_start=tile_start,
            )
        )
        next_id += 1

        score_id = f"rec_{next_id:04d}"
        records.append(
            _attention_compute_record(
                score_id,
                "AttentionScore",
                layer_index,
                "attention_score_gemv",
                manifest,
                head_index=head_index,
                tile_index=tile_index,
                tile_tokens=tile_tokens,
                inputs=[q_tensor, f"{k_tensor}.resident"],
                outputs=[score_tensor],
                dependencies=[k_load_id],
                tile_start=tile_start,
            )
        )
        score_record_ids.append(score_id)
        next_id += 1
    return next_id


def _emit_q_operand_residency_per_head(
    records: list[dict],
    next_id: int,
    manifest: dict,
    *,
    layer_index: int,
    head_index: int,
) -> tuple[int, str, str, str]:
    """Emit Q PIMOperandResidency before score tiles, return (next_id, residency_id, resident_tensor)."""
    q_tensor = f"L{layer_index}.H{head_index}.Q"
    q_resident = f"{q_tensor}.pim_visible"
    q_residency_id = f"rec_{next_id:04d}"
    records.append(
        _semantic_operand_record(
            q_residency_id,
            "PIMOperandResidency",
            layer_index,
            "q_operand_residency",
            manifest,
            operator_family="attention",
            stage="operand_residency",
            inputs=[q_tensor],
            outputs=[q_resident],
            dependencies=[],
            operand_role="activation_input",
            residency="pim_visible_operand",
            materialized=False,
            reuse_scope="attention_score",
            lowering="semantic_only_operand_residency",
            context_fields={"head_index": head_index, "distribution_policy": "producer_aligned"},
        )
    )
    return next_id + 1, q_residency_id, q_tensor, q_resident


def _append_attention_softmax_context_records(
    records: list[dict],
    next_id: int,
    manifest: dict,
    *,
    layer_index: int,
    head_index: int,
    head_tiles: list[tuple[int, int]],
    score_record_ids: list[str],
) -> int:
    head_group_size = _attention_head_group_size(manifest)
    kv_head = head_index // head_group_size
    softmax_id = f"rec_{next_id:04d}"
    score_tensors = [f"L{layer_index}.H{head_index}.T{tile_index}.score" for tile_index, _ in enumerate(head_tiles)]
    probability_tensors = [f"L{layer_index}.H{head_index}.T{tile_index}.probability" for tile_index, _ in enumerate(head_tiles)]
    records.append(
        _softmax_record(
            softmax_id,
            layer_index,
            manifest,
            head_index=head_index,
            # Softmax is head-global: tile_index=0 and tile_tokens=past_len encode the full head span.
            tile_index=0,
            tile_tokens=int(manifest["past_len"]),
            score_record_ids=score_record_ids,
            score_tensors=score_tensors,
            probability_tensors=probability_tensors,
            tile_start=0,
        )
    )
    next_id += 1

    context_record_ids: list[str] = []
    context_tensors: list[str] = []
    for tile_index, (tile_start, tile_tokens) in enumerate(head_tiles):
        prob_tensor = f"L{layer_index}.H{head_index}.T{tile_index}.probability"
        v_tensor = f"L{layer_index}.H{head_index}.T{tile_index}.V"
        context_tensor = f"L{layer_index}.H{head_index}.T{tile_index}.context"

        # P4.22 fix: probability operand residency
        prob_residency_id = f"rec_{next_id:04d}"
        prob_resident = f"{prob_tensor}.pim_visible"
        records.append(
            _semantic_operand_record(
                prob_residency_id,
                "PIMOperandResidency",
                layer_index,
                "attention_probability_residency",
                manifest,
                operator_family="attention",
                stage="operand_residency",
                inputs=[prob_tensor],
                outputs=[prob_resident],
                dependencies=[softmax_id],
                operand_role="activation_input",
                residency="pim_visible_operand",
                materialized=False,
                reuse_scope="attention_context",
                lowering="semantic_only_operand_residency",
                context_fields={
                    "head_index": head_index,
                    "tile_index": tile_index,
                    "distribution_policy": "producer_aligned",
                },
            )
        )
        next_id += 1

        v_load_id = f"rec_{next_id:04d}"
        records.append(
            _host_access_record(
                v_load_id,
                "HostRead",
                layer_index,
                "kv_cache_v_tile_read",
                manifest,
                head_index=head_index,
                address_head_index=kv_head,
                tile_index=tile_index,
                tile_tokens=tile_tokens,
                inputs=[v_tensor],
                outputs=[f"{v_tensor}.resident"],
                dependencies=[softmax_id],
                byte_elements=tile_tokens * int(manifest["head_dim"]),
                address_scope="kv_cache_v_tile",
                tile_start=tile_start,
            )
        )
        next_id += 1

        context_id = f"rec_{next_id:04d}"
        records.append(
            _attention_compute_record(
                context_id,
                "AttentionContext",
                layer_index,
                "attention_context_gemv",
                manifest,
                head_index=head_index,
                tile_index=tile_index,
                tile_tokens=tile_tokens,
                inputs=[prob_resident, f"{v_tensor}.resident"],
                outputs=[context_tensor],
                dependencies=[prob_residency_id, softmax_id, v_load_id],
                tile_start=tile_start,
            )
        )
        context_record_ids.append(context_id)
        context_tensors.append(context_tensor)
        next_id += 1

    if len(context_record_ids) > 1:
        records.append(
            _context_reduction_record(
                f"rec_{next_id:04d}",
                layer_index,
                manifest,
                head_index=head_index,
                context_record_ids=context_record_ids,
                context_tensors=context_tensors,
            )
        )
        next_id += 1
    return next_id


def _generate_attention_records_overlapped_heads(manifest: dict) -> list[dict]:
    records: list[dict] = []
    next_id = 0
    for layer_index in range(int(manifest["num_layers"])):
        head_tiles = list(_tile_ranges(int(manifest["past_len"]), int(manifest["score_tile_tokens"])))
        score_ids_by_head: dict[int, list[str]] = {}
        for head_index in range(int(manifest["num_heads"])):
            score_ids_by_head[head_index] = []
            # P4.22 fix: emit Q residency BEFORE scores so dependency precedes consumer
            next_id, q_residency_id, q_tensor, q_resident = _emit_q_operand_residency_per_head(
                records, next_id, manifest,
                layer_index=layer_index, head_index=head_index,
            )
            next_id = _append_attention_score_records(
                records,
                next_id,
                manifest,
                layer_index=layer_index,
                head_index=head_index,
                head_tiles=head_tiles,
                score_record_ids=score_ids_by_head[head_index],
            )
            # Update score records to consume resident Q and depend on Q residency
            for record in records:
                if record.get("record_id") in score_ids_by_head[head_index]:
                    old_inputs = record["tensor_io"]["inputs"]
                    record["tensor_io"]["inputs"] = [q_resident if inp == q_tensor else inp for inp in old_inputs]
                    if q_residency_id not in record["logical_dependencies"]:
                        record["logical_dependencies"] = [q_residency_id] + record["logical_dependencies"]

        for head_index in range(int(manifest["num_heads"])):
            next_id = _append_attention_softmax_context_records(
                records,
                next_id,
                manifest,
                layer_index=layer_index,
                head_index=head_index,
                head_tiles=head_tiles,
                score_record_ids=score_ids_by_head[head_index],
            )
    _add_dag_hint(records, "heads_are_independent_safe_to_parallelize")
    return records


def generate_attention_records(manifest: dict | None = None) -> list[dict]:
    """Generate attention records.

    ``overlap_independent_heads`` reorders the flat offline IR so independent
    head score phases are adjacent before head-local softmax/context phases.
    It does not add runtime parallelism constructs to the sequential concrete
    opcode stream; ``operator_context.dag_hint`` marks future-safe DAG parallelism.
    """
    manifest = get_tiny_attention_manifest() if manifest is None else manifest
    _validate_attention_manifest(manifest)
    if manifest["phase"] == "prefill":
        return generate_prefill_attention_records(manifest)

    if manifest["schedule_policy"] == "overlap_independent_heads":
        records = _generate_attention_records_overlapped_heads(manifest)
        for record in records:
            validate_record(record)
        return records

    records: list[dict] = []
    next_id = 0
    for layer_index in range(int(manifest["num_layers"])):
        for head_index in range(int(manifest["num_heads"])):
            head_tiles = list(_tile_ranges(int(manifest["past_len"]), int(manifest["score_tile_tokens"])))
            score_record_ids: list[str] = []
            # P4.22 fix: emit Q residency BEFORE scores so dependency precedes consumer
            next_id, q_residency_id, q_tensor, q_resident = _emit_q_operand_residency_per_head(
                records, next_id, manifest,
                layer_index=layer_index, head_index=head_index,
            )
            next_id = _append_attention_score_records(
                records,
                next_id,
                manifest,
                layer_index=layer_index,
                head_index=head_index,
                head_tiles=head_tiles,
                score_record_ids=score_record_ids,
            )
            # Update score records to consume resident Q and depend on Q residency
            for record in records:
                if record.get("record_id") in score_record_ids:
                    old_inputs = record["tensor_io"]["inputs"]
                    record["tensor_io"]["inputs"] = [q_resident if inp == q_tensor else inp for inp in old_inputs]
                    if q_residency_id not in record["logical_dependencies"]:
                        record["logical_dependencies"] = [q_residency_id] + record["logical_dependencies"]
            next_id = _append_attention_softmax_context_records(
                records,
                next_id,
                manifest,
                layer_index=layer_index,
                head_index=head_index,
                head_tiles=head_tiles,
                score_record_ids=score_record_ids,
            )

    for record in records:
        validate_record(record)
    return records


def generate_prefill_attention_records(manifest: dict | None = None) -> list[dict]:
    """Generate causal-prefill attention records without KV-cache HostRead records."""
    if manifest is None:
        manifest = get_llama2_dense_prefill_attention_manifest(LLAMA2_7B_MODEL_SPEC, prompt_len=1024)
    _validate_attention_manifest(manifest)
    if manifest["phase"] != "prefill":
        raise ValueError("generate_prefill_attention_records requires phase='prefill'")

    records: list[dict] = []
    next_id = 0
    prompt_len = int(manifest["seq_len"])
    q_ranges = _tile_ranges(prompt_len, int(manifest["score_tile_tokens"]))
    kv_ranges = _tile_ranges(prompt_len, int(manifest["context_tile_tokens"]))
    expected_pairs = prompt_len * (prompt_len + 1) // 2

    # Prefill K/V operands are produced by the same layer's K/V projections and
    # made PIM-visible by semantic PIMOperandResidency records.  The trace still
    # emits HostWrite records to populate the persistent KV cache for later
    # decode, but it intentionally emits no HostRead KV-cache tile traffic during
    # prefill attention.

    for layer_index in range(int(manifest["num_layers"])):
        for head_index in range(int(manifest["num_heads"])):
            observed_pairs = 0
            for q_tile_index, (q_start, q_tokens) in enumerate(q_ranges):
                q_tensor = f"L{layer_index}.H{head_index}.Q.T{q_tile_index}"
                q_resident = f"{q_tensor}.pim_visible"
                q_residency_id = f"rec_{next_id:04d}"
                records.append(
                    _semantic_operand_record(
                        q_residency_id,
                        "PIMOperandResidency",
                        layer_index,
                        "q_operand_residency",
                        manifest,
                        operator_family="attention",
                        stage="operand_residency",
                        inputs=[q_tensor],
                        outputs=[q_resident],
                        dependencies=[],
                        operand_role="activation_input",
                        residency="pim_visible_operand",
                        materialized=False,
                        reuse_scope="prefill_attention_score",
                        lowering="semantic_only_operand_residency",
                        context_fields={"head_index": head_index, "query_tile_index": q_tile_index, "query_start": q_start},
                    )
                )
                next_id += 1

                score_ids: list[str] = []
                score_tensors: list[str] = []
                legal_k_tiles: list[tuple[int, int, int, int]] = []
                for k_tile_index, (k_start, k_tokens) in enumerate(kv_ranges):
                    pair_count = _causal_pair_count(q_start, q_tokens, k_start, k_tokens)
                    if pair_count <= 0:
                        continue
                    observed_pairs += pair_count
                    legal_k_tiles.append((k_tile_index, k_start, k_tokens, pair_count))
                    k_tensor = f"L{layer_index}.H{head_index}.K.T{k_tile_index}"
                    k_resident = f"{k_tensor}.pim_visible"
                    k_residency_id = f"rec_{next_id:04d}"
                    records.append(
                        _semantic_operand_record(
                            k_residency_id,
                            "PIMOperandResidency",
                            layer_index,
                            "k_prefill_tile_residency",
                            manifest,
                            operator_family="attention",
                            stage="operand_residency",
                            inputs=[k_tensor],
                            outputs=[k_resident],
                            dependencies=[],
                            operand_role="activation_input",
                            residency="pim_visible_operand",
                            materialized=False,
                            reuse_scope="prefill_attention_score",
                            lowering="semantic_only_operand_residency",
                            context_fields={"head_index": head_index, "tile_index": k_tile_index, "tile_start": k_start},
                        )
                    )
                    next_id += 1
                    score_id = f"rec_{next_id:04d}"
                    score_tensor = f"L{layer_index}.H{head_index}.Q{q_tile_index}.K{k_tile_index}.score"
                    records.append(
                        _attention_compute_record(
                            score_id,
                            "AttentionScore",
                            layer_index,
                            "attention_score_gemm",
                            manifest,
                            head_index=head_index,
                            tile_index=k_tile_index,
                            tile_tokens=k_tokens,
                            inputs=[q_resident, k_resident],
                            outputs=[score_tensor],
                            dependencies=[q_residency_id, k_residency_id],
                            tile_start=k_start,
                            query_tokens=q_tokens,
                            effective_pairs=pair_count,
                            attention_mode="causal_prefill",
                            key_start=k_start,
                            query_start=q_start,
                        )
                    )
                    score_ids.append(score_id)
                    score_tensors.append(score_tensor)
                    next_id += 1

                softmax_id = f"rec_{next_id:04d}"
                prob_tensors = [tensor.replace(".score", ".probability") for tensor in score_tensors]
                records.append(
                    _softmax_record(
                        softmax_id,
                        layer_index,
                        manifest,
                        head_index=head_index,
                        tile_index=q_tile_index,
                        tile_tokens=q_tokens,
                        score_record_ids=score_ids,
                        score_tensors=score_tensors,
                        probability_tensors=prob_tensors,
                        tile_start=q_start,
                    )
                )
                records[-1]["operator_context"]["attention_mode"] = "causal_prefill"
                records[-1]["operator_context"]["query_start"] = q_start
                records[-1]["operator_context"]["softmax_causal_span_tokens"] = q_start + q_tokens
                records[-1]["operator_context"]["causal_pair_count"] = sum(pair_count for *_, pair_count in legal_k_tiles)
                records[-1]["accounting_metadata"]["elements"] = sum(pair_count for *_, pair_count in legal_k_tiles)
                next_id += 1

                context_ids: list[str] = []
                context_tensors: list[str] = []
                for (k_tile_index, k_start, k_tokens, pair_count), prob_tensor in zip(legal_k_tiles, prob_tensors):
                    prob_resident = f"{prob_tensor}.pim_visible"
                    prob_residency_id = f"rec_{next_id:04d}"
                    records.append(
                        _semantic_operand_record(
                            prob_residency_id,
                            "PIMOperandResidency",
                            layer_index,
                            "attention_probability_residency",
                            manifest,
                            operator_family="attention",
                            stage="operand_residency",
                            inputs=[prob_tensor],
                            outputs=[prob_resident],
                            dependencies=[softmax_id],
                            operand_role="activation_input",
                            residency="pim_visible_operand",
                            materialized=False,
                            reuse_scope="prefill_attention_context",
                            lowering="semantic_only_operand_residency",
                            context_fields={"head_index": head_index, "query_tile_index": q_tile_index, "tile_index": k_tile_index},
                        )
                    )
                    next_id += 1
                    v_tensor = f"L{layer_index}.H{head_index}.V.T{k_tile_index}"
                    v_resident = f"{v_tensor}.pim_visible"
                    v_residency_id = f"rec_{next_id:04d}"
                    records.append(
                        _semantic_operand_record(
                            v_residency_id,
                            "PIMOperandResidency",
                            layer_index,
                            "v_prefill_tile_residency",
                            manifest,
                            operator_family="attention",
                            stage="operand_residency",
                            inputs=[v_tensor],
                            outputs=[v_resident],
                            dependencies=[],
                            operand_role="activation_input",
                            residency="pim_visible_operand",
                            materialized=False,
                            reuse_scope="prefill_attention_context",
                            lowering="semantic_only_operand_residency",
                            context_fields={"head_index": head_index, "tile_index": k_tile_index, "tile_start": k_start},
                        )
                    )
                    next_id += 1
                    context_id = f"rec_{next_id:04d}"
                    context_tensor = f"L{layer_index}.H{head_index}.Q{q_tile_index}.K{k_tile_index}.context"
                    records.append(
                        _attention_compute_record(
                            context_id,
                            "AttentionContext",
                            layer_index,
                            "attention_context_gemm",
                            manifest,
                            head_index=head_index,
                            tile_index=k_tile_index,
                            tile_tokens=k_tokens,
                            inputs=[prob_resident, v_resident],
                            outputs=[context_tensor],
                            dependencies=[prob_residency_id, softmax_id, v_residency_id],
                            tile_start=k_start,
                            query_tokens=q_tokens,
                            effective_pairs=pair_count,
                            attention_mode="causal_prefill",
                            key_start=k_start,
                            query_start=q_start,
                        )
                    )
                    context_ids.append(context_id)
                    context_tensors.append(context_tensor)
                    next_id += 1
                if len(context_ids) > 1:
                    records.append(
                        _context_reduction_record(
                            f"rec_{next_id:04d}",
                            layer_index,
                            manifest,
                            head_index=head_index,
                            context_record_ids=context_ids,
                            context_tensors=context_tensors,
                            output_tensor=f"L{layer_index}.H{head_index}.Q{q_tile_index}.context_reduced",
                            context_fields={"query_tile_index": q_tile_index, "query_start": q_start},
                            query_tokens=q_tokens,
                        )
                    )
                    next_id += 1
            if observed_pairs != expected_pairs:
                raise ValueError("Internal prefill causal tiling error: pair count mismatch")

    for record in records:
        validate_record(record)
    return records


def generate_ffn_records(manifest: dict | None = None) -> list[dict]:
    manifest = get_tiny_ffn_manifest() if manifest is None else manifest
    _validate_ffn_manifest(manifest)

    records: list[dict] = []
    next_id = 0
    ffn_activation_tile_size = int(manifest["ffn_activation_tile_size"])
    hidden_size = int(manifest["hidden_size"])
    variant = ffn_variant(manifest)
    is_gated_3proj = variant in {FFN_VARIANT_SWIGLU_3PROJ, FFN_VARIANT_GEGLU_3PROJ}
    for layer_index in range(int(manifest["num_layers"])):
        hidden_reused = f"L{layer_index}.hidden.reused_for_up_gate" if is_gated_3proj else f"L{layer_index}.hidden.reused_for_up_projection"
        hidden_reuse_op = "ffn_hidden_reuse_for_up_gate" if is_gated_3proj else "ffn_hidden_reuse_for_up_projection"
        up_weight = f"L{layer_index}.ffn.up_weight"
        gate_weight = f"L{layer_index}.ffn.gate_weight"
        down_weight = f"L{layer_index}.ffn.down_weight"
        up_intermediate = f"L{layer_index}.ffn.up_intermediate"
        gate_intermediate = f"L{layer_index}.ffn.gate_intermediate"
        activated_intermediate = f"L{layer_index}.ffn.activated_gate" if is_gated_3proj else f"L{layer_index}.ffn.relu_intermediate"
        ffn_intermediate = f"L{layer_index}.ffn.gated_multiply" if is_gated_3proj else f"L{layer_index}.ffn.relu_intermediate"
        ffn_intermediate_resident = f"{ffn_intermediate}.bank_local"
        hidden_out = f"L{layer_index}.hidden_out"

        # Per-tile hidden activation setup: emit one PIMDataMove per tile
        tile_ranges = list(_tile_ranges(hidden_size, ffn_activation_tile_size))
        hidden_setup_ids: list[str] = []
        tile_hidden_residents: list[str] = []
        for tile_index, (tile_start, tile_elements) in enumerate(tile_ranges):
            tile_hidden = f"L{layer_index}.T{tile_index}.hidden"
            tile_hidden_resident = f"L{layer_index}.T{tile_index}.hidden.resident"
            setup_id = f"rec_{next_id:04d}"
            records.append(
                _ffn_data_move_record(
                    setup_id,
                    layer_index,
                    manifest,
                    op="ffn_hidden_activation_tile_setup",
                    stage_index=0,
                    inputs=[tile_hidden],
                    outputs=[tile_hidden_resident],
                    dependencies=[],
                    operand_role="activation_input",
                    residency="dynamic_activation_tile",
                    reuse_scope="ffn_layer",
                    tile_index=tile_index,
                    tile_elements=tile_elements,
                    tile_start=tile_start,
                )
            )
            hidden_setup_ids.append(setup_id)
            tile_hidden_residents.append(tile_hidden_resident)
            next_id += 1

        weight_ids: dict[str, str] = {}
        weight_specs = [("up", "ffn_up_weight_residency", up_weight)]
        if is_gated_3proj:
            weight_specs.append(("gate", "ffn_gate_weight_residency", gate_weight))
        weight_specs.append(("down", "ffn_down_weight_residency", down_weight))
        for offset, (weight_name, op_name, output_name) in enumerate(weight_specs):
            weight_record_id = f"rec_{next_id:04d}"
            records.append(
                _ffn_data_move_record(
                    weight_record_id,
                    layer_index,
                    manifest,
                    op=op_name,
                    stage_index=offset,
                    inputs=[output_name],
                    outputs=[f"{output_name}.resident"],
                    dependencies=[],
                    operand_role="weight",
                    residency="preloaded_stationary",
                    reuse_scope=f"ffn_{weight_name}_projection",
                    movement_elements=hidden_size * int(manifest["ffn_hidden_size"]),
                )
            )
            weight_ids[weight_name] = weight_record_id
            next_id += 1

        hidden_reuse_id = f"rec_{next_id:04d}"
        records.append(
            _semantic_operand_record(
                hidden_reuse_id,
                "PIMOperandReuse",
                layer_index,
                hidden_reuse_op,
                manifest,
                operator_family=_ffn_operator_family(manifest),
                stage="operand_reuse",
                inputs=tile_hidden_residents,
                outputs=[hidden_reused],
                dependencies=hidden_setup_ids,
                operand_role="activation_input",
                residency="dynamic_activation_tile",
                materialized=False,
                reuse_scope="ffn_up_gate_pair" if is_gated_3proj else "ffn_up_projection",
                lowering="semantic_only_operand_reuse",
                context_fields={"stage_index": 0},
            )
        )
        next_id += 1

        up_id = f"rec_{next_id:04d}"
        records.append(
            _ffn_projection_record(
                up_id,
                layer_index,
                manifest,
                op="ffn_up_projection",
                stage_index=0,
                inputs=[hidden_reused, f"{up_weight}.resident"],
                outputs=[up_intermediate],
                dependencies=[hidden_reuse_id, weight_ids["up"]],
            )
        )
        next_id += 1

        # P4.23 fix: up_intermediate residency
        up_residency_id = f"rec_{next_id:04d}"
        up_resident = f"{up_intermediate}.pim_visible"
        records.append(
            _semantic_operand_record(
                up_residency_id,
                "PIMOperandResidency",
                layer_index,
                "ffn_up_intermediate_residency",
                manifest,
                operator_family=_ffn_operator_family(manifest),
                stage="operand_residency",
                inputs=[up_intermediate],
                outputs=[up_resident],
                dependencies=[up_id],
                operand_role="activation_intermediate",
                residency="pim_visible_operand",
                materialized=False,
                reuse_scope="ffn_gated_multiply" if is_gated_3proj else "ffn_down_projection",
                lowering="semantic_only_operand_residency",
                context_fields={"distribution_policy": "producer_aligned"},
            )
        )
        next_id += 1

        if is_gated_3proj:
            gate_id = f"rec_{next_id:04d}"
            records.append(
                _ffn_projection_record(
                    gate_id,
                    layer_index,
                    manifest,
                    op="ffn_gate_projection",
                    stage_index=1,
                    inputs=[hidden_reused, f"{gate_weight}.resident"],
                    outputs=[gate_intermediate],
                    dependencies=[hidden_reuse_id, weight_ids["gate"]],
                )
            )
            next_id += 1

            # P4.23 fix: gate_intermediate residency before activation elementwise
            gate_residency_id = f"rec_{next_id:04d}"
            gate_resident = f"{gate_intermediate}.pim_visible"
            records.append(
                _semantic_operand_record(
                    gate_residency_id,
                    "PIMOperandResidency",
                    layer_index,
                    "ffn_gate_intermediate_residency",
                    manifest,
                    operator_family=_ffn_operator_family(manifest),
                    stage="operand_residency",
                    inputs=[gate_intermediate],
                    outputs=[gate_resident],
                    dependencies=[gate_id],
                    operand_role="activation_intermediate",
                    residency="pim_visible_operand",
                    materialized=False,
                    reuse_scope="ffn_gate_activation",
                    lowering="semantic_only_operand_residency",
                    context_fields={"distribution_policy": "producer_aligned"},
                )
            )
            next_id += 1

            activation_id = f"rec_{next_id:04d}"
            records.append(
                _ffn_elementwise_record(
                    activation_id,
                    layer_index,
                    manifest,
                    op="ffn_gate_activation_accounting",
                    stage_index=2,
                    inputs=[gate_resident],
                    outputs=[activated_intermediate],
                    dependencies=[gate_residency_id],
                    elementwise_kind="semantic_only_activation",
                )
            )
            next_id += 1

            # P4.23 fix: activated_gate residency before gated_multiply elementwise
            activated_residency_id = f"rec_{next_id:04d}"
            activated_resident = f"{activated_intermediate}.pim_visible"
            records.append(
                _semantic_operand_record(
                    activated_residency_id,
                    "PIMOperandResidency",
                    layer_index,
                    "ffn_activated_gate_residency",
                    manifest,
                    operator_family=_ffn_operator_family(manifest),
                    stage="operand_residency",
                    inputs=[activated_intermediate],
                    outputs=[activated_resident],
                    dependencies=[activation_id],
                    operand_role="activation_intermediate",
                    residency="pim_visible_operand",
                    materialized=False,
                    reuse_scope="ffn_gated_multiply",
                    lowering="semantic_only_operand_residency",
                    context_fields={"distribution_policy": "producer_aligned"},
                )
            )
            next_id += 1

            gated_id = f"rec_{next_id:04d}"
            records.append(
                _ffn_elementwise_record(
                    gated_id,
                    layer_index,
                    manifest,
                    op="ffn_gated_multiply_accounting",
                    stage_index=3,
                    inputs=[activated_resident, up_resident],
                    outputs=[ffn_intermediate],
                    dependencies=[activated_residency_id, up_residency_id],
                    elementwise_kind="semantic_only_gated_multiply",
                )
            )
            next_id += 1
            intermediate_inputs = [ffn_intermediate]
            intermediate_dependencies = [gated_id]
            intermediate_stage_index = 4
        else:
            activation_id = f"rec_{next_id:04d}"
            records.append(
                _ffn_elementwise_record(
                    activation_id,
                    layer_index,
                    manifest,
                    op="ffn_relu_activation_accounting",
                    stage_index=1,
                    inputs=[up_resident],
                    outputs=[activated_intermediate],
                    dependencies=[up_residency_id],
                    elementwise_kind="semantic_only_activation",
                )
            )
            next_id += 1
            intermediate_inputs = [activated_intermediate]
            intermediate_dependencies = [activation_id]
            intermediate_stage_index = 2

        intermediate_residency_id = f"rec_{next_id:04d}"
        records.append(
            _semantic_operand_record(
                intermediate_residency_id,
                "PIMOperandResidency",
                layer_index,
                "ffn_intermediate_bank_local_residency",
                manifest,
                operator_family=_ffn_operator_family(manifest),
                stage="operand_residency",
                inputs=intermediate_inputs,
                outputs=[ffn_intermediate_resident],
                dependencies=intermediate_dependencies,
                operand_role="activation_intermediate",
                residency="bank_local_capacity_controlled",
                materialized=False,
                reuse_scope="ffn_down_projection",
                lowering="semantic_only_bank_local_capacity_controlled_operand",
                context_fields={"stage_index": intermediate_stage_index},
            )
        )
        next_id += 1

        records.append(
            _ffn_projection_record(
                f"rec_{next_id:04d}",
                layer_index,
                manifest,
                op="ffn_down_projection",
                stage_index=intermediate_stage_index,
                inputs=[ffn_intermediate_resident, f"{down_weight}.resident"],
                outputs=[hidden_out],
                dependencies=[intermediate_residency_id, weight_ids["down"]],
            )
        )
        next_id += 1

    for record in records:
        validate_record(record)
    return records


def generate_moe_records(manifest: dict | None = None) -> list[dict]:
    """Generate MoE records.

    ``overlap_selected_experts`` reorders independent selected expert groups in
    the flat offline IR. Concrete replay remains a sequential opcode stream;
    ``operator_context.dag_hint`` marks future-safe expert-level parallelism.
    """
    manifest = get_tiny_moe_manifest() if manifest is None else manifest
    _validate_moe_manifest(manifest)

    records: list[dict] = []
    next_id = 0
    for layer_index in range(int(manifest["num_layers"])):
        hidden = f"L{layer_index}.hidden"
        router_input = f"L{layer_index}.moe.router_input.resident"
        router_weight = f"L{layer_index}.moe.router_weight"
        router_logits = f"L{layer_index}.moe.router_logits"
        topk_tensor = f"L{layer_index}.moe.topk_experts"
        dispatch_accounted_tensor = f"L{layer_index}.moe.dispatch_accounted_tokens"
        dispatch_tensor = f"L{layer_index}.moe.dispatched_tokens"

        router_setup_id = f"rec_{next_id:04d}"
        records.append(
            _moe_data_move_record(
                router_setup_id,
                layer_index,
                manifest,
                op="moe_router_input_setup",
                stage_index=0,
                inputs=[hidden],
                outputs=[router_input],
                dependencies=[],
                operand_role="activation_input",
                residency="dynamic_activation_tile",
                reuse_scope="moe_router",
            )
        )
        next_id += 1

        router_weight_id = f"rec_{next_id:04d}"
        records.append(
            _moe_data_move_record(
                router_weight_id,
                layer_index,
                manifest,
                op="moe_router_weight_residency",
                stage_index=0,
                inputs=[router_weight],
                outputs=[f"{router_weight}.resident"],
                dependencies=[],
                operand_role="weight",
                residency="preloaded_stationary",
                reuse_scope="moe_router",
                expert_id=None,
                movement_elements=int(manifest["hidden_size"]) * int(manifest["num_experts"]),
            )
        )
        next_id += 1

        router_id = f"rec_{next_id:04d}"
        records.append(
            _moe_compute_record(
                router_id,
                "MoERouter",
                layer_index,
                manifest,
                op="moe_router_projection",
                stage_index=0,
                inputs=[router_input, f"{router_weight}.resident"],
                outputs=[router_logits],
                dependencies=[router_setup_id, router_weight_id],
            )
        )
        next_id += 1

        # P4.24 fix: router logits residency
        router_logits_residency_id = f"rec_{next_id:04d}"
        router_logits_resident = f"{router_logits}.pim_visible"
        records.append(
            _semantic_operand_record(
                router_logits_residency_id,
                "PIMOperandResidency",
                layer_index,
                "moe_router_logits_residency",
                manifest,
                operator_family="moe",
                stage="operand_residency",
                inputs=[router_logits],
                outputs=[router_logits_resident],
                dependencies=[router_id],
                operand_role="activation_intermediate",
                residency="pim_visible_operand",
                materialized=False,
                reuse_scope="moe_topk",
                lowering="semantic_only_operand_residency",
                context_fields={"distribution_policy": "producer_aligned"},
            )
        )
        next_id += 1

        topk_id = f"rec_{next_id:04d}"
        records.append(
            _moe_accounting_record(
                topk_id,
                "MoETopK",
                layer_index,
                manifest,
                op="moe_topk_select_accounting",
                stage_index=1,
                inputs=[router_logits_resident],
                outputs=[topk_tensor],
                dependencies=[router_logits_residency_id],
                accounting_kind="semantic_only_topk_select",
            )
        )
        next_id += 1

        dispatch_id = f"rec_{next_id:04d}"
        records.append(
            _moe_accounting_record(
                dispatch_id,
                "MoEDispatch",
                layer_index,
                manifest,
                op="moe_expert_dispatch_accounting",
                stage_index=2,
                inputs=[hidden, topk_tensor],
                outputs=[dispatch_accounted_tensor],
                dependencies=[topk_id],
                accounting_kind="semantic_only_expert_dispatch",
            )
        )
        next_id += 1

        dispatch_move_id = f"rec_{next_id:04d}"
        records.append(
            _moe_data_move_record(
                dispatch_move_id,
                layer_index,
                manifest,
                op="moe_token_dispatch_materialized",
                stage_index=2,
                inputs=[dispatch_accounted_tensor],
                outputs=[dispatch_tensor],
                dependencies=[dispatch_id],
                operand_role="token_dispatch",
                residency="dynamic_dispatched_activation_tile",
                reuse_scope="moe_selected_experts",
            )
        )
        next_id += 1

        expert_ids: list[str] = []
        expert_outputs: list[str] = []
        selected_experts = [int(expert) for expert in manifest["selected_experts"]]
        if manifest["schedule_policy"] == "overlap_selected_experts":
            selected_experts = selected_experts[::2] + selected_experts[1::2]
        for expert_id in selected_experts:
            projection_records: dict[str, str] = {}
            projection_outputs: dict[str, str] = {}
            for projection in ("up", "gate", "down"):
                expert_weight_id = f"rec_{next_id:04d}"
                expert_weight = f"L{layer_index}.moe.expert_{expert_id}.{projection}_weight"
                records.append(
                    _moe_data_move_record(
                        expert_weight_id,
                        layer_index,
                        manifest,
                        op=f"moe_expert_{expert_id}_{projection}_weight_residency",
                        stage_index=3 if projection in {"up", "gate"} else 4,
                        inputs=[expert_weight],
                        outputs=[f"{expert_weight}.resident"],
                        dependencies=[],
                        operand_role="weight",
                        residency="preloaded_stationary",
                        reuse_scope=f"moe_expert_{expert_id}_{projection}",
                        expert_id=expert_id,
                        movement_elements=int(manifest["hidden_size"]) * int(manifest["expert_hidden_size"]),
                    )
                )
                next_id += 1

                if projection == "down":
                    # P4.24 fix: expert activation + gated-multiply accounting (mirrors dense FFN lines 2347-2376)
                    # Insert activation elementwise after gate_projection
                    expert_activation_intermediate = f"L{layer_index}.moe.expert_{expert_id}.activated_gate"
                    expert_activation_id = f"rec_{next_id:04d}"
                    records.append(
                        _ffn_elementwise_record(
                            expert_activation_id,
                            layer_index,
                            manifest,
                            op=f"moe_expert_{expert_id}_gate_activation_accounting",
                            stage_index=4,
                            inputs=[projection_outputs["gate"]],
                            outputs=[expert_activation_intermediate],
                            dependencies=[projection_records["gate"]],
                            elementwise_kind="semantic_only_activation",
                        )
                    )
                    next_id += 1

                    # Insert gated-multiply elementwise
                    expert_gated_intermediate = f"L{layer_index}.moe.expert_{expert_id}.gated"
                    expert_gated_id = f"rec_{next_id:04d}"
                    records.append(
                        _ffn_elementwise_record(
                            expert_gated_id,
                            layer_index,
                            manifest,
                            op=f"moe_expert_{expert_id}_gated_multiply_accounting",
                            stage_index=4,
                            inputs=[expert_activation_intermediate, projection_outputs["up"]],
                            outputs=[expert_gated_intermediate],
                            dependencies=[expert_activation_id, projection_records["up"]],
                            elementwise_kind="semantic_only_gated_multiply",
                        )
                    )
                    next_id += 1

                    # Insert intermediate residency before down projection (mirrors dense FFN line 2401)
                    expert_gated_resident = f"{expert_gated_intermediate}.bank_local"
                    expert_gated_residency_id = f"rec_{next_id:04d}"
                    records.append(
                        _semantic_operand_record(
                            expert_gated_residency_id,
                            "PIMOperandResidency",
                            layer_index,
                            f"moe_expert_{expert_id}_intermediate_residency",
                            manifest,
                            operator_family="moe",
                            stage="operand_residency",
                            inputs=[expert_gated_intermediate],
                            outputs=[expert_gated_resident],
                            dependencies=[expert_gated_id],
                            operand_role="activation_intermediate",
                            residency="bank_local_capacity_controlled",
                            materialized=False,
                            reuse_scope=f"moe_expert_{expert_id}_down_projection",
                            lowering="semantic_only_operand_residency",
                            context_fields={"expert_id": expert_id, "distribution_policy": "producer_aligned"},
                        )
                    )
                    next_id += 1

                expert_record_id = f"rec_{next_id:04d}"
                if projection == "down":
                    expert_inputs = [
                        expert_gated_resident,
                        f"{expert_weight}.resident",
                    ]
                    expert_deps = [expert_gated_residency_id, expert_weight_id]
                    expert_output = f"L{layer_index}.moe.expert_{expert_id}.output"
                    stage_index = 4
                else:
                    expert_inputs = [dispatch_tensor, f"{expert_weight}.resident"]
                    expert_deps = [dispatch_move_id, expert_weight_id]
                    expert_output = f"L{layer_index}.moe.expert_{expert_id}.{projection}_intermediate"
                    stage_index = 3
                records.append(
                    _moe_compute_record(
                        expert_record_id,
                        "MoEExpertFFN",
                        layer_index,
                        manifest,
                        op=f"moe_expert_{expert_id}_{projection}_projection",
                        stage_index=stage_index,
                        inputs=expert_inputs,
                        outputs=[expert_output],
                        dependencies=expert_deps,
                        expert_id=expert_id,
                        projection=projection,
                    )
                )
                projection_records[projection] = expert_record_id
                projection_outputs[projection] = expert_output
                next_id += 1

            # P4.24 fix: expert output residency before combine
            expert_output_tensor = projection_outputs["down"]
            expert_output_residency_id = f"rec_{next_id:04d}"
            expert_output_resident = f"{expert_output_tensor}.pim_visible"
            records.append(
                _semantic_operand_record(
                    expert_output_residency_id,
                    "PIMOperandResidency",
                    layer_index,
                    f"moe_expert_{expert_id}_output_residency",
                    manifest,
                    operator_family="moe",
                    stage="operand_residency",
                    inputs=[expert_output_tensor],
                    outputs=[expert_output_resident],
                    dependencies=[projection_records["down"]],
                    operand_role="activation_intermediate",
                    residency="pim_visible_operand",
                    materialized=False,
                    reuse_scope="moe_combine",
                    lowering="semantic_only_operand_residency",
                    context_fields={"expert_id": expert_id, "distribution_policy": "producer_aligned"},
                )
            )
            next_id += 1

            expert_ids.append(expert_output_residency_id)
            expert_outputs.append(expert_output_resident)

        combine_id = f"rec_{next_id:04d}"
        combined_output = f"L{layer_index}.moe.combined_output"
        records.append(
            _moe_accounting_record(
                combine_id,
                "MoECombine",
                layer_index,
                manifest,
                op="moe_expert_combine_accounting",
                stage_index=5,
                inputs=expert_outputs,
                outputs=[combined_output],
                dependencies=expert_ids,
                accounting_kind="semantic_only_expert_combine",
            )
        )
        next_id += 1

        records.append(
            _moe_data_move_record(
                f"rec_{next_id:04d}",
                layer_index,
                manifest,
                op="moe_expert_output_combine_materialized",
                stage_index=5,
                inputs=[combined_output],
                outputs=[f"{combined_output}.materialized"],
                dependencies=[combine_id],
                operand_role="expert_output_combine",
                residency="dynamic_combined_expert_output",
                reuse_scope="moe_layer_output",
            )
        )
        next_id += 1

    for record in records:
        if manifest["schedule_policy"] == "overlap_selected_experts":
            record["operator_context"]["dag_hint"] = "selected_experts_are_independent_safe_to_parallelize"
        validate_record(record)
    return records


def _renumber_records(records: list[dict], start_id: int) -> tuple[list[dict], int]:
    id_map = {record["record_id"]: f"rec_{start_id + index:04d}" for index, record in enumerate(records)}
    renumbered: list[dict] = []
    for record in records:
        updated = dict(record)
        updated["record_id"] = id_map[record["record_id"]]
        updated["logical_dependencies"] = [id_map.get(dependency, dependency) for dependency in record.get("logical_dependencies", [])]
        renumbered.append(updated)
    return renumbered, start_id + len(records)


def _retarget_layer_zero_record(record: dict, layer_index: int) -> dict:
    def retarget(value):
        if isinstance(value, str):
            return value.replace("layer_00", f"layer_{layer_index:02d}").replace("L0.", f"L{layer_index}.")
        if isinstance(value, list):
            return [retarget(item) for item in value]
        if isinstance(value, dict):
            return {key: retarget(item) for key, item in value.items()}
        return value

    updated = retarget(record)
    if "operator_context" in updated:
        updated["operator_context"]["layer_id"] = layer_index
    if "barrier_scope" in updated:
        updated["barrier_scope"]["layer_id"] = layer_index
    if "drain_scope" in updated:
        updated["drain_scope"]["layer_id"] = layer_index
    return updated


def _one_layer_manifest(manifest: dict) -> dict:
    one_layer = dict(manifest)
    one_layer["num_layers"] = 1
    return one_layer


def _validate_dense_decode_v2_supported_manifests(attention_manifest: dict, ffn_manifest: dict) -> None:
    num_heads = int(attention_manifest["num_heads"])
    head_dim = int(attention_manifest["head_dim"])
    hidden_size = int(attention_manifest["hidden_size"])
    _attention_head_group_size(attention_manifest)
    if num_heads * head_dim <= 0:
        raise ValueError("decode-block v2 requires positive num_heads * head_dim for Q/O projection shapes")
    if int(ffn_manifest["hidden_size"]) != hidden_size:
        raise ValueError("Dense decoder attention/FFN manifests must use the same hidden_size")
    if attention_manifest["datatype"] != ffn_manifest["datatype"]:
        raise ValueError("Dense decoder attention/FFN manifests must use the same datatype")


def _validate_dense_prefill_supported_manifests(attention_manifest: dict, ffn_manifest: dict) -> None:
    _validate_attention_manifest(attention_manifest)
    _validate_ffn_manifest(ffn_manifest)
    if attention_manifest["phase"] != "prefill" or ffn_manifest["phase"] != "prefill":
        raise ValueError("Dense prefill generation requires prefill attention and FFN manifests")
    _validate_dense_decode_v2_supported_manifests(attention_manifest, ffn_manifest)
    if int(attention_manifest["seq_len"]) != int(ffn_manifest["seq_len"]):
        raise ValueError("Dense prefill attention/FFN manifests must use the same seq_len")


def _validate_moe_decode_v2_supported_manifests(attention_manifest: dict, moe_manifest: dict) -> None:
    num_heads = int(attention_manifest["num_heads"])
    head_dim = int(attention_manifest["head_dim"])
    hidden_size = int(attention_manifest["hidden_size"])
    _attention_head_group_size(attention_manifest)
    if num_heads * head_dim <= 0:
        raise ValueError("decode-block v2 requires positive num_heads * head_dim")
    moe_hidden = int(moe_manifest["hidden_size"])
    if moe_hidden != hidden_size:
        raise ValueError(
            f"MoE decoder attention hidden_size ({hidden_size}) and MoE hidden_size ({moe_hidden}) must match"
        )
    if attention_manifest["datatype"] != moe_manifest["datatype"]:
        raise ValueError("MoE decoder attention/MoE manifests must use the same datatype")


def generate_moe_transformer_layer_records(
    *, attention_manifest: dict, moe_manifest: dict
) -> list[dict]:
    """Generate a full-depth MoE decoder trace: attention + MoE per layer.

    For every layer the pipeline emits Q/K/V/O projections, KV-cache writes,
    tiled attention (score / softmax / context), and a complete MoE block
    (router -> top-k -> dispatch -> selected expert FFNs -> combine).

    The attention manifest drives the GQA-aware attention side; the MoE
    manifest drives the router and per-expert compute.  Both must agree on
    ``hidden_size``, ``datatype``, and ``num_layers``.
    """
    _validate_moe_decode_v2_supported_manifests(attention_manifest, moe_manifest)
    combined: list[dict] = []
    next_id = 0
    num_layers = int(attention_manifest["num_layers"])
    if int(moe_manifest["num_layers"]) != num_layers:
        raise ValueError("MoE decoder attention/MoE manifests must use the same num_layers")

    one_attention = _one_layer_manifest(attention_manifest)
    one_moe = _one_layer_manifest(moe_manifest)
    num_kv = _attention_num_kv_heads(one_attention)
    hd = int(one_attention["head_dim"])

    for layer_index in range(num_layers):
        layer_start, next_id = _renumber_records(
            [_barrier_record("rec_0000", layer_index, attention_manifest, op="layer_start")], next_id
        )
        combined.extend(layer_start)

        qkv, next_id = _renumber_records(
            _generate_decode_v2_qkvo_projection_records(one_moe, num_kv_heads=num_kv, head_dim=hd), next_id
        )
        qkv = [_retarget_layer_zero_record(record, layer_index) for record in qkv]
        q_id = [record for record in qkv if record["op"] == "q_projection"][0]["record_id"]
        k_id = [record for record in qkv if record["op"] == "k_projection"][0]["record_id"]
        v_id = [record for record in qkv if record["op"] == "v_projection"][0]["record_id"]
        combined.extend(qkv)

        kv_writes, next_id = _renumber_records(
            _generate_decode_v2_kv_cache_write_records(one_attention), next_id
        )
        kv_writes = [_retarget_layer_zero_record(record, layer_index) for record in kv_writes]
        for record in kv_writes:
            if record["op"] == "kv_cache_k_append":
                record["logical_dependencies"] = [k_id]
            elif record["op"] == "kv_cache_v_append":
                record["logical_dependencies"] = [v_id]
        k_append_id = [record for record in kv_writes if record["op"] == "kv_cache_k_append"][0]["record_id"]
        v_append_id = [record for record in kv_writes if record["op"] == "kv_cache_v_append"][0]["record_id"]
        combined.extend(kv_writes)

        attention, next_id = _renumber_records(generate_attention_records(one_attention), next_id)
        attention = [_retarget_layer_zero_record(record, layer_index) for record in attention]

        # P4.22 fix: deduplicate per-head Q residencies to single canonical global Q residency
        qr_ids = [r["record_id"] for r in attention
                   if r["kind"] == "PIMOperandResidency" and r.get("op") == "q_operand_residency"]
        keep_qr = qr_ids[0] if qr_ids else None
        for record in attention:
            if record["kind"] == "AttentionScore":
                record["tensor_io"]["inputs"][0] = f"L{layer_index}.Q.pim_visible"
                old_dependencies = list(record["logical_dependencies"])
                canonical_dependencies = [q_id] + ([keep_qr] if keep_qr else [])
                replaced_q_residencies = set(qr_ids)
                record["logical_dependencies"] = canonical_dependencies + [
                    dependency
                    for dependency in old_dependencies
                    if dependency not in replaced_q_residencies and dependency not in canonical_dependencies
                ]
            elif record["kind"] == "PIMOperandResidency" and record.get("op") == "q_operand_residency":
                if record["record_id"] == keep_qr:
                    record["tensor_io"]["inputs"] = [f"L{layer_index}.Q"]
                    record["tensor_io"]["outputs"] = [f"L{layer_index}.Q.pim_visible"]
                    record["logical_dependencies"] = [q_id]
            elif record["op"] == "kv_cache_k_tile_read":
                record["logical_dependencies"] = [k_append_id]
            elif record["op"] == "kv_cache_v_tile_read":
                record["logical_dependencies"] = [v_append_id, *record["logical_dependencies"]]
        # Remove duplicate Q residency records (keep only canonical one)
        attention = [r for r in attention
                     if not (r["kind"] == "PIMOperandResidency"
                             and r.get("op") == "q_operand_residency"
                             and r["record_id"] != keep_qr)]
        terminal_attention_ids = _terminal_attention_record_ids(attention)
        terminal_attention_tensors = [
            output
            for record in attention
            if record["record_id"] in terminal_attention_ids
            for output in record.get("tensor_io", {}).get("outputs", [])
        ]
        combined.extend(attention)

        output_projection, next_id = _renumber_records(
            _generate_decode_v2_output_projection_records(one_moe), next_id
        )
        output_projection = [_retarget_layer_zero_record(record, layer_index) for record in output_projection]
        output_projection[0]["tensor_io"]["inputs"] = terminal_attention_tensors + [f"L{layer_index}.o_projection.weight.resident"]
        output_projection[0]["logical_dependencies"] = terminal_attention_ids
        o_id = output_projection[0]["record_id"]
        combined.extend(output_projection)

        # MoE block: router → top-k → dispatch → selected expert FFNs → combine
        moe, next_id = _renumber_records(generate_moe_records(one_moe), next_id)
        moe = [_retarget_layer_zero_record(record, layer_index) for record in moe]
        for record in moe:
            if record.get("op") == "moe_router_input_setup":
                record["tensor_io"]["inputs"] = [f"L{layer_index}.attention_output"]
                record["logical_dependencies"] = [o_id]
        combined.extend(moe)

        barrier, next_id = _renumber_records(
            [_barrier_record("rec_0000", layer_index, attention_manifest)], next_id
        )
        combined.extend(barrier)

    drain, next_id = _renumber_records(
        [_drain_record("rec_0000", max(0, num_layers - 1), attention_manifest)], next_id
    )
    combined.extend(drain)
    for record in combined:
        validate_record(record)
    return combined


def generate_mixtral_8x7b_decoder_records(
    *, attention_manifest: dict | None = None, moe_manifest: dict | None = None
) -> list[dict]:
    if attention_manifest is None or moe_manifest is None:
        default_attention, default_moe = get_mixtral_8x7b_moe_decoder_manifests()
        attention_manifest = default_attention if attention_manifest is None else attention_manifest
        moe_manifest = default_moe if moe_manifest is None else moe_manifest
    return generate_moe_transformer_layer_records(
        attention_manifest=attention_manifest, moe_manifest=moe_manifest
    )


def _generate_decode_v2_qkvo_projection_records(
    manifest: dict,
    *,
    num_heads: int | None = None,
    num_kv_heads: int | None = None,
    head_dim: int | None = None,
) -> list[dict]:
    records: list[dict] = []
    next_id = 0
    q_output_dim = (num_heads * head_dim) if (num_heads is not None and head_dim is not None) else None
    kv_output_dim = (num_kv_heads * head_dim) if (num_kv_heads is not None and head_dim is not None) else None
    for layer_index in range(int(manifest["num_layers"])):
        hidden = f"L{layer_index}.hidden"
        for stage_index, op in enumerate(["q_projection", "k_projection", "v_projection"]):
            output = f"L{layer_index}.{op.removesuffix('_projection').upper()}"
            weight = f"L{layer_index}.{op}.weight.resident"
            proj_dim = kv_output_dim if op in {"k_projection", "v_projection"} else q_output_dim
            records.append(
                _ffn_projection_record(
                    f"rec_{next_id:04d}",
                    layer_index,
                    manifest,
                    op=op,
                    stage_index=stage_index,
                    inputs=[hidden, weight],
                    outputs=[output],
                    dependencies=[],
                    projection_output_dim=proj_dim,
                )
            )
            next_id += 1
    for record in records:
        validate_record(record)
    return records


def _generate_decode_v2_output_projection_records(manifest: dict, *, q_input_dim: int | None = None) -> list[dict]:
    records: list[dict] = []
    next_id = 0
    for layer_index in range(int(manifest["num_layers"])):
        records.append(
            _ffn_projection_record(
                f"rec_{next_id:04d}",
                layer_index,
                manifest,
                op="o_projection",
                stage_index=3,
                inputs=[f"L{layer_index}.attention_context", f"L{layer_index}.o_projection.weight.resident"],
                outputs=[f"L{layer_index}.attention_output"],
                dependencies=[],
                projection_input_dim=q_input_dim,
            )
        )
        next_id += 1
    for record in records:
        validate_record(record)
    return records


def _generate_decode_v2_kv_cache_write_records(manifest: dict) -> list[dict]:
    records: list[dict] = []
    next_id = 0
    hidden_size = int(manifest["hidden_size"])
    num_kv = _attention_num_kv_heads(manifest)
    hd = int(manifest.get("head_dim", hidden_size // max(1, num_kv)))
    kv_byte_elements = num_kv * hd
    for layer_index in range(int(manifest["num_layers"])):
        for op, tensor in [("kv_cache_k_append", "K"), ("kv_cache_v_append", "V")]:
            records.append(
                _host_access_record(
                    f"rec_{next_id:04d}",
                    "HostWrite",
                    layer_index,
                    op,
                    manifest,
                    inputs=[f"L{layer_index}.{tensor}"],
                    outputs=[f"L{layer_index}.kv_cache.{tensor}.current_token"],
                    dependencies=[],
                    byte_elements=kv_byte_elements,
                    address_scope=op,
                )
            )
            next_id += 1
    for record in records:
        validate_record(record)
    return records


def _generate_prefill_kv_cache_write_records(manifest: dict) -> list[dict]:
    records: list[dict] = []
    next_id = 0
    prompt_len = int(manifest["seq_len"])
    num_kv = _attention_num_kv_heads(manifest)
    hd = int(manifest["head_dim"])
    kv_byte_elements = prompt_len * num_kv * hd
    for layer_index in range(int(manifest["num_layers"])):
        for op, tensor, scope in [
            ("kv_cache_k_prefill_population", "K", "kv_cache_k_prefill_population"),
            ("kv_cache_v_prefill_population", "V", "kv_cache_v_prefill_population"),
        ]:
            records.append(
                _host_access_record(
                    f"rec_{next_id:04d}",
                    "HostWrite",
                    layer_index,
                    op,
                    manifest,
                    inputs=[f"L{layer_index}.{tensor}"],
                    outputs=[f"L{layer_index}.kv_cache.{tensor}.prefill_tokens"],
                    dependencies=[],
                    byte_elements=kv_byte_elements,
                    address_scope=scope,
                )
            )
            next_id += 1
    for record in records:
        validate_record(record)
    return records


def _generate_decode_v2_boundary_records(manifest: dict) -> list[dict]:
    records: list[dict] = []
    next_id = 0
    num_layers = int(manifest["num_layers"])
    for layer_index in range(num_layers):
        records.append(_barrier_record(f"rec_{next_id:04d}", layer_index, manifest))
        next_id += 1
    records.append(_drain_record(f"rec_{next_id:04d}", max(0, num_layers - 1), manifest))
    for record in records:
        validate_record(record)
    return records


def generate_dense_transformer_layer_records(*, attention_manifest: dict, ffn_manifest: dict) -> list[dict]:
    _validate_dense_decode_v2_supported_manifests(attention_manifest, ffn_manifest)
    combined: list[dict] = []
    next_id = 0
    num_layers = int(attention_manifest["num_layers"])
    if int(ffn_manifest["num_layers"]) != num_layers:
        raise ValueError("Dense decoder attention/FFN manifests must use the same num_layers")

    one_attention = _one_layer_manifest(attention_manifest)
    one_ffn = _one_layer_manifest(ffn_manifest)
    num_kv = _attention_num_kv_heads(one_attention)
    num_heads = int(one_attention["num_heads"])
    hd = int(one_attention["head_dim"])
    q_dim = num_heads * hd
    for layer_index in range(num_layers):
        layer_start, next_id = _renumber_records([_barrier_record("rec_0000", layer_index, attention_manifest, op="layer_start")], next_id)
        combined.extend(layer_start)

        qkv, next_id = _renumber_records(
            _generate_decode_v2_qkvo_projection_records(one_ffn, num_heads=num_heads, num_kv_heads=num_kv, head_dim=hd),
            next_id,
        )
        qkv = [_retarget_layer_zero_record(record, layer_index) for record in qkv]
        q_id = [record for record in qkv if record["op"] == "q_projection"][0]["record_id"]
        k_id = [record for record in qkv if record["op"] == "k_projection"][0]["record_id"]
        v_id = [record for record in qkv if record["op"] == "v_projection"][0]["record_id"]
        combined.extend(qkv)

        kv_writes, next_id = _renumber_records(_generate_decode_v2_kv_cache_write_records(one_attention), next_id)
        kv_writes = [_retarget_layer_zero_record(record, layer_index) for record in kv_writes]
        for record in kv_writes:
            if record["op"] == "kv_cache_k_append":
                record["logical_dependencies"] = [k_id]
            elif record["op"] == "kv_cache_v_append":
                record["logical_dependencies"] = [v_id]
        k_append_id = [record for record in kv_writes if record["op"] == "kv_cache_k_append"][0]["record_id"]
        v_append_id = [record for record in kv_writes if record["op"] == "kv_cache_v_append"][0]["record_id"]
        combined.extend(kv_writes)

        attention, next_id = _renumber_records(generate_attention_records(one_attention), next_id)
        attention = [_retarget_layer_zero_record(record, layer_index) for record in attention]

        # P4.22 fix: deduplicate per-head Q residencies to single canonical global Q residency
        qr_ids = [r["record_id"] for r in attention
                   if r["kind"] == "PIMOperandResidency" and r.get("op") == "q_operand_residency"]
        keep_qr = qr_ids[0] if qr_ids else None
        for record in attention:
            if record["kind"] == "AttentionScore":
                record["tensor_io"]["inputs"][0] = f"L{layer_index}.Q.pim_visible"
                old_dependencies = list(record["logical_dependencies"])
                canonical_dependencies = [q_id] + ([keep_qr] if keep_qr else [])
                replaced_q_residencies = set(qr_ids)
                record["logical_dependencies"] = canonical_dependencies + [
                    dependency
                    for dependency in old_dependencies
                    if dependency not in replaced_q_residencies and dependency not in canonical_dependencies
                ]
            elif record["kind"] == "PIMOperandResidency" and record.get("op") == "q_operand_residency":
                if record["record_id"] == keep_qr:
                    record["tensor_io"]["inputs"] = [f"L{layer_index}.Q"]
                    record["tensor_io"]["outputs"] = [f"L{layer_index}.Q.pim_visible"]
                    record["logical_dependencies"] = [q_id]
            elif record["op"] == "kv_cache_k_tile_read":
                record["logical_dependencies"] = [k_append_id]
            elif record["op"] == "kv_cache_v_tile_read":
                record["logical_dependencies"] = [v_append_id, *record["logical_dependencies"]]
        # Remove duplicate Q residency records (keep only canonical one)
        attention = [r for r in attention
                     if not (r["kind"] == "PIMOperandResidency"
                             and r.get("op") == "q_operand_residency"
                             and r["record_id"] != keep_qr)]
        terminal_attention_ids = _terminal_attention_record_ids(attention)
        terminal_attention_tensors = [
            output
            for record in attention
            if record["record_id"] in terminal_attention_ids
            for output in record.get("tensor_io", {}).get("outputs", [])
        ]
        combined.extend(attention)

        output_projection, next_id = _renumber_records(_generate_decode_v2_output_projection_records(one_ffn, q_input_dim=q_dim), next_id)
        output_projection = [_retarget_layer_zero_record(record, layer_index) for record in output_projection]
        output_projection[0]["tensor_io"]["inputs"] = terminal_attention_tensors + [f"L{layer_index}.o_projection.weight.resident"]
        output_projection[0]["logical_dependencies"] = terminal_attention_ids
        o_id = output_projection[0]["record_id"]
        combined.extend(output_projection)

        ffn, next_id = _renumber_records(generate_ffn_records(one_ffn), next_id)
        ffn = [_retarget_layer_zero_record(record, layer_index) for record in ffn]
        for record in ffn:
            if record["op"] == "ffn_hidden_activation_tile_setup":
                record["tensor_io"]["inputs"] = [f"L{layer_index}.attention_output"]
                record["logical_dependencies"] = [o_id]
        combined.extend(ffn)

        barrier, next_id = _renumber_records([_barrier_record("rec_0000", layer_index, attention_manifest)], next_id)
        combined.extend(barrier)

    drain, next_id = _renumber_records([_drain_record("rec_0000", max(0, num_layers - 1), attention_manifest)], next_id)
    combined.extend(drain)
    for record in combined:
        validate_record(record)
    return combined


def generate_dense_prefill_transformer_layer_records(*, attention_manifest: dict, ffn_manifest: dict) -> list[dict]:
    _validate_dense_prefill_supported_manifests(attention_manifest, ffn_manifest)
    combined: list[dict] = []
    next_id = 0
    num_layers = int(attention_manifest["num_layers"])
    if int(ffn_manifest["num_layers"]) != num_layers:
        raise ValueError("Dense prefill attention/FFN manifests must use the same num_layers")

    one_attention = _one_layer_manifest(attention_manifest)
    one_ffn = _one_layer_manifest(ffn_manifest)
    num_kv = _attention_num_kv_heads(one_attention)
    num_heads = int(one_attention["num_heads"])
    hd = int(one_attention["head_dim"])
    q_dim = num_heads * hd
    for layer_index in range(num_layers):
        layer_start, next_id = _renumber_records([_barrier_record("rec_0000", layer_index, attention_manifest, op="layer_start")], next_id)
        combined.extend(layer_start)

        qkv, next_id = _renumber_records(
            _generate_decode_v2_qkvo_projection_records(one_ffn, num_heads=num_heads, num_kv_heads=num_kv, head_dim=hd),
            next_id,
        )
        qkv = [_retarget_layer_zero_record(record, layer_index) for record in qkv]
        q_id = [record for record in qkv if record["op"] == "q_projection"][0]["record_id"]
        k_id = [record for record in qkv if record["op"] == "k_projection"][0]["record_id"]
        v_id = [record for record in qkv if record["op"] == "v_projection"][0]["record_id"]
        combined.extend(qkv)

        kv_writes, next_id = _renumber_records(_generate_prefill_kv_cache_write_records(one_attention), next_id)
        kv_writes = [_retarget_layer_zero_record(record, layer_index) for record in kv_writes]
        for record in kv_writes:
            if record["op"] == "kv_cache_k_prefill_population":
                record["logical_dependencies"] = [k_id]
            elif record["op"] == "kv_cache_v_prefill_population":
                record["logical_dependencies"] = [v_id]
        combined.extend(kv_writes)

        attention, next_id = _renumber_records(generate_prefill_attention_records(one_attention), next_id)
        attention = [_retarget_layer_zero_record(record, layer_index) for record in attention]
        for record in attention:
            if record["kind"] == "PIMOperandResidency" and record.get("op") == "q_operand_residency":
                record["tensor_io"]["inputs"] = [f"L{layer_index}.Q"]
                record["logical_dependencies"] = [q_id]
            elif record["kind"] == "PIMOperandResidency" and record.get("op") == "k_prefill_tile_residency":
                record["tensor_io"]["inputs"] = [f"L{layer_index}.K"]
                record["logical_dependencies"] = [k_id]
            elif record["kind"] == "PIMOperandResidency" and record.get("op") == "v_prefill_tile_residency":
                record["tensor_io"]["inputs"] = [f"L{layer_index}.V"]
                record["logical_dependencies"] = [v_id]
        terminal_attention_ids = _terminal_attention_record_ids(attention)
        terminal_attention_tensors = [
            output
            for record in attention
            if record["record_id"] in terminal_attention_ids
            for output in record.get("tensor_io", {}).get("outputs", [])
        ]
        combined.extend(attention)

        output_projection, next_id = _renumber_records(_generate_decode_v2_output_projection_records(one_ffn, q_input_dim=q_dim), next_id)
        output_projection = [_retarget_layer_zero_record(record, layer_index) for record in output_projection]
        output_projection[0]["tensor_io"]["inputs"] = terminal_attention_tensors + [f"L{layer_index}.o_projection.weight.resident"]
        output_projection[0]["logical_dependencies"] = terminal_attention_ids
        o_id = output_projection[0]["record_id"]
        combined.extend(output_projection)

        ffn, next_id = _renumber_records(generate_ffn_records(one_ffn), next_id)
        ffn = [_retarget_layer_zero_record(record, layer_index) for record in ffn]
        for record in ffn:
            if record["op"] == "ffn_hidden_activation_tile_setup":
                record["tensor_io"]["inputs"] = [f"L{layer_index}.attention_output"]
                record["logical_dependencies"] = [o_id]
        combined.extend(ffn)

        barrier, next_id = _renumber_records([_barrier_record("rec_0000", layer_index, attention_manifest)], next_id)
        combined.extend(barrier)

    drain, next_id = _renumber_records([_drain_record("rec_0000", max(0, num_layers - 1), attention_manifest)], next_id)
    combined.extend(drain)
    for record in combined:
        validate_record(record)
    return combined


def generate_llama2_7b_dense_decoder_records(
    *, attention_manifest: dict | None = None, ffn_manifest: dict | None = None
) -> list[dict]:
    if attention_manifest is None or ffn_manifest is None:
        default_attention, default_ffn = get_llama2_7b_dense_decoder_manifests()
        attention_manifest = default_attention if attention_manifest is None else attention_manifest
        ffn_manifest = default_ffn if ffn_manifest is None else ffn_manifest
    return generate_dense_transformer_layer_records(attention_manifest=attention_manifest, ffn_manifest=ffn_manifest)


def generate_llama2_13b_dense_decoder_records(
    *, attention_manifest: dict | None = None, ffn_manifest: dict | None = None
) -> list[dict]:
    if attention_manifest is None or ffn_manifest is None:
        default_attention, default_ffn = get_llama2_13b_dense_decoder_manifests()
        attention_manifest = default_attention if attention_manifest is None else attention_manifest
        ffn_manifest = default_ffn if ffn_manifest is None else ffn_manifest
    return generate_dense_transformer_layer_records(attention_manifest=attention_manifest, ffn_manifest=ffn_manifest)


def generate_dense_prefill_records_from_spec(
    spec: ModelSpec,
    *,
    prompt_len: int,
    schedule_policy: str = "serialized",
) -> list[dict]:
    """Generate full dense prefill records from a ModelSpec."""
    attn, ffn = get_dense_prefill_manifests(spec, prompt_len=prompt_len, schedule_policy=schedule_policy)
    return generate_dense_prefill_transformer_layer_records(attention_manifest=attn, ffn_manifest=ffn)


def generate_dense_prefill_records_for_model(
    model_name: str,
    *,
    prompt_len: int,
    schedule_policy: str = "serialized",
) -> list[dict]:
    """Generate full dense prefill records by model registry name."""
    return generate_dense_prefill_records_from_spec(
        get_model_spec(model_name), prompt_len=prompt_len, schedule_policy=schedule_policy
    )


def get_dense_decoder_manifests(
    spec: ModelSpec,
    *,
    past_len: int | None = None,
    schedule_policy: str = "serialized",
) -> tuple[dict, dict]:
    """Build attention+FFN manifest pair for any dense decoder ModelSpec."""
    attn = get_llama2_dense_decoder_attention_manifest(
        spec,
        past_len=past_len if past_len is not None else LLAMA2_7B_DEFAULT_PAST_LEN,
        schedule_policy=schedule_policy,
    )
    ffn = get_llama2_dense_decoder_ffn_manifest(spec, schedule_policy=schedule_policy)
    return attn, ffn


def generate_dense_decoder_records_from_spec(
    spec: ModelSpec,
    *,
    past_len: int | None = None,
    schedule_policy: str = "serialized",
) -> list[dict]:
    """Generate full dense decoder records from a ModelSpec."""
    attn, ffn = get_dense_decoder_manifests(spec, past_len=past_len, schedule_policy=schedule_policy)
    return generate_dense_transformer_layer_records(attention_manifest=attn, ffn_manifest=ffn)


def generate_dense_decoder_records_for_model(
    model_name: str,
    *,
    past_len: int | None = None,
    schedule_policy: str = "serialized",
) -> list[dict]:
    """Generate full dense decoder records by model registry name."""
    return generate_dense_decoder_records_from_spec(
        get_model_spec(model_name), past_len=past_len, schedule_policy=schedule_policy
    )


# ── Qwen2.5 convenience generators ────────────────────────────────────

def generate_qwen25_7b_dense_decoder_records(
    *, attention_manifest: dict | None = None, ffn_manifest: dict | None = None
) -> list[dict]:
    if attention_manifest is None or ffn_manifest is None:
        default_attn, default_ffn = get_dense_decoder_manifests(QWEN25_7B_MODEL_SPEC)
        attention_manifest = default_attn if attention_manifest is None else attention_manifest
        ffn_manifest = default_ffn if ffn_manifest is None else ffn_manifest
    return generate_dense_transformer_layer_records(attention_manifest=attention_manifest, ffn_manifest=ffn_manifest)


def generate_qwen25_14b_dense_decoder_records(
    *, attention_manifest: dict | None = None, ffn_manifest: dict | None = None
) -> list[dict]:
    if attention_manifest is None or ffn_manifest is None:
        default_attn, default_ffn = get_dense_decoder_manifests(QWEN25_14B_MODEL_SPEC)
        attention_manifest = default_attn if attention_manifest is None else attention_manifest
        ffn_manifest = default_ffn if ffn_manifest is None else ffn_manifest
    return generate_dense_transformer_layer_records(attention_manifest=attention_manifest, ffn_manifest=ffn_manifest)


def generate_qwen25_32b_dense_decoder_records(
    *, attention_manifest: dict | None = None, ffn_manifest: dict | None = None
) -> list[dict]:
    if attention_manifest is None or ffn_manifest is None:
        default_attn, default_ffn = get_dense_decoder_manifests(QWEN25_32B_MODEL_SPEC)
        attention_manifest = default_attn if attention_manifest is None else attention_manifest
        ffn_manifest = default_ffn if ffn_manifest is None else ffn_manifest
    return generate_dense_transformer_layer_records(attention_manifest=attention_manifest, ffn_manifest=ffn_manifest)


def generate_qwen25_72b_dense_decoder_records(
    *, attention_manifest: dict | None = None, ffn_manifest: dict | None = None
) -> list[dict]:
    if attention_manifest is None or ffn_manifest is None:
        default_attn, default_ffn = get_dense_decoder_manifests(QWEN25_72B_MODEL_SPEC)
        attention_manifest = default_attn if attention_manifest is None else attention_manifest
        ffn_manifest = default_ffn if ffn_manifest is None else ffn_manifest
    return generate_dense_transformer_layer_records(attention_manifest=attention_manifest, ffn_manifest=ffn_manifest)


# ── Gemma convenience generators ──────────────────────────────────────

def generate_gemma_2b_dense_decoder_records(
    *, attention_manifest: dict | None = None, ffn_manifest: dict | None = None
) -> list[dict]:
    if attention_manifest is None or ffn_manifest is None:
        default_attn, default_ffn = get_dense_decoder_manifests(GEMMA_2B_MODEL_SPEC)
        attention_manifest = default_attn if attention_manifest is None else attention_manifest
        ffn_manifest = default_ffn if ffn_manifest is None else ffn_manifest
    return generate_dense_transformer_layer_records(attention_manifest=attention_manifest, ffn_manifest=ffn_manifest)


def generate_gemma_7b_dense_decoder_records(
    *, attention_manifest: dict | None = None, ffn_manifest: dict | None = None
) -> list[dict]:
    if attention_manifest is None or ffn_manifest is None:
        default_attn, default_ffn = get_dense_decoder_manifests(GEMMA_7B_MODEL_SPEC)
        attention_manifest = default_attn if attention_manifest is None else attention_manifest
        ffn_manifest = default_ffn if ffn_manifest is None else ffn_manifest
    return generate_dense_transformer_layer_records(attention_manifest=attention_manifest, ffn_manifest=ffn_manifest)


def generate_gemma2_9b_dense_decoder_records(
    *, attention_manifest: dict | None = None, ffn_manifest: dict | None = None
) -> list[dict]:
    if attention_manifest is None or ffn_manifest is None:
        default_attn, default_ffn = get_dense_decoder_manifests(GEMMA2_9B_MODEL_SPEC)
        attention_manifest = default_attn if attention_manifest is None else attention_manifest
        ffn_manifest = default_ffn if ffn_manifest is None else ffn_manifest
    return generate_dense_transformer_layer_records(attention_manifest=attention_manifest, ffn_manifest=ffn_manifest)


def generate_gemma2_27b_dense_decoder_records(
    *, attention_manifest: dict | None = None, ffn_manifest: dict | None = None
) -> list[dict]:
    if attention_manifest is None or ffn_manifest is None:
        default_attn, default_ffn = get_dense_decoder_manifests(GEMMA2_27B_MODEL_SPEC)
        attention_manifest = default_attn if attention_manifest is None else attention_manifest
        ffn_manifest = default_ffn if ffn_manifest is None else ffn_manifest
    return generate_dense_transformer_layer_records(attention_manifest=attention_manifest, ffn_manifest=ffn_manifest)


def _summary_manifest_fields(manifest: dict | None = None, manifest_name: str | None = None) -> dict:
    if manifest is not None:
        return {
            "manifest_version": manifest.get("manifest_version"),
            "manifest_name": manifest.get("manifest_name"),
            "phase": manifest.get("phase"),
            "datatype": manifest.get("datatype"),
            "non_claims": list(manifest.get("non_claims", [])),
            "mapping_policy": _mapping_policy(manifest),
        }
    return {
        "manifest_name": manifest_name or "unknown",
        "non_claims": [
            "not_runtime_replay",
            "not_vllm_replay",
            "not_numerical_correctness",
            "not_silicon_faithful_softmax_or_data_movement",
            "not_raw_attacc_schema",
        ],
    }


def _build_provenance_summary(
    records: list[dict],
    manifest_or_name: dict | str | None = None,
    *,
    manifest_name: str | None = None,
    notes: str,
    compute_kinds: set[str],
) -> dict:
    counts: dict[str, int] = {}
    for record in records:
        counts[record["kind"]] = counts.get(record["kind"], 0) + 1
    manifest_fields = _summary_manifest_fields(
        manifest_or_name if isinstance(manifest_or_name, dict) else None,
        manifest_name if manifest_name is not None else (manifest_or_name if isinstance(manifest_or_name, str) else None),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": FULL_TRANSFORMER_GENERATOR_VERSION,
        **manifest_fields,
        "record_counts_by_kind": counts,
        "total_logical_records": len(records),
        "total_expanded_records": expanded_record_count(records),
        "estimated_pim_compute_requests": sum(
            int(record.get("num_requests", 0))
            for record in records
            if record["kind"] in compute_kinds
        ),
        "notes": notes,
    }


def build_full_transformer_provenance_summary(records: list[dict], manifest_or_name: dict | str | None = None, *, manifest_name: str | None = None) -> dict:
    return _build_provenance_summary(
        records,
        manifest_or_name,
        manifest_name=manifest_name,
        notes="combined P4 attention+FFN+MoE offline semantic DAG summary; concrete opcode trace is the replay path",
        compute_kinds={"AttentionScore", "AttentionContext", "FFNProjection", "MoERouter", "MoEExpertFFN"},
    )


def build_dense_decoder_provenance_summary(
    records: list[dict], manifest_or_name: dict | str | None = None, *, manifest_name: str | None = None
) -> dict:
    return _build_provenance_summary(
        records,
        manifest_or_name,
        manifest_name=manifest_name,
        notes="dense P4 attention+FFN offline semantic DAG summary; concrete opcode trace is the replay path",
        compute_kinds={"AttentionScore", "AttentionContext", "FFNProjection"},
    )


def build_provenance_summary(records: list[dict], manifest_or_name: dict | str | None = None, *, manifest_name: str | None = None) -> dict:
    return _build_provenance_summary(
        records,
        manifest_or_name,
        manifest_name=manifest_name,
        notes="P4 semantic DAG summary; softmax remains semantic/accounting-only",
        compute_kinds={"AttentionScore", "AttentionContext"},
    )


def generate_attention_artifacts(output_dir: Path | str = DEFAULT_OUTPUT_DIR, manifest: dict | None = None) -> tuple[Path, Path]:
    manifest = get_tiny_attention_manifest() if manifest is None else manifest
    _validate_attention_manifest(manifest)
    records = generate_attention_records(manifest)
    summary = build_provenance_summary(records, manifest)
    output_dir = Path(output_dir)
    trace_path = output_dir / "structured_trace.jsonl"
    summary_path = output_dir / "provenance_summary.json"
    write_jsonl(records, trace_path)
    write_json(summary, summary_path)
    return trace_path, summary_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate P4 transformer semantic tensor-DAG artifacts")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name from registry to generate dense decoder traces for "
        f"(choices: {sorted(MODEL_REGISTRY)}). "
        "If omitted, generates tiny attention artifacts.",
    )
    parser.add_argument(
        "--past-len",
        type=int,
        default=None,
        help="Override past sequence length (default: model-specific).",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Override number of layers to replay (default: model total layers).",
    )
    return parser


def main() -> int:
    opts = _build_arg_parser().parse_args()
    if opts.model:
        spec = get_model_spec(opts.model)
        past_len = opts.past_len if opts.past_len is not None else LLAMA2_7B_DEFAULT_PAST_LEN
        attn, ffn = get_dense_decoder_manifests(spec, past_len=past_len)
        if opts.num_layers is not None:
            attn["num_layers"] = opts.num_layers
            ffn["num_layers"] = opts.num_layers
        records = generate_dense_transformer_layer_records(attention_manifest=attn, ffn_manifest=ffn)
        summary = build_dense_decoder_provenance_summary(records, attn)
        output_dir = Path(opts.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_slug = spec.name.lower().replace("-", "_").replace(".", "")
        trace_path = output_dir / f"{model_slug}_dense_decoder_trace.jsonl"
        summary_path = output_dir / f"{model_slug}_dense_decoder_provenance.json"
        write_jsonl(records, trace_path)
        write_json(summary, summary_path)
        print(f"Generated {len(records)} records for {spec.name}")
        print(f"  Trace: {trace_path}")
        print(f"  Summary: {summary_path}")
    else:
        trace_path, summary_path = generate_attention_artifacts(output_dir=opts.output_dir)
        print(f"Generated: {trace_path}")
        print(f"Generated: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
