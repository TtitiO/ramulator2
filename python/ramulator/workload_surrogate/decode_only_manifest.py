"""Machine-readable decode-only workload-surrogate tuple manifest."""

from __future__ import annotations

from copy import deepcopy


DECODE_ONLY_MANIFEST = {
    "manifest_name": "decode_only_tuple",
    "manifest_version": "v0.1",
    "trace_contract_version": "v0.1",
    "lifecycle_status": "legacy_deprecated_p2_mvp",
    "replacement_generator": "ramulator.workload_surrogate.generate_full_transformer.generate_llama2_7b_dense_decoder_records",
    "deprecation_note": (
        "Legacy P2 MVP replay fixture; does not implement "
        "paper/algorithms/llama2_decode_trace_algorithm.tex. "
        "Use generate_full_transformer.generate_llama2_7b_dense_decoder_records for decode-block v2."
    ),
    "workload_class": "decode-heavy structured surrogate",
    "phase": "decode",
    "batch": 1,
    "generated_tokens": 1,
    "past_len": 128,
    "hidden_size": 4096,
    "num_layers": 32,
    "num_heads": 32,
    "head_dim": 128,
    "ffn_hidden_size": 11008,
    "datatype": "int8",
    "model_family": "LLaMA-like decoder-only transformer",
    "pim_compute_operator_classes": [
        "q_projection_gemv",
        "k_projection_gemv",
        "v_projection_gemv",
        "o_projection_gemv",
        "ffn_up_projection_gemv",
        "ffn_gate_projection_gemv",
        "ffn_down_projection_gemv",
    ],
    "host_support_record_classes": [
        "attention_k_cache_read",
        "attention_v_cache_read",
    ],
    "host_only_provenance_classes": [
        "softmax",
        "layernorm",
        "sampling",
        "token_selection",
        "rope_or_position_update",
        "control_flow",
        "kv_cache_append_accounting",
    ],
    "non_claims": [
        "does_not_implement_paper_algorithm",
        "legacy_phase2_mvp_replay_fixture",
    ],
    "ramulator_visible_defaults": {
        "bank_sequence": [0, 1, 2, 3],
        "bank_sequence_order": "controller",
        "pim_banks_per_mpu": 2,
        "burst_length": 8,
        "row_start": 0,
        "row_count": 2,
        "dependency_count": 2,
        "tx_bytes": 64,
        "column_start": 0,
    },
    "pim_operator_request_widths": {
        "q_projection_gemv": "hidden_size",
        "k_projection_gemv": "hidden_size",
        "v_projection_gemv": "hidden_size",
        "o_projection_gemv": "hidden_size",
        "ffn_up_projection_gemv": "ffn_hidden_size",
        "ffn_gate_projection_gemv": "ffn_hidden_size",
        "ffn_down_projection_gemv": "hidden_size",
    },
    "scaffolding_notes": {
        "deterministic_mvp_scaffolding": True,
        "bank_count": 4,
        "bank_sequence": [0, 1, 2, 3],
        "mpu_per_bank_pair": 2,
        "row_count": 2,
        "column_dependency_count": 2,
        "dependency_notes": "fixed bank/dependency defaults are deterministic MVP scaffolding, not universal decode semantics",
    },
    "mapping_policy": {
        "host_policy": "bounded_sequential_host_requests",
        "pim_policy": "per_bank_pimcompute_using_shared_mpu_serial",
        "bank_sequence_policy": "controller_visible_round_robin_4bank",
        "mpu_grouping_policy": "pim_banks_per_mpu=2_shared_mpu_serial",
        "controller_bank_order": True,
        "row_layout_policy": "bounded_row_window_decode_only",
        "kv_cache_policy": "host_visible_kv_reads_and_bounded_append_accounting",
    },
    "literature_anchors": [
        "2411.17309:decode_is_memory_bound_gemv",
        "FACIL:decode_bottleneck_and_gemv_profile",
        "2601.12298:decode_gemv_and_memory_bandwidth_bottleneck",
        "SamsungBreakthrough:generation_phase_dominates",
    ],
}


def get_decode_only_manifest() -> dict:
    """Return a deep copy of the frozen decode-only manifest."""
    return deepcopy(DECODE_ONLY_MANIFEST)
