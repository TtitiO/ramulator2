"""Structured transformer workload configuration and trace generation."""

from typing import Any

DENSE_MODEL_KEYS = (
    "llama2-7b",
    "llama2-13b",
    "llama2-70b",
    "opt-125m",
    "opt-350m",
    "opt-1.3b",
    "qwen25-7b",
    "qwen25-14b",
    "qwen25-32b",
    "qwen25-72b",
    "gemma-2b",
    "gemma-7b",
    "gemma2-9b",
    "gemma2-27b",
)
MIXTRAL_MODEL_KEY = "mixtral-8x7b"
SUPPORTED_MODEL_PHASES = {
    **{model: ("decode", "prefill") for model in DENSE_MODEL_KEYS},
    MIXTRAL_MODEL_KEY: ("decode",),
}


def generate_workload(workload: dict[str, Any]) -> tuple[list[dict], str]:
    """Generate semantic records from a resolved workload configuration."""
    from ramulator.workload_surrogate.generate_full_transformer import (
        ModelSpec,
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
    if model == MIXTRAL_MODEL_KEY:
        attention, moe = get_mixtral_8x7b_moe_decoder_manifests(
            past_len=workload["past_len"], schedule_policy=schedule
        )
        records = generate_mixtral_8x7b_decoder_records(
            attention_manifest=attention, moe_manifest=moe
        )
        return records, model
    if isinstance(model, dict):
        spec = ModelSpec(
            name=model["name"],
            num_layers=model["num_layers"],
            hidden_size=model["hidden_size"],
            num_heads=model["num_heads"],
            num_kv_heads=model.get("num_kv_heads"),
            head_dim=model["head_dim"],
            ffn_hidden_size=model["ffn_hidden_size"],
            datatype=workload["datatype"],
            citation=model.get("citation"),
            paper_anchor=model.get("paper_anchor"),
            ffn_variant=model.get("ffn_variant", "swiglu_3proj"),
            activation=model.get("activation", "silu"),
        )
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
