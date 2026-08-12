"""Public PIMScope experiment API."""

from ramulator.pimscope.backend import (
    count_concrete_opcodes,
    create_address_mapper,
    create_concrete_frontend,
    create_dram,
    create_memory_system,
    generate_and_replay,
    hardware_config_from_manifest,
    infer_model_family,
    pim_cfg_per_bank,
    pim_cfg_shared,
    prefill_formula,
    replay_concrete_trace,
    time_unit_ns,
)
from ramulator.pimscope.capabilities import (
    PIM_BACKEND_CAPABILITIES,
    pim_backend_capabilities,
    require_supported_pim_backend,
)
from ramulator.pimscope.config import (
    MANIFEST_SCHEMA_VERSION,
    ResolvedExperiment,
    apply_overrides,
    load_experiment_manifest,
    load_raw_manifest,
    resolve_experiment_manifest,
)
from ramulator.pimscope.experiment import (
    estimate_concrete_trace,
    retarget_semantic_banks,
    run_experiment,
    validate_backend,
)
from ramulator.pimscope.runner import run_single
from ramulator.pimscope.schema import (
    AGGREGATE_SCHEMA_NAMES,
    AGGREGATE_SCHEMA_VERSION,
    RESULT_SCHEMA_NAME,
    RESULT_SCHEMA_VERSION,
    load_json_object,
    validate_aggregate,
    validate_result,
    validate_trace_file,
)
from ramulator.pimscope.workloads import SUPPORTED_MODEL_PHASES, generate_workload

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "PIM_BACKEND_CAPABILITIES",
    "pim_backend_capabilities",
    "require_supported_pim_backend",
    "AGGREGATE_SCHEMA_NAMES",
    "AGGREGATE_SCHEMA_VERSION",
    "RESULT_SCHEMA_NAME",
    "RESULT_SCHEMA_VERSION",
    "load_json_object",
    "validate_aggregate",
    "validate_result",
    "validate_trace_file",
    "ResolvedExperiment",
    "apply_overrides",
    "load_experiment_manifest",
    "load_raw_manifest",
    "resolve_experiment_manifest",
    "validate_backend",
    "estimate_concrete_trace",
    "retarget_semantic_banks",
    "run_experiment",
    "count_concrete_opcodes",
    "create_address_mapper",
    "create_concrete_frontend",
    "create_dram",
    "create_memory_system",
    "generate_and_replay",
    "hardware_config_from_manifest",
    "infer_model_family",
    "pim_cfg_per_bank",
    "pim_cfg_shared",
    "prefill_formula",
    "replay_concrete_trace",
    "time_unit_ns",
    "run_single",
    "generate_workload",
    "SUPPORTED_MODEL_PHASES",
]
