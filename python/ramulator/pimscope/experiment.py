"""Run validated LPDDR-PIM workload experiments."""

import multiprocessing
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping

from ramulator.dram.addressing import extract_dram_layout
from ramulator.pimscope.backend import (
    count_concrete_opcodes,
    create_dram,
    hardware_config_from_manifest,
    replay_concrete_trace,
)
from ramulator.pimscope.config import ResolvedExperiment
from ramulator.pimscope.schema import RESULT_SCHEMA_NAME, RESULT_SCHEMA_VERSION
from ramulator.pimscope.workloads import generate_workload
from ramulator.workload_surrogate.lpddr5_pim_concrete_trace import (
    build_header,
    expanded_record_count,
    slim_record,
    stable_json_dumps,
)

RAMULATOR_ROOT = Path(__file__).resolve().parents[3]


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in ("ramulator", "pimscope", "numpy", "matplotlib", "PyYAML"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


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
            "subchannel_model": layout["subchannel_model"],
        },
        "dram_config": dram.to_config(),
    }


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


def _process_worker(connection, function, args: tuple, kwargs: dict[str, Any]) -> None:
    try:
        connection.send((True, function(*args, **kwargs)))
    except Exception as exc:
        connection.send((False, f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


def _run_process(function, args: tuple, kwargs: dict[str, Any], timeout_seconds: int):
    method = "spawn"
    main_file = getattr(sys.modules["__main__"], "__file__", None)
    if os.name != "nt" and (not main_file or str(main_file).startswith("<")):
        method = "fork"
    context = multiprocessing.get_context(method)
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_process_worker, args=(child, function, args, kwargs))
    process.start()
    child.close()
    deadline = time.monotonic() + timeout_seconds
    while process.is_alive() and not parent.poll(0.1):
        if time.monotonic() >= deadline:
            process.terminate()
            process.join()
            parent.close()
            raise TimeoutError(f"operation exceeded {timeout_seconds} seconds")
    if not parent.poll():
        process.join()
        parent.close()
        raise RuntimeError(f"operation process exited with code {process.exitcode}")
    try:
        success, payload = parent.recv()
    except EOFError as exc:
        process.join()
        parent.close()
        raise RuntimeError(f"operation process exited with code {process.exitcode}") from exc
    parent.close()
    process.join()
    if not success:
        raise RuntimeError(payload)
    return payload


def _replay(concrete: list[dict], kwargs: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
    if timeout_seconds == 0:
        return replay_concrete_trace(concrete, **kwargs)
    return _run_process(replay_concrete_trace, (concrete,), kwargs, timeout_seconds)


def estimate_concrete_trace(concrete: list[dict], dram_class: str) -> dict[str, int]:
    trace_bytes = len(stable_json_dumps(build_header(dram_class=dram_class)).encode("utf-8")) + 1
    trace_bytes += sum(
        len(stable_json_dumps(slim_record(record)).encode("utf-8")) + 1 for record in concrete
    )
    return {
        "records": len(concrete),
        "expanded_records": expanded_record_count(concrete),
        "trace_bytes": trace_bytes,
    }


def _materialization_summary(
    concrete: list[dict], workload: Mapping[str, Any], tx_bytes: int
) -> dict[str, Any]:
    write_records = [
        record
        for record in concrete
        if record["opcode"] == "WRITE"
        and record.get("provenance", {}).get("semantic_source", {}).get("kind") == "PIMDataMove"
    ]
    write_transactions = sum(int(record.get("repeat", 1)) for record in write_records)
    return {
        "policy": workload["weight_residency"],
        "synthetic": True,
        "address_policy": "bounded_surrogate_v1",
        "write_records": len(write_records),
        "write_transactions": write_transactions,
        "generated_bytes": write_transactions * tx_bytes,
    }


def _check_preflight(preflight: Mapping[str, int], workload: Mapping[str, Any]) -> None:
    for field in ("expanded_records", "trace_bytes"):
        limit = workload[f"max_{field}"]
        if preflight[field] > limit:
            raise ValueError(
                f"workload.max_{field}: estimated {preflight[field]} exceeds limit {limit}"
            )


def _check_materialization(summary: Mapping[str, Any], policy: str) -> None:
    transactions = summary["write_transactions"]
    if policy == "resident" and transactions:
        raise ValueError("resident weight policy must not emit synthetic preload writes")
    if policy == "full_preload" and not transactions:
        raise ValueError("full_preload weight policy must emit synthetic preload writes")


def _effective_inflight(workload: Mapping[str, Any], semantic: list[dict]) -> int:
    if workload["mac_mode"] == "all_bank":
        return 1
    requested = workload["max_inflight_requests"]
    if requested <= 1 or workload["mac_mode"] not in {"per_kind", "per_bank"}:
        return requested
    max_span = max(
        (len(record["bank_sequence"]) for record in semantic if record.get("bank_sequence")),
        default=1,
    )
    return max(requested, max_span * workload["interleave_depth"])


def _provenance(resolved: ResolvedExperiment, supplied: Mapping[str, Any] | None) -> dict[str, Any]:
    workload = resolved.manifest["workload"]
    payload: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "seed": workload["seed"],
        "ramulator2_commit": _git_revision(RAMULATOR_ROOT),
        "python_packages": _package_versions(),
        "config_source": (
            Path(resolved.source).name if resolved.source != "<memory>" else resolved.source
        ),
    }
    if supplied:
        payload.update(dict(supplied))
    return payload


def run_experiment(
    resolved: ResolvedExperiment,
    *,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one validated workload manifest."""
    from ramulator.workload_surrogate.generate_lpddr5_pim_concrete import (
        lower_semantic_records_to_concrete,
    )

    manifest = resolved.manifest
    workload = manifest["workload"]
    backend_cfg = hardware_config_from_manifest(resolved)
    dram = create_dram(backend_cfg)
    organization, timing = dram.resolve()
    semantic, model_name = generate_workload(workload)
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
    preflight = estimate_concrete_trace(concrete, manifest["hardware"]["dram_class"])
    _check_preflight(preflight, workload)
    materialization = _materialization_summary(concrete, workload, layout["tx_bytes"])
    _check_materialization(materialization, workload["weight_residency"])
    effective_inflight = _effective_inflight(workload, semantic)

    replay = _replay(
        concrete,
        {
            "max_trace_bytes": workload["max_trace_bytes"],
            "max_expanded_records": workload["max_expanded_records"],
            "max_inflight_requests": effective_inflight,
            "backend_cfg": backend_cfg,
        },
        workload["simulation_timeout_seconds"],
    )
    provenance_payload = _provenance(resolved, provenance)

    result = {
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
                "dram_class": manifest["hardware"]["dram_class"],
                "mapping_version": layout["mapping_version"],
                "level_names": layout["level_names"],
                "level_sizes": layout["level_sizes"],
                "address_level_sizes": layout["address_level_sizes"],
                "internal_prefetch_size": layout["internal_prefetch_size"],
                "tx_bytes": layout["tx_bytes"],
                "capacity_bytes": layout["capacity_bytes"],
                "subchannel_model": layout["subchannel_model"],
            },
            "effective_max_inflight_requests": effective_inflight,
            "preflight": {
                **preflight,
                "max_expanded_records": workload["max_expanded_records"],
                "max_trace_bytes": workload["max_trace_bytes"],
            },
        },
        "workload_summary": {
            "seed": workload["seed"],
            "workload_type": workload["workload_type"],
            "model": model_name,
            "phase": workload["phase"],
            "semantic_records": len(semantic),
            "concrete_records": len(concrete),
            "concrete_opcode_counts": count_concrete_opcodes(concrete),
            "host_address_policy": "bounded_surrogate_v1",
            "weight_materialization": materialization,
            "surrogate_scope": {
                "architecture_dimensions": "published_or_user_supplied",
                "runtime_trace": False,
                "numerical_execution": False,
                "silicon_calibration": False,
                "weights": "synthetic_not_materialized",
                "host_operator_model": "partial",
            },
            "claim_boundary": "structured memory-side surrogate, not runtime or silicon trace",
        },
        "simulation": replay,
        "provenance": provenance_payload,
    }
    return result
