"""Backend-backed data collection for P4 full-transformer figures.

Runs P4 attention/FFN/MoE traces through the actual Ramulator2 LPDDR5-PIM
backend simulator and collects: command traces, per-opcode command counts,
controller timing/stall stats, and energy observability.

This produces real backend simulation data, not just generator output counts.
"""

from __future__ import annotations

import csv
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import ramulator

from tests.analysis.testcases.lpddr5_pim import CONFIG as LPDDR5_PIM_CONFIG
from tests.utils.dram import create_dram
from tests.analysis.figures._sim_helpers import _frontend


# ─── Trace reader helpers (adapted from tests/analysis/runner.py) ────


def _read_command_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not path.exists():
        return counts
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        command, count = line.split(",", maxsplit=1)
        counts[command.strip()] = int(count.strip())
    return dict(sorted(counts.items()))


def _read_command_trace(trace_path: Path) -> list[dict[str, Any]]:
    """Read a CmdTraceRecorder CSV output and return list of (clk, command) rows."""
    rows: list[dict[str, Any]] = []
    if not trace_path.exists():
        return rows
    with trace_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({
                "clk": int(row.get("clock", row.get("clk", 0))),
                "command": row.get("command", ""),
            })
    return rows


# ─── Simulation builder with observability plugins ────────────────────


def _make_mem_with_plugins(dram: Any, tmpdir: Path) -> Any:
    """Build memory system with CmdTraceRecorder and CommandCounter plugins."""
    counts_path = str(tmpdir / "command_counts.csv")
    trace_prefix = str(tmpdir / "command_trace.csv")

    command_counter = ramulator.controller_plugin.CommandCounter(
        commands_to_count=[
            "ACT1", "ACT2", "CAS_RD", "CAS_WR", "RD", "WR", "RDA", "WRA",
            "SB", "HAB", "HAB_PIM", "PIM_BCAST", "PIM_MAC", "PIM_MAC_AB",
            "PREpb", "PREab", "REFab",
        ],
        path=counts_path,
    )
    cmd_trace = ramulator.controller_plugin.CmdTraceRecorder(path=trace_prefix)

    ctrl = ramulator.controller.LPDDR5PIM(
        dram=dram,
        scheduler=ramulator.scheduler.FRFCFS(),
        refresh_manager=ramulator.refresh_manager.NoRefresh(),
        row_policy=ramulator.row_policy.Open(),
        addr_mapper=ramulator.addr_mapper.PassThroughAddrMapper(),
        controller_plugins=[command_counter, cmd_trace],
    )
    return ramulator.memory_system.GenericDRAM(
        clock_ratio=1,
        controllers=[ctrl],
        channel_mapper=ramulator.channel_mapper.CacheLineInterleave(),
    )


# ─── Main entry point: run one trace through the backend ──────────────


def run_trace_through_backend(
    semantic_records: list[dict],
    manifest_name: str,
    *,
    tmpdir: Path | None = None,
) -> dict[str, Any]:
    """Run a P4 semantic trace through the full Ramulator2 LPDDR5-PIM backend.

    Returns a dict with:
      - command_trace: list of {clk, command} rows from CmdTraceRecorder
      - command_counts: {command: count}
      - controller_stats: dict of key controller counters
      - cycles: total simulation cycles
      - frontend_stats: key frontend counters
    """
    from ramulator.workload_surrogate.generate_lpddr5_pim_concrete import (
        lower_semantic_records_to_concrete,
    )
    from ramulator.workload_surrogate.lpddr5_pim_concrete_trace import write_jsonl

    concrete = lower_semantic_records_to_concrete(
        semantic_records, manifest_name=manifest_name
    )

    use_own_tmpdir = tmpdir is None
    if use_own_tmpdir:
        tmpdir = Path(tempfile.mkdtemp())

    td = Path(tmpdir)
    trace_path = td / f"{manifest_name}_trace.jsonl"
    write_jsonl(concrete, trace_path)

    dram = create_dram(LPDDR5_PIM_CONFIG)
    frontend = _frontend(trace_path, dram, max_trace_bytes=1024 * 1024 * 1024)
    mem = _make_mem_with_plugins(dram, td)
    sim = ramulator.Simulation(frontend, mem)
    sim.run()

    ctrl = dict(sim.stats["memory_system"]["controller"])
    frontend_stats = dict(sim.stats["frontend"])

    cycles = int(ctrl.get("cycles", 0))
    tCK_ns = 0.625  # LPDDR5_6400

    # Read command trace
    cmd_trace_path = td / "command_trace.csv"
    # CmdTraceRecorder writes per-channel files like command_trace.csv.ch0
    cmd_traces: list[dict[str, Any]] = []
    ch_files = sorted(td.glob("command_trace.csv.ch*"))
    if ch_files:
        cmd_traces = _read_command_trace(ch_files[0])
    elif cmd_trace_path.exists():
        cmd_traces = _read_command_trace(cmd_trace_path)

    # Read command counts
    counts_path = td / "command_counts.csv"
    cmd_counts = _read_command_counts(counts_path)

    result = {
        "manifest_name": manifest_name,
        "semantic_records": len(semantic_records),
        "concrete_records": len(concrete),
        "cycles": cycles,
        "runtime_ns": cycles * tCK_ns,
        "command_trace": cmd_traces,
        "command_trace_total_commands": len(cmd_traces),
        "command_counts": cmd_counts,
        "controller_stats": ctrl,
        "frontend_stats": frontend_stats,
        "pim_mac_issued": ctrl.get("num_issued_pim_mac", ctrl.get("num_pim_reqs_served", 0)),
        "avg_pim_latency_cycles": ctrl.get("avg_pim_latency", 0),
        "pim_inflight_peak": ctrl.get("pim_inflight_peak", 0),
        "pim_capacity_stalls": ctrl.get("pim_capacity_stalls", 0),
        "pim_dependency_stalls": ctrl.get("pim_dependency_stalls", 0),
        "mpu_group_busy_cycles": ctrl.get("num_mpu_group_busy_blocked_cycles", 0),
        "effective_mpu_groups": ctrl.get("effective_mpu_groups", 0),
        "replay_ok": (
            frontend_stats.get("opcode_requests_completed", 0)
            == frontend_stats.get("opcode_requests_sent", 0)
        ),
    }

    if use_own_tmpdir:
        import shutil
        shutil.rmtree(str(td), ignore_errors=True)

    return result


def collect_all_backend_stats() -> dict[str, dict[str, Any]]:
    """Run attention, FFN, MoE, and combined traces through the backend.

    Returns dict keyed by operator type with full backend stats.
    """
    from ramulator.workload_surrogate.generate_full_transformer import (
        generate_attention_records,
        generate_ffn_records,
        generate_moe_records,
        generate_full_transformer_layer_records,
        get_tiny_attention_manifest,
        get_tiny_ffn_manifest,
        get_tiny_moe_manifest,
    )

    results = {}

    # Attention (serialized)
    attn = get_tiny_attention_manifest()
    attn["past_len"] = 32  # keep fast
    attn["schedule_policy"] = "serialized"
    results["attention_serialized"] = run_trace_through_backend(
        generate_attention_records(attn), "attention_serialized"
    )

    # Attention (overlapped)
    attn_ov = dict(attn)
    attn_ov["schedule_policy"] = "overlap_independent_heads"
    attn_ov["num_heads"] = 2
    results["attention_overlapped"] = run_trace_through_backend(
        generate_attention_records(attn_ov), "attention_overlapped"
    )

    # FFN/SwiGLU
    results["ffn_swiglu"] = run_trace_through_backend(
        generate_ffn_records(), "ffn_swiglu"
    )

    # MoE
    results["moe_top2"] = run_trace_through_backend(
        generate_moe_records(), "moe_top2"
    )

    # Combined
    results["combined_layer"] = run_trace_through_backend(
        generate_full_transformer_layer_records(), "combined_layer"
    )

    return results


def collect_all_backend_stats_paper() -> dict[str, dict[str, Any]]:
    """Run paper-scale traces through the backend using realistic models.

    Uses OPT-125M for attention/FFN and Mixtral-style for MoE.
    Returns dict keyed by operator type with full backend stats.
    """
    from ramulator.workload_surrogate.generate_full_transformer import (
        generate_attention_records,
        generate_ffn_records,
        generate_moe_records,
        generate_full_transformer_layer_records,
    )

    try:
        from tests.analysis.figures.p4_paper_manifests import (
            get_opt_125m_attention_manifest,
            get_opt_125m_ffn_manifest,
            get_mixtral_style_moe_manifest,
        )
        attn_default = get_opt_125m_attention_manifest
        ffn_default = get_opt_125m_ffn_manifest
        moe_default = get_mixtral_style_moe_manifest
    except ImportError:
        from ramulator.workload_surrogate.generate_full_transformer import (
            get_tiny_attention_manifest,
            get_tiny_ffn_manifest,
            get_tiny_moe_manifest,
        )
        attn_default = get_tiny_attention_manifest
        ffn_default = get_tiny_ffn_manifest
        moe_default = get_tiny_moe_manifest

    results = {}

    # Attention (serialized) — OPT-125M
    attn = attn_default(past_len=256, schedule_policy="serialized")
    results["attention_serialized"] = run_trace_through_backend(
        generate_attention_records(attn), "opt125m_attention_serialized"
    )

    # Attention (overlapped)
    attn_ov = attn_default(past_len=256, schedule_policy="overlap_independent_heads")
    results["attention_overlapped"] = run_trace_through_backend(
        generate_attention_records(attn_ov), "opt125m_attention_overlapped"
    )

    # FFN/SwiGLU — OPT-125M
    ffn = ffn_default(schedule_policy="serialized")
    results["ffn_swiglu"] = run_trace_through_backend(
        generate_ffn_records(ffn), "opt125m_ffn_swiglu"
    )

    # MoE — Mixtral-style
    moe = moe_default(schedule_policy="serialized")
    results["moe_top2"] = run_trace_through_backend(
        generate_moe_records(moe), "mixtral_moe_top2"
    )

    # Combined layer — OPT-125M attention + FFN + Mixtral MoE
    combined = generate_full_transformer_layer_records(
        attention_manifest=attn,
        ffn_manifest=ffn,
        moe_manifest=moe,
    )
    results["combined_layer"] = run_trace_through_backend(
        combined, "paper_combined_layer"
    )

    return results
