"""Small direct Ramulator 2.1 runner used for smoke and observability checks."""

from __future__ import annotations

import copy
import csv
import tempfile
from collections import Counter
from pathlib import Path

from ramulator.dram.addressing import extract_dram_layout

DEFAULT_CFG = {
    "org_preset": "LPDDR5_8Gb_x16",
    "timing_preset": "LPDDR5_6400",
    "dram_kwargs": {"pim_datatype": "int8"},
    "frontend_clock_ratio": 4,
    "stream_cls": 8,
    "seed": 12345,
}

COMMANDS_TO_COUNT = [
    "ACT1",
    "ACT2",
    "CAS_RD",
    "CAS_WR",
    "RD",
    "WR",
    "RDA",
    "WRA",
    "SB",
    "HAB",
    "HAB_PIM",
    "PIM_BCAST",
    "PIM_MAC",
    "PIM_MAC_AB",
    "PREpb",
    "PREab",
    "REFab",
]


def _merge_cfg(base: dict, override: dict | None) -> dict:
    merged = copy.deepcopy(base)
    if not override:
        return merged
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_cfg(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _extract_dram_layout(dram) -> dict:
    """Compatibility wrapper for the canonical hierarchy-aware layout."""
    return extract_dram_layout(dram)


def _make_dram(ramulator, cfg: dict):
    return ramulator.dram.LPDDR5PIM(
        org_preset=cfg["org_preset"],
        timing_preset=cfg["timing_preset"],
        **cfg.get("dram_kwargs", {}),
    )


def _read_command_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not path.exists():
        return counts
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        command, count = [part.strip() for part in line.split(",", maxsplit=1)]
        counts[command] = int(count)
    return dict(sorted(counts.items()))


def _read_command_traces(prefix: Path) -> list[dict]:
    traces = []
    for trace_path in sorted(prefix.parent.glob(f"{prefix.name}.ch*")):
        with trace_path.open(newline="", encoding="utf-8") as handle:
            commands = [row["command"] for row in csv.DictReader(handle)]
        traces.append(
            {
                "channel": trace_path.suffix.replace(".ch", ""),
                "command_count": len(commands),
                "command_counts": dict(sorted(Counter(commands).items())),
                "commands_preview": commands[:12],
            }
        )
    return traces


def _attach_plugins(ramulator, tmpdir: Path):
    counts_path = tmpdir / "command_counts.csv"
    trace_prefix = tmpdir / "command_trace.csv"
    return [
        ramulator.controller_plugin.CommandCounter(
            commands_to_count=COMMANDS_TO_COUNT, path=str(counts_path)
        ),
        ramulator.controller_plugin.CmdTraceRecorder(path=str(trace_prefix)),
    ]


def _collect_observability(stats: dict, tmpdir: Path, cfg: dict) -> dict:
    ctrl = stats.get("memory_system", {}).get("controller", {})
    selected = {}
    for key in (
        "cycles",
        "num_pim_reqs_served",
        "num_issued_pim_mac",
        "avg_pim_latency",
        "avg_pim_service_latency",
        "avg_pim_launch_wait",
        "avg_pim_response_latency",
        "pim_capacity_stalls",
        "pim_shared_block_stalls",
        "pim_dependency_stalls",
        "pim_inflight_peak",
        "pim_banks_per_block",
        "pim_shared_block_count",
        "total_banks",
        "effective_shared_blocks",
        "pim_ab_completion_latency_cycles",
    ):
        if key in ctrl:
            selected[key] = ctrl[key]
    return {
        "modeled": {
            "command_counts": _read_command_counts(tmpdir / "command_counts.csv"),
            "command_traces": _read_command_traces(tmpdir / "command_trace.csv"),
            "controller_stats": selected,
            "pim_datatype": cfg.get("dram_kwargs", {}).get("pim_datatype", "unknown"),
        }
    }


def _make_controller_and_mem(ramulator, dram, plugins):
    ctrl = ramulator.controller.LPDDR5PIM(
        dram=dram,
        scheduler=ramulator.scheduler.FRFCFS(),
        refresh_manager=ramulator.refresh_manager.NoRefresh(),
        row_policy=ramulator.row_policy.Open(),
        addr_mapper=ramulator.addr_mapper.PassThroughAddrMapper(),
        controller_plugins=plugins,
    )
    return ramulator.memory_system.GenericDRAM(
        clock_ratio=1,
        controllers=[ctrl],
        channel_mapper=ramulator.channel_mapper.PassThroughChannelMapper(),
    )


def run_single(
    dram=None,
    cfg_override: dict | None = None,
    nop: int = 1,
    num_probes: int = 100,
    warmup: int = 100,
    read_ratio: int = 100,
    seed: int | None = None,
    observability_dir: Path | None = None,
) -> dict:
    """Run one host-traffic LPDDR5-PIM smoke point.

    PIM command replay is handled by :mod:`ramulator.pimscope.backend`. This
    helper intentionally uses Ramulator 2.1's generic latency-throughput
    frontend and no longer passes parameters removed from that frontend.
    """
    import ramulator

    cfg = _merge_cfg(DEFAULT_CFG, cfg_override)
    resolved_seed = int(cfg["seed"] if seed is None else seed)
    if resolved_seed < 0:
        raise ValueError("seed must be a non-negative integer")
    dram = dram if dram is not None else _make_dram(ramulator, cfg)
    layout = _extract_dram_layout(dram)
    frontend = ramulator.frontend.LatencyThroughputTrace(
        clock_ratio=int(cfg["frontend_clock_ratio"]),
        nop_counter=int(nop),
        num_probe_requests=int(num_probes),
        latency_sample_count=int(num_probes),
        warmup_cycles=int(warmup),
        seed=resolved_seed,
        read_ratio=int(read_ratio),
        stream_cls=int(cfg.get("stream_cls", 8)),
        **layout,
    )
    with tempfile.TemporaryDirectory(dir=observability_dir) as tmp:
        tmpdir = Path(tmp)
        mem = _make_controller_and_mem(ramulator, dram, _attach_plugins(ramulator, tmpdir))
        sim = ramulator.Simulation(frontend, mem)
        sim.run()
        sim.finalize()
        stats = sim.stats
        stats.setdefault("evidence", {})["pim_energy_observability"] = _collect_observability(
            stats, tmpdir, cfg
        )
        stats["evidence"]["seed"] = resolved_seed
        return stats
