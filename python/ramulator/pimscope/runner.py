"""Direct Ramulator runner for smoke and observability checks."""

import copy
import csv
import tempfile
from collections import Counter
from pathlib import Path
from typing import Literal

from ramulator.dram.addressing import extract_dram_layout
from ramulator.pimscope.backend import create_dram, create_memory_system

DEFAULT_CFG = {
    "dram_class": "LPDDR5PIM",
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
    "CAS",
    "CAS_RD",
    "CAS_WR",
    "RD",
    "WR",
    "RDA",
    "WRA",
    "RD_S",
    "WR_S",
    "RDA_S",
    "WRA_S",
    "RD_L",
    "WR_L",
    "RDA_L",
    "WRA_L",
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


def _attach_plugins(ramulator, tmpdir: Path, dram):
    counts_path = tmpdir / "command_counts.csv"
    trace_prefix = tmpdir / "command_trace.csv"
    commands = [command for command in COMMANDS_TO_COUNT if command in type(dram).commands]
    return [
        ramulator.controller_plugin.CommandCounter(
            commands_to_count=commands, path=str(counts_path)
        ),
        ramulator.controller_plugin.CmdTraceRecorder(path=str(trace_prefix)),
    ]


def _collect_observability(stats: dict, output_dir: Path, cfg: dict) -> dict:
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
            "dram_class": cfg.get("dram_class", "LPDDR5PIM"),
            "command_counts": _read_command_counts(output_dir / "command_counts.csv"),
            "command_traces": _read_command_traces(output_dir / "command_trace.csv"),
            "controller_stats": selected,
            "pim_datatype": cfg.get("dram_kwargs", {}).get("pim_datatype", "unknown"),
        }
    }


def run_single(
    dram=None,
    cfg_override: dict | None = None,
    nop: int = 1,
    num_probes: int = 100,
    warmup: int = 100,
    read_ratio: int = 100,
    seed: int | None = None,
    observability_dir: Path | None = None,
    observability: Literal["preview", "persistent", "disabled"] = "preview",
) -> dict:
    """Run one host-traffic LPDDR PIM smoke point.

    PIM command replay is handled by :mod:`ramulator.pimscope.backend`. This
    helper uses Ramulator's generic latency-throughput frontend and selects the
    controller from the resolved PIM DRAM class.
    ``preview`` returns bounded in-memory observability without paths,
    ``persistent`` writes plugin output below ``observability_dir``, and
    ``disabled`` runs without observability plugins. This low-level diagnostic
    helper is separate from the manifest-based one-workload command.
    """
    import ramulator

    cfg = _merge_cfg(DEFAULT_CFG, cfg_override)
    resolved_seed = cfg["seed"] if seed is None else seed
    if isinstance(resolved_seed, bool) or not isinstance(resolved_seed, int) or resolved_seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if observability not in {"preview", "persistent", "disabled"}:
        raise ValueError("observability must be 'preview', 'persistent', or 'disabled'")
    if observability == "persistent" and observability_dir is None:
        raise ValueError("observability_dir is required for persistent observability")

    dram = dram if dram is not None else create_dram(cfg)
    layout = extract_dram_layout(dram)
    frontend = ramulator.frontend.LatencyThroughputTrace(
        clock_ratio=int(cfg["frontend_clock_ratio"]),
        nop_counter=int(nop),
        num_probe_requests=int(num_probes),
        latency_sample_count=int(num_probes),
        warmup_cycles=int(warmup),
        seed=resolved_seed,
        read_ratio=int(read_ratio),
        stream_cls=int(cfg.get("stream_cls", 8)),
        addr_vec_size=layout["addr_vec_size"],
        total_bank_units=layout["total_bank_units"],
        row_pos=layout["row_pos"],
        col_pos=layout["col_pos"],
        num_rows=layout["num_rows"],
        num_cols=layout["num_cols"],
        internal_prefetch_size=layout["internal_prefetch_size"],
        num_cls=layout["num_cls"],
        bank_positions=layout["bank_positions"],
        bank_counts=layout["bank_counts"],
    )

    def run(output_dir: Path | None) -> dict:
        plugins = _attach_plugins(ramulator, output_dir, dram) if output_dir else []
        memory_cfg = {
            **cfg,
            "controller": {
                "scheduler": "FRFCFS",
                "refresh_manager": "NoRefresh",
                "row_policy": "Open",
                "addr_mapper": "PassThroughAddrMapper",
            },
            "memory_system": {
                "clock_ratio": 1,
                "channel_mapper": "PassThroughChannelMapper",
            },
        }
        mem = create_memory_system(dram, memory_cfg, controller_plugins=plugins)
        sim = ramulator.Simulation(frontend, mem)
        sim.run()
        sim.finalize()
        stats = sim.stats
        evidence = stats.setdefault("evidence", {})
        evidence["seed"] = resolved_seed
        if output_dir is not None:
            observed = _collect_observability(stats, output_dir, cfg)
            observed["mode"] = observability
            if observability == "persistent":
                observed["outputs"] = [
                    str(path) for path in sorted(output_dir.glob("command_*")) if path.is_file()
                ]
            evidence["pim_energy_observability"] = observed
        return stats

    if observability == "persistent":
        output_dir = Path(observability_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        return run(output_dir)
    if observability == "disabled":
        return run(None)
    with tempfile.TemporaryDirectory() as tmp:
        return run(Path(tmp))
