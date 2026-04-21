"""Latency-throughput suite simulation runners.

All component choices are visible here — no hidden defaults in shared code.
run_single is a top-level function (not a closure) so it can be pickled
for ProcessPoolExecutor.
"""

import copy
import csv
import tempfile
from collections import Counter
from pathlib import Path

from tests.analysis.testcases import STANDARDS
from tests.utils import create_dram, extract_dram_layout


def _serialize_int_list(values):
    return ",".join(str(v) for v in values)


def _merge_cfg(base_cfg, override_cfg):
    merged = copy.deepcopy(base_cfg)
    for key, value in override_cfg.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def resolve_cfg(std_name, cfg_override=None):
    cfg = copy.deepcopy(STANDARDS[std_name])
    if cfg_override:
        cfg = _merge_cfg(cfg, cfg_override)
    return cfg


def _read_command_counts(path):
    counts = {}
    csv_path = Path(path)
    if not csv_path.exists():
        return counts
    for raw_line in csv_path.read_text().splitlines():
        if not raw_line.strip():
            continue
        command, count = [part.strip() for part in raw_line.split(",", maxsplit=1)]
        counts[command] = int(count)
    return dict(sorted(counts.items()))


def _read_command_traces(trace_prefix):
    traces = []
    for trace_path in sorted(Path(trace_prefix).parent.glob(f"{Path(trace_prefix).name}.ch*")):
        raw_text = trace_path.read_text()
        with trace_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            commands = [row["command"] for row in reader]
        trace_command_counts = dict(sorted(Counter(commands).items()))
        traces.append(
            {
                "path": str(trace_path),
                "channel": trace_path.suffix.replace(".ch", ""),
                "command_count": len(commands),
                "command_counts": trace_command_counts,
                "commands_preview": commands[:12],
                "raw_text": raw_text,
            }
        )
    return traces


def _collect_pim_observability(stats, tmpdir):
    counts_path = Path(tmpdir) / "command_counts.csv"
    trace_prefix = Path(tmpdir) / "command_trace.csv"
    observability = {
        "modeled": {
            "command_counts": _read_command_counts(counts_path),
            "command_traces": _read_command_traces(trace_prefix),
            "controller_stats": {},
        }
    }

    controller_stats = stats["memory_system"]["controller"]
    for key in (
        "cycles",
        "num_pim_reqs_served",
        "pim_capacity_stalls",
        "pim_dependency_stalls",
        "pim_inflight_peak",
        "pim_latency",
        "avg_pim_latency",
        "cas_issued",
        "cas_skipped",
        "act2_deadline_forced",
        "act2_deferred",
    ):
        if key in controller_stats:
            observability["modeled"]["controller_stats"][key] = controller_stats[key]

    return observability


def _build_controller(cfg, dram, *, full=False):
    """Instantiate the configured controller for fast or full validation."""
    import ramulator

    ctrl_cls = getattr(ramulator.controller, cfg["controller_class"])
    ctrl_kwargs = dict(
        dram=dram,
        scheduler=ramulator.scheduler.FRFCFS(),
        row_policy=ramulator.row_policy.Open(),
        addr_mapper=ramulator.addr_mapper.PassThroughAddrMapper(),
    )
    if "controller_plugins" in cfg:
        ctrl_kwargs["controller_plugins"] = cfg["controller_plugins"]
    extra_key = "full_ctrl_extra_kwargs" if full else "fast_ctrl_extra_kwargs"
    ctrl_kwargs.update(cfg[extra_key])
    return ctrl_cls(**ctrl_kwargs)


def run_single(
    std_name,
    nop_counter,
    read_ratio=100,
    num_probes=10000,
    warmup=10000,
    clock_ratio=None,
    stream_cols=None,
    full=False,
    cfg_override=None,
    observability_dir=None,
):
    """Run one simulation point and return sim.stats.

    This is a top-level function (not a closure) so it can be pickled
    for ProcessPoolExecutor.  It re-imports ramulator in each subprocess.
    """
    import ramulator

    cfg = resolve_cfg(std_name, cfg_override=cfg_override)
    if clock_ratio is None:
        clock_ratio = cfg["frontend_clock_ratio"]
    if stream_cols is None:
        stream_cols = cfg["stream_cols"]
    pim_mode = cfg.get("pim_mode", False)
    num_pim_requests = cfg.get("num_pim_requests", 0)
    pim_distribution_mode = cfg.get("pim_distribution_mode", "same_bank")
    pim_same_bank = cfg.get("pim_same_bank", True)
    pim_dependency_count = cfg.get("pim_dependency_count", 1)
    pim_bank_group_size = cfg.get("pim_bank_group_size", 0)
    pim_bank_sequence = _serialize_int_list(cfg.get("pim_bank_sequence", []))
    pim_burst_length = cfg.get("pim_burst_length", 1)
    pim_row_start = cfg.get("pim_row_start", 0)
    pim_row_count = cfg.get("pim_row_count", 1)
    pim_split_all_bank = cfg.get("pim_split_all_bank", False)

    dram = create_dram(cfg)
    layout = extract_dram_layout(dram)

    pim_request_type_id = -1
    pim_load_request_type_id = -1
    pim_compute_all_request_type_id = -1
    if pim_mode:
        request_names = list(type(dram).supported_requests.keys())
        pim_request_type_id = request_names.index("PIMCompute")
        if pim_split_all_bank:
            pim_load_request_type_id = request_names.index("PIMLoadAll")
            pim_compute_all_request_type_id = request_names.index("PIMComputeAll")

    frontend = ramulator.frontend.LatencyThroughputTrace(
        clock_ratio=clock_ratio,
        nop_counter=nop_counter,
        num_probe_requests=num_probes,
        pim_mode=pim_mode,
        num_pim_requests=num_pim_requests,
        pim_distribution_mode=pim_distribution_mode,
        pim_same_bank=pim_same_bank,
        pim_dependency_count=pim_dependency_count,
        pim_bank_group_size=pim_bank_group_size,
        pim_bank_sequence=pim_bank_sequence,
        pim_burst_length=pim_burst_length,
        pim_row_start=pim_row_start,
        pim_row_count=pim_row_count,
        pim_request_type_id=pim_request_type_id,
        pim_load_request_type_id=pim_load_request_type_id,
        pim_compute_all_request_type_id=pim_compute_all_request_type_id,
        pim_split_all_bank=pim_split_all_bank,
        warmup_cycles=warmup,
        seed=12345,
        read_ratio=read_ratio,
        stream_cols=stream_cols,
        **layout,
    )

    with tempfile.TemporaryDirectory(dir=observability_dir) as tmpdir:
        if pim_mode:
            counts_path = str(Path(tmpdir) / "command_counts.csv")
            trace_prefix = str(Path(tmpdir) / "command_trace.csv")
            command_counter = ramulator.controller_plugin.CommandCounter(
                commands_to_count=[
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
                ],
                path=counts_path,
            )
            cmd_trace = ramulator.controller_plugin.CmdTraceRecorder(path=trace_prefix)
            extra_plugins = list(cfg.get("controller_plugins", [])) + [command_counter, cmd_trace]
            cfg = _merge_cfg(cfg, {"controller_plugins": extra_plugins})

        ctrl = _build_controller(cfg, dram, full=full)

        mem = ramulator.memory_system.GenericDRAM(
            clock_ratio=1,
            controllers=[ctrl],
            channel_mapper=ramulator.channel_mapper.PassThroughChannelMapper(),
        )

        sim = ramulator.Simulation(frontend, mem)
        sim.run()
        stats = sim.stats
        if pim_mode:
            stats.setdefault("evidence", {})["pim_energy_observability"] = _collect_pim_observability(
                stats, tmpdir
            )
        return stats


def run_streaming_only(std_name, num_requests=50000, full=False, cfg_override=None):
    """Run a streaming-only simulation (no probes) at maximum throughput.

    Fires sequential read requests as fast as the memory system can accept
    them, with no probe interference or NOP rate-limiting.

    Returns the stats dict from the completed simulation.
    """
    import ramulator

    cfg = resolve_cfg(std_name, cfg_override=cfg_override)
    clock_ratio = cfg["frontend_clock_ratio"]
    stream_cols = cfg["stream_cols"]
    pim_mode = cfg.get("pim_mode", False)
    num_pim_requests = cfg.get("num_pim_requests", 0)
    pim_distribution_mode = cfg.get("pim_distribution_mode", "same_bank")
    pim_same_bank = cfg.get("pim_same_bank", True)
    pim_dependency_count = cfg.get("pim_dependency_count", 1)
    pim_bank_group_size = cfg.get("pim_bank_group_size", 0)
    pim_bank_sequence = _serialize_int_list(cfg.get("pim_bank_sequence", []))
    pim_burst_length = cfg.get("pim_burst_length", 1)
    pim_row_start = cfg.get("pim_row_start", 0)
    pim_row_count = cfg.get("pim_row_count", 1)
    pim_split_all_bank = cfg.get("pim_split_all_bank", False)

    dram = create_dram(cfg)
    layout = extract_dram_layout(dram)

    pim_request_type_id = -1
    pim_load_request_type_id = -1
    pim_compute_all_request_type_id = -1
    if pim_mode:
        request_names = list(type(dram).supported_requests.keys())
        pim_request_type_id = request_names.index("PIMCompute")
        if pim_split_all_bank:
            pim_load_request_type_id = request_names.index("PIMLoadAll")
            pim_compute_all_request_type_id = request_names.index("PIMComputeAll")

    frontend = ramulator.frontend.LatencyThroughputTrace(
        clock_ratio=clock_ratio,
        nop_counter=1,
        num_probe_requests=0,
        streaming_only=True,
        num_streaming_requests=num_requests,
        pim_mode=pim_mode,
        num_pim_requests=num_pim_requests,
        pim_distribution_mode=pim_distribution_mode,
        pim_same_bank=pim_same_bank,
        pim_dependency_count=pim_dependency_count,
        pim_bank_group_size=pim_bank_group_size,
        pim_bank_sequence=pim_bank_sequence,
        pim_burst_length=pim_burst_length,
        pim_row_start=pim_row_start,
        pim_row_count=pim_row_count,
        pim_request_type_id=pim_request_type_id,
        pim_load_request_type_id=pim_load_request_type_id,
        pim_compute_all_request_type_id=pim_compute_all_request_type_id,
        pim_split_all_bank=pim_split_all_bank,
        read_ratio=100,
        stream_cols=stream_cols,
        **layout,
    )

    ctrl = _build_controller(cfg, dram, full=full)

    mem = ramulator.memory_system.GenericDRAM(
        clock_ratio=1,
        controllers=[ctrl],
        channel_mapper=ramulator.channel_mapper.PassThroughChannelMapper(),
    )

    sim = ramulator.Simulation(frontend, mem)
    sim.run()
    return sim.stats
