"""Shared simulation helpers for P4 figure data collection.

Extracted from tests/workload_surrogate/test_full_transformer_generator.py
to avoid duplicating the replay helper functions.
"""

from __future__ import annotations

from pathlib import Path

import ramulator
from tests.analysis.testcases.lpddr5_pim import CONFIG as LPDDR5_PIM_CONFIG
from tests.utils.dram import create_dram
from tests.utils.sim import extract_dram_layout


def _make_mem(dram):
    ctrl = ramulator.controller.LPDDR5PIM(
        dram=dram,
        scheduler=ramulator.scheduler.FRFCFS(),
        refresh_manager=ramulator.refresh_manager.NoRefresh(),
        row_policy=ramulator.row_policy.Open(),
        addr_mapper=ramulator.addr_mapper.PassThroughAddrMapper(),
    )
    return ramulator.memory_system.GenericDRAM(
        clock_ratio=1,
        controllers=[ctrl],
        channel_mapper=ramulator.channel_mapper.CacheLineInterleave(),
    )


def _frontend(trace_path: Path, dram, max_trace_bytes: int | None = None):
    request_type_ids = {
        name: index for index, name in enumerate(type(dram).supported_requests.keys())
    }
    command_ids = {
        name: index for index, name in enumerate(type(dram).commands)
    }
    layout = extract_dram_layout(dram)
    kwargs = dict(
        clock_ratio=LPDDR5_PIM_CONFIG["frontend_clock_ratio"],
        path=str(trace_path),
        pim_compute_request_type_id=request_type_ids["PIMCompute"],
        pim_load_all_request_type_id=request_type_ids["PIMLoadAll"],
        pim_compute_all_request_type_id=request_type_ids["PIMComputeAll"],
        sb_command_id=command_ids["SB"],
        hab_command_id=command_ids["HAB"],
        hab_pim_command_id=command_ids["HAB_PIM"],
        addr_vec_size=layout["addr_vec_size"],
    )
    if max_trace_bytes is not None:
        kwargs["max_trace_bytes"] = max_trace_bytes
    return ramulator.frontend.LPDDR5PIMConcreteTrace(**kwargs)
