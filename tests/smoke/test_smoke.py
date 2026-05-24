"""Tier 1: Smoke tests — verify each DRAM standard runs without crashing."""

import copy

pytest = __import__("pytest")

from tests.smoke.runner import run_single
from tests.smoke.testcases import STANDARDS
from tests.utils import create_dram, extract_dram_layout


@pytest.mark.smoke
@pytest.mark.parametrize("standard", sorted(STANDARDS.keys()))
def test_smoke(standard):
    """Run a short simulation (100 probes) and verify basic stats are sane."""
    stats = run_single(standard, nop_counter=5, num_probes=100, warmup=100)

    assert "memory_system" in stats
    assert "frontend" in stats
    # Verify the controller has run
    ctrl_stats = stats["memory_system"]["controller"]
    assert ctrl_stats["num_read_reqs"] > 0
    assert stats["frontend"]["avg_probe_latency"] > 0


@pytest.mark.smoke
def test_lpddr5pim_host_only_exposes_builtin_energy_stats():
    """LPDDR5PIM should expose built-in memory energy stats under host-only traffic."""
    ramulator = __import__("ramulator")

    cfg = copy.deepcopy(STANDARDS["LPDDR5PIM"])
    cfg["dram_kwargs"]["power"] = {
        "enabled": True,
        "VDD1": 1.80,
        "VDD2H": 1.05,
        "VDD2L": 0.90,
        "VDDQ": 0.50,
        "IDD01": 2.80,
        "IDD02H": 32.00,
        "IDD02L": 0.25,
        "IDD0Q": 0.75,
        "IDD2N1": 1.20,
        "IDD2N2H": 16.00,
        "IDD2N2L": 0.25,
        "IDD2NQ": 0.75,
        "IDD3N1": 1.20,
        "IDD3N2H": 16.00,
        "IDD3N2L": 0.25,
        "IDD3NQ": 0.75,
        "IDD4R1": 2.00,
        "IDD4R2H": 18.00,
        "IDD4R2L": 0.30,
        "IDD4RQ": 0.85,
        "IDD4W1": 2.10,
        "IDD4W2H": 19.00,
        "IDD4W2L": 0.35,
        "IDD4WQ": 0.90,
        "IDD5AB1": 2.20,
        "IDD5AB2H": 35.00,
        "IDD5AB2L": 0.25,
        "IDD5ABQ": 0.75,
    }

    dram = create_dram(cfg)
    layout = extract_dram_layout(dram)

    frontend = ramulator.frontend.LatencyThroughputTrace(
        clock_ratio=cfg["frontend_clock_ratio"],
        nop_counter=5,
        num_probe_requests=64,
        warmup_cycles=64,
        seed=12345,
        read_ratio=100,
        stream_cols=cfg["stream_cols"],
        **layout,
    )

    ctrl_cls = getattr(ramulator.controller, cfg["controller_class"])
    ctrl = ctrl_cls(
        dram=dram,
        scheduler=ramulator.scheduler.FRFCFS(),
        row_policy=ramulator.row_policy.Open(),
        addr_mapper=ramulator.addr_mapper.PassThroughAddrMapper(),
        refresh_manager=ramulator.refresh_manager.NoRefresh(),
    )

    mem = ramulator.memory_system.GenericDRAM(
        clock_ratio=1,
        controllers=[ctrl],
        channel_mapper=ramulator.channel_mapper.PassThroughChannelMapper(),
    )

    sim = ramulator.Simulation(frontend, mem)
    sim.run()

    ctrl_stats = sim.stats["memory_system"]["controller"]
    assert ctrl_stats["num_read_reqs"] > 0
    assert "total_background_energy" in ctrl_stats
    assert "total_cmd_energy" in ctrl_stats
    assert "total_energy" in ctrl_stats
    assert ctrl_stats["total_energy"] >= 0
