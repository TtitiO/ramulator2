"""Pytest configuration and explicit test-suite tiers for Ramulator 2."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--verbose-plot",
        action="store_true",
        default=False,
        help="Include reference lines and metrics in latency-throughput plots",
    )
    parser.addoption(
        "--run-full",
        action="store_true",
        default=False,
        help="Run long refresh-enabled latency/throughput sweeps",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "smoke: Tier 1 basic functional tests")
    config.addinivalue_line(
        "markers", "latency_throughput_fast: Fast no-refresh latency-throughput tests"
    )
    config.addinivalue_line(
        "markers", "latency_throughput_full: Full refresh-enabled latency-throughput tests"
    )
    config.addinivalue_line("markers", "device_timings: DRAM device timing and legality tests")
    config.addinivalue_line(
        "markers", "controller_scheduling: Controller request scheduling tests"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-full"):
        return
    skip = pytest.mark.skip(reason="long suite; pass --run-full to run")
    for item in items:
        if "latency_throughput_full" in item.keywords:
            item.add_marker(skip)
