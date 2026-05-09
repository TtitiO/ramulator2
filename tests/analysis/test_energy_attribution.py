import pytest

from tests.analysis.utils.energy import build_pim_energy_report


ENERGY_COMPONENT_FIELDS = (
    "energy_background_pJ",
    "energy_lpddr_base_cmd_pJ",
    "energy_incremental_pim_builtin_pJ",
    "energy_pim_setup_cmd_pJ",
    "energy_pim_mac_cmd_pJ",
    "energy_pim_movement_pJ",
    "energy_pim_writeback_pJ",
    "energy_mode_switch_pJ",
    "energy_fallback_or_unattributed_pJ",
)


def _stats(ctrl_overrides, command_counts):
    ctrl = {
        "cycles": 1_000,
        "num_pim_reqs_served": command_counts.get("PIM_MAC", 0)
        + command_counts.get("PIM_MAC_AB", 0),
        "pim_lanes": 32,
        "pim_ops_per_mac": 2.0,
        "pim_ops_per_request": 64.0,
        "avg_pim_response_latency": 0,
        "avg_pim_latency": 0,
        "avg_pim_service_latency": 0,
        "avg_pim_launch_wait": 0,
        "pim_service_latency": 0,
        "pim_launch_wait": 0,
        "total_background_energy": 0.0,
        "total_cmd_energy": 0.0,
        "total_energy": 0.0,
        "total_incremental_cmd_energy": 0.0,
    }
    ctrl.update(ctrl_overrides)
    return {
        "memory_system": {"controller": ctrl},
        "frontend": {},
        "evidence": {
            "pim_energy_observability": {
                "modeled": {
                    "command_counts": command_counts,
                    "command_traces": [],
                }
            }
        },
    }


def _component_sum(report):
    return sum(report[field] for field in ENERGY_COMPONENT_FIELDS)


def test_builtin_total_mode_excludes_placeholder_pim_mac_energy():
    stats = _stats(
        {
            "total_background_energy": 226.0158,
            "total_cmd_energy": 3.3456,
            "total_energy": 229.3614,
            "total_incremental_cmd_energy": 148.889,
        },
        {"PIM_MAC": 4096},
    )

    report = build_pim_energy_report(stats, "lpddr5_pim")["derived"]

    assert report["energy_attribution_mode"] == "builtin_total"
    assert report["energy_pim_mac_cmd_pJ"] == 0.0
    assert report["energy_incremental_pim_builtin_pJ"] == pytest.approx(148.889)
    assert report["energy_fallback_or_unattributed_pJ"] >= 0.0
    assert not (
        report["energy_pim_mac_cmd_pJ"] > 1e6
        and report["energy_fallback_or_unattributed_pJ"] < 0.0
    )
    assert report["energy_total_pJ"] == pytest.approx(_component_sum(report))
    assert report["energy_attribution_sum_pJ"] == pytest.approx(report["energy_total_pJ"])
    assert "excluded_placeholder_analytical_command_coefficients" in report[
        "coefficient_source_summary"
    ]


def test_analytical_component_mode_total_is_component_sum():
    stats = _stats(
        {
            "total_background_energy": 100.0,
            "total_cmd_energy": 50.0,
            "total_energy": 150.0,
            "total_incremental_cmd_energy": 1.0,
            "pim_compute_energy_pJ_per_mac": 0.5,
            "pim_cell_to_pim_energy_pJ_per_256b": 1.0,
            "pim_interconnect_energy_pJ_per_256b": 2.0,
            "pim_vrf_access_energy_pJ": 3.0,
            "pim_srf_access_energy_pJ": 4.0,
            "pim_mode_switch_energy_pJ": 5.0,
        },
        {"PIM_MAC": 10, "SB": 2},
    )

    report = build_pim_energy_report(stats, "lpddr5_pim")["derived"]

    assert report["energy_attribution_mode"] == "analytical_component"
    assert report["energy_incremental_pim_builtin_pJ"] == 0.0
    assert report["energy_pim_mac_cmd_pJ"] == pytest.approx(10 * 32 * 0.5)
    assert report["energy_pim_movement_pJ"] == pytest.approx(10 * (1.0 + 2.0))
    assert report["energy_pim_writeback_pJ"] == pytest.approx(10 * (3.0 + 4.0))
    assert report["energy_mode_switch_pJ"] == pytest.approx(2 * 5.0)
    assert report["energy_fallback_or_unattributed_pJ"] == 0.0
    assert report["energy_total_pJ"] == pytest.approx(_component_sum(report))
    assert report["energy_attribution_sum_pJ"] == pytest.approx(report["energy_total_pJ"])
    for field in ENERGY_COMPONENT_FIELDS:
        assert report[field] >= 0.0
