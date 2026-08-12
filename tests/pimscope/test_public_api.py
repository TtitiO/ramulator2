import json
import os
import time
from pathlib import Path

import pytest
from ramulator.pimscope import (
    create_dram,
    create_memory_system,
    hardware_config_from_manifest,
    load_experiment_manifest,
    pim_backend_capabilities,
    resolve_experiment_manifest,
    run_experiment,
    validate_backend,
)
from ramulator.pimscope.experiment import _run_process

ROOT = Path(__file__).parents[2]


def _sleep(seconds):
    time.sleep(seconds)


def _manifest():
    return json.loads(
        (ROOT / "examples" / "pimscope_custom_model.json").read_text(encoding="utf-8")
    )


def test_lpddr6_capability_is_declared_experimental():
    capabilities = pim_backend_capabilities()
    assert capabilities["LPDDR5PIM"]["status"] == "supported"
    assert capabilities["LPDDR6PIM"]["status"] == "experimental"
    assert capabilities["LPDDR6PIM"]["available_base_dram_model"]
    assert capabilities["LPDDR6PIM"]["validated_rank_counts"] == [1, 2]
    assert capabilities["LPDDR6PIM"]["validated_topology"] == {
        "controllers": 1,
        "channels": 1,
    }
    assert "notes" not in capabilities["LPDDR6PIM"]
    assert capabilities["LPDDR6PIM"]["paper_artifact_backend"] is False
    assert capabilities["LPDDR6PIM"]["paper_artifact_backend_name"] == "LPDDR5PIM"
    assert capabilities["LPDDR6PIM"]["standard_power_calibration"] == (
        "drampower_test_fixture_not_device_datasheet"
    )
    assert capabilities["LPDDR6PIM"]["metadata_documentation"].endswith(
        "PIMScope-metadata.md"
    )
    assert capabilities["LPDDR6PIM"]["subchannel_model"] == {
        "status": "single_subchannel_only",
        "modeled_subchannels_per_channel": 1,
        "refresh_density_reference_subchannels": 2,
        "independent_subchannel_scheduling": False,
    }

    raw = _manifest()
    raw["hardware"]["dram_class"] = "LPDDR6"
    with pytest.raises(ValueError, match=r"generic LPDDR6.*select LPDDR6PIM explicitly"):
        resolve_experiment_manifest(raw, source="test")


def test_unvalidated_rank_count_is_rejected_by_public_manifest():
    raw = _manifest()
    raw["hardware"]["org_overrides"] = {"rank": 3}
    with pytest.raises(ValueError, match=r"hardware\.org_overrides\.rank.*one- and two-rank"):
        resolve_experiment_manifest(raw, source="test")


def test_unsupported_topology_is_rejected_by_public_manifest():
    raw = _manifest()
    raw["hardware"]["topology"] = {"controllers": 1, "channels": 2}
    with pytest.raises(ValueError, match=r"hardware\.topology\.channels.*unsupported"):
        resolve_experiment_manifest(raw, source="test")


def test_manifest_and_backend_are_public():
    resolved = resolve_experiment_manifest(_manifest(), source="test")
    backend = validate_backend(resolved)
    assert backend["address_layout"]["mapping_version"] == 1
    assert backend["address_layout"]["capacity_bytes"] == 1 << 30


def test_nested_rit_mapper_is_constructed_by_public_api():
    raw = _manifest()
    raw["hardware"]["controller"] = {"addr_mapper": "RITAddrMapper"}
    resolved = resolve_experiment_manifest(raw, source="test")
    config = hardware_config_from_manifest(resolved)
    memory = create_memory_system(create_dram(config), config)
    mapper = memory.to_config()["controllers"][0]["addr_mapper"]
    assert mapper["impl"] == "RITAddrMapper"
    assert mapper["addr_mapper"]["impl"] == "PassThroughAddrMapper"


def test_public_experiment_api_replays_example():
    result = run_experiment(
        load_experiment_manifest(ROOT / "examples" / "pimscope_custom_model.json")
    )
    assert result["status"] == "PASS"
    assert result["simulation"]["replay_ok"]
    assert result["simulation"]["cycles"] == 5981
    assert result["simulation"]["refresh_manager"] == "NoRefresh"
    assert (
        result["simulation"]["trace_opcode_counts"]
        == result["workload_summary"]["concrete_opcode_counts"]
    )
    preflight = result["resolved_hardware"]["preflight"]
    assert preflight["records"] == result["workload_summary"]["concrete_records"]
    assert preflight["expanded_records"] > preflight["records"]
    assert 0 < preflight["trace_bytes"] <= preflight["max_trace_bytes"]
    assert result["provenance"]["config_source"] == "pimscope_custom_model.json"
    assert result["provenance"]["seed"] == 12345
    assert result["workload_summary"]["seed"] == 12345
    energy = result["simulation"]["power_accounting"]
    assert energy["status"] == "paper_two_layer"
    assert energy["equation"] == "E = E_LPDDR + E_PIM"
    assert energy["power_profile"] == "PAPER_LPDDR5_POWER"
    assert energy["energy_scope"] == "standard_memory_plus_incremental_pim_events"
    assert "notes" not in energy
    assert energy["pim_energy_method"] == "paper_event_coefficients"
    assert energy["metadata_documentation"].endswith("PIMScope-metadata.md")
    assert energy["standard_energy_units"] == {
        "current": "mA",
        "voltage": "V",
        "time": "ns",
        "energy": "pJ",
    }
    assert energy["standard_power_source"]["profile_scope"] == "camera_ready_lpddr5_6400_analysis"
    assert energy["standard_power_source"]["calibration"] == "not_device_calibrated"
    assert energy["standard_power_source"]["dimensional_equation"] == (
        "V[V] * I[mA] * t[ns] = E[pJ]"
    )
    assert energy["standard_power_source"]["legacy_conversion_scale"] == pytest.approx(1e-3)
    assert energy["standard_power_source"]["legacy_conversion"] == (
        "multiply_by_1e-3_after_V_mA_ns_product"
    )
    assert energy["coefficients"]["pim_compute_energy_pJ_per_mac"] == pytest.approx(0.35)
    assert energy["coefficients"]["pim_cell_to_pim_energy_pJ_per_256b"] == pytest.approx(2.68)
    assert energy["total_standard_energy_pJ"] == pytest.approx(199.166)
    assert energy["total_pim_event_energy_pJ"] == pytest.approx(4786.6752)
    assert energy["total_energy_pJ"] == pytest.approx(4985.8412)
    assert energy["total_energy_pJ"] == pytest.approx(
        energy["total_standard_energy_pJ"] + energy["total_pim_event_energy_pJ"]
    )


def test_process_timeout_terminates_worker():
    with pytest.raises(TimeoutError, match="operation exceeded"):
        _run_process(_sleep, (2,), {}, 1)


def test_process_isolated_replay_completes_before_timeout():
    raw = _manifest()
    raw["workload"]["simulation_timeout_seconds"] = 60
    result = run_experiment(resolve_experiment_manifest(raw, source="timeout-test"))
    assert result["simulation"]["cycles"] == 5981


def test_preflight_limits_fail_before_native_replay(monkeypatch):
    raw = _manifest()
    raw["workload"]["max_expanded_records"] = 1
    resolved = resolve_experiment_manifest(raw, source="test")

    def unexpected_replay(*args, **kwargs):
        raise AssertionError("native replay must not start after a preflight failure")

    monkeypatch.setattr("ramulator.pimscope.experiment.replay_concrete_trace", unexpected_replay)
    with pytest.raises(ValueError, match=r"max_expanded_records.*estimated.*exceeds"):
        run_experiment(resolved)


def test_public_api_does_not_mutate_global_expanded_record_limit(monkeypatch):
    monkeypatch.setenv("RAMULATOR_MAX_EXPANDED_RECORDS", "1")
    resolved = load_experiment_manifest(ROOT / "examples" / "pimscope_custom_model.json")
    assert run_experiment(resolved)["status"] == "PASS"
    assert os.environ["RAMULATOR_MAX_EXPANDED_RECORDS"] == "1"
