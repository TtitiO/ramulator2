import json
import os
from pathlib import Path

from ramulator.pimscope import (
    create_dram,
    create_memory_system,
    hardware_config_from_manifest,
    load_experiment_manifest,
    resolve_experiment_manifest,
    run_experiment,
    validate_backend,
)

ROOT = Path(__file__).parents[2]


def _manifest():
    return json.loads(
        (ROOT / "examples" / "pimscope_custom_model.json").read_text(encoding="utf-8")
    )


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
    assert result["provenance"]["config_source"] == "pimscope_custom_model.json"


def test_public_api_does_not_mutate_global_expanded_record_limit(monkeypatch):
    monkeypatch.setenv("RAMULATOR_MAX_EXPANDED_RECORDS", "1")
    resolved = load_experiment_manifest(ROOT / "examples" / "pimscope_custom_model.json")
    assert run_experiment(resolved)["status"] == "PASS"
    assert os.environ["RAMULATOR_MAX_EXPANDED_RECORDS"] == "1"
