import json
from pathlib import Path

import pytest
from ramulator.dram.addressing import extract_dram_layout
from ramulator.pimscope.backend import (
    create_concrete_frontend,
    create_dram,
    create_memory_system,
)
from ramulator.pimscope.schema import validate_trace_file
from ramulator.workload_surrogate.lpddr5_pim_concrete_trace import (
    CONCRETE_SCHEMA_VERSIONS,
    read_jsonl,
)

import ramulator

pytestmark = pytest.mark.smoke

FIXTURES = Path(__file__).parent / "fixtures"
LPDDR6_CONFIG = {
    "dram_class": "LPDDR6PIM",
    "org_preset": "LPDDR6_16Gb_x12",
    "timing_preset": "LPDDR6_10667_BL24",
    "dram_kwargs": {
        "pim_datatype": "int8",
        "pim_banks_per_block": 2,
        "pim_mac_execution_model": "shared_block_serial",
    },
    "frontend_clock_ratio": 4,
}


def _dram_and_layout():
    dram = create_dram(LPDDR6_CONFIG)
    return dram, extract_dram_layout(dram)


def _run_native_trace(trace_path: Path) -> dict:
    dram, _ = _dram_and_layout()
    frontend = create_concrete_frontend(
        trace_path,
        dram,
        clock_ratio=LPDDR6_CONFIG["frontend_clock_ratio"],
    )
    memory = create_memory_system(dram, LPDDR6_CONFIG)
    simulation = ramulator.Simulation(frontend, memory)
    simulation.run()
    simulation.finalize()
    return simulation.stats


def test_lpddr6_fixture_has_one_schema_and_one_backend_across_python_layers():
    trace_path = FIXTURES / "lpddr6_pim_minimal.jsonl"
    _, layout = _dram_and_layout()

    summary = validate_trace_file(trace_path, address_layout=layout)
    header, records = read_jsonl(
        trace_path,
        address_layout=layout,
        expected_dram_class="LPDDR6PIM",
    )

    assert header["dram_class"] == "LPDDR6PIM"
    assert header["schema_version"] == CONCRETE_SCHEMA_VERSIONS["LPDDR6PIM"]
    assert summary["records"] == len(records) == 1
    assert summary["expanded_records"] == 1


def test_lpddr6_fixture_is_consumed_by_native_frontend():
    stats = _run_native_trace(FIXTURES / "lpddr6_pim_minimal.jsonl")
    frontend = stats["frontend"]
    controller = stats["memory_system"]["controller"]

    assert frontend["records_loaded"] == 1
    assert frontend["records_expanded"] == 1
    assert frontend["pim_mac_records"] == 1
    assert controller["num_issued_pim_mac"] == 1


def test_lpddr6_all_bank_sequence_is_shared_by_python_and_native_frontend():
    trace_path = FIXTURES / "lpddr6_pim_all_bank.jsonl"
    _, layout = _dram_and_layout()

    summary = validate_trace_file(trace_path, address_layout=layout)
    stats = _run_native_trace(trace_path)

    assert summary["records"] == 4
    assert stats["frontend"]["records_loaded"] == 4
    assert stats["frontend"]["hab_records"] == 1
    assert stats["frontend"]["pim_bcast_records"] == 1
    assert stats["frontend"]["hab_pim_records"] == 1
    assert stats["frontend"]["pim_mac_ab_records"] == 1
    assert stats["memory_system"]["controller"]["num_issued_pim_mac_ab"] == 1


def test_lpddr6_host_address_mapping_is_shared_by_python_and_native_frontend():
    trace_path = FIXTURES / "lpddr6_pim_host_read.jsonl"
    _, layout = _dram_and_layout()

    validate_trace_file(trace_path, address_layout=layout)
    stats = _run_native_trace(trace_path)

    assert stats["frontend"]["read_records"] == 1
    assert stats["memory_system"]["controller"]["num_read_reqs"] == 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda header, record: header.update(schema_version="lpddr5-pim-opcode-v0.2"),
            "schema_version",
        ),
        (
            lambda header, record: header.update(dram_class="LPDDR5PIM"),
            "dram_class",
        ),
        (
            lambda header, record: record.update(opcode="CAS_RD"),
            "opcode",
        ),
        (
            lambda header, record: record.update(addr_vec=[0, 0, 0, 0, 65536, 0]),
            "addr_vec",
        ),
    ],
)
def test_lpddr6_trace_rejection_cases_fail_in_python_and_native(
    tmp_path: Path, mutation, message: str
):
    source = FIXTURES / "lpddr6_pim_minimal.jsonl"
    lines = source.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    record = json.loads(lines[1])
    mutation(header, record)
    invalid = tmp_path / f"invalid-{message}.jsonl"
    invalid.write_text(
        "\n".join((json.dumps(header), json.dumps(record))) + "\n",
        encoding="utf-8",
    )
    _, layout = _dram_and_layout()

    with pytest.raises(ValueError, match=message):
        validate_trace_file(invalid, address_layout=layout)
    with pytest.raises((RuntimeError, ValueError), match=message):
        _run_native_trace(invalid)


def test_lpddr6_trace_requires_an_explicit_header(tmp_path: Path):
    invalid = tmp_path / "missing-header.jsonl"
    invalid.write_text(
        json.dumps({"addr_vec": [0, 0, 0, 0, 9, 0], "opcode": "PIM_MAC", "repeat": 1})
        + "\n",
        encoding="utf-8",
    )
    _, layout = _dram_and_layout()

    with pytest.raises(ValueError, match="header"):
        validate_trace_file(invalid, address_layout=layout)
    with pytest.raises((RuntimeError, ValueError), match="header"):
        _run_native_trace(invalid)
