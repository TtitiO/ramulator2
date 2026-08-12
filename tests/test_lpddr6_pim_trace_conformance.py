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
from ramulator.workload_surrogate.generate_lpddr5_pim_concrete import (
    lower_semantic_records_to_concrete,
)
from ramulator.workload_surrogate.lpddr5_pim_concrete_trace import (
    CONCRETE_SCHEMA_VERSIONS,
    read_jsonl,
)

import ramulator

pytestmark = pytest.mark.smoke

FIXTURES = Path(__file__).parent / "fixtures"


def _lpddr6_config(*, rank: int = 1) -> dict:
    return {
        "dram_class": "LPDDR6PIM",
        "org_preset": "LPDDR6_16Gb_x12",
        "timing_preset": "LPDDR6_10667_BL24",
        "dram_kwargs": {
            "rank": rank,
            "pim_datatype": "int8",
            "pim_banks_per_block": 2,
            "pim_mac_execution_model": "shared_block_serial",
        },
        "frontend_clock_ratio": 4,
    }


LPDDR6_CONFIG = _lpddr6_config()


def _dram_and_layout(config: dict | None = None):
    dram = create_dram(config or LPDDR6_CONFIG)
    return dram, extract_dram_layout(dram)


def _run_native_trace(trace_path: Path, *, config: dict | None = None) -> dict:
    resolved_config = config or LPDDR6_CONFIG
    dram, _ = _dram_and_layout(resolved_config)
    frontend = create_concrete_frontend(
        trace_path,
        dram,
        clock_ratio=resolved_config["frontend_clock_ratio"],
    )
    memory = create_memory_system(dram, resolved_config)
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


def test_lpddr6_two_rank_all_bank_fixture_is_rank_scoped_end_to_end():
    config = _lpddr6_config(rank=2)
    trace_path = FIXTURES / "lpddr6_pim_two_rank_all_bank.jsonl"
    _, layout = _dram_and_layout(config)

    summary = validate_trace_file(trace_path, address_layout=layout)
    stats = _run_native_trace(trace_path, config=config)
    frontend = stats["frontend"]
    controller = stats["memory_system"]["controller"]

    assert layout["level_sizes"][layout["rank_pos"]] == 2
    assert layout["capacity_bytes"] == 1 << 32
    assert summary["records"] == 10
    assert frontend["hab_records"] == 2
    assert frontend["pim_bcast_records"] == 2
    assert frontend["hab_pim_records"] == 2
    assert frontend["pim_mac_ab_records"] == 2
    assert frontend["sb_records"] == 2
    assert controller["num_issued_pim_mac_ab"] == 2
    assert controller["num_pim_ab_reqs_served"] == 2
    assert controller["total_banks"] == 32
    assert controller["pim_shared_block_count"] == 16
    assert controller["pim_inflight_peak"] == 16
    assert controller["pim_simultaneous_active_banks_peak"] == 16
    assert [controller[f"pim_launches_bank_{bank}"] for bank in range(32)] == [1] * 32


def test_lpddr6_cross_rank_mode_sequence_is_rejected_by_python_and_native():
    config = _lpddr6_config(rank=2)
    trace_path = FIXTURES / "lpddr6_pim_invalid_cross_rank_sequence.jsonl"
    _, layout = _dram_and_layout(config)

    with pytest.raises(ValueError, match="PIM_BCAST requires HAB mode"):
        validate_trace_file(trace_path, address_layout=layout)
    with pytest.raises((RuntimeError, ValueError), match="PIM_BCAST requires HAB mode"):
        _run_native_trace(trace_path, config=config)


def test_lpddr6_compact_pim_mac_checks_every_expanded_rank(tmp_path: Path):
    config = _lpddr6_config(rank=2)
    _, layout = _dram_and_layout(config)
    trace_path = tmp_path / "invalid-cross-rank-compact-mac.jsonl"
    records = [
        {"addr_vec": [0, 1, 0, 0, 0, 0], "opcode": "HAB", "repeat": 1},
        {
            "addr_vec": [0, 0, 0, 0, 0, 0],
            "opcode": "PIM_MAC",
            "repeat": 2,
            "bank_sequence": [0, 16],
            "bank_positions": layout["bank_positions"],
            "bank_counts": layout["bank_counts"],
            "dependency_count": 1,
            "row_count": 1,
            "row_start": 0,
            "column_start": 0,
            "resolved_row_offset": 0,
            "resolved_col_offset": 0,
            "interleave_depth": 1,
            "row_level": layout["row_pos"],
            "col_level": layout["col_pos"],
        },
    ]
    lines = [
        json.dumps(
            {
                "dram_class": "LPDDR6PIM",
                "schema_version": CONCRETE_SCHEMA_VERSIONS["LPDDR6PIM"],
            }
        ),
        *(json.dumps(record) for record in records),
    ]
    trace_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="PIM_MAC requires SB mode"):
        validate_trace_file(trace_path, address_layout=layout)
    with pytest.raises((RuntimeError, ValueError), match="PIM_MAC requires SB mode"):
        _run_native_trace(trace_path, config=config)


def test_lpddr6_host_address_mapping_is_shared_by_python_and_native_frontend():
    trace_path = FIXTURES / "lpddr6_pim_host_read.jsonl"
    _, layout = _dram_and_layout()

    validate_trace_file(trace_path, address_layout=layout)
    stats = _run_native_trace(trace_path)

    assert stats["frontend"]["read_records"] == 1
    assert stats["memory_system"]["controller"]["num_read_reqs"] == 1


def test_lpddr6_two_rank_host_boundary_maps_to_rank_one(tmp_path: Path):
    config = _lpddr6_config(rank=2)
    _, layout = _dram_and_layout(config)
    bytes_per_rank = layout["capacity_bytes"] // 2
    trace_path = tmp_path / "rank-one-read.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "dram_class": "LPDDR6PIM",
                "schema_version": CONCRETE_SCHEMA_VERSIONS["LPDDR6PIM"],
            }
        )
        + "\n"
        + json.dumps(
            {
                "addr_byte": bytes_per_rank,
                "addr_vec": [0, 1, 0, 0, 0, 0],
                "opcode": "READ",
                "repeat": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    validate_trace_file(trace_path, address_layout=layout)
    stats = _run_native_trace(trace_path, config=config)

    assert stats["frontend"]["read_records"] == 1
    assert stats["memory_system"]["controller"]["num_read_reqs_served"] == 1


def test_lpddr6_all_bank_lowering_covers_each_resolved_rank():
    _, layout = _dram_and_layout(_lpddr6_config(rank=2))
    semantic = {
        "schema_version": "v0.1",
        "record_id": "ranked_ffn",
        "kind": "FFNProjection",
        "phase": "decode",
        "layer": "layer_00",
        "op": "ffn_projection",
        "repeat": 1,
        "num_requests": 32,
        "bank_sequence": list(range(layout["total_bank_units"])),
        "dependency_context": {"dependency_count": 1},
        "row_policy": {"row_start": 0, "row_count": 1, "resolved_row": 0},
        "column_policy": {"column_start": 0, "resolved_column": 0},
        "burst_length": 1,
        "provenance": {},
        "mapping_policy": {},
    }

    concrete = lower_semantic_records_to_concrete(
        [semantic],
        interleave_banks=True,
        mac_mode="all_bank",
        address_layout=layout,
        addr_vec_size=layout["addr_vec_size"],
        bank_positions=layout["bank_positions"],
        bank_counts=layout["bank_counts"],
        row_level=layout["row_pos"],
        col_level=layout["col_pos"],
    )

    assert [record["opcode"] for record in concrete] == [
        "HAB",
        "PIM_BCAST",
        "HAB_PIM",
        "PIM_MAC_AB",
        "SB",
        "HAB",
        "PIM_BCAST",
        "HAB_PIM",
        "PIM_MAC_AB",
        "SB",
    ]
    rank_pos = layout["rank_pos"]
    assert [record["addr_vec"][rank_pos] for record in concrete] == [0] * 5 + [1] * 5


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
        json.dumps({"addr_vec": [0, 0, 0, 0, 9, 0], "opcode": "PIM_MAC", "repeat": 1}) + "\n",
        encoding="utf-8",
    )
    _, layout = _dram_and_layout()

    with pytest.raises(ValueError, match="header"):
        validate_trace_file(invalid, address_layout=layout)
    with pytest.raises((RuntimeError, ValueError), match="header"):
        _run_native_trace(invalid)
