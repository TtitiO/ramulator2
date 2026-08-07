import pytest

import ramulator
from ramulator.dram.addressing import addr_vec_from_byte_address, extract_dram_layout
from ramulator.workload_surrogate.generate_lpddr5_pim_concrete import (
    _host_address_chunks,
)
from ramulator.workload_surrogate.lpddr5_pim_concrete_trace import validate_record


def _layout(preset="LPDDR5_8Gb_x16", **overrides):
    dram = ramulator.dram.LPDDR5PIM(
        org_preset=preset,
        timing_preset="LPDDR5_6400",
        **overrides,
    )
    return extract_dram_layout(dram)


def _map(address, layout):
    return addr_vec_from_byte_address(
        address,
        level_names=layout["level_names"],
        level_sizes=layout["level_sizes"],
        internal_prefetch_size=layout["internal_prefetch_size"],
        tx_bytes=layout["tx_bytes"],
    )


@pytest.mark.parametrize(
    ("preset", "rows", "capacity"),
    [
        ("LPDDR5_8Gb_x16", 32768, 1 << 30),
        ("LPDDR5_16Gb_x16", 65536, 1 << 31),
    ],
)
def test_resolved_mapping_boundaries(preset, rows, capacity):
    layout = _layout(preset)
    assert layout["level_sizes"] == [1, 1, 4, 4, rows, 1024]
    assert layout["address_level_sizes"] == [1, 1, 4, 4, rows, 64]
    assert layout["capacity_bytes"] == capacity
    assert _map(0, layout) == [0, 0, 0, 0, 0, 0]
    assert _map(layout["tx_bytes"], layout) == [0, 0, 0, 0, 0, 1]
    assert _map(capacity - 1, layout) == [0, 0, 3, 3, rows - 1, 63]
    with pytest.raises(ValueError, match="exceeds configured addressable capacity"):
        _map(capacity, layout)


def test_rank_override_is_part_of_address_mapping():
    layout = _layout(rank=2)
    bytes_per_rank = layout["capacity_bytes"] // 2
    assert _map(bytes_per_rank, layout)[layout["rank_pos"]] == 1


def test_repeated_range_crossing_capacity_is_rejected():
    layout = _layout()
    start = layout["capacity_bytes"] - layout["tx_bytes"]
    with pytest.raises(ValueError, match="exceeds configured addressable capacity"):
        validate_record(
            {
                "opcode": "READ",
                "repeat": 2,
                "addr_byte": start,
                "addr_byte_stride": layout["tx_bytes"],
                "addr_vec": _map(start, layout),
            },
            address_layout=layout,
        )


def test_every_flat_bank_maps_to_one_unique_coordinate():
    from ramulator.workload_surrogate.generate_lpddr5_pim_concrete import (
        _decompose_flat_bank,
    )

    for layout in (_layout(), _layout(rank=2)):
        seen = set()
        for flat_bank in range(layout["total_bank_units"]):
            addr_vec = [0] * layout["addr_vec_size"]
            _decompose_flat_bank(
                flat_bank,
                addr_vec,
                bank_positions=layout["bank_positions"],
                bank_counts=layout["bank_counts"],
                controller_order=False,
            )
            coordinate = tuple(addr_vec[index] for index in layout["bank_positions"])
            assert coordinate not in seen
            seen.add(coordinate)
        assert len(seen) == layout["total_bank_units"]


def test_bounded_surrogate_policy_wraps_without_out_of_range_records():
    layout = _layout()
    base = 64_063_963_520
    chunks = list(
        _host_address_chunks(
            base_byte=base,
            stride_bytes=64,
            count=100,
            max_repeat=1_000_000,
            address_layout=layout,
            synthetic_address_policy="bounded_surrogate_v1",
        )
    )
    assert chunks
    assert sum(repeat for _, _, repeat in chunks) == 100
    for addr_byte, stride, repeat in chunks:
        assert 0 <= addr_byte < layout["capacity_bytes"]
        record = {
            "opcode": "WRITE",
            "repeat": repeat,
            "addr_byte": addr_byte,
            "addr_vec": _map(addr_byte, layout),
        }
        if repeat > 1 and stride > 0:
            record["addr_byte_stride"] = stride
        validate_record(record, address_layout=layout)


def test_strict_policy_does_not_wrap_out_of_range_addresses():
    layout = _layout()
    chunk = next(
        _host_address_chunks(
            base_byte=layout["capacity_bytes"],
            stride_bytes=64,
            count=1,
            max_repeat=1_000_000,
            address_layout=layout,
            synthetic_address_policy="strict_bytes",
        )
    )
    with pytest.raises(ValueError, match="exceeds configured addressable capacity"):
        _map(chunk[0], layout)
