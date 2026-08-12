"""Hand-authored fixtures for the canonical rank-local PIM bank mapping."""

import pytest

import ramulator
from tests.device_timings.harness import DeviceUnderTest

pytestmark = pytest.mark.device_timings


def make_dut(rank: int) -> DeviceUnderTest:
    return DeviceUnderTest(
        ramulator.dram.LPDDR5PIM(
            org_preset="LPDDR5_8Gb_x16",
            timing_preset="LPDDR5_6400",
            rank=rank,
        )
    )


@pytest.mark.parametrize("rank_count", [1, 2])
def test_every_bank_maps_once_to_a_rank_local_shared_block(rank_count: int):
    dut = make_dut(rank_count)
    seen_flat_banks: set[int] = set()
    groups: dict[tuple[int, int], set[int]] = {}

    # LPDDR5_8Gb_x16 has 4 bank groups x 4 banks = 16 banks per rank.
    for rank in range(rank_count):
        for bank_group in range(4):
            for bank in range(4):
                mapping = dut.bank_mapping(
                    dut.addr_vec(Rank=rank, BankGroup=bank_group, Bank=bank),
                    banks_per_group=2,
                )
                expected_local = bank_group * 4 + bank
                expected_flat = rank * 16 + expected_local
                expected_group = expected_local // 2
                expected_members = [rank * 16 + expected_group * 2 + offset for offset in range(2)]

                assert mapping == {
                    "flat_bank": expected_flat,
                    "rank": rank,
                    "rank_local_bank": expected_local,
                    "rank_local_group": expected_group,
                    "global_group": rank * 8 + expected_group,
                    "group_banks": expected_members,
                }
                assert all(member // 16 == rank for member in mapping["group_banks"])
                seen_flat_banks.add(mapping["flat_bank"])
                groups.setdefault((rank, expected_group), set()).add(mapping["flat_bank"])

    assert seen_flat_banks == set(range(rank_count * 16))
    assert set(groups) == {(rank, group) for rank in range(rank_count) for group in range(8)}
    assert all(len(members) == 2 for members in groups.values())
    if rank_count == 2:
        assert groups[(0, 7)].isdisjoint(groups[(1, 0)])


def test_invalid_shared_block_width_fails_closed():
    dut = make_dut(1)
    addr = dut.addr_vec(Rank=0, BankGroup=0, Bank=0)
    with pytest.raises(RuntimeError, match="banks_per_group"):
        dut.bank_mapping(addr, banks_per_group=3)
