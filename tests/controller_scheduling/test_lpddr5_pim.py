import pytest

import ramulator
import tests.controller_scheduling.harness as cs


pytestmark = pytest.mark.controller_scheduling


def make_dut(pim_blocks_per_bank: int = 1):
    dram = ramulator.dram.LPDDR5PIM(
        org_preset="LPDDR5_8Gb_x16",
        timing_preset="LPDDR5_6400",
        rank=1,
        pim_enabled=True,
        pim_blocks_per_bank=pim_blocks_per_bank,
    )
    controller = ramulator.controller.LPDDR5PIM(
        scheduler=ramulator.scheduler.FRFCFS(),
        refresh_manager=ramulator.refresh_manager.NoRefresh(),
        row_policy=ramulator.row_policy.Open(),
        addr_mapper=ramulator.addr_mapper.PassThroughAddrMapper(),
        dram=dram,
        controller_plugins=[],
    )
    return cs.ControllerUnderTest(controller)


def run_until_pim_mac_count(dut: cs.ControllerUnderTest, count: int, max_ticks: int = 256):
    history = []
    for _ in range(max_ticks):
        issued = dut.tick()
        history.extend(issued)
        if [cmd.command for cmd in history].count("PIM_MAC") >= count:
            return history
    raise AssertionError(f"PIM_MAC count did not reach {count} within {max_ticks} ticks")


def tick_without_new_commands(dut: cs.ControllerUnderTest, cycles: int):
    for _ in range(max(0, cycles)):
        issued = dut.tick()
        assert issued == []


def run_until_pim_reqs_served(dut: cs.ControllerUnderTest, count: int, max_ticks: int = 256):
    for _ in range(max_ticks):
        if dut.stats()["num_pim_reqs_served"] >= count:
            return
        issued = dut.tick()
        assert issued == []
    raise AssertionError(f"num_pim_reqs_served did not reach {count} within {max_ticks} ticks")


def test_pimcompute_issues_act1_act2_pim_mac():
    dut = make_dut()
    a = dut.addr_vec(Rank=0, BankGroup=0, Bank=0, Row=9, Column=0)

    dut.send_request("PIMCompute", a)
    history = dut.run_until_idle(max_ticks=256)

    dut.assert_commands(["ACT1", "ACT2", "PIM_MAC"], history=history)


def test_host_read_then_pimcompute_respects_issue_ordering():
    dut = make_dut()
    host = dut.addr_vec(Rank=0, BankGroup=0, Bank=0, Row=4, Column=0)
    pim = dut.addr_vec(Rank=0, BankGroup=0, Bank=1, Row=5, Column=0)

    dut.send_request("Read", host)
    dut.send_request("PIMCompute", pim)
    history = dut.run_until_idle(max_ticks=512)

    dut.assert_commands(["ACT1", "ACT2", "ACT1", "ACT2", "CAS_RD", "RD", "PIM_MAC"], history=history)


def test_same_bank_conflicting_host_read_waits_for_inflight_pim_completion():
    dut = make_dut()
    pim = dut.addr_vec(Rank=0, BankGroup=0, Bank=0, Row=9, Column=0)
    host = dut.addr_vec(Rank=0, BankGroup=0, Bank=0, Row=10, Column=0)

    dut.send_request("PIMCompute", pim)
    dut.send_request("Read", host)

    history = []
    for _ in range(256):
        issued = dut.tick()
        history.extend(issued)
        if [c.command for c in history].count("PIM_MAC") == 1:
            break
    else:
        raise AssertionError("PIM_MAC was not issued within 256 ticks")

    for _ in range(max(0, dut.timings["nPIM_MAC_LAT"] - 1)):
        issued = dut.tick()
        history.extend(issued)
        assert [c.command for c in issued] == []

    history.extend(dut.run_until_idle(max_ticks=512))

    dut.assert_commands(["ACT1", "ACT2", "PIM_MAC", "PREpb", "ACT1", "ACT2", "CAS_RD", "RD"], history=history)


def test_same_bank_refresh_waits_for_inflight_pim_completion():
    dut = make_dut()
    pim = dut.addr_vec(Rank=0, BankGroup=0, Bank=0, Row=9, Column=0)
    refresh = dut.addr_vec(Rank=0, BankGroup=0, Bank=0, Row=dut.ALL, Column=0)

    dut.send_request("PIMCompute", pim)

    history = []
    for _ in range(256):
        issued = dut.tick()
        history.extend(issued)
        if [c.command for c in history].count("PIM_MAC") == 1:
            break
    else:
        raise AssertionError("PIM_MAC was not issued within 256 ticks")

    dut.priority_send("REFpb", refresh)

    for _ in range(max(0, dut.timings["nPIM_MAC_LAT"] - 1)):
        issued = dut.tick()
        history.extend(issued)
        assert [c.command for c in issued] == []

    history.extend(dut.run_until_idle(max_ticks=512))

    dut.assert_commands(["ACT1", "ACT2", "PIM_MAC", "PREpb", "REFpb"], history=history)


def test_omitted_pim_blocks_per_bank_matches_explicit_one():
    addr = dict(Rank=0, BankGroup=0, Bank=0, Row=9, Column=0)

    dut_default = make_dut()
    a_default = dut_default.addr_vec(**addr)
    dut_default.send_request("PIMCompute", a_default)
    dut_default.send_request("PIMCompute", a_default)
    history_default = dut_default.run_until_idle(max_ticks=1024)

    dut_explicit_one = make_dut(pim_blocks_per_bank=1)
    a_one = dut_explicit_one.addr_vec(**addr)
    dut_explicit_one.send_request("PIMCompute", a_one)
    dut_explicit_one.send_request("PIMCompute", a_one)
    history_one = dut_explicit_one.run_until_idle(max_ticks=1024)

    stats_default = dut_default.stats()
    stats_one = dut_explicit_one.stats()

    assert stats_default["pim_capacity_stalls"] == stats_one["pim_capacity_stalls"]
    assert [c.command for c in history_default] == [c.command for c in history_one]


def test_pimcompute_completion_is_delayed_beyond_launch():
    dut = make_dut()
    a = dut.addr_vec(Rank=0, BankGroup=0, Bank=0, Row=9, Column=0)

    dut.send_request("PIMCompute", a)

    history = run_until_pim_mac_count(dut, count=1)
    tick_without_new_commands(dut, dut.timings["nPIM_MAC_LAT"] - 1)

    stats = dut.stats()

    dut.assert_commands(["ACT1", "ACT2", "PIM_MAC"], history=history)
    assert stats["num_pim_reqs_served"] == 0
    assert stats["pim_inflight_peak"] == 1


def test_same_bank_dependent_pim_launches_stall_on_dependency_even_with_two_slots():
    dut = make_dut(pim_blocks_per_bank=2)
    a = dut.addr_vec(Rank=0, BankGroup=0, Bank=0, Row=9, Column=0)

    dut.send_request("PIMCompute", a)
    dut.send_request("PIMCompute", a)

    history = run_until_pim_mac_count(dut, count=2)
    run_until_pim_reqs_served(dut, count=2, max_ticks=dut.timings["nPIM_MAC_LAT"] + 2)

    stats = dut.stats()

    dut.assert_commands(["ACT1", "ACT2", "PIM_MAC", "PIM_MAC"], history=history)
    dut.assert_gap(2, 3, dut.timings["nPIM_MAC_LAT"] + 1, history=history)
    assert stats["num_pim_reqs_served"] == 2
    assert stats["pim_dependency_stalls"] > 0
    assert stats["pim_capacity_stalls"] == 0
    assert stats["pim_inflight_peak"] == 1


def test_same_bank_independent_pim_launches_can_overlap_with_two_slots():
    dut = make_dut(pim_blocks_per_bank=2)
    a0 = dut.addr_vec(Rank=0, BankGroup=0, Bank=0, Row=9, Column=0)
    a1 = dut.addr_vec(Rank=0, BankGroup=0, Bank=0, Row=9, Column=32)

    dut.send_request("PIMCompute", a0)
    dut.send_request("PIMCompute", a1)

    history = run_until_pim_mac_count(dut, count=2)
    run_until_pim_reqs_served(dut, count=2, max_ticks=dut.timings["nPIM_MAC_LAT"] + 2)

    stats = dut.stats()

    dut.assert_commands(["ACT1", "ACT2", "PIM_MAC", "PIM_MAC"], history=history)
    dut.assert_gap(2, 3, dut.timings["nPIM_MAC_LAT"], history=history)
    assert stats["num_pim_reqs_served"] == 2
    assert stats["pim_dependency_stalls"] == 0
    assert stats["pim_capacity_stalls"] == 0
    assert stats["pim_inflight_peak"] == 2


def test_same_bank_independent_pim_launches_serialize_when_bank_has_one_slot():
    dut = make_dut(pim_blocks_per_bank=1)
    a0 = dut.addr_vec(Rank=0, BankGroup=0, Bank=0, Row=9, Column=0)
    a1 = dut.addr_vec(Rank=0, BankGroup=0, Bank=0, Row=9, Column=32)

    dut.send_request("PIMCompute", a0)
    dut.send_request("PIMCompute", a1)

    history = run_until_pim_mac_count(dut, count=2)
    run_until_pim_reqs_served(dut, count=2, max_ticks=dut.timings["nPIM_MAC_LAT"] + 2)

    stats = dut.stats()

    dut.assert_commands(["ACT1", "ACT2", "PIM_MAC", "PIM_MAC"], history=history)
    dut.assert_gap(2, 3, dut.timings["nPIM_MAC_LAT"] + 1, history=history)
    assert stats["num_pim_reqs_served"] == 2
    assert stats["pim_dependency_stalls"] == 0
    assert stats["pim_capacity_stalls"] > 0
    assert stats["pim_inflight_peak"] == 1
