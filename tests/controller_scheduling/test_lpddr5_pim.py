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


def test_same_bank_pim_capacity_stalls_differ_between_one_and_two_blocks():
    addr = dict(Rank=0, BankGroup=0, Bank=0, Row=9, Column=0)

    dut_cap1 = make_dut(pim_blocks_per_bank=1)
    a1 = dut_cap1.addr_vec(**addr)
    dut_cap1.send_request("PIMCompute", a1)
    dut_cap1.send_request("PIMCompute", a1)
    history_cap1 = dut_cap1.run_until_idle(max_ticks=1024)
    stalls_cap1 = dut_cap1.stats()["pim_capacity_stalls"]

    dut_cap2 = make_dut(pim_blocks_per_bank=2)
    a2 = dut_cap2.addr_vec(**addr)
    dut_cap2.send_request("PIMCompute", a2)
    dut_cap2.send_request("PIMCompute", a2)
    history_cap2 = dut_cap2.run_until_idle(max_ticks=1024)
    stalls_cap2 = dut_cap2.stats()["pim_capacity_stalls"]

    assert stalls_cap1 > stalls_cap2
    assert stalls_cap1 > 0
    assert stalls_cap2 == 0
    assert [c.command for c in history_cap1] == [c.command for c in history_cap2]


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
