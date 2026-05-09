import pytest

import ramulator
import tests.device_timings.harness as device_timings


pytestmark = pytest.mark.device_timings


def make_dut():
    dram = ramulator.dram.LPDDR5PIM(
        org_preset="LPDDR5_8Gb_x16",
        timing_preset="LPDDR5_6400",
        rank=1,
        pim_enabled=True,
    )
    return device_timings.DeviceUnderTest(dram)


def test_pim_mac_requires_act1_then_act2_before_timing_gate():
    dut = make_dut()
    a = dut.addr_vec(Rank=0, BankGroup=0, Bank=0, Row=12, Column=0)

    closed = dut.probe("PIM_MAC", a, clk=0)
    assert closed.preq == "ACT1"
    assert closed.ready is False

    dut.issue("ACT1", a, clk=0)
    after_act1 = dut.probe("PIM_MAC", a, clk=1)
    assert after_act1.preq == "ACT2"
    assert after_act1.ready is False

    dut.issue("ACT2", a, clk=1)
    early = dut.probe("PIM_MAC", a, clk=dut.timings["nRCD"] - 1)
    ontime = dut.probe("PIM_MAC", a, clk=dut.timings["nRCD"])

    assert early.preq == "PIM_MAC"
    assert early.timing_OK is False
    assert early.ready is False
    assert ontime.preq == "PIM_MAC"
    assert ontime.timing_OK is True
    assert ontime.ready is True


def test_pim_mac_to_pim_mac_gap_respects_npim_mac_ii_for_launch_legality():
    dut = make_dut()
    a = dut.addr_vec(Rank=0, BankGroup=0, Bank=0, Row=7, Column=0)

    dut.issue("ACT1", a, clk=0)
    dut.issue("ACT2", a, clk=1)
    first_mac_clk = 1 + dut.timings["nRCD"]
    dut.issue("PIM_MAC", a, clk=first_mac_clk)

    # This is a device-level launch-timing check only. Controller tests own the
    # post-issue execution-overlap behavior because in-flight residency is not
    # visible through DeviceUnderTest probes.
    early = dut.probe("PIM_MAC", a, clk=first_mac_clk + dut.timings["nPIM_MAC_II"] - 1)
    ontime = dut.probe("PIM_MAC", a, clk=first_mac_clk + dut.timings["nPIM_MAC_II"])

    assert early.preq == "PIM_MAC"
    assert early.timing_OK is False
    assert early.ready is False
    assert ontime.preq == "PIM_MAC"
    assert ontime.timing_OK is True
    assert ontime.ready is True


def test_pim_mac_keeps_single_bank_legality_scoped_per_bank():
    dut = make_dut()
    bank0 = dut.addr_vec(Rank=0, BankGroup=0, Bank=0, Row=7, Column=0)
    bank1 = dut.addr_vec(Rank=0, BankGroup=0, Bank=1, Row=7, Column=0)

    dut.issue("ACT1", bank0, clk=0)
    dut.issue("ACT2", bank0, clk=1)
    first_mac_clk = 1 + dut.timings["nRCD"]
    dut.issue("PIM_MAC", bank0, clk=first_mac_clk)

    closed_other_bank = dut.probe("PIM_MAC", bank1, clk=first_mac_clk + 1)
    assert closed_other_bank.preq == "ACT1"
    assert closed_other_bank.ready is False

    dut.issue("ACT1", bank1, clk=first_mac_clk + 1)
    after_other_act1 = dut.probe("PIM_MAC", bank1, clk=first_mac_clk + 2)
    assert after_other_act1.preq == "ACT2"
    assert after_other_act1.ready is False

    dut.issue("ACT2", bank1, clk=first_mac_clk + 2)
    ready_other = dut.probe("PIM_MAC", bank1, clk=first_mac_clk + 2 + dut.timings["nRCD"] - 1)

    assert ready_other.preq == "PIM_MAC"
    assert ready_other.timing_OK is True
    assert ready_other.ready is True


def test_pim_bcast_requires_hab_mode_first():
    dut = make_dut()
    a = dut.addr_vec(Rank=0, BankGroup=dut.ALL, Bank=dut.ALL, Row=0, Column=0)

    before = dut.probe("PIM_BCAST", a, clk=0)
    assert before.preq == "HAB"
    assert before.ready is False

    dut.issue("HAB", a, clk=0)
    after = dut.probe("PIM_BCAST", a, clk=1)
    assert after.preq == "PIM_BCAST"
    assert after.ready is True


def test_lpddr5_pim_declares_rank_pim_bcast_to_pim_bcast_nbl_timing_constraint():
    matches = [
        constraint
        for constraint in ramulator.dram.LPDDR5PIM.timing_constraints
        if constraint.level == "Rank"
        and constraint.preceding == ["PIM_BCAST"]
        and constraint.following == ["PIM_BCAST"]
        and constraint.latency == "nBL"
    ]
    assert len(matches) == 1


def test_pim_mac_ab_requires_hab_pim_mode_first():
    dut = make_dut()
    a = dut.addr_vec(Rank=0, BankGroup=dut.ALL, Bank=dut.ALL, Row=0, Column=0)

    before = dut.probe("PIM_MAC_AB", a, clk=0)
    assert before.preq == "HAB_PIM"
    assert before.ready is False

    dut.issue("HAB_PIM", a, clk=0)
    after = dut.probe("PIM_MAC_AB", a, clk=1)
    assert after.preq == "PIM_MAC_AB"
    assert after.ready is True


def test_pim_mac_ab_requires_hab_pim_even_after_pim_bcast_load_phase():
    dut = make_dut()
    a = dut.addr_vec(Rank=0, BankGroup=dut.ALL, Bank=dut.ALL, Row=0, Column=0)

    dut.issue("HAB", a, clk=0)
    dut.issue("PIM_BCAST", a, clk=1)

    before_hab_pim = dut.probe("PIM_MAC_AB", a, clk=2)
    assert before_hab_pim.preq == "HAB_PIM"
    assert before_hab_pim.ready is False

    dut.issue("HAB_PIM", a, clk=2)
    after_hab_pim = dut.probe("PIM_MAC_AB", a, clk=3)
    assert after_hab_pim.preq == "PIM_MAC_AB"
    assert after_hab_pim.ready is True


def test_sb_returns_all_bank_rank_mode_to_single_bank_mode():
    dut = make_dut()
    a = dut.addr_vec(Rank=0, BankGroup=dut.ALL, Bank=dut.ALL, Row=0, Column=0)

    dut.issue("HAB_PIM", a, clk=0)
    dut.issue("SB", a, clk=1)

    after_sb_bcast = dut.probe("PIM_BCAST", a, clk=2)
    after_sb_mac_ab = dut.probe("PIM_MAC_AB", a, clk=2)

    assert after_sb_bcast.preq == "HAB"
    assert after_sb_bcast.ready is False
    assert after_sb_mac_ab.preq == "HAB_PIM"
    assert after_sb_mac_ab.ready is False
