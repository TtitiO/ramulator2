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


def test_pim_mac_to_pim_mac_gap_respects_npim_mac_lat():
    dut = make_dut()
    a = dut.addr_vec(Rank=0, BankGroup=0, Bank=0, Row=7, Column=0)

    dut.issue("ACT1", a, clk=0)
    dut.issue("ACT2", a, clk=1)
    first_mac_clk = 1 + dut.timings["nRCD"]
    dut.issue("PIM_MAC", a, clk=first_mac_clk)

    early = dut.probe("PIM_MAC", a, clk=first_mac_clk + dut.timings["nPIM_MAC_LAT"] - 1)
    ontime = dut.probe("PIM_MAC", a, clk=first_mac_clk + dut.timings["nPIM_MAC_LAT"])

    assert early.preq == "PIM_MAC"
    assert early.timing_OK is False
    assert early.ready is False
    assert ontime.preq == "PIM_MAC"
    assert ontime.timing_OK is True
    assert ontime.ready is True
