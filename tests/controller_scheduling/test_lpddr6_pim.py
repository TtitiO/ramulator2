import pytest

import ramulator
import tests.controller_scheduling.harness as cs

pytestmark = pytest.mark.controller_scheduling


LPDDR6_ORG = "LPDDR6_16Gb_x12"
LPDDR6_TIMING = "LPDDR6_10667_BL24"


def make_dut(
    *,
    pim_blocks_per_bank: int = 1,
    pim_banks_per_block: int = 2,
    pim_datatype: str = "int8",
    pim_datatype_behavior_enabled: bool = False,
    rank: int = 1,
    refresh_manager=None,
):
    dram = ramulator.dram.LPDDR6PIM(
        org_preset=LPDDR6_ORG,
        timing_preset=LPDDR6_TIMING,
        rank=rank,
        pim_blocks_per_bank=pim_blocks_per_bank,
        pim_banks_per_block=pim_banks_per_block,
        pim_mac_execution_model="shared_block_serial",
        pim_datatype=pim_datatype,
        pim_datatype_class=pim_datatype,
        pim_datatype_behavior_enabled=pim_datatype_behavior_enabled,
    )
    return cs.ControllerUnderTest.make_lpddr_pim(
        dram,
        refresh_manager=refresh_manager,
    )


def run_until_command(dut, command: str, count: int = 1, max_ticks: int = 512):
    history = []
    for _ in range(max_ticks):
        history.extend(dut.tick())
        if sum(item.command == command for item in history) >= count:
            return history
    raise AssertionError(f"Did not observe {count} {command} commands")


def test_lpddr6_pim_host_read_uses_cas_and_short_read_vocabulary():
    dut = make_dut()
    address = dut.addr_vec(Rank=0, BankGroup=0, Bank=0, Row=4, Column=0)
    dut.send_request("Read", address)

    history = dut.run_until_idle(max_ticks=512)

    dut.assert_commands(["ACT1", "ACT2", "CAS", "RD_S"], history=history)


def test_lpddr6_pim_host_write_uses_cas_and_short_write_vocabulary():
    dut = make_dut()
    address = dut.addr_vec(Rank=0, BankGroup=0, Bank=0, Row=4, Column=0)
    dut.send_request("Write", address)

    history = dut.run_until_idle(max_ticks=512)

    dut.assert_commands(["ACT1", "ACT2", "CAS", "WR_S"], history=history)


def test_lpddr6_pim_compute_uses_shared_controller_and_delayed_completion():
    dut = make_dut()
    address = dut.addr_vec(Rank=0, BankGroup=0, Bank=0, Row=9, Column=0)
    dut.send_request("PIMCompute", address)

    history = run_until_command(dut, "PIM_MAC")
    assert [item.command for item in history] == ["ACT1", "ACT2", "PIM_MAC"]
    assert dut.stats()["num_pim_reqs_served"] == 0

    while dut.stats()["num_pim_reqs_served"] < 1:
        dut.tick()
    assert dut.stats()["num_pim_reqs_served"] == 1


def test_lpddr6_pim_shared_block_serialization_is_preserved():
    dut = make_dut(pim_blocks_per_bank=2, pim_banks_per_block=2)
    first = dut.addr_vec(Rank=0, BankGroup=0, Bank=0, Row=9, Column=0)
    second = dut.addr_vec(Rank=0, BankGroup=0, Bank=1, Row=9, Column=0)
    dut.send_request("PIMCompute", first)
    dut.send_request("PIMCompute", second)

    history = run_until_command(dut, "PIM_MAC", count=2, max_ticks=512)
    macs = [item for item in history if item.command == "PIM_MAC"]

    assert macs[1].clk - macs[0].clk >= dut.timings["nPIM_MAC_LAT"] + 1
    assert dut.stats()["pim_shared_block_stalls"] > 0


def test_lpddr6_pim_all_bank_commands_use_rank_scope():
    dut = make_dut()
    address = dut.addr_vec(Rank=0, BankGroup=0, Bank=0, Row=0, Column=0)
    dut.send_request("PIMLoadAll", address)
    dut.send_request("PIMComputeAll", address)

    history = dut.run_until_idle(max_ticks=512)
    names = [item.command for item in history]

    assert names == ["HAB", "PIM_BCAST", "HAB_PIM", "PIM_MAC_AB"]
    for item in history:
        if item.command in {"HAB", "PIM_BCAST", "HAB_PIM", "PIM_MAC_AB", "SB"}:
            assert item.addr_vec[dut.level_names.index("Rank")] == 0
            assert item.addr_vec[dut.level_names.index("BankGroup")] == 0
            assert item.addr_vec[dut.level_names.index("Bank")] == 0


def test_lpddr6_pim_refresh_uses_rank_scope_and_waits_for_compute():
    dut = make_dut(refresh_manager=ramulator.refresh_manager.AllBank())
    address = dut.addr_vec(Rank=0, BankGroup=0, Bank=0, Row=9, Column=0)
    dut.send_request("PIMCompute", address)
    run_until_command(dut, "PIM_MAC")

    refresh = dut.addr_vec(Rank=0, BankGroup=dut.ALL, Bank=dut.ALL, Row=0, Column=0)
    dut.priority_send("REFab", refresh)
    history = dut.run_until_idle(max_ticks=1024)
    refreshes = [item for item in history if item.command == "REFab"]

    assert refreshes
    assert refreshes[0].addr_vec[dut.level_names.index("Rank")] == 0
    assert refreshes[0].addr_vec[dut.level_names.index("BankGroup")] == dut.ALL
    assert refreshes[0].addr_vec[dut.level_names.index("Bank")] == dut.ALL


def test_lpddr6_pim_power_accounting_is_explicitly_unavailable_for_standard_energy():
    dram = ramulator.dram.LPDDR6PIM(
        org_preset=LPDDR6_ORG,
        timing_preset=LPDDR6_TIMING,
    )
    config = dram.to_config()
    assert config["impl"] == "LPDDR6PIM"
    assert config.get("power", {}).get("enabled", False) is False
    assert config["pim_compute_energy_pJ_per_mac"] == pytest.approx(0.35)
    assert config["pim_cell_to_pim_energy_pJ_per_256b"] == pytest.approx(2.68)


def test_lpddr6_pim_datatype_resources_change_only_when_behavior_enabled():
    int8 = make_dut(pim_datatype="int8", pim_datatype_behavior_enabled=True)
    fp16 = make_dut(pim_datatype="fp16", pim_datatype_behavior_enabled=True)

    assert int8.timings["nPIM_MAC_II"] == 4
    assert fp16.timings["nPIM_MAC_II"] == 4
    assert int8.timings["nPIM_MAC_LAT"] == 8
    assert fp16.timings["nPIM_MAC_LAT"] == 8
    assert int8.stats()["pim_lanes"] == 32
    assert fp16.stats()["pim_lanes"] == 16


def test_lpddr6_pim_rejects_lpddr5_command_names_in_device_api():
    dut = make_dut()
    address = dut.addr_vec(Rank=0, BankGroup=0, Bank=0, Row=4, Column=0)

    with pytest.raises(ValueError, match="Unknown command"):
        dut.priority_send("CAS_RD", address)
