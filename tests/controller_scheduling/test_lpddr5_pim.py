import pytest

import ramulator
import tests.controller_scheduling.harness as cs

pytestmark = pytest.mark.controller_scheduling


def make_dut(
    pim_blocks_per_bank: int = 1,
    pim_banks_per_block: int = 2,
    pim_mac_execution_model: str = "shared_block_serial",
    pim_datatype: str = "int8",
    pim_datatype_class: str | None = None,
    pim_datatype_behavior_enabled: bool = False,
    rank: int = 1,
    controller_plugins=None,
    **dram_kwargs,
):
    dram = ramulator.dram.LPDDR5PIM(
        org_preset="LPDDR5_8Gb_x16",
        timing_preset="LPDDR5_6400",
        rank=rank,
        pim_blocks_per_bank=pim_blocks_per_bank,
        pim_banks_per_block=pim_banks_per_block,
        pim_mac_execution_model=pim_mac_execution_model,
        pim_datatype=pim_datatype,
        pim_datatype_class=pim_datatype_class,
        pim_datatype_behavior_enabled=pim_datatype_behavior_enabled,
        **dram_kwargs,
    )
    controller = ramulator.controller.LPDDR5PIM(
        scheduler=ramulator.scheduler.FRFCFS(),
        refresh_manager=ramulator.refresh_manager.NoRefresh(),
        row_policy=ramulator.row_policy.Open(),
        addr_mapper=ramulator.addr_mapper.PassThroughAddrMapper(),
        dram=dram,
        controller_plugins=controller_plugins or [],
    )
    return cs.ControllerUnderTest(controller)


def make_experimental_dut(**kwargs):
    return make_dut(pim_mac_execution_model="subbank_overlap_experimental", **kwargs)


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


def level_value(dut: cs.ControllerUnderTest, addr_vec: list[int], level_name: str) -> int:
    return addr_vec[dut.level_names.index(level_name)]


def bank_coords(dut: cs.ControllerUnderTest, cmd) -> tuple[int, int]:
    return (
        level_value(dut, cmd.addr_vec, "BankGroup"),
        level_value(dut, cmd.addr_vec, "Bank"),
    )


def rank_bank_coords(dut: cs.ControllerUnderTest, cmd) -> tuple[int, int, int]:
    return (
        level_value(dut, cmd.addr_vec, "Rank"),
        level_value(dut, cmd.addr_vec, "BankGroup"),
        level_value(dut, cmd.addr_vec, "Bank"),
    )


def per_bank_stats(stats: dict, prefix: str, bank_count: int = 4) -> list[int]:
    return [stats[f"{prefix}{bank_id}"] for bank_id in range(bank_count)]


def assert_pim_latency_split_identity(stats: dict):
    assert stats["pim_latency"] == stats["pim_response_latency"]
    assert stats["pim_response_latency"] == stats["pim_launch_wait"] + stats["pim_service_latency"]
    if stats["num_pim_reqs_served"] > 0:
        assert stats["avg_pim_latency"] == pytest.approx(stats["avg_pim_response_latency"])
        assert stats["avg_pim_response_latency"] == pytest.approx(
            stats["avg_pim_launch_wait"] + stats["avg_pim_service_latency"]
        )


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

    dut.assert_commands(
        ["ACT1", "ACT2", "ACT1", "ACT2", "CAS_RD", "RD", "PIM_MAC"], history=history
    )


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

    dut.assert_commands(
        ["ACT1", "ACT2", "PIM_MAC", "PREpb", "ACT1", "ACT2", "CAS_RD", "RD"], history=history
    )


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


def test_pim_datatype_labels_do_not_change_lpddr5_pim_scheduling_or_stats():
    addr_specs = [
        dict(Rank=0, BankGroup=0, Bank=0, Row=9, Column=0),
        dict(Rank=0, BankGroup=0, Bank=0, Row=9, Column=32),
    ]

    dut_int8 = make_experimental_dut(
        pim_blocks_per_bank=2, pim_datatype="int8", pim_datatype_class="int8"
    )
    for addr_spec in addr_specs:
        dut_int8.send_request("PIMCompute", dut_int8.addr_vec(**addr_spec))
    history_int8 = dut_int8.run_until_idle(max_ticks=256)

    dut_fp16 = make_experimental_dut(
        pim_blocks_per_bank=2, pim_datatype="fp16", pim_datatype_class="fp16"
    )
    for addr_spec in addr_specs:
        dut_fp16.send_request("PIMCompute", dut_fp16.addr_vec(**addr_spec))
    history_fp16 = dut_fp16.run_until_idle(max_ticks=256)

    stats_int8 = dut_int8.stats()
    stats_fp16 = dut_fp16.stats()
    stat_names = [
        "pim_capacity_stalls",
        "pim_dependency_stalls",
        "num_pim_reqs_served",
        "pim_inflight_peak",
    ]

    assert [(cmd.command, cmd.clk) for cmd in history_int8] == [
        (cmd.command, cmd.clk) for cmd in history_fp16
    ]
    assert {name: stats_int8[name] for name in stat_names} == {
        name: stats_fp16[name] for name in stat_names
    }


def test_pim_datatype_behavior_uses_explicit_resource_fields_not_scales():
    addr_specs = [
        dict(Rank=0, BankGroup=0, Bank=0, Row=9, Column=0),
        dict(Rank=0, BankGroup=0, Bank=0, Row=9, Column=32),
    ]

    dut_int8 = make_experimental_dut(
        pim_blocks_per_bank=2,
        pim_datatype="int8",
        pim_datatype_class="int8",
        pim_datatype_behavior_enabled=True,
    )
    for addr_spec in addr_specs:
        dut_int8.send_request("PIMCompute", dut_int8.addr_vec(**addr_spec))
    history_int8 = run_until_pim_mac_count(dut_int8, count=2)
    run_until_pim_reqs_served(dut_int8, count=2, max_ticks=dut_int8.timings["nPIM_MAC_LAT"] + 4)

    dut_fp16 = make_experimental_dut(
        pim_blocks_per_bank=2,
        pim_datatype="fp16",
        pim_datatype_class="fp16",
        pim_datatype_behavior_enabled=True,
    )
    for addr_spec in addr_specs:
        dut_fp16.send_request("PIMCompute", dut_fp16.addr_vec(**addr_spec))
    history_fp16 = run_until_pim_mac_count(dut_fp16, count=2, max_ticks=512)
    run_until_pim_reqs_served(dut_fp16, count=2, max_ticks=dut_fp16.timings["nPIM_MAC_LAT"] + 4)

    stats_int8 = dut_int8.stats()
    stats_fp16 = dut_fp16.stats()

    assert dut_int8.timings["nPIM_MAC_II"] == 4
    assert dut_fp16.timings["nPIM_MAC_II"] == 4
    assert dut_int8.timings["nPIM_MAC_LAT"] == 8
    assert dut_fp16.timings["nPIM_MAC_LAT"] == 8
    assert stats_int8["pim_lanes"] == 32
    assert stats_fp16["pim_lanes"] == 16
    assert stats_int8["pim_ops_per_block_issue"] == 64.0
    assert stats_fp16["pim_ops_per_block_issue"] == 32.0
    assert stats_int8["pim_ops_per_request"] == 64.0
    assert stats_fp16["pim_ops_per_request"] == 32.0
    assert stats_int8["pim_mac_issue_interval_cycles"] == 4
    assert stats_fp16["pim_mac_issue_interval_cycles"] == 4
    assert stats_int8["pim_mac_pipeline_latency_cycles"] == 8
    assert stats_fp16["pim_mac_pipeline_latency_cycles"] == 8
    assert stats_int8["pim_movement_cycles"] == 1
    assert stats_fp16["pim_movement_cycles"] == 1
    assert stats_int8["pim_completion_latency_cycles"] == 9
    assert stats_fp16["pim_completion_latency_cycles"] == 9
    assert stats_int8["pim_slots_per_request"] == 1
    assert stats_fp16["pim_slots_per_request"] == 1
    assert stats_int8["pim_capacity_stalls"] == stats_fp16["pim_capacity_stalls"]
    assert [(cmd.command, cmd.clk) for cmd in history_int8] == [
        (cmd.command, cmd.clk) for cmd in history_fp16
    ]
    dut_int8.assert_gap(2, 3, dut_int8.timings["nPIM_MAC_II"], history=history_int8)
    dut_fp16.assert_gap(2, 3, dut_fp16.timings["nPIM_MAC_II"], history=history_fp16)


def test_pim_datatype_legacy_scale_knobs_are_rejected():
    with pytest.raises(ValueError, match="pim_mac_latency_scale is deprecated"):
        make_dut(pim_datatype_behavior_enabled=True, pim_mac_latency_scale=2.0)

    with pytest.raises(ValueError, match="pim_incremental_energy_scale is deprecated"):
        make_dut(pim_datatype_behavior_enabled=True, pim_incremental_energy_scale=2.0)


def test_invalid_pim_mac_execution_model_is_rejected():
    with pytest.raises(ValueError, match="unknown pim_mac_execution_model"):
        make_dut(pim_mac_execution_model="same_bank_overlap")


def test_invalid_pim_banks_per_block_is_rejected():
    with pytest.raises(ValueError, match="pim_banks_per_block must be positive"):
        make_dut(pim_banks_per_block=0)

    with pytest.raises(ValueError, match="pim_banks_per_block must be positive"):
        make_dut(pim_banks_per_block=-1)


def test_incompatible_pim_banks_per_block_is_rejected():
    with pytest.raises(
        ValueError,
        match=r"banks per rank \(16\) must be divisible by pim_banks_per_block \(3\)",
    ):
        make_dut(pim_banks_per_block=3)


def test_pim_banks_per_block_larger_than_rank_scope_is_rejected():
    with pytest.raises(
        ValueError,
        match=r"pim_banks_per_block \(32\) exceeds banks per rank \(16\)",
    ):
        make_dut(rank=2, pim_banks_per_block=32)


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
    dut = make_experimental_dut(pim_blocks_per_bank=2)
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


def test_default_same_bank_independent_pim_launches_serialize_with_two_slots():
    dut = make_dut(pim_blocks_per_bank=2)
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
    assert stats["pim_shared_block_stalls"] == 0
    assert stats["num_bank_timing_blocked_cycles"] > 0
    assert stats["pim_capacity_stalls"] == 0
    assert stats["pim_inflight_peak"] == 1
    assert stats["pim_mac_execution_model"] == 0


def test_experimental_same_bank_independent_pim_launches_can_overlap_with_two_slots():
    dut = make_experimental_dut(pim_blocks_per_bank=2)
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
    assert stats["pim_shared_block_stalls"] == 0
    assert stats["pim_capacity_stalls"] == 0
    assert stats["pim_inflight_peak"] == 2
    assert stats["pim_mac_execution_model"] == 1


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
    assert stats["pim_shared_block_stalls"] == 0
    assert stats["num_bank_timing_blocked_cycles"] > 0
    assert stats["pim_capacity_stalls"] == 0
    assert stats["pim_inflight_peak"] == 1


def test_shared_block_serial_same_shared_block_paired_banks_serialize_by_default():
    dut = make_dut(pim_blocks_per_bank=1, pim_banks_per_block=2)
    bank0 = dut.addr_vec(Rank=0, BankGroup=0, Bank=0, Row=9, Column=0)
    bank1 = dut.addr_vec(Rank=0, BankGroup=0, Bank=1, Row=9, Column=0)

    dut.send_request("PIMCompute", bank0)
    dut.send_request("PIMCompute", bank1)

    history = run_until_pim_mac_count(dut, count=2)
    run_until_pim_reqs_served(dut, count=2, max_ticks=dut.timings["nPIM_MAC_LAT"] + 8)

    pim_cmds = [cmd for cmd in history if cmd.command == "PIM_MAC"]
    stats = dut.stats()

    assert len(pim_cmds) == 2
    assert [bank_coords(dut, cmd) for cmd in pim_cmds] == [(0, 0), (0, 1)]
    dut.assert_gap(0, 1, dut.timings["nPIM_MAC_LAT"] + 1, history=pim_cmds)
    assert stats["num_pim_reqs_served"] == 2
    assert stats["pim_dependency_stalls"] == 0
    assert stats["pim_shared_block_stalls"] > 0
    assert stats["pim_capacity_stalls"] == 0
    assert stats["pim_inflight_peak"] == 1
    assert stats["pim_banks_per_block"] == 2
    assert stats["pim_shared_block_count"] == 8
    assert stats["total_banks"] == 16
    assert stats["effective_shared_blocks"] == 8
    assert stats["num_shared_block_busy_blocked_cycles"] > 0
    assert_pim_latency_split_identity(stats)
    assert stats["pim_service_latency"] == 2 * stats["pim_completion_latency_cycles"]
    assert stats["pim_launch_wait"] > 0


def test_shared_block_serial_different_shared_blocks_can_overlap():
    dut = make_dut(pim_blocks_per_bank=1, pim_banks_per_block=2)
    bank0 = dut.addr_vec(Rank=0, BankGroup=0, Bank=0, Row=9, Column=0)
    bank2 = dut.addr_vec(Rank=0, BankGroup=0, Bank=2, Row=9, Column=0)

    dut.send_request("PIMCompute", bank0)
    dut.send_request("PIMCompute", bank2)

    history = run_until_pim_mac_count(dut, count=2)
    run_until_pim_reqs_served(dut, count=2, max_ticks=dut.timings["nPIM_MAC_LAT"] + 8)

    pim_cmds = [cmd for cmd in history if cmd.command == "PIM_MAC"]
    stats = dut.stats()

    assert len(pim_cmds) == 2
    assert [bank_coords(dut, cmd) for cmd in pim_cmds] == [(0, 0), (0, 2)]
    dut.assert_gap(0, 1, 4, history=pim_cmds)
    assert stats["num_pim_reqs_served"] == 2
    assert stats["pim_dependency_stalls"] == 0
    assert stats["pim_shared_block_stalls"] == 0
    assert stats["pim_capacity_stalls"] == 0
    assert stats["pim_inflight_peak"] == 1
    assert stats["pim_simultaneous_active_banks_peak"] == 2


def test_shared_block_serial_one_bank_per_block_allows_cross_bank_overlap():
    dut = make_dut(pim_blocks_per_bank=1, pim_banks_per_block=1)
    bank0 = dut.addr_vec(Rank=0, BankGroup=0, Bank=0, Row=9, Column=0)
    bank1 = dut.addr_vec(Rank=0, BankGroup=0, Bank=1, Row=9, Column=0)

    dut.send_request("PIMCompute", bank0)
    dut.send_request("PIMCompute", bank1)

    history = run_until_pim_mac_count(dut, count=2)
    run_until_pim_reqs_served(dut, count=2, max_ticks=dut.timings["nPIM_MAC_LAT"] + 8)

    pim_cmds = [cmd for cmd in history if cmd.command == "PIM_MAC"]
    stats = dut.stats()

    assert len(pim_cmds) == 2
    assert [bank_coords(dut, cmd) for cmd in pim_cmds] == [(0, 0), (0, 1)]
    dut.assert_gap(0, 1, 4, history=pim_cmds)
    assert stats["num_pim_reqs_served"] == 2
    assert stats["pim_dependency_stalls"] == 0
    assert stats["pim_shared_block_stalls"] == 0
    assert stats["pim_capacity_stalls"] == 0
    assert stats["pim_inflight_peak"] == 1
    assert stats["pim_simultaneous_active_banks_peak"] == 2
    assert stats["pim_banks_per_block"] == 1
    assert stats["pim_shared_block_count"] == 16
    assert stats["total_banks"] == 16
    assert stats["effective_shared_blocks"] == 16
    assert_pim_latency_split_identity(stats)
    assert stats["pim_service_latency"] == 2 * stats["pim_completion_latency_cycles"]


def test_shared_block_serial_all_rank_banks_share_one_block_serializes_cross_bankgroup():
    dut = make_dut(pim_blocks_per_bank=1, pim_banks_per_block=16)
    first_bank = dut.addr_vec(Rank=0, BankGroup=0, Bank=0, Row=9, Column=0)
    last_bank = dut.addr_vec(Rank=0, BankGroup=3, Bank=3, Row=9, Column=0)

    dut.send_request("PIMCompute", first_bank)
    dut.send_request("PIMCompute", last_bank)

    history = run_until_pim_mac_count(dut, count=2)
    run_until_pim_reqs_served(dut, count=2, max_ticks=dut.timings["nPIM_MAC_LAT"] + 8)

    pim_cmds = [cmd for cmd in history if cmd.command == "PIM_MAC"]
    stats = dut.stats()

    assert len(pim_cmds) == 2
    assert [bank_coords(dut, cmd) for cmd in pim_cmds] == [(0, 0), (3, 3)]
    dut.assert_gap(0, 1, dut.timings["nPIM_MAC_LAT"] + 1, history=pim_cmds)
    assert stats["num_pim_reqs_served"] == 2
    assert stats["pim_dependency_stalls"] == 0
    assert stats["pim_shared_block_stalls"] > 0
    assert stats["pim_capacity_stalls"] == 0
    assert stats["pim_inflight_peak"] == 1
    assert stats["pim_banks_per_block"] == 16
    assert stats["pim_shared_block_count"] == 1
    assert stats["effective_shared_blocks"] == 1


def test_shared_block_serial_four_bank_grouping_aligns_with_bankgroup_boundary():
    dut = make_dut(pim_blocks_per_bank=1, pim_banks_per_block=4)
    bank0 = dut.addr_vec(Rank=0, BankGroup=0, Bank=0, Row=9, Column=0)
    next_bankgroup_bank0 = dut.addr_vec(Rank=0, BankGroup=1, Bank=0, Row=9, Column=0)

    dut.send_request("PIMCompute", bank0)
    dut.send_request("PIMCompute", next_bankgroup_bank0)

    history = run_until_pim_mac_count(dut, count=2)
    run_until_pim_reqs_served(dut, count=2, max_ticks=dut.timings["nPIM_MAC_LAT"] + 8)

    pim_cmds = [cmd for cmd in history if cmd.command == "PIM_MAC"]
    stats = dut.stats()

    assert len(pim_cmds) == 2
    assert [bank_coords(dut, cmd) for cmd in pim_cmds] == [(0, 0), (1, 0)]
    dut.assert_gap(0, 1, 4, history=pim_cmds)
    assert stats["num_pim_reqs_served"] == 2
    assert stats["pim_dependency_stalls"] == 0
    assert stats["pim_capacity_stalls"] == 0
    assert stats["pim_inflight_peak"] == 1
    assert stats["pim_banks_per_block"] == 4
    assert stats["pim_shared_block_count"] == 4
    assert stats["pim_simultaneous_active_banks_peak"] == 2


def test_shared_block_serial_groups_do_not_cross_rank_boundary():
    dut = make_dut(rank=2, pim_blocks_per_bank=1, pim_banks_per_block=4)
    rank0_bank0 = dut.addr_vec(Rank=0, BankGroup=0, Bank=0, Row=9, Column=0)
    rank1_bank0 = dut.addr_vec(Rank=1, BankGroup=0, Bank=0, Row=9, Column=0)

    dut.send_request("PIMCompute", rank0_bank0)
    dut.send_request("PIMCompute", rank1_bank0)

    history = run_until_pim_mac_count(dut, count=2)
    run_until_pim_reqs_served(dut, count=2, max_ticks=dut.timings["nPIM_MAC_LAT"] + 8)

    pim_cmds = [cmd for cmd in history if cmd.command == "PIM_MAC"]
    stats = dut.stats()

    assert len(pim_cmds) == 2
    assert [rank_bank_coords(dut, cmd) for cmd in pim_cmds] == [(0, 0, 0), (1, 0, 0)]
    dut.assert_gap(0, 1, 1, history=pim_cmds)
    assert stats["num_pim_reqs_served"] == 2
    assert stats["pim_dependency_stalls"] == 0
    assert stats["pim_capacity_stalls"] == 0
    assert stats["pim_inflight_peak"] == 1
    assert stats["pim_banks_per_block"] == 4
    assert stats["pim_shared_block_count"] == 8
    assert stats["pim_simultaneous_active_banks_peak"] == 2


def test_experimental_same_shared_block_paired_banks_can_overlap_when_slots_permit():
    dut = make_experimental_dut(pim_blocks_per_bank=1, pim_banks_per_block=2)
    bank0 = dut.addr_vec(Rank=0, BankGroup=0, Bank=0, Row=9, Column=0)
    bank1 = dut.addr_vec(Rank=0, BankGroup=0, Bank=1, Row=9, Column=0)

    dut.send_request("PIMCompute", bank0)
    dut.send_request("PIMCompute", bank1)

    history = run_until_pim_mac_count(dut, count=2)
    run_until_pim_reqs_served(dut, count=2, max_ticks=dut.timings["nPIM_MAC_LAT"] + 8)

    pim_cmds = [cmd for cmd in history if cmd.command == "PIM_MAC"]
    stats = dut.stats()

    assert len(pim_cmds) == 2
    assert [bank_coords(dut, cmd) for cmd in pim_cmds] == [(0, 0), (0, 1)]
    dut.assert_gap(0, 1, 4, history=pim_cmds)
    assert stats["num_pim_reqs_served"] == 2
    assert stats["pim_dependency_stalls"] == 0
    assert stats["pim_capacity_stalls"] == 0
    assert stats["pim_inflight_peak"] == 1
    assert stats["pim_simultaneous_active_banks_peak"] == 2
    assert stats["pim_mac_execution_model"] == 1


def test_bounded_multi_bank_round_robin_keeps_one_inflight_slot_per_bank():
    dut = make_experimental_dut(pim_blocks_per_bank=2)
    addrs = [dut.addr_vec(Rank=0, BankGroup=0, Bank=bank, Row=9, Column=0) for bank in range(4)]

    for addr in addrs:
        dut.send_request("PIMCompute", addr)

    history = run_until_pim_mac_count(dut, count=4)
    run_until_pim_reqs_served(dut, count=4, max_ticks=dut.timings["nPIM_MAC_LAT"] + 12)

    pim_cmds = [cmd for cmd in history if cmd.command == "PIM_MAC"]
    stats = dut.stats()

    assert len(pim_cmds) == 4
    assert [bank_coords(dut, cmd) for cmd in pim_cmds] == [(0, 0), (0, 1), (0, 2), (0, 3)]
    assert [cmd.clk for cmd in pim_cmds] == [16, 20, 24, 28]
    assert stats["num_pim_reqs_served"] == 4
    assert stats["pim_dependency_stalls"] == 0
    assert stats["pim_capacity_stalls"] == 0
    assert stats["pim_inflight_peak"] == 1
    assert stats["pim_simultaneous_active_banks_peak"] == 3
    assert per_bank_stats(stats, "pim_launches_bank_") == [1, 1, 1, 1]
    assert per_bank_stats(stats, "pim_inflight_peak_bank_") == [1, 1, 1, 1]


def test_cross_bank_refpb_waits_only_for_target_bank_while_other_bank_remains_inflight():
    dut = make_experimental_dut(pim_blocks_per_bank=1)
    bank0 = dut.addr_vec(Rank=0, BankGroup=0, Bank=0, Row=9, Column=0)
    bank1 = dut.addr_vec(Rank=0, BankGroup=0, Bank=1, Row=9, Column=0)
    refresh_bank0 = dut.addr_vec(Rank=0, BankGroup=0, Bank=0, Row=dut.ALL, Column=0)

    dut.send_request("PIMCompute", bank0)
    dut.send_request("PIMCompute", bank1)

    history = run_until_pim_mac_count(dut, count=2)
    pim_cmds = [cmd for cmd in history if cmd.command == "PIM_MAC"]
    assert [bank_coords(dut, cmd) for cmd in pim_cmds] == [(0, 0), (0, 1)]

    dut.priority_send("REFpb", refresh_bank0)

    first_to_second_gap = pim_cmds[1].clk - pim_cmds[0].clk
    blocked_cycles = dut.timings["nPIM_MAC_LAT"] - first_to_second_gap - 1
    for _ in range(max(0, blocked_cycles)):
        issued = dut.tick()
        history.extend(issued)
        assert issued == []

    history.extend(dut.run_until_idle(max_ticks=512))
    stats = dut.stats()

    dut.assert_commands(
        ["ACT1", "ACT2", "ACT1", "ACT2", "PIM_MAC", "PIM_MAC", "PREpb", "REFpb"], history=history
    )
    assert history[6].clk > pim_cmds[1].clk
    assert history[6].clk > pim_cmds[0].clk + dut.timings["nPIM_MAC_LAT"]
    assert stats["num_pim_reqs_served"] == 2
    assert stats["pim_simultaneous_active_banks_peak"] == 2
    assert per_bank_stats(stats, "pim_launches_bank_") == [1, 1, 0, 0]
    assert per_bank_stats(stats, "pim_inflight_peak_bank_") == [1, 1, 0, 0]


def test_all_bank_load_then_execute_requires_mode_and_load_ordering():
    dut = make_dut()
    all_banks = dut.addr_vec(Rank=0, BankGroup=dut.ALL, Bank=dut.ALL, Row=0, Column=0)
    concrete = dut.addr_vec(Rank=0, BankGroup=0, Bank=0, Row=0, Column=0)

    dut.priority_send("HAB", all_banks)
    history = []
    history.extend(dut.tick())
    dut.send_request("PIMLoadAll", concrete)
    history.extend(dut.tick())
    dut.priority_send("HAB_PIM", all_banks)
    dut.send_request("PIMComputeAll", concrete)

    for _ in range(64):
        history.extend(dut.tick())
        if [cmd.command for cmd in history] == ["HAB", "PIM_BCAST", "HAB_PIM", "PIM_MAC_AB"]:
            break
    dut.assert_commands(["HAB", "PIM_BCAST", "HAB_PIM", "PIM_MAC_AB"], history=history)

    completion_limit = 2 * (dut.timings["nPIM_MAC_LAT"] + 1) + 2
    for _ in range(completion_limit):
        if dut.stats()["num_pim_ab_reqs_served"] == 1:
            break
        dut.tick()

    stats = dut.stats()
    assert stats["num_pim_ab_reqs_served"] == 1
    assert stats["pim_load_stalls"] == 0
    assert stats["pim_mode_stalls"] == 0
    assert stats["pim_ab_inflight_peak"] == 1
    assert stats["pim_inflight_peak"] == 16
    assert per_bank_stats(stats, "pim_launches_bank_") == [1, 1, 1, 1]
    assert per_bank_stats(stats, "pim_inflight_peak_bank_") == [1, 1, 1, 1]
    assert_pim_latency_split_identity(stats)
    assert stats["pim_service_latency"] == stats["pim_ab_completion_latency_cycles"]
    assert stats["pim_ab_completion_latency_cycles"] == (
        stats["pim_completion_latency_cycles"] * stats["pim_banks_per_block"]
    )
    assert stats["pim_launch_wait"] > 0


def test_all_bank_execute_stalls_without_load():
    dut = make_dut()
    all_banks = dut.addr_vec(Rank=0, BankGroup=dut.ALL, Bank=dut.ALL, Row=0, Column=0)
    concrete = dut.addr_vec(Rank=0, BankGroup=0, Bank=0, Row=0, Column=0)

    dut.priority_send("HAB_PIM", all_banks)
    dut.send_request("PIMComputeAll", concrete)

    history = []
    for _ in range(32):
        history.extend(dut.tick())

    assert [cmd.command for cmd in history] == ["HAB_PIM"]
    stats = dut.stats()
    assert stats["num_pim_ab_reqs_served"] == 0
    assert stats["pim_load_stalls"] > 0


def test_host_reads_do_not_issue_while_rank_is_in_hab_mode():
    dut = make_dut()
    all_banks = dut.addr_vec(Rank=0, BankGroup=dut.ALL, Bank=dut.ALL, Row=0, Column=0)
    host = dut.addr_vec(Rank=0, BankGroup=0, Bank=0, Row=4, Column=0)

    dut.priority_send("HAB", all_banks)
    history = []
    history.extend(dut.tick())
    dut.send_request("Read", host)

    for _ in range(16):
        history.extend(dut.tick())

    assert [cmd.command for cmd in history] == ["HAB"]


def test_host_reads_do_not_issue_while_rank_is_in_hab_pim_mode():
    dut = make_dut()
    all_banks = dut.addr_vec(Rank=0, BankGroup=dut.ALL, Bank=dut.ALL, Row=0, Column=0)
    host = dut.addr_vec(Rank=0, BankGroup=0, Bank=0, Row=4, Column=0)

    dut.priority_send("HAB_PIM", all_banks)
    history = []
    history.extend(dut.tick())
    dut.send_request("Read", host)

    for _ in range(16):
        history.extend(dut.tick())

    assert [cmd.command for cmd in history] == ["HAB_PIM"]


def test_all_bank_refresh_waits_for_pim_mac_ab_completion():
    dut = make_dut()
    all_banks = dut.addr_vec(Rank=0, BankGroup=dut.ALL, Bank=dut.ALL, Row=0, Column=0)
    concrete = dut.addr_vec(Rank=0, BankGroup=0, Bank=0, Row=0, Column=0)

    dut.priority_send("HAB", all_banks)
    history = []
    history.extend(dut.tick())
    dut.send_request("PIMLoadAll", concrete)
    history.extend(dut.tick())
    dut.priority_send("HAB_PIM", all_banks)
    dut.send_request("PIMComputeAll", concrete)

    for _ in range(64):
        issued = dut.tick()
        history.extend(issued)
        if [cmd.command for cmd in history] == ["HAB", "PIM_BCAST", "HAB_PIM", "PIM_MAC_AB"]:
            break
    else:
        raise AssertionError("PIM_MAC_AB was not issued within 64 ticks")

    dut.priority_send("REFab", all_banks)

    for _ in range(max(0, dut.timings["nPIM_MAC_LAT"] - 1)):
        issued = dut.tick()
        history.extend(issued)
        assert issued == []

    history.extend(dut.run_until_idle(max_ticks=512))

    commands = [cmd.command for cmd in history]
    assert commands[:4] == ["HAB", "PIM_BCAST", "HAB_PIM", "PIM_MAC_AB"]
    assert commands[-1] == "REFab"
    assert commands.index("PIM_MAC_AB") < commands.index("REFab")


def test_all_bank_completion_allows_sb_transition_and_host_read_progress():
    dut = make_dut()
    all_banks = dut.addr_vec(Rank=0, BankGroup=dut.ALL, Bank=dut.ALL, Row=0, Column=0)
    concrete = dut.addr_vec(Rank=0, BankGroup=0, Bank=0, Row=0, Column=0)
    host = dut.addr_vec(Rank=0, BankGroup=0, Bank=0, Row=4, Column=0)

    dut.priority_send("HAB", all_banks)
    history = []
    history.extend(dut.tick())
    dut.send_request("PIMLoadAll", concrete)
    history.extend(dut.tick())
    dut.priority_send("HAB_PIM", all_banks)
    dut.send_request("PIMComputeAll", concrete)

    for _ in range(64):
        issued = dut.tick()
        history.extend(issued)
        if [cmd.command for cmd in history] == ["HAB", "PIM_BCAST", "HAB_PIM", "PIM_MAC_AB"]:
            break
    else:
        raise AssertionError("PIM_MAC_AB was not issued within 64 ticks")

    for _ in range(dut.timings["nPIM_MAC_LAT"] + 2):
        history.extend(dut.tick())

    dut.priority_send("SB", all_banks)
    dut.send_request("Read", host)
    history.extend(dut.run_until_idle(max_ticks=512))

    dut.assert_commands(
        ["HAB", "PIM_BCAST", "HAB_PIM", "PIM_MAC_AB", "SB", "ACT1", "ACT2", "CAS_RD", "RD"],
        history=history,
    )
    stats = dut.stats()
    assert stats["num_pim_ab_reqs_served"] == 1
