import pytest

import ramulator

BASE_KWARGS = {
    "org_preset": "LPDDR5_8Gb_x16",
    "timing_preset": "LPDDR5_6400",
}


def make_dram(**kwargs):
    return ramulator.dram.LPDDR5PIM(**BASE_KWARGS, **kwargs)


def make_lpddr6_pim(**kwargs):
    return ramulator.dram.LPDDR6PIM(
        org_preset="LPDDR6_16Gb_x12",
        timing_preset="LPDDR6_10667_BL24",
        **kwargs,
    )


def test_default_config_serializes_explicit_execution_contract():
    cfg = make_dram().to_config()

    assert cfg["pim_blocks_per_bank"] == 1
    assert cfg["pim_banks_per_block"] == 2
    assert cfg["pim_mac_execution_model"] == "shared_block_serial"
    assert cfg["pim_datatype"] == "int8"
    assert cfg["pim_datatype_class"] == "int8"
    assert cfg["pim_datatype_behavior_enabled"] is False
    assert cfg["pim_datatype_bits"] == 8
    assert cfg["pim_simd_width_bits"] == 256
    assert cfg["pim_lanes"] == 32
    assert cfg["pim_slots_per_request"] == 1


@pytest.mark.parametrize("field", ["pim_blocks_per_bank", "pim_banks_per_block"])
def test_positive_integer_resource_counts_are_required(field):
    with pytest.raises(ValueError, match=field):
        make_dram(**{field: 0})
    with pytest.raises(ValueError, match=field):
        make_dram(**{field: True})


def test_shared_block_grouping_is_validated_against_resolved_rank_organization():
    with pytest.raises(ValueError, match=r"banks per rank \(16\).*pim_banks_per_block \(3\)"):
        make_dram(pim_banks_per_block=3).to_config()

    with pytest.raises(ValueError, match=r"pim_banks_per_block \(32\).*banks per rank \(16\)"):
        make_dram(rank=2, pim_banks_per_block=32).to_config()


def test_unknown_datatype_names_do_not_fall_back_to_int8():
    with pytest.raises(ValueError, match="unknown pim_datatype 'fp8'"):
        make_dram(pim_datatype="fp8")

    with pytest.raises(ValueError, match="unknown pim_datatype_class 'fp8'"):
        make_dram(pim_datatype_class="fp8")


def test_datatype_class_cannot_silently_substitute_another_resource_profile():
    with pytest.raises(ValueError, match="must match pim_datatype"):
        make_dram(pim_datatype="int8", pim_datatype_class="fp16")


def test_simd_width_and_lane_count_must_describe_the_same_resource():
    with pytest.raises(ValueError, match="divisible by pim_datatype_bits"):
        make_dram(pim_datatype_bits=10, pim_simd_width_bits=256)

    with pytest.raises(ValueError, match=r"pim_lanes must equal.*\(32\), got 16"):
        make_dram(pim_datatype_bits=8, pim_simd_width_bits=256, pim_lanes=16)


def test_legacy_shared_block_names_normalize_with_deprecation_warnings():
    with pytest.warns(DeprecationWarning, match="pim_banks_per_mpu"):
        cfg = make_dram(pim_banks_per_mpu=1).to_config()
    assert cfg["pim_banks_per_block"] == 1
    assert "pim_banks_per_mpu" not in cfg

    with pytest.warns(DeprecationWarning, match="shared_mpu_serial"):
        cfg = make_dram(pim_mac_execution_model="shared_mpu_serial").to_config()
    assert cfg["pim_mac_execution_model"] == "shared_block_serial"


def test_lpddr6_pim_rejects_obsolete_compatibility_fields():
    for field, value in (
        ("pim_banks_per_mpu", 2),
        ("pim_slot_cost", 1),
        ("pim_mac_latency_scale", 2.0),
        ("pim_incremental_energy_scale", 2.0),
    ):
        with pytest.raises(ValueError, match=field):
            make_lpddr6_pim(**{field: value})
    with pytest.raises(ValueError, match="shared_mpu_serial"):
        make_lpddr6_pim(pim_mac_execution_model="shared_mpu_serial")


def test_legacy_shared_block_field_conflicts_fail_closed():
    with pytest.raises(ValueError, match="conflicts"):
        make_dram(pim_banks_per_block=1, pim_banks_per_mpu=2)


def test_legacy_slot_cost_alias_cannot_conflict_with_canonical_slot_count():
    with pytest.raises(ValueError, match="compatibility alias"):
        make_dram(pim_slots_per_request=2, pim_slot_cost=1)


def test_datatype_behavior_requires_enough_bank_slots_for_each_request():
    with pytest.raises(ValueError, match="at least pim_slots_per_request"):
        make_dram(
            pim_datatype_behavior_enabled=True,
            pim_blocks_per_bank=1,
            pim_slots_per_request=2,
        ).to_config()


def test_unsupported_source_backed_datatype_behavior_is_explicitly_rejected():
    with pytest.raises(ValueError, match="source-backed datatype resources"):
        make_dram(pim_datatype="bf16", pim_datatype_behavior_enabled=True)
