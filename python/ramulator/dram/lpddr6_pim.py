from ramulator.dram.lpddr6 import LPDDR6
from ramulator.dram.pim_validation import optional_int, optional_number
from ramulator.dram.spec import TimingConstraint

ResourceValue = int | float


PIM_EVENT_ENERGY_FIELDS = [
    "pim_compute_energy_pJ_per_mac",
    "pim_array_local_energy_pJ",
    "pim_cell_to_pim_energy_pJ_per_256b",
    "pim_vrf_access_energy_pJ",
    "pim_srf_access_energy_pJ",
    "pim_mode_switch_energy_pJ",
]


PIM_DATATYPE_RESOURCES: dict[str, dict[str, ResourceValue]] = {
    "int8": {
        "pim_datatype_bits": 8,
        "pim_simd_width_bits": 256,
        "pim_ops_per_mac": 2,
        "pim_lanes": 32,
        "pim_ops_per_block_issue": 64,
        "pim_mac_issue_interval_cycles": 4,
        "pim_mac_pipeline_latency_cycles": 8,
        "pim_movement_cycles": 1,
        "pim_writeback_cycles": 0,
        "pim_slots_per_request": 1,
    },
    "fp16": {
        "pim_datatype_bits": 16,
        "pim_simd_width_bits": 256,
        "pim_ops_per_mac": 2,
        "pim_lanes": 16,
        "pim_ops_per_block_issue": 32,
        "pim_mac_issue_interval_cycles": 4,
        "pim_mac_pipeline_latency_cycles": 8,
        "pim_movement_cycles": 1,
        "pim_writeback_cycles": 0,
        "pim_slots_per_request": 1,
    },
}


PIM_DATATYPE_METADATA: dict[str, dict[str, ResourceValue]] = {
    **PIM_DATATYPE_RESOURCES,
    "int16": {
        "pim_datatype_bits": 16,
        "pim_simd_width_bits": 256,
        "pim_ops_per_mac": 2,
        "pim_lanes": 16,
        "pim_ops_per_block_issue": 32,
        "pim_mac_issue_interval_cycles": 8,
        "pim_mac_pipeline_latency_cycles": 8,
        "pim_movement_cycles": 1,
        "pim_writeback_cycles": 0,
        "pim_slots_per_request": 1,
    },
    "bf16": {
        "pim_datatype_bits": 16,
        "pim_simd_width_bits": 256,
        "pim_ops_per_mac": 2,
        "pim_lanes": 16,
        "pim_ops_per_block_issue": 32,
        "pim_mac_issue_interval_cycles": 8,
        "pim_mac_pipeline_latency_cycles": 8,
        "pim_movement_cycles": 1,
        "pim_writeback_cycles": 0,
        "pim_slots_per_request": 1,
    },
}


PIM_MAC_EXECUTION_MODELS = {
    "shared_block_serial",
    "subbank_overlap_experimental",
}
# Literature-anchored energy defaults:
#   compute:   int8=0.35 pJ/MAC (CD-PIM, LPDDR6-PIM-native)
#              fp16=0.69, int16/bf16=0.55 (P3-LLM/LP-Spec ratios)
#   movement:  cell_to_pim=2.68 pJ/256b (paper Table III, O'Connor FGDRAM)
#   RF access: vrf=3.17 pJ/256b, srf=0.40 pJ/32b (paper Table III)
#   mode_switch_energy: 0.0 (no public number)
#   array_local_energy: 0.0 (folded into movement; charged in layer-1)
#
# Per-bank PIM_MAC energy does NOT include rank-level bus energy (JEDEC
# IDD0-IDD2N × nBL_min). Rank-level bus energy for PIM_BCAST is accounted
# by the inherited LPDDR6 power model.
_PIM_ENERGY_DEFAULTS_BY_DTYPE: dict[str, dict[str, float]] = {
    "int8": {"pim_compute_energy_pJ_per_mac": 0.35},
    "fp16": {"pim_compute_energy_pJ_per_mac": 0.69},
    "int16": {"pim_compute_energy_pJ_per_mac": 0.55},
    "bf16": {"pim_compute_energy_pJ_per_mac": 0.55},
}
_PIM_ENERGY_SHARED_DEFAULTS: dict[str, float] = {
    "pim_array_local_energy_pJ": 0.0,
    "pim_cell_to_pim_energy_pJ_per_256b": 2.68,
    "pim_vrf_access_energy_pJ": 3.17,
    "pim_srf_access_energy_pJ": 0.40,
    "pim_mode_switch_energy_pJ": 0.0,
}

# DRAMPower v6.2.0 tests/tests_drampower/resources/lpddr6.json, converted
# from A to mA for PIMScope's V * mA * ns = pJ equations. The upstream file
# is a validation fixture with synthetic/zero fields, not a device datasheet.
# Keep the provenance explicit and do not describe this profile as calibrated
# LPDDR6 silicon power.
DRAMPOWER_V620_LPDDR6_TEST_PROFILE: dict[str, float | bool] = {
    "enabled": True,
    "VDD1": 1.2,
    "VDD2C": 1.2,
    "VDD2D": 1.2,
    "IDD01": 56.25,
    "IDD02C": 0.0,
    "IDD02D": 0.0,
    "IDD2N1": 33.75,
    "IDD2N2C": 0.0,
    "IDD2N2D": 0.0,
    "IDD3N1": 35.0,
    "IDD3N2C": 0.0,
    "IDD3N2D": 0.0,
    "IDD4R1": 157.5,
    "IDD4R2C": 0.0,
    "IDD4R2D": 0.0,
    "IDD4W1": 135.0,
    "IDD4W2C": 0.0,
    "IDD4W2D": 0.0,
    "IDD51": 118.0,
    "IDD52C": 0.0,
    "IDD52D": 0.0,
}

for _dtype, _resource in PIM_DATATYPE_METADATA.items():
    for _field in PIM_EVENT_ENERGY_FIELDS:
        _resource[_field] = (
            _PIM_ENERGY_DEFAULTS_BY_DTYPE.get(_dtype, {}).get(_field)
            or _PIM_ENERGY_SHARED_DEFAULTS[_field]
        )


class LPDDR6PIM(LPDDR6):
    """LPDDR6 plus the PIMScope execution-resource and mode abstractions.

    ``nPIM_MAC_II`` constrains command launch spacing. Request completion is
    modeled separately by the controller as pipeline + movement + writeback
    residency, subject to bank slots and optional shared-block serialization.
    """

    name = "LPDDR6PIM"

    # Keep inherited LPDDR6 standard terms separate from PIM-only event energy.
    # The default standard profile is the DRAMPower v6.2 test fixture above and
    # is intentionally reported as experimental rather than silicon-calibrated.
    power_incremental_commands_counted = [
        "PIM_MAC",
        "PIM_MAC_AB",
        "PIM_BCAST",
        "HAB",
        "HAB_PIM",
        "SB",
    ]
    power_incremental_command_hooks = [
        ("Rank", command, "COUNT_PIM_INCREMENTAL_ENERGY")
        for command in power_incremental_commands_counted
    ]
    power_incremental_command_energy_timings = {
        "PIM_MAC": "nPIM_MAC_LAT",
        "PIM_MAC_AB": "nPIM_MAC_LAT",
        "PIM_BCAST": "nBL_min",
        "HAB": "nBL_min",
        "HAB_PIM": "nBL_min",
        "SB": "nBL_min",
    }
    _pim_mac_event_energy_expr = (
        "pim_array_local_energy_pJ + "
        "pim_lanes * pim_compute_energy_pJ_per_mac + "
        "pim_cell_to_pim_energy_pJ_per_256b + "
        "pim_vrf_access_energy_pJ + "
        "pim_srf_access_energy_pJ"
    )
    power_incremental_command_event_energy_exprs = {
        "PIM_MAC": _pim_mac_event_energy_expr,
        "PIM_MAC_AB": _pim_mac_event_energy_expr,
        "PIM_BCAST": "pim_cell_to_pim_energy_pJ_per_256b",
        "HAB": "pim_mode_switch_energy_pJ",
        "HAB_PIM": "pim_mode_switch_energy_pJ",
        "SB": "pim_mode_switch_energy_pJ",
    }
    _rank_transfer_terms = [
        (f"VDD{rail}", f"IDD0{rail}", f"IDD2N{rail}") for rail in ("1", "2C", "2D")
    ]
    power_incremental_command_energy_terms = {
        "PIM_MAC": [],
        "PIM_MAC_AB": [],
        "PIM_BCAST": _rank_transfer_terms,
        "HAB": _rank_transfer_terms,
        "HAB_PIM": _rank_transfer_terms,
        "SB": _rank_transfer_terms,
    }

    levels = {
        **LPDDR6.levels,
        "Rank": "PIM_SB",
    }

    states = LPDDR6.states + ["PIM_SB", "PIM_HAB", "PIM_HAB_PIM"]

    commands = LPDDR6.commands + [
        "SB",
        "HAB",
        "HAB_PIM",
        "PIM_BCAST",
        "PIM_MAC",
        "PIM_MAC_AB",
    ]

    timing_params = LPDDR6.timing_params + ["nPIM_MAC_LAT", "nPIM_MAC_II"]

    supported_requests = {
        **LPDDR6.supported_requests,
        "PIMCompute": "PIM_MAC",
        "PIMLoadAll": "PIM_BCAST",
        "PIMComputeAll": "PIM_MAC_AB",
    }

    timing_constraints = LPDDR6.timing_constraints + [
        TimingConstraint(level="Bank", preceding=["ACT1"], following=["PIM_MAC"], latency="nRCDr"),
        TimingConstraint(
            level="Bank",
            preceding=["PIM_MAC"],
            following=["PIM_MAC"],
            latency="nPIM_MAC_II",
        ),
        TimingConstraint(
            level="Rank",
            preceding=["PIM_MAC_AB"],
            following=["PIM_MAC_AB"],
            latency="nPIM_MAC_II",
        ),
        # Bounded spacing abstraction for the synthetic PIM_BCAST opcode; exact
        # LPDDR6-PIM broadcast/source timing is not public silicon ground truth.
        TimingConstraint(
            level="Rank",
            preceding=["PIM_BCAST"],
            following=["PIM_BCAST"],
            latency="nBL_min",
        ),
    ]

    org_presets = LPDDR6.org_presets

    timing_presets = {
        preset_name: {
            **preset,
            "nPIM_MAC_LAT": 8,
            "nPIM_MAC_II": 8,
        }
        for preset_name, preset in LPDDR6.timing_presets.items()
    }

    def __init__(
        self,
        *,
        org_preset,
        timing_preset,
        power=None,
        pim_blocks_per_bank=1,
        pim_banks_per_block=2,
        pim_mac_execution_model="shared_block_serial",
        pim_datatype="int8",
        pim_datatype_class=None,
        pim_datatype_behavior_enabled=False,
        pim_datatype_bits=None,
        pim_simd_width_bits=None,
        pim_lanes=None,
        pim_ops_per_mac=None,
        pim_ops_per_block_issue=None,
        pim_ops_per_request=None,
        pim_mac_issue_interval_cycles=None,
        pim_mac_pipeline_latency_cycles=None,
        pim_mac_latency_cycles=None,
        pim_movement_cycles=None,
        pim_writeback_cycles=None,
        pim_slots_per_request=None,
        pim_compute_energy_pJ_per_mac=None,
        pim_array_local_energy_pJ=None,
        pim_cell_to_pim_energy_pJ_per_256b=None,
        pim_vrf_access_energy_pJ=None,
        pim_srf_access_energy_pJ=None,
        pim_mode_switch_energy_pJ=None,
        **overrides,
    ):
        pim_datatype = str(pim_datatype).lower()
        if pim_datatype not in PIM_DATATYPE_METADATA:
            supported = ", ".join(sorted(PIM_DATATYPE_METADATA))
            raise ValueError(
                f"LPDDR6PIM unknown pim_datatype '{pim_datatype}'; supported datatypes: {supported}"
            )
        if not pim_datatype_class:
            pim_datatype_class = pim_datatype
        pim_datatype_class = str(pim_datatype_class).lower()
        if pim_datatype_class not in PIM_DATATYPE_METADATA:
            supported = ", ".join(sorted(PIM_DATATYPE_METADATA))
            raise ValueError(
                f"LPDDR6PIM unknown pim_datatype_class '{pim_datatype_class}'; "
                f"supported datatype classes: {supported}"
            )
        if pim_datatype_class != pim_datatype:
            raise ValueError(
                "LPDDR6PIM pim_datatype_class must match pim_datatype; "
                "cross-datatype resource substitution is not a defined hardware model"
            )
        pim_mac_execution_model = str(pim_mac_execution_model)
        if pim_mac_execution_model not in PIM_MAC_EXECUTION_MODELS:
            supported = ", ".join(sorted(PIM_MAC_EXECUTION_MODELS))
            raise ValueError(
                f"LPDDR6PIM unknown pim_mac_execution_model '{pim_mac_execution_model}'; "
                f"supported values: {supported}"
            )

        integer_overrides = {
            "pim_datatype_bits": (pim_datatype_bits, 1),
            "pim_simd_width_bits": (pim_simd_width_bits, 1),
            "pim_lanes": (pim_lanes, 1),
            "pim_mac_issue_interval_cycles": (pim_mac_issue_interval_cycles, 1),
            "pim_mac_pipeline_latency_cycles": (pim_mac_pipeline_latency_cycles, 1),
            "pim_mac_latency_cycles": (pim_mac_latency_cycles, 1),
            "pim_movement_cycles": (pim_movement_cycles, 0),
            "pim_writeback_cycles": (pim_writeback_cycles, 0),
            "pim_slots_per_request": (pim_slots_per_request, 1),
            "pim_ops_per_mac": (pim_ops_per_mac, 1),
            "pim_ops_per_block_issue": (pim_ops_per_block_issue, 1),
            "pim_ops_per_request": (pim_ops_per_request, 1),
        }
        for field, (value, minimum) in integer_overrides.items():
            optional_int(value, f"LPDDR6PIM {field}", minimum=minimum)
        if not isinstance(pim_datatype_behavior_enabled, bool):
            raise ValueError("LPDDR6PIM pim_datatype_behavior_enabled must be a boolean")
        for field, value in {
            "pim_compute_energy_pJ_per_mac": pim_compute_energy_pJ_per_mac,
            "pim_array_local_energy_pJ": pim_array_local_energy_pJ,
            "pim_cell_to_pim_energy_pJ_per_256b": pim_cell_to_pim_energy_pJ_per_256b,
            "pim_vrf_access_energy_pJ": pim_vrf_access_energy_pJ,
            "pim_srf_access_energy_pJ": pim_srf_access_energy_pJ,
            "pim_mode_switch_energy_pJ": pim_mode_switch_energy_pJ,
        }.items():
            optional_number(value, f"LPDDR6PIM {field}", positive=False)

        resource = dict(PIM_DATATYPE_METADATA[pim_datatype_class])
        if pim_datatype_behavior_enabled and pim_datatype_class not in PIM_DATATYPE_RESOURCES:
            supported = ", ".join(sorted(PIM_DATATYPE_RESOURCES))
            raise ValueError(
                f"LPDDR6PIM source-backed datatype resources for '{pim_datatype_class}' are unsupported; "
                f"supported datatype classes: {supported}"
            )
        if pim_datatype_bits is not None:
            resource["pim_datatype_bits"] = int(pim_datatype_bits)
        if pim_simd_width_bits is not None:
            resource["pim_simd_width_bits"] = int(pim_simd_width_bits)
        if pim_lanes is not None:
            resource["pim_lanes"] = int(pim_lanes)
        if pim_ops_per_mac is not None:
            resource["pim_ops_per_mac"] = float(pim_ops_per_mac)
        if pim_ops_per_block_issue is not None:
            resource["pim_ops_per_block_issue"] = float(pim_ops_per_block_issue)
        if pim_ops_per_request is not None and pim_ops_per_block_issue is None:
            resource["pim_ops_per_block_issue"] = float(pim_ops_per_request)
        if pim_mac_issue_interval_cycles is not None:
            resource["pim_mac_issue_interval_cycles"] = int(pim_mac_issue_interval_cycles)
        if pim_mac_pipeline_latency_cycles is not None:
            resource["pim_mac_pipeline_latency_cycles"] = int(pim_mac_pipeline_latency_cycles)
        if pim_mac_latency_cycles is not None and pim_mac_pipeline_latency_cycles is None:
            resource["pim_mac_pipeline_latency_cycles"] = int(pim_mac_latency_cycles)
        if pim_movement_cycles is not None:
            resource["pim_movement_cycles"] = int(pim_movement_cycles)
        if pim_writeback_cycles is not None:
            resource["pim_writeback_cycles"] = int(pim_writeback_cycles)
        if pim_slots_per_request is not None:
            resource["pim_slots_per_request"] = int(pim_slots_per_request)

        energy_overrides = {
            "pim_compute_energy_pJ_per_mac": pim_compute_energy_pJ_per_mac,
            "pim_array_local_energy_pJ": pim_array_local_energy_pJ,
            "pim_cell_to_pim_energy_pJ_per_256b": pim_cell_to_pim_energy_pJ_per_256b,
            "pim_vrf_access_energy_pJ": pim_vrf_access_energy_pJ,
            "pim_srf_access_energy_pJ": pim_srf_access_energy_pJ,
            "pim_mode_switch_energy_pJ": pim_mode_switch_energy_pJ,
        }
        for energy_field, energy_value in energy_overrides.items():
            if energy_value is not None:
                resource[energy_field] = float(energy_value)

        if pim_lanes is None:
            resource["pim_lanes"] = int(
                resource["pim_simd_width_bits"] // resource["pim_datatype_bits"]
            )
        if pim_ops_per_block_issue is None and pim_ops_per_request is None:
            resource["pim_ops_per_block_issue"] = float(resource["pim_lanes"]) * float(
                resource["pim_ops_per_mac"]
            )
        resource["pim_ops_per_request"] = float(resource["pim_ops_per_block_issue"])

        if not isinstance(pim_blocks_per_bank, int) or isinstance(pim_blocks_per_bank, bool):
            raise ValueError("LPDDR6PIM pim_blocks_per_bank must be an integer")
        if not isinstance(pim_banks_per_block, int) or isinstance(pim_banks_per_block, bool):
            raise ValueError("LPDDR6PIM pim_banks_per_block must be an integer")
        if pim_blocks_per_bank <= 0:
            raise ValueError("LPDDR6PIM pim_blocks_per_bank must be positive")
        if resource["pim_datatype_bits"] <= 0:
            raise ValueError("LPDDR6PIM pim_datatype_bits must be positive")
        if resource["pim_simd_width_bits"] <= 0:
            raise ValueError("LPDDR6PIM pim_simd_width_bits must be positive")
        if resource["pim_simd_width_bits"] % resource["pim_datatype_bits"] != 0:
            raise ValueError("LPDDR6PIM pim_simd_width_bits must be divisible by pim_datatype_bits")
        expected_lanes = resource["pim_simd_width_bits"] // resource["pim_datatype_bits"]
        if resource["pim_lanes"] <= 0:
            raise ValueError("LPDDR6PIM pim_lanes must be positive")
        if resource["pim_lanes"] != expected_lanes:
            raise ValueError(
                f"LPDDR6PIM pim_lanes must equal pim_simd_width_bits / "
                f"pim_datatype_bits ({expected_lanes}), got {resource['pim_lanes']}"
            )
        if resource["pim_ops_per_mac"] <= 0:
            raise ValueError("LPDDR6PIM pim_ops_per_mac must be positive")
        if resource["pim_ops_per_block_issue"] <= 0:
            raise ValueError("LPDDR6PIM pim_ops_per_block_issue must be positive")
        if resource["pim_ops_per_request"] <= 0:
            raise ValueError("LPDDR6PIM pim_ops_per_request must be positive")
        if resource["pim_mac_issue_interval_cycles"] <= 0:
            raise ValueError("LPDDR6PIM pim_mac_issue_interval_cycles must be positive")
        if resource["pim_mac_pipeline_latency_cycles"] <= 0:
            raise ValueError("LPDDR6PIM pim_mac_pipeline_latency_cycles must be positive")
        if resource["pim_movement_cycles"] < 0:
            raise ValueError("LPDDR6PIM pim_movement_cycles must be non-negative")
        if resource["pim_writeback_cycles"] < 0:
            raise ValueError("LPDDR6PIM pim_writeback_cycles must be non-negative")
        if resource["pim_slots_per_request"] <= 0:
            raise ValueError("LPDDR6PIM pim_slots_per_request must be positive")
        if pim_banks_per_block <= 0:
            raise ValueError("LPDDR6PIM pim_banks_per_block must be positive")
        for energy_field in PIM_EVENT_ENERGY_FIELDS:
            if resource[energy_field] < 0:
                raise ValueError(f"LPDDR6PIM {energy_field} must be non-negative")

        self.pim_blocks_per_bank = pim_blocks_per_bank
        self.pim_banks_per_block = pim_banks_per_block
        self.pim_mac_execution_model = pim_mac_execution_model
        self.pim_datatype = pim_datatype
        self.pim_datatype_class = pim_datatype_class
        self.pim_datatype_behavior_enabled = pim_datatype_behavior_enabled
        self.pim_datatype_resource = resource
        if power is None:
            power = dict(DRAMPOWER_V620_LPDDR6_TEST_PROFILE)
        super().__init__(
            org_preset=org_preset, timing_preset=timing_preset, power=power, **overrides
        )

    def _resolve_pim_resource_cycles(self, base_pim_mac_latency, base_pim_mac_issue_interval):
        if self.pim_datatype_behavior_enabled:
            return (
                int(self.pim_datatype_resource["pim_mac_pipeline_latency_cycles"]),
                int(self.pim_datatype_resource["pim_mac_issue_interval_cycles"]),
            )
        return base_pim_mac_latency, base_pim_mac_issue_interval

    def resolve(self):
        org_dict, timing_dict = super().resolve()
        banks_per_rank = 1
        for level_name in list(type(self).levels)[2:]:
            if level_name in {"Row", "Column"}:
                continue
            banks_per_rank *= int(org_dict[level_name.lower()])
        if self.pim_banks_per_block > banks_per_rank:
            raise ValueError(
                f"LPDDR6PIM pim_banks_per_block ({self.pim_banks_per_block}) exceeds "
                f"banks per rank ({banks_per_rank})"
            )
        if banks_per_rank % self.pim_banks_per_block != 0:
            raise ValueError(
                f"LPDDR6PIM banks per rank ({banks_per_rank}) must be divisible by "
                f"pim_banks_per_block ({self.pim_banks_per_block})"
            )
        if self.pim_datatype_behavior_enabled and self.pim_blocks_per_bank < int(
            self.pim_datatype_resource["pim_slots_per_request"]
        ):
            raise ValueError(
                "LPDDR6PIM pim_blocks_per_bank must be at least pim_slots_per_request "
                "when datatype behavior is enabled"
            )

        base_pim_mac_latency = timing_dict["nPIM_MAC_LAT"]
        base_pim_mac_issue_interval = timing_dict["nPIM_MAC_II"]
        pim_mac_latency_cycles, pim_mac_issue_interval_cycles = self._resolve_pim_resource_cycles(
            base_pim_mac_latency,
            base_pim_mac_issue_interval,
        )
        timing_dict["nPIM_MAC_LAT"] = pim_mac_latency_cycles
        timing_dict["nPIM_MAC_II"] = pim_mac_issue_interval_cycles
        self._resolved_pim_mac_latency_cycles = pim_mac_latency_cycles
        self._resolved_pim_mac_issue_interval_cycles = pim_mac_issue_interval_cycles
        return org_dict, timing_dict

    def to_config(self):
        cfg = super().to_config()
        if self.pim_datatype_behavior_enabled:
            pim_mac_latency_cycles = self._resolved_pim_mac_latency_cycles
            pim_mac_issue_interval_cycles = self._resolved_pim_mac_issue_interval_cycles
            pim_movement_cycles = self.pim_datatype_resource["pim_movement_cycles"]
            pim_writeback_cycles = self.pim_datatype_resource["pim_writeback_cycles"]
            pim_slots_per_request = self.pim_datatype_resource["pim_slots_per_request"]
        else:
            pim_mac_latency_cycles = self._resolved_pim_mac_latency_cycles
            pim_mac_issue_interval_cycles = self._resolved_pim_mac_issue_interval_cycles
            pim_movement_cycles = 1
            pim_writeback_cycles = 0
            pim_slots_per_request = 1

        cfg["pim_blocks_per_bank"] = self.pim_blocks_per_bank
        cfg["pim_banks_per_block"] = self.pim_banks_per_block
        cfg["pim_mac_execution_model"] = self.pim_mac_execution_model
        cfg["pim_datatype"] = self.pim_datatype
        cfg["pim_datatype_class"] = self.pim_datatype_class
        cfg["pim_datatype_behavior_enabled"] = self.pim_datatype_behavior_enabled
        cfg["pim_datatype_bits"] = self.pim_datatype_resource["pim_datatype_bits"]
        cfg["pim_simd_width_bits"] = self.pim_datatype_resource["pim_simd_width_bits"]
        cfg["pim_lanes"] = self.pim_datatype_resource["pim_lanes"]
        cfg["pim_ops_per_mac"] = self.pim_datatype_resource["pim_ops_per_mac"]
        cfg["pim_ops_per_block_issue"] = self.pim_datatype_resource["pim_ops_per_block_issue"]
        cfg["pim_ops_per_request"] = self.pim_datatype_resource["pim_ops_per_request"]
        cfg["pim_mac_issue_interval_cycles"] = pim_mac_issue_interval_cycles
        cfg["pim_mac_latency_cycles"] = pim_mac_latency_cycles
        cfg["pim_mac_pipeline_latency_cycles"] = pim_mac_latency_cycles
        cfg["pim_movement_cycles"] = pim_movement_cycles
        cfg["pim_writeback_cycles"] = pim_writeback_cycles
        cfg["pim_slots_per_request"] = pim_slots_per_request
        for energy_field in PIM_EVENT_ENERGY_FIELDS:
            cfg[energy_field] = self.pim_datatype_resource[energy_field]
        return cfg
