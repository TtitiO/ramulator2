from ramulator.dram.lpddr5 import LPDDR5
from ramulator.dram.spec import TimingConstraint


ResourceValue = int | float


PIM_EVENT_ENERGY_FIELDS = [
    "pim_compute_energy_pJ_per_mac",
    "pim_array_local_energy_pJ",
    "pim_cell_to_pim_energy_pJ_per_256b",
    "pim_interconnect_energy_pJ_per_256b",
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
    "shared_mpu_serial",
    "subbank_overlap_experimental",
}

for resource in PIM_DATATYPE_METADATA.values():
    for energy_field in PIM_EVENT_ENERGY_FIELDS:
        resource[energy_field] = 0.0


class LPDDR5PIM(LPDDR5):
    name = "LPDDR5PIM"

    # Keep inherited LPDDR5 power terms as-is for standard memory energy,
    # and add PIM-only incremental command energy as a separate category.
    power_incremental_commands_counted = [
        "PIM_MAC",
        "PIM_MAC_AB",
        "PIM_BCAST",
        "HAB",
        "HAB_PIM",
        "SB",
    ]
    power_incremental_command_hooks = [
        ("Rank", "PIM_MAC", "COUNT_PIM_INCREMENTAL_ENERGY"),
        ("Rank", "PIM_MAC_AB", "COUNT_PIM_INCREMENTAL_ENERGY"),
        ("Rank", "PIM_BCAST", "COUNT_PIM_INCREMENTAL_ENERGY"),
        ("Rank", "HAB", "COUNT_PIM_INCREMENTAL_ENERGY"),
        ("Rank", "HAB_PIM", "COUNT_PIM_INCREMENTAL_ENERGY"),
        ("Rank", "SB", "COUNT_PIM_INCREMENTAL_ENERGY"),
    ]
    power_incremental_command_energy_timings = {
        "PIM_MAC": "nPIM_MAC_LAT",
        "PIM_MAC_AB": "nPIM_MAC_LAT",
        "PIM_BCAST": "nBL",
        "HAB": "nBL",
        "HAB_PIM": "nBL",
        "SB": "nBL",
    }
    # PIM_MAC/PIM_MAC_AB incremental energy is explicit and parameterized.
    # Do not proxy PIM compute with JEDEC host-read current (IDD4R - IDD3N):
    # active-bank background is already charged by the inherited LPDDR5 IDD3N
    # background model, while PIM-specific array/local-transfer/compute/RF
    # costs must come from user- or literature-supplied coefficients.
    _pim_mac_event_energy_expr = (
        "pim_array_local_energy_pJ + "
        "pim_lanes * pim_compute_energy_pJ_per_mac + "
        "pim_cell_to_pim_energy_pJ_per_256b + "
        "pim_interconnect_energy_pJ_per_256b + "
        "pim_vrf_access_energy_pJ + "
        "pim_srf_access_energy_pJ"
    )
    # Bounded attribution only: PIM_BCAST models all-bank setup/broadcast
    # pressure. Public Samsung-style PIM sources describe this as HAB/all-bank
    # WR-like broadcast behavior; these event terms are not silicon-calibrated.
    power_incremental_command_event_energy_exprs = {
        "PIM_MAC": _pim_mac_event_energy_expr,
        "PIM_MAC_AB": _pim_mac_event_energy_expr,
        "PIM_BCAST": "pim_cell_to_pim_energy_pJ_per_256b + pim_interconnect_energy_pJ_per_256b",
        "HAB": "pim_mode_switch_energy_pJ",
        "HAB_PIM": "pim_mode_switch_energy_pJ",
        "SB": "pim_mode_switch_energy_pJ",
    }
    power_incremental_command_energy_terms = {
        # PIM_MAC banks remain active, so inherited LPDDR5 background accounts
        # for IDD3N. The incremental PIM_MAC layer is event-coefficient only.
        "PIM_MAC": [],
        "PIM_MAC_AB": [],
        "PIM_BCAST": [
            ("VDD1", "IDD01", "IDD2N1"),
            ("VDD2H", "IDD02H", "IDD2N2H"),
            ("VDD2L", "IDD02L", "IDD2N2L"),
            ("VDDQ", "IDD0Q", "IDD2NQ"),
        ],
        "HAB": [
            ("VDD1", "IDD01", "IDD2N1"),
            ("VDD2H", "IDD02H", "IDD2N2H"),
            ("VDD2L", "IDD02L", "IDD2N2L"),
            ("VDDQ", "IDD0Q", "IDD2NQ"),
        ],
        "HAB_PIM": [
            ("VDD1", "IDD01", "IDD2N1"),
            ("VDD2H", "IDD02H", "IDD2N2H"),
            ("VDD2L", "IDD02L", "IDD2N2L"),
            ("VDDQ", "IDD0Q", "IDD2NQ"),
        ],
        "SB": [
            ("VDD1", "IDD01", "IDD2N1"),
            ("VDD2H", "IDD02H", "IDD2N2H"),
            ("VDD2L", "IDD02L", "IDD2N2L"),
            ("VDDQ", "IDD0Q", "IDD2NQ"),
        ],
    }

    levels = {
        **LPDDR5.levels,
        "Rank": "PIM_SB",
    }

    states = LPDDR5.states + ["PIM_SB", "PIM_HAB", "PIM_HAB_PIM"]

    commands = LPDDR5.commands + [
        "SB",
        "HAB",
        "HAB_PIM",
        "PIM_BCAST",
        "PIM_MAC",
        "PIM_MAC_AB",
    ]

    timing_params = LPDDR5.timing_params + ["nPIM_MAC_LAT", "nPIM_MAC_II"]

    supported_requests = {
        **LPDDR5.supported_requests,
        "PIMCompute": "PIM_MAC",
        "PIMLoadAll": "PIM_BCAST",
        "PIMComputeAll": "PIM_MAC_AB",
    }

    timing_constraints = LPDDR5.timing_constraints + [
        TimingConstraint(level="Bank", preceding=["ACT1"], following=["PIM_MAC"], latency="nRCD"),
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
        # LPDDR5-PIM broadcast/source timing is not public silicon ground truth.
        TimingConstraint(
            level="Rank",
            preceding=["PIM_BCAST"],
            following=["PIM_BCAST"],
            latency="nBL",
        ),
    ]

    org_presets = LPDDR5.org_presets

    timing_presets = {
        preset_name: {
            **preset,
            "nPIM_MAC_LAT": 8,
            "nPIM_MAC_II": 8,
        }
        for preset_name, preset in LPDDR5.timing_presets.items()
    }

    def __init__(
        self,
        *,
        org_preset,
        timing_preset,
        power=None,
        pim_enabled=False,
        pim_mode="bank",
        pim_blocks_per_bank=1,
        pim_banks_per_mpu=2,
        pim_mac_execution_model="shared_mpu_serial",
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
        pim_slot_cost=None,
        pim_compute_energy_pJ_per_mac=None,
        pim_array_local_energy_pJ=None,
        pim_cell_to_pim_energy_pJ_per_256b=None,
        pim_interconnect_energy_pJ_per_256b=None,
        pim_vrf_access_energy_pJ=None,
        pim_srf_access_energy_pJ=None,
        pim_mode_switch_energy_pJ=None,
        pim_mac_latency_scale=None,
        pim_incremental_energy_scale=None,
        **overrides,
    ):
        if not pim_datatype_class:
            pim_datatype_class = pim_datatype
        pim_datatype_class = str(pim_datatype_class).lower()
        if pim_mac_latency_scale is not None:
            raise ValueError("LPDDR5PIM pim_mac_latency_scale is deprecated; use explicit pipeline/II cycles")
        if pim_incremental_energy_scale is not None:
            raise ValueError("LPDDR5PIM pim_incremental_energy_scale is deprecated; use explicit event-energy terms")

        pim_mac_execution_model = str(pim_mac_execution_model)
        if pim_mac_execution_model not in PIM_MAC_EXECUTION_MODELS:
            supported = ", ".join(sorted(PIM_MAC_EXECUTION_MODELS))
            raise ValueError(
                f"LPDDR5PIM unknown pim_mac_execution_model '{pim_mac_execution_model}'; "
                f"supported values: {supported}"
            )

        resource = dict(PIM_DATATYPE_METADATA.get(pim_datatype_class, PIM_DATATYPE_METADATA["int8"]))
        if pim_datatype_behavior_enabled and pim_datatype_class not in PIM_DATATYPE_RESOURCES:
            supported = ", ".join(sorted(PIM_DATATYPE_RESOURCES))
            raise ValueError(
                f"LPDDR5PIM source-backed datatype resources for '{pim_datatype_class}' are unsupported; "
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
        if pim_slot_cost is not None and pim_slots_per_request is None:
            resource["pim_slots_per_request"] = int(pim_slot_cost)

        energy_overrides = {
            "pim_compute_energy_pJ_per_mac": pim_compute_energy_pJ_per_mac,
            "pim_array_local_energy_pJ": pim_array_local_energy_pJ,
            "pim_cell_to_pim_energy_pJ_per_256b": pim_cell_to_pim_energy_pJ_per_256b,
            "pim_interconnect_energy_pJ_per_256b": pim_interconnect_energy_pJ_per_256b,
            "pim_vrf_access_energy_pJ": pim_vrf_access_energy_pJ,
            "pim_srf_access_energy_pJ": pim_srf_access_energy_pJ,
            "pim_mode_switch_energy_pJ": pim_mode_switch_energy_pJ,
        }
        for energy_field, energy_value in energy_overrides.items():
            if energy_value is not None:
                resource[energy_field] = float(energy_value)

        if pim_lanes is None:
            resource["pim_lanes"] = int(resource["pim_simd_width_bits"] // resource["pim_datatype_bits"])
        if pim_ops_per_block_issue is None and pim_ops_per_request is None:
            resource["pim_ops_per_block_issue"] = (
                float(resource["pim_lanes"]) * float(resource["pim_ops_per_mac"])
            )
        resource["pim_ops_per_request"] = float(resource["pim_ops_per_block_issue"])

        if resource["pim_datatype_bits"] <= 0:
            raise ValueError("LPDDR5PIM pim_datatype_bits must be positive")
        if resource["pim_simd_width_bits"] <= 0:
            raise ValueError("LPDDR5PIM pim_simd_width_bits must be positive")
        if resource["pim_lanes"] <= 0:
            raise ValueError("LPDDR5PIM pim_lanes must be positive")
        if resource["pim_ops_per_mac"] <= 0:
            raise ValueError("LPDDR5PIM pim_ops_per_mac must be positive")
        if resource["pim_ops_per_block_issue"] <= 0:
            raise ValueError("LPDDR5PIM pim_ops_per_block_issue must be positive")
        if resource["pim_ops_per_request"] <= 0:
            raise ValueError("LPDDR5PIM pim_ops_per_request must be positive")
        if resource["pim_mac_issue_interval_cycles"] <= 0:
            raise ValueError("LPDDR5PIM pim_mac_issue_interval_cycles must be positive")
        if resource["pim_mac_pipeline_latency_cycles"] <= 0:
            raise ValueError("LPDDR5PIM pim_mac_pipeline_latency_cycles must be positive")
        if resource["pim_movement_cycles"] < 0:
            raise ValueError("LPDDR5PIM pim_movement_cycles must be non-negative")
        if resource["pim_writeback_cycles"] < 0:
            raise ValueError("LPDDR5PIM pim_writeback_cycles must be non-negative")
        if resource["pim_slots_per_request"] <= 0:
            raise ValueError("LPDDR5PIM pim_slots_per_request must be positive")
        if pim_banks_per_mpu <= 0:
            raise ValueError("LPDDR5PIM pim_banks_per_mpu must be positive")
        for energy_field in PIM_EVENT_ENERGY_FIELDS:
            if resource[energy_field] < 0:
                raise ValueError(f"LPDDR5PIM {energy_field} must be non-negative")

        self.pim_enabled = pim_enabled
        self.pim_mode = pim_mode
        self.pim_blocks_per_bank = pim_blocks_per_bank
        self.pim_banks_per_mpu = pim_banks_per_mpu
        self.pim_mac_execution_model = pim_mac_execution_model
        self.pim_datatype = pim_datatype
        self.pim_datatype_class = pim_datatype_class
        self.pim_datatype_behavior_enabled = pim_datatype_behavior_enabled
        self.pim_datatype_resource = resource
        super().__init__(org_preset=org_preset, timing_preset=timing_preset, power=power, **overrides)

    def _resolve_pim_resource_cycles(self, base_pim_mac_latency, base_pim_mac_issue_interval):
        if self.pim_datatype_behavior_enabled:
            return (
                int(self.pim_datatype_resource["pim_mac_pipeline_latency_cycles"]),
                int(self.pim_datatype_resource["pim_mac_issue_interval_cycles"]),
            )
        return base_pim_mac_latency, base_pim_mac_issue_interval

    def resolve(self):
        org_dict, timing_dict = super().resolve()
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
        cfg["pim_banks_per_mpu"] = self.pim_banks_per_mpu
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
        cfg["pim_slot_cost"] = pim_slots_per_request
        for energy_field in PIM_EVENT_ENERGY_FIELDS:
            cfg[energy_field] = self.pim_datatype_resource[energy_field]
        return cfg
