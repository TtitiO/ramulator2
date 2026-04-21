from ramulator.dram.lpddr5 import LPDDR5
from ramulator.dram.spec import TimingConstraint


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
    power_incremental_command_energy_terms = {
        "PIM_MAC": [
            ("VDD1", "IDD4R1", "IDD3N1"),
            ("VDD2H", "IDD4R2H", "IDD3N2H"),
            ("VDD2L", "IDD4R2L", "IDD3N2L"),
            ("VDDQ", "IDD4RQ", "IDD3NQ"),
        ],
        "PIM_MAC_AB": [
            ("VDD1", "IDD4R1", "IDD3N1"),
            ("VDD2H", "IDD4R2H", "IDD3N2H"),
            ("VDD2L", "IDD4R2L", "IDD3N2L"),
            ("VDDQ", "IDD4RQ", "IDD3NQ"),
        ],
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

    datatype_capability_matrix = {
        "int8": {
            "kind": "integer",
            "status": "modeled",
            "scope": "config-only framing",
            "behavior": "no datatype-specific timing or slot behavior change",
        },
        "int16": {
            "kind": "integer",
            "status": "derived",
            "scope": "assumption-only contrast",
            "behavior": "no datatype-specific timing or slot behavior change",
        },
        "fp16": {
            "kind": "floating-point",
            "status": "assumed",
            "scope": "config-only framing",
            "behavior": "no datatype-specific timing or slot behavior change",
        },
    }

    commands = LPDDR5.commands + [
        "SB",
        "HAB",
        "HAB_PIM",
        "PIM_BCAST",
        "PIM_MAC",
        "PIM_MAC_AB",
    ]

    timing_params = LPDDR5.timing_params + ["nPIM_MAC_LAT"]

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
            latency="nPIM_MAC_LAT",
        ),
        TimingConstraint(
            level="Rank",
            preceding=["PIM_MAC_AB"],
            following=["PIM_MAC_AB"],
            latency="nPIM_MAC_LAT",
        ),
    ]

    org_presets = LPDDR5.org_presets

    timing_presets = {
        preset_name: {
            **preset,
            "nPIM_MAC_LAT": 8,
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
        pim_datatype="int8",
        pim_datatype_class=None,
        **overrides,
    ):
        if pim_datatype_class is None:
            pim_datatype_class = pim_datatype
        self.pim_enabled = pim_enabled
        self.pim_mode = pim_mode
        self.pim_blocks_per_bank = pim_blocks_per_bank
        self.pim_datatype = pim_datatype
        self.pim_datatype_class = pim_datatype_class
        self.pim_datatype_assumption = self.datatype_capability_matrix.get(
            pim_datatype_class,
            {
                "kind": "unknown",
                "status": "assumed",
                "scope": "assumption-only contrast",
                "behavior": "no datatype-specific timing or slot behavior change",
            },
        )
        super().__init__(org_preset=org_preset, timing_preset=timing_preset, power=power, **overrides)

    def to_config(self):
        cfg = super().to_config()
        cfg["pim_blocks_per_bank"] = self.pim_blocks_per_bank
        cfg["pim_datatype"] = self.pim_datatype
        cfg["pim_datatype_class"] = self.pim_datatype_class
        cfg["pim_datatype_assumption"] = self.pim_datatype_assumption
        return cfg
