from ramulator.dram.lpddr5 import LPDDR5
from ramulator.dram.spec import TimingConstraint


class LPDDR5PIM(LPDDR5):
    name = "LPDDR5PIM"

    commands = LPDDR5.commands + ["PIM_MAC"]

    timing_params = LPDDR5.timing_params + ["nPIM_MAC_LAT"]

    supported_requests = {
        **LPDDR5.supported_requests,
        "PIMCompute": "PIM_MAC",
    }

    timing_constraints = LPDDR5.timing_constraints + [
        TimingConstraint(level="Bank", preceding=["ACT1"], following=["PIM_MAC"], latency="nRCD"),
        TimingConstraint(
            level="Bank",
            preceding=["PIM_MAC"],
            following=["PIM_MAC"],
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
        pim_enabled=False,
        pim_mode="bank",
        pim_blocks_per_bank=1,
        pim_datatype="int8",
        **overrides,
    ):
        self.pim_enabled = pim_enabled
        self.pim_mode = pim_mode
        self.pim_blocks_per_bank = pim_blocks_per_bank
        self.pim_datatype = pim_datatype
        super().__init__(org_preset=org_preset, timing_preset=timing_preset, **overrides)

    def to_config(self):
        cfg = super().to_config()
        cfg["pim_blocks_per_bank"] = self.pim_blocks_per_bank
        return cfg
