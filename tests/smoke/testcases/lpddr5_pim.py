import ramulator

CONFIG = dict(
    dram_class="LPDDR5PIM",
    org_preset="LPDDR5_8Gb_x16",
    timing_preset="LPDDR5_6400",
    dram_kwargs=dict(
        pim_enabled=True,
        pim_mode="bank",
        pim_blocks_per_bank=1,
        pim_datatype="int8",
    ),
    controller_class="LPDDR5PIM",
    fast_ctrl_extra_kwargs=dict(
        refresh_manager=ramulator.refresh_manager.NoRefresh(),
    ),
    frontend_clock_ratio=4,
    stream_cols=8,
)
