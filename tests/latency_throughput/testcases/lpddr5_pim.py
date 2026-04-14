import importlib

ramulator = importlib.import_module("ramulator")

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
    full_ctrl_extra_kwargs=dict(
        refresh_manager=ramulator.refresh_manager.AllBank(scope="Rank"),
    ),
    full_streaming_requests=1_000_000,
    frontend_clock_ratio=4,
    stream_cols=8,
    pim_mode=True,
    num_pim_requests=512,
    pim_same_bank=True,
    pim_bank_group_size=4,
    pim_burst_length=16,
    nop_counters=[
        1,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        20,
        30,
        50,
        100,
        1000,
        2000,
        5000,
        10000,
    ],
)
