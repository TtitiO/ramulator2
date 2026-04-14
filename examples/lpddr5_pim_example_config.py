import ramulator

frontend = ramulator.frontend.SimpleO3(
    clock_ratio=8,
    traces=["./examples/traces/example_inst.trace"],
    num_expected_insts=500000,
    llc_linesize=32,
    translation=ramulator.translation.NoTranslation(max_addr=2147483648),
)

lpddr5_pim = ramulator.dram.LPDDR5PIM(
    org_preset="LPDDR5_8Gb_x16",
    timing_preset="LPDDR5_6400",
    pim_enabled=True,
    pim_mode="bank",
    pim_blocks_per_bank=1,
    pim_datatype="int8",
)
ctrl = ramulator.controller.LPDDR5PIM(
    dram=lpddr5_pim,
    scheduler=ramulator.scheduler.FRFCFS(),
    refresh_manager=ramulator.refresh_manager.AllBank(scope="Rank"),
    row_policy=ramulator.row_policy.Open(),
    addr_mapper=ramulator.addr_mapper.RoBaRaCoCh(),
)
mem = ramulator.memory_system.GenericDRAM(
    clock_ratio=3,
    controllers=[ctrl],
    channel_mapper=ramulator.channel_mapper.CacheLineInterleave(),
)

sim = ramulator.Simulation(frontend, mem)
sim.run()

stats = sim.stats

if stats:
    ctrl_stats = stats["memory_system"]["controller"]

    print(f"Controller cycles:     {ctrl_stats['cycles']}")
    print(f"Avg read latency:      {ctrl_stats['avg_read_latency']:.1f} cycles")
    print(f"Read requests:         {ctrl_stats['num_read_reqs']}")
    print(f"Write requests:        {ctrl_stats['num_write_reqs']}")
    print(f"Row hits:              {ctrl_stats['row_hits']}")
    print(f"Row misses:            {ctrl_stats['row_misses']}")
    print(f"Row conflicts:         {ctrl_stats['row_conflicts']}")
