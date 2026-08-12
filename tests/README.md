# Ramulator test tiers

The default suite excludes only the long refresh-enabled latency/throughput
sweeps. It still runs smoke, PIMScope API, trace conformance, device timing,
controller scheduling, unit, configuration, and fast latency/throughput tests.

```bash
python -m pytest -q tests
```

Run the long performance-characterization tier explicitly:

```bash
python -m pytest -q tests --run-full
# or only that tier
python -m pytest -q tests -m latency_throughput_full --run-full
```

Controller scheduling and device timing tests require the optional native test
binding. Configure the build with `-DRAMULATOR_TEST_BINDINGS=ON` before running
the default suite. Performance-characterization tests are not part of the
PIMScope release smoke test because they sweep unrelated DRAM standards and can
take substantially longer than correctness tests.
