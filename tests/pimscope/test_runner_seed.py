import pytest
from ramulator.pimscope.runner import run_single


@pytest.mark.parametrize("seed", [True, 1.5, "7", -1])
def test_runner_rejects_invalid_seed_types(seed):
    with pytest.raises(ValueError, match="seed must be a non-negative integer"):
        run_single(seed=seed, observability="disabled", num_probes=1, warmup=1)


def test_runner_seed_is_deterministic():
    kwargs = {
        "observability": "preview",
        "num_probes": 8,
        "warmup": 4,
        "nop": 1,
    }
    first = run_single(seed=7, **kwargs)
    second = run_single(seed=7, **kwargs)

    assert first["evidence"]["seed"] == 7
    assert second["evidence"]["seed"] == 7
    assert (
        first["evidence"]["pim_energy_observability"]["modeled"]["command_counts"]
        == second["evidence"]["pim_energy_observability"]["modeled"]["command_counts"]
    )
