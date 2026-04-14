"""Fast latency-throughput: no-refresh checks and plots."""

import pytest

from tests.latency_throughput.testcases import STANDARDS
from tests.latency_throughput.utils.checks import (
    check_peak_bandwidth,
    check_pim_dependency_pattern_movement,
    check_pim_latency_throughput,
    check_streaming_peak_bandwidth,
    check_unloaded_latency,
    compare_at_same_nop,
    curves_to_nop_dict,
)
from tests.latency_throughput.utils.plot import (
    plot_lat_tp,
    plot_pim_dependency_pattern_same_nop,
    plot_pim_dependency_pattern_same_nop_plotly,
    plot_pim_lat_tp,
)
from tests.latency_throughput.utils.sweep import extract_curves, extract_pim_curves, run_sweep
from tests.latency_throughput.runner import run_streaming_only

# Sweep parameters
CI_READ_RATIOS = [100, 90, 80, 70, 60, 50]
CI_NUM_PROBES = 10000


# Cache sweep results per standard (expensive to compute)
_sweep_cache = {}
_streaming_cache = {}
_dependency_pattern_cache = {}


def _get_sweep(std_name):
    """Run sweep once per standard, cache for reuse."""
    if std_name not in _sweep_cache:
        nops = STANDARDS[std_name]["nop_counters"]
        raw = run_sweep(std_name, nops, CI_READ_RATIOS, CI_NUM_PROBES, full=False)
        if STANDARDS[std_name].get("pim_mode", False):
            curves = extract_pim_curves(raw, std_name)
        else:
            curves = extract_curves(raw, std_name)
        _sweep_cache[std_name] = curves
    return _sweep_cache[std_name]


def _get_streaming(std_name):
    """Run streaming-only once per standard, cache for reuse."""
    if STANDARDS[std_name].get("pim_mode", False):
        return None
    if std_name not in _streaming_cache:
        _streaming_cache[std_name] = run_streaming_only(std_name, full=False)
    return _streaming_cache[std_name]


def _get_dependency_pattern_results(std_name):
    cfg = STANDARDS[std_name]
    patterns = cfg.get("dependency_patterns")
    if not patterns:
        return None

    if std_name not in _dependency_pattern_cache:
        nops = cfg.get("dependency_pattern_nop_counters", cfg["nop_counters"])
        results = {}
        for pattern_name, cfg_override in sorted(patterns.items()):
            raw = run_sweep(
                std_name,
                nops,
                CI_READ_RATIOS,
                CI_NUM_PROBES,
                full=False,
                cfg_override=cfg_override,
            )
            curves = extract_pim_curves(raw, std_name)
            nop_dict = curves_to_nop_dict(curves)
            summary = check_pim_latency_throughput(curves)
            results[pattern_name] = {
                "curves": curves,
                "nop_dict": nop_dict,
                "summary": summary,
            }
        _dependency_pattern_cache[std_name] = results

    return _dependency_pattern_cache[std_name]


@pytest.mark.latency_throughput_fast
@pytest.mark.parametrize("standard", sorted(STANDARDS.keys()))
def test_latency_throughput_fast(request, standard):
    """Run no-refresh formula checks, print % deviations, generate lat-tp plot."""
    verbose = request.config.getoption("--verbose-plot")
    curves = _get_sweep(standard)
    pim_mode = STANDARDS[standard].get("pim_mode", False)

    if pim_mode:
        pim_result = check_pim_latency_throughput(curves)
        dependency_pattern_results = _get_dependency_pattern_results(standard)

        output_dir = "tests/latency_throughput/plots/fast"
        if verbose:
            output_dir = "tests/latency_throughput/plots/fast_verbose"

        print(f"\n{'=' * 60}")
        print(f"  {standard} Fast PIM Latency-Throughput Results")
        print(f"{'=' * 60}")
        print(f"  Measured PIMCompute latency = {pim_result['measured_latency_ns']:.1f} ns")
        print(
            f"  Measured PIMCompute throughput = {pim_result['measured_throughput']:.6f} requests/ns"
        )
        print(f"  PIM dependency stalls = {pim_result['pim_dependency_stalls']}")
        print(f"  PIM capacity stalls = {pim_result['pim_capacity_stalls']}")
        print(f"  PIM requests served = {pim_result['num_pim_reqs_served']}")

        if dependency_pattern_results:
            dep_data = dependency_pattern_results["same_bank_dependent"]
            ind_data = dependency_pattern_results["same_bank_independent"]
            dependent = dep_data["summary"]
            independent = ind_data["summary"]
            comparison = check_pim_dependency_pattern_movement(dependent, independent)

            dep_nop_dict = dep_data["nop_dict"]
            ind_nop_dict = ind_data["nop_dict"]
            common_nops = sorted(set(dep_nop_dict.keys()) & set(ind_nop_dict.keys()), reverse=True)

            print("  Same-bank dependency-pattern comparison (pim_blocks_per_bank=2):")
            print()
            print("  [Best-throughput points per pattern]")
            print(
                "    dependent: "
                f"avg_pim_latency={dependent['avg_pim_latency']:.1f} ns, "
                f"throughput={dependent['measured_throughput']:.6f} req/ns, "
                f"pim_dependency_stalls={dependent['pim_dependency_stalls']}, "
                f"pim_capacity_stalls={dependent['pim_capacity_stalls']}, "
                f"num_pim_reqs_served={dependent['num_pim_reqs_served']}, "
                f"nop={dependent['nop']}"
            )
            print(
                "    independent: "
                f"avg_pim_latency={independent['avg_pim_latency']:.1f} ns, "
                f"throughput={independent['measured_throughput']:.6f} req/ns, "
                f"pim_dependency_stalls={independent['pim_dependency_stalls']}, "
                f"pim_capacity_stalls={independent['pim_capacity_stalls']}, "
                f"num_pim_reqs_served={independent['num_pim_reqs_served']}, "
                f"nop={independent['nop']}"
            )
            print(
                "    delta (independent - dependent): "
                f"avg_pim_latency={comparison['latency_delta_ns']:+.3f} ns, "
                f"throughput={comparison['throughput_delta']:+.9f} req/ns, "
                f"pim_dependency_stalls={comparison['dependency_stall_delta']:+d}, "
                f"pim_capacity_stalls={comparison['capacity_stall_delta']:+d}, "
                f"num_pim_reqs_served={comparison['served_delta']:+d}"
            )
            print()
            print("  [Point-by-point at same NOP]")
            for nop in common_nops:
                dep_pt = dep_nop_dict[nop]
                ind_pt = ind_nop_dict[nop]
                pt_cmp = compare_at_same_nop(dep_pt, ind_pt)
                print(f"    nop={nop}:")
                print(
                    f"      dependent: "
                    f"avg_pim_latency={dep_pt['avg_pim_latency']:.1f} ns, "
                    f"throughput={dep_pt['measured_throughput']:.6f} req/ns, "
                    f"pim_dependency_stalls={dep_pt['pim_dependency_stalls']}, "
                    f"pim_capacity_stalls={dep_pt['pim_capacity_stalls']}, "
                    f"num_pim_reqs_served={dep_pt['num_pim_reqs_served']}"
                )
                print(
                    f"      independent: "
                    f"avg_pim_latency={ind_pt['avg_pim_latency']:.1f} ns, "
                    f"throughput={ind_pt['measured_throughput']:.6f} req/ns, "
                    f"pim_dependency_stalls={ind_pt['pim_dependency_stalls']}, "
                    f"pim_capacity_stalls={ind_pt['pim_capacity_stalls']}, "
                    f"num_pim_reqs_served={ind_pt['num_pim_reqs_served']}"
                )
                print(
                    f"      delta: "
                    f"avg_pim_latency={pt_cmp['latency_delta_ns']:+.3f} ns, "
                    f"throughput={pt_cmp['throughput_delta']:+.9f} req/ns, "
                    f"pim_dependency_stalls={pt_cmp['dependency_stall_delta']:+d}, "
                    f"pim_capacity_stalls={pt_cmp['capacity_stall_delta']:+d}, "
                    f"num_pim_reqs_served={pt_cmp['served_delta']:+d}"
                )
            plot_paths = plot_pim_dependency_pattern_same_nop(
                dependency_pattern_results, standard, output_dir=output_dir
            )
            print()
            print("  [Dependency-pattern same-NOP plots (PNG)]")
            print(f"    latency_vs_nop: {plot_paths['latency_vs_nop']}")
            print(f"    throughput_vs_nop: {plot_paths['throughput_vs_nop']}")
            print(f"    dependency_stalls_vs_nop: {plot_paths['dependency_stalls_vs_nop']}")

            if verbose:
                plotly_paths = plot_pim_dependency_pattern_same_nop_plotly(
                    dependency_pattern_results, standard, output_dir=output_dir
                )
                print()
                print("  [Dependency-pattern same-NOP plots (Plotly HTML)]")
                print(f"    latency_vs_nop: {plotly_paths['latency_vs_nop']}")
                print(f"    throughput_vs_nop: {plotly_paths['throughput_vs_nop']}")
                print(f"    dependency_stalls_vs_nop: {plotly_paths['dependency_stalls_vs_nop']}")
        print(f"{'=' * 60}")

        png_path = plot_pim_lat_tp(curves, standard, output_dir=output_dir)
        print(f"  Baseline PIM plot saved: {png_path}")
        return

    streaming_stats = _get_streaming(standard)

    lat_result = check_unloaded_latency(curves, standard)
    bw_result = check_peak_bandwidth(curves, standard)
    streaming_result = check_streaming_peak_bandwidth(standard, streaming_stats)

    print(f"\n{'=' * 60}")
    print(f"  {standard} Fast Latency-Throughput Results")
    print(f"{'=' * 60}")
    print("  Unloaded Latency:")
    print(
        f"    Expected (nRP+nRCD+nCL)*tCK = "
        f"({lat_result['nRP']}+{lat_result['nRCD']}+{lat_result['nCL']})"
        f" * {lat_result['tCK_ns']:.3f} ns = "
        f"{lat_result['expected_ns']:.1f} ns"
    )
    print(f"    Measured = {lat_result['measured_ns']:.1f} ns")
    print(f"    Deviation = {lat_result['deviation_pct']:+.1f}%")
    print()
    print("  Max Bandwidth (probed sweep):")
    print(f"    Theoretical peak = {bw_result['max_theoretical_bw']:.1f} GB/s")
    print(f"    Measured max = {bw_result['measured_max_bw']:.1f} GB/s")
    print(f"    Deviation = {bw_result['deviation_from_theoretical_pct']:+.1f}%")
    print()
    print("  Streaming-Only Bandwidth (no probes):")
    print(f"    Theoretical peak = {streaming_result['max_theoretical_bw']:.1f} GB/s")
    print(f"    Measured = {streaming_result['measured_streaming_bw']:.1f} GB/s")
    print(f"    Deviation = {streaming_result['deviation_from_theoretical_pct']:+.1f}%")
    print(f"{'=' * 60}")

    output_dir = "tests/latency_throughput/plots/fast"
    if verbose:
        output_dir = "tests/latency_throughput/plots/fast_verbose"

    png_path = plot_lat_tp(
        curves,
        standard,
        {"latency": lat_result, "bandwidth": bw_result, "streaming": streaming_result},
        output_dir=output_dir,
        verbose=verbose,
    )
    print(f"  Plot saved: {png_path}")
