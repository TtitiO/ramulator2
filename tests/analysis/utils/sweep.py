"""Analysis sweep and curve extraction utilities.

Provides functions for parallel sweep execution and result extraction
specific to the latency-throughput validation workflow.
"""

import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from tests.analysis.runner import run_single
from tests.analysis.utils.energy import build_pim_energy_report


def run_sweep(
    std_name,
    nop_counters,
    read_ratios,
    num_probes=5000,
    warmup=10000,
    max_workers=4,
    full=False,
    cfg_override=None,
):
    """Run a parallel NOP x read_ratio sweep.

    Returns: {(nop, read_ratio): stats_dict}
    """
    jobs = [(nop, rr) for rr in read_ratios for nop in nop_counters]
    total = len(jobs)

    t_start = time.time()
    raw_results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                run_single,
                std_name,
                nop,
                read_ratio=rr,
                num_probes=num_probes,
                warmup=warmup,
                full=full,
                cfg_override=cfg_override,
            ): (nop, rr)
            for nop, rr in jobs
        }
        done = 0
        for fut in as_completed(futures):
            nop, rr = futures[fut]
            raw_results[(nop, rr)] = fut.result()
            done += 1
            if done % 10 == 0 or done == total:
                print(f"  [{std_name}] {done}/{total} completed")

    elapsed = time.time() - t_start
    print(f"  [{std_name}] All {total} runs done in {elapsed:.1f}s")
    return raw_results


def extract_curves(raw_results, std_name):
    """Convert raw sweep results into per-read-ratio lat-tp curves.

    Returns: {
        read_ratio: {
            "bw": [float, ...],   # GB/s, sorted low-BW first
            "lat": [float, ...],  # ns
            "nops": [int, ...],   # corresponding NOP values
        }
    }
    """
    from tests.analysis.utils.spec import resolve_spec

    spec = resolve_spec(std_name)
    time_unit_ns = spec["time_unit_ns"]
    bytes_per_req = spec["bytes_per_req"]

    # Group by read ratio
    by_rr = defaultdict(list)
    for (nop, rr), stats in raw_results.items():
        by_rr[rr].append((nop, stats))

    curves = {}
    for rr, entries in sorted(by_rr.items()):
        # Sort by NOP descending (low BW first)
        entries.sort(key=lambda x: -x[0])
        bw_list, lat_list, nop_list = [], [], []
        for nop, stats in entries:
            ms = stats["memory_system"]
            fe = stats["frontend"]
            cycles = ms["controller"]["cycles"]
            streaming = fe["streaming_requests_sent"]
            lat_cycles = fe["avg_probe_latency"]

            bw = (streaming * bytes_per_req) / (cycles * time_unit_ns)
            lat = lat_cycles * time_unit_ns

            bw_list.append(bw)
            lat_list.append(lat)
            nop_list.append(nop)

        curves[rr] = {"bw": bw_list, "lat": lat_list, "nops": nop_list}

    return curves


def extract_pim_curves(raw_results, std_name):
    from tests.analysis.utils.spec import resolve_spec

    spec = resolve_spec(std_name)
    time_unit_ns = spec["time_unit_ns"]

    by_rr = defaultdict(list)
    for (nop, rr), stats in raw_results.items():
        by_rr[rr].append((nop, stats))

    curves = {}
    for rr, entries in sorted(by_rr.items()):
        entries.sort(key=lambda x: -x[0])
        latency_list = []
        service_latency_list = []
        launch_wait_list = []
        response_latency_list = []
        throughput_list = []
        all_bank_throughput_list = []
        nop_list = []
        capacity_stalls_list = []
        mpu_group_stalls_list = []
        dependency_stalls_list = []
        global_issue_stalls_per_request_list = []
        mpu_group_busy_stalls_per_request_list = []
        bank_timing_stalls_per_request_list = []
        rr_head_of_line_stalls_per_request_list = []
        queue_empty_cycles_per_request_list = []
        effective_mpu_groups_list = []
        total_banks_list = []
        total_energy_list = []
        full_system_edp_list = []
        total_simulation_time_list = []
        request_throughput_list = []
        inflight_peak_list = []
        served_list = []
        served_ab_list = []
        ab_inflight_peak_list = []
        mode_stalls_list = []
        load_stalls_list = []
        energy_proxy_list = []
        energy_proxy_per_request_list = []
        simulated_gops_list = []
        simulated_display_value_list = []
        simulated_display_unit_list = []
        simulated_display_label_list = []
        datatype_list = []
        datatype_class_list = []
        datatype_kind_list = []
        elements_per_request_list = []
        ops_per_request_list = []
        command_count_list = []
        attribution_fields = (
            "energy_total_pJ",
            "energy_background_pJ",
            "energy_lpddr_base_cmd_pJ",
            "energy_incremental_pim_builtin_pJ",
            "energy_pim_setup_cmd_pJ",
            "energy_pim_mac_cmd_pJ",
            "energy_pim_movement_pJ",
            "energy_pim_writeback_pJ",
            "energy_mode_switch_pJ",
            "energy_fallback_or_unattributed_pJ",
            "energy_attribution_sum_pJ",
            "average_power_mW",
            "runtime_full_system_edp_pJ_ns",
            "legacy_full_system_edp_pJ_ns",
            "latency_full_system_edp_pJ_ns",
            "throughput_edp_pJ_ns",
            "count_ACT",
            "count_PRE",
            "count_RD",
            "count_WR",
            "count_REF",
            "count_PIM_MAC",
            "count_PIM_MAC_AB",
            "count_PIM_setup_commands",
            "count_completed_pim_requests",
            "count_frontend_pim_requests",
        )
        attribution_lists = {field: [] for field in attribution_fields}
        for nop, stats in entries:
            ctrl = stats["memory_system"]["controller"]
            cycles = ctrl["cycles"]
            completed = ctrl["num_pim_reqs_served"]
            all_bank_completed = ctrl.get("num_pim_ab_reqs_served", 0)
            avg_latency = ctrl["avg_pim_latency"] * time_unit_ns
            avg_service_latency = ctrl.get("avg_pim_service_latency", 0) * time_unit_ns
            avg_launch_wait = ctrl.get("avg_pim_launch_wait", 0) * time_unit_ns
            avg_response_latency = ctrl.get("avg_pim_response_latency", ctrl["avg_pim_latency"]) * time_unit_ns
            throughput = completed / (cycles * time_unit_ns)
            all_bank_throughput = all_bank_completed / (cycles * time_unit_ns)
            energy_report = build_pim_energy_report(stats, std_name)

            latency_list.append(avg_latency)
            service_latency_list.append(avg_service_latency)
            launch_wait_list.append(avg_launch_wait)
            response_latency_list.append(avg_response_latency)
            throughput_list.append(throughput)
            all_bank_throughput_list.append(all_bank_throughput)
            capacity_stalls_list.append(ctrl["pim_capacity_stalls"])
            mpu_group_stalls_list.append(ctrl.get("pim_mpu_group_stalls", 0))
            dependency_stalls_list.append(ctrl.get("pim_dependency_stalls", 0))
            global_issue_stalls_per_request_list.append(energy_report["derived"]["global_issue_stall_per_request"])
            mpu_group_busy_stalls_per_request_list.append(energy_report["derived"]["mpu_group_busy_stall_per_request"])
            bank_timing_stalls_per_request_list.append(energy_report["derived"]["bank_timing_stall_per_request"])
            rr_head_of_line_stalls_per_request_list.append(energy_report["derived"]["rr_head_of_line_stall_per_request"])
            queue_empty_cycles_per_request_list.append(energy_report["derived"]["queue_empty_cycles_per_request"])
            effective_mpu_groups_list.append(energy_report["derived"]["effective_mpu_groups"])
            total_banks_list.append(energy_report["derived"]["total_banks"])
            total_energy_list.append(energy_report["derived"]["total_energy_pJ"])
            full_system_edp_list.append(energy_report["derived"]["full_system_edp_pJ_ns"])
            total_simulation_time_list.append(energy_report["derived"]["total_simulation_time_ns"])
            request_throughput_list.append(energy_report["derived"]["request_throughput"])
            inflight_peak_list.append(ctrl.get("pim_inflight_peak", 0))
            served_list.append(completed)
            served_ab_list.append(ctrl.get("num_pim_ab_reqs_served", 0))
            ab_inflight_peak_list.append(ctrl.get("pim_ab_inflight_peak", 0))
            mode_stalls_list.append(ctrl.get("pim_mode_stalls", 0))
            load_stalls_list.append(ctrl.get("pim_load_stalls", 0))
            energy_proxy_list.append(energy_report["derived"]["analytical_energy_estimate_pJ"])
            energy_proxy_per_request_list.append(
                energy_report["derived"]["analytical_energy_per_pim_request_pJ"]
            )
            simulated_gops_list.append(energy_report["derived"]["simulated_pim_gops"])
            simulated_display_value_list.append(energy_report["derived"]["simulated_pim_display_value"])
            simulated_display_unit_list.append(energy_report["derived"]["simulated_pim_display_unit"])
            simulated_display_label_list.append(energy_report["derived"]["simulated_pim_display_label"])
            datatype_list.append(energy_report["derived"]["pim_datatype"])
            datatype_class_list.append(energy_report["derived"]["pim_datatype_class"])
            datatype_kind_list.append(energy_report["derived"]["pim_datatype_kind"])
            elements_per_request_list.append(energy_report["derived"]["pim_elements_per_request"])
            ops_per_request_list.append(energy_report["derived"]["pim_ops_per_request"])
            command_count_list.append(energy_report["derived"]["total_command_count"])
            for field in attribution_fields:
                attribution_lists[field].append(energy_report["derived"][field])
            nop_list.append(nop)

        curves[rr] = {
            "pim_lat": latency_list,
            "pim_throughput": throughput_list,
            "pim_all_bank_throughput": all_bank_throughput_list,
            "avg_pim_latency_ns": latency_list,
            "avg_pim_service_latency_ns": service_latency_list,
            "avg_pim_launch_wait_ns": launch_wait_list,
            "avg_pim_response_latency_ns": response_latency_list,
            "num_pim_reqs_served": served_list,
            "num_pim_ab_reqs_served": served_ab_list,
            "pim_capacity_stalls": capacity_stalls_list,
            "pim_mpu_group_stalls": mpu_group_stalls_list,
            "pim_dependency_stalls": dependency_stalls_list,
            "global_issue_stall_per_request": global_issue_stalls_per_request_list,
            "mpu_group_busy_stall_per_request": mpu_group_busy_stalls_per_request_list,
            "bank_timing_stall_per_request": bank_timing_stalls_per_request_list,
            "rr_head_of_line_stall_per_request": rr_head_of_line_stalls_per_request_list,
            "queue_empty_cycles_per_request": queue_empty_cycles_per_request_list,
            "effective_mpu_groups": effective_mpu_groups_list,
            "total_banks": total_banks_list,
            "total_energy_pJ": total_energy_list,
            "full_system_edp_pJ_ns": full_system_edp_list,
            "total_simulation_time_ns": total_simulation_time_list,
            "request_throughput": request_throughput_list,
            "pim_inflight_peak": inflight_peak_list,
            "pim_ab_inflight_peak": ab_inflight_peak_list,
            "pim_mode_stalls": mode_stalls_list,
            "pim_load_stalls": load_stalls_list,
            "analytical_energy_estimate_pJ": energy_proxy_list,
            "analytical_energy_per_pim_request_pJ": energy_proxy_per_request_list,
            "simulated_pim_gops": simulated_gops_list,
            "simulated_pim_display_value": simulated_display_value_list,
            "simulated_pim_display_unit": simulated_display_unit_list,
            "simulated_pim_display_label": simulated_display_label_list,
            "pim_datatype": datatype_list,
            "pim_datatype_class": datatype_class_list,
            "pim_datatype_kind": datatype_kind_list,
            "pim_elements_per_request": elements_per_request_list,
            "pim_ops_per_request": ops_per_request_list,
            "total_command_count": command_count_list,
            **attribution_lists,
            "nops": nop_list,
        }

    return curves
