#include "ramulator/dram/dram_spec.h"

namespace Ramulator {

void DRAMSpec::load_config(const ConfigNode& config) {
  const ConfigNode dram = config["dram"];

  // Optional PIM capacity knob (defaults to 1 for non-PIM configs)
  pim_blocks_per_bank = dram["pim_blocks_per_bank"].as<int>(1);
  pim_banks_per_mpu = dram["pim_banks_per_mpu"].as<int>(2);
  pim_mac_execution_model = dram["pim_mac_execution_model"].as<std::string>("shared_mpu_serial");
  if (pim_mac_execution_model != "shared_mpu_serial" &&
      pim_mac_execution_model != "subbank_overlap_experimental") {
    throw std::runtime_error(
        "DRAMSpec: unknown pim_mac_execution_model '" + pim_mac_execution_model +
        "'; supported values: shared_mpu_serial, subbank_overlap_experimental");
  }
  pim_datatype = dram["pim_datatype"].as<std::string>("int8");
  pim_datatype_class = dram["pim_datatype_class"].as<std::string>(pim_datatype);
  if (pim_datatype_class.empty()) {
    pim_datatype_class = pim_datatype;
  }
  pim_datatype_behavior_enabled = dram["pim_datatype_behavior_enabled"].as<bool>(false);
  pim_datatype_bits = dram["pim_datatype_bits"].as<int>(8);
  pim_simd_width_bits = dram["pim_simd_width_bits"].as<int>(256);
  pim_lanes = dram["pim_lanes"].as<int>(32);
  pim_ops_per_mac = dram["pim_ops_per_mac"].as<double>(2.0);
  pim_ops_per_block_issue = dram["pim_ops_per_block_issue"].as<double>(64.0);
  pim_ops_per_request = dram["pim_ops_per_request"].as<double>(64.0);
  pim_mac_latency_cycles = dram["pim_mac_latency_cycles"].as<int>(-1);
  pim_mac_issue_interval_cycles = dram["pim_mac_issue_interval_cycles"].as<int>(-1);
  pim_mac_pipeline_latency_cycles = dram["pim_mac_pipeline_latency_cycles"].as<int>(pim_mac_latency_cycles);
  pim_movement_cycles = dram["pim_movement_cycles"].as<int>(1);
  pim_writeback_cycles = dram["pim_writeback_cycles"].as<int>(0);
  pim_slot_cost = dram["pim_slot_cost"].as<int>(1);
  pim_slots_per_request = dram["pim_slots_per_request"].as<int>(pim_slot_cost);
  pim_slot_cost = pim_slots_per_request;
  pim_compute_energy_pJ_per_mac = dram["pim_compute_energy_pJ_per_mac"].as<double>(0.0);
  pim_array_local_energy_pJ = dram["pim_array_local_energy_pJ"].as<double>(0.0);
  pim_cell_to_pim_energy_pJ_per_256b = dram["pim_cell_to_pim_energy_pJ_per_256b"].as<double>(0.0);
  pim_interconnect_energy_pJ_per_256b = dram["pim_interconnect_energy_pJ_per_256b"].as<double>(0.0);
  pim_vrf_access_energy_pJ = dram["pim_vrf_access_energy_pJ"].as<double>(0.0);
  pim_srf_access_energy_pJ = dram["pim_srf_access_energy_pJ"].as<double>(0.0);
  pim_mode_switch_energy_pJ = dram["pim_mode_switch_energy_pJ"].as<double>(0.0);
  if (pim_datatype_bits <= 0) {
    throw std::runtime_error("DRAMSpec: pim_datatype_bits must be positive");
  }
  if (pim_simd_width_bits <= 0) {
    throw std::runtime_error("DRAMSpec: pim_simd_width_bits must be positive");
  }
  if (pim_lanes <= 0) {
    throw std::runtime_error("DRAMSpec: pim_lanes must be positive");
  }
  if (pim_ops_per_mac <= 0.0) {
    throw std::runtime_error("DRAMSpec: pim_ops_per_mac must be positive");
  }
  if (pim_ops_per_block_issue <= 0.0) {
    throw std::runtime_error("DRAMSpec: pim_ops_per_block_issue must be positive");
  }
  if (pim_ops_per_request <= 0.0) {
    throw std::runtime_error("DRAMSpec: pim_ops_per_request must be positive");
  }
  if (pim_movement_cycles < 0) {
    throw std::runtime_error("DRAMSpec: pim_movement_cycles must be non-negative");
  }
  if (pim_writeback_cycles < 0) {
    throw std::runtime_error("DRAMSpec: pim_writeback_cycles must be non-negative");
  }
  if (pim_slots_per_request <= 0) {
    throw std::runtime_error("DRAMSpec: pim_slots_per_request must be positive");
  }
  if (pim_banks_per_mpu <= 0) {
    throw std::runtime_error("DRAMSpec: pim_banks_per_mpu must be positive");
  }
  if (pim_compute_energy_pJ_per_mac < 0.0 || pim_array_local_energy_pJ < 0.0 ||
      pim_cell_to_pim_energy_pJ_per_256b < 0.0 || pim_interconnect_energy_pJ_per_256b < 0.0 ||
      pim_vrf_access_energy_pJ < 0.0 || pim_srf_access_energy_pJ < 0.0 || pim_mode_switch_energy_pJ < 0.0) {
    throw std::runtime_error("DRAMSpec: PIM event energy terms must be non-negative");
  }

  // Optional built-in DRAM power parameters
  const ConfigNode power = dram["power"];
  power_params.clear();
  if (power && power.is_map()) {
    drampower_enable = power["enabled"].as<bool>(false);
    power_debug = power["debug"].as<bool>(false);
    for (const auto& kv : power.map()) {
      if (kv.first == "enabled" || kv.first == "debug") {
        continue;
      }
      power_params[kv.first] = kv.second.as<double>(0.0);
    }
  } else {
    drampower_enable = false;
    power_debug = false;
  }

  // Organization
  channel_width = dram["channel_width"].as<int>();
  ConfigNode org = dram["org"];
  organization.dq = org["dq"].as<int>();
  ConfigNode count_node = org["count"];
  const auto& counts = count_node.seq();
  organization.level_sizes.resize(counts.size());
  for (size_t i = 0; i < counts.size(); i++) {
    organization.level_sizes[i] = counts[i].as<int>();
  }
  // Each controller handles a single channel
  if (organization.level_sizes[0] != 1) {
    throw std::runtime_error(
        "DRAMSpec: level_sizes[0] (Channel) must be 1 — "
        "multi-channel is configured at the system level, not in the DRAM spec. "
        "Got: " + std::to_string(organization.level_sizes[0]));
  }

  // Timing values
  ConfigNode timing_node = dram["timing"];
  const auto& timing = timing_node.seq();
  timing_vals.resize(timing.size());
  for (size_t i = 0; i < timing.size(); i++) {
    timing_vals[i] = timing[i].as<int>();
  }
  if (has_timing("nPIM_MAC_LAT")) {
    int pim_mac_latency_value = get_timing_value("nPIM_MAC_LAT");
    int pim_mac_issue_interval_value = has_timing("nPIM_MAC_II") ? get_timing_value("nPIM_MAC_II") : pim_mac_latency_value;
    if (pim_mac_latency_cycles <= 0) {
      pim_mac_latency_cycles = pim_mac_latency_value;
    }
    if (pim_mac_pipeline_latency_cycles <= 0) {
      pim_mac_pipeline_latency_cycles = pim_mac_latency_cycles;
    }
    if (pim_mac_issue_interval_cycles <= 0) {
      pim_mac_issue_interval_cycles = pim_mac_issue_interval_value;
    }
    if (pim_datatype_behavior_enabled) {
      timing_vals[timings["nPIM_MAC_LAT"]] = pim_mac_pipeline_latency_cycles;
      if (has_timing("nPIM_MAC_II")) {
        timing_vals[timings["nPIM_MAC_II"]] = pim_mac_issue_interval_cycles;
      }
    } else {
      pim_mac_latency_cycles = pim_mac_latency_value;
      pim_mac_pipeline_latency_cycles = pim_mac_latency_value;
      pim_mac_issue_interval_cycles = pim_mac_issue_interval_value;
      pim_movement_cycles = 1;
      pim_writeback_cycles = 0;
      pim_slots_per_request = 1;
      pim_slot_cost = 1;
    }
    if (pim_mac_issue_interval_cycles <= 0) {
      throw std::runtime_error("DRAMSpec: pim_mac_issue_interval_cycles must be positive");
    }
    if (pim_mac_pipeline_latency_cycles <= 0) {
      throw std::runtime_error("DRAMSpec: pim_mac_pipeline_latency_cycles must be positive");
    }
  }

  // Read latency (pre-computed by Python)
  read_latency = dram["read_latency"].as<int>();

  // Timing constraints (pre-computed by Python)
  timing_cons.resize(level_count, std::vector<std::vector<TimingConsEntry>>(command_count));

  ConfigNode tc_node = dram["timing_constraints"];
  for (const auto& entry : tc_node.seq()) {
    const auto& f = entry.seq();
    int level = f[0].as<int>();
    int latency = f[3].as<int>();
    int window = f.size() > 4 ? f[4].as<int>() : 1;
    bool sibling = f.size() > 5 ? f[5].as<bool>() : false;
    for (const auto& p : f[1].seq()) {
      for (const auto& fc : f[2].seq()) {
        timing_cons[level][p.as<int>()].push_back({fc.as<int>(), latency, window, sibling});
      }
    }
  }

  // Precompute sibling-constraint flags to skip unnecessary traversal
  has_sibling_cons.assign(level_count, std::vector<int8_t>(command_count, 0));
  for (int lvl = 0; lvl < level_count; ++lvl) {
    for (int cmd = 0; cmd < command_count; ++cmd) {
      for (const auto& t : timing_cons[lvl][cmd]) {
        if (t.sibling) {
          has_sibling_cons[lvl][cmd] = 1;
          break;
        }
      }
    }
  }
}

std::map<std::string, DRAMSpec::Creator>& DRAMSpec::registry() {
  static std::map<std::string, Creator> r;
  return r;
}

bool DRAMSpec::register_standard(const std::string& name, Creator c) {
  registry()[name] = std::move(c);
  return true;
}

std::unique_ptr<DRAMSpec> DRAMSpec::create(const std::string& name, const ConfigNode& config) {
  auto it = registry().find(name);
  if (it == registry().end()) {
    throw std::runtime_error("Unknown DRAM standard: " + name);
  }
  return it->second(config);
}

}  // namespace Ramulator
