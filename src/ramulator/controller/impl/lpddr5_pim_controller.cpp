#include <fmt/format.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <vector>

#include "ramulator/base/base.h"
#include "ramulator/controller/controller_base.h"
#include "ramulator/controller/refresh/i_refresh_manager.h"
#include "ramulator/controller/rowpolicy/i_row_policy.h"
#include "ramulator/dram/dram_spec.h"

namespace Ramulator {

class LPDDR5PIMController : public ControllerBase {
  RAMULATOR_REGISTER_IMPLEMENTATION_DERIVED(IController, LPDDR5PIMController, ControllerBase, "LPDDR5PIM")

 public:
  void init() override;
  void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override;
  bool send(Request& req) override;
  void tick() override;

 private:
  enum class Act2IssueKind {
    Urgent,
    Deferred,
  };

  enum class PIMRankMode {
    SingleBank,
    HostAllBank,
    PIMAllBank,
  };

  enum class PIMMACExecutionModel {
    SharedMPUSerial,
    SubbankOverlapExperimental,
  };

  ReqBuffer m_activating_buffer;
  int m_cmd_act1 = -1;
  int m_cmd_act2 = -1;
  std::vector<bool> m_act2_owner_valid;
  std::vector<Clk_t> m_act2_deadline;
  int m_nAAD = 0;

  Clk_t m_wck_expiry = 0;
  int m_cmd_cas_rd = -1;
  int m_cmd_cas_wr = -1;
  int m_cmd_rd = -1;
  int m_cmd_wr = -1;
  int m_cmd_rda = -1;
  int m_cmd_wra = -1;
  int m_nCL = 0;
  int m_nCWL = 0;
  int m_nBL = 0;
  int m_nWCKPST = 0;

  int m_cmd_pim_mac = -1;
  int m_cmd_sb = -1;
  int m_cmd_hab = -1;
  int m_cmd_hab_pim = -1;
  int m_cmd_pim_bcast = -1;
  int m_cmd_pim_mac_ab = -1;
  int m_pim_blocks_per_bank = 1;
  int m_pim_banks_per_mpu = 1;
  int m_pim_mpu_group_count = 0;
  int m_pim_slots_per_request = 1;
  PIMMACExecutionModel m_pim_mac_execution_model = PIMMACExecutionModel::SharedMPUSerial;
  int m_rank_level = -1;
  int m_bank_level = -1;
  int m_pim_banks_per_rank = 0;
  int m_row_level = -1;
  int m_column_level = -1;
  int m_pim_mac_issue_interval_cycles = 0;
  int m_pim_mac_pipeline_latency_cycles = 0;
  int m_pim_movement_cycles = 1;
  int m_pim_writeback_cycles = 0;
  int m_pim_completion_latency_cycles = 0;
  PIMRankMode m_pim_rank_mode = PIMRankMode::SingleBank;
  bool m_pim_all_bank_load_ready = false;
  bool m_pim_ab_inflight = false;
  Clk_t m_pim_ab_start_clk = -1;
  Clk_t m_pim_ab_done_clk = -1;
  Request m_pim_ab_request;
  // dependency-aware MVP semantics:
  // - frontend-derived dependency identity comes from the frontend address pattern; do not add Request metadata.
  // - PIM_MAC issue is launch not completion.
  // - same-bank dependents serialize, but ACT1 -> ACT2 -> PIM_MAC launch legality stays intact.
  // - overlap comes from controller-side in-flight execution residency after issue, bounded by pim_blocks_per_bank.
  struct InflightPIM {
    uint64_t dependency_id = 0;
    int slot_cost = 1;
    Clk_t start_clk = -1;
    Clk_t done_clk = -1;
    Request request;
  };
  std::vector<int> m_pim_slots_in_use;
  std::vector<std::vector<InflightPIM>> m_inflight_pim;

  bool m_cas_issued = false;
  ReqBuffer::iterator m_cas_req_it;
  ReqBuffer* m_cas_buffer = nullptr;

  size_t s_cas_issued = 0;
  size_t s_cas_skipped = 0;
  size_t s_act2_deadline_forced = 0;
  size_t s_act2_deferred = 0;
  size_t s_pim_capacity_stalls = 0;
  size_t s_pim_mpu_group_stalls = 0;
  size_t s_pim_dependency_stalls = 0;
  size_t s_num_global_issue_blocked_cycles = 0;
  size_t s_num_mpu_group_busy_blocked_cycles = 0;
  size_t s_num_bank_timing_blocked_cycles = 0;
  size_t s_num_rr_head_of_line_blocked_cycles = 0;
  size_t s_num_queue_empty_cycles = 0;
  size_t s_num_issued_pim_mac = 0;
  size_t s_num_pim_reqs_served = 0;
  size_t s_pim_inflight_peak = 0;
  size_t s_pim_simultaneous_active_banks_peak = 0;
  size_t s_pim_mode_stalls = 0;
  size_t s_pim_load_stalls = 0;
  size_t s_num_pim_ab_reqs_served = 0;
  size_t s_pim_ab_inflight_peak = 0;
  int64_t s_pim_latency = 0;
  float s_avg_pim_latency = 0.0f;
  int64_t s_pim_service_latency = 0;
  float s_avg_pim_service_latency = 0.0f;
  int64_t s_pim_launch_wait = 0;
  float s_avg_pim_launch_wait = 0.0f;
  int64_t s_pim_response_latency = 0;
  float s_avg_pim_response_latency = 0.0f;
  int s_pim_datatype_behavior_enabled = 0;
  int s_pim_datatype_bits = 8;
  int s_pim_simd_width_bits = 256;
  int s_pim_lanes = 32;
  double s_pim_ops_per_mac = 2.0;
  double s_pim_ops_per_block_issue = 64.0;
  double s_pim_ops_per_request = 64.0;
  int s_pim_mac_issue_interval_cycles = 0;
  int s_pim_mac_pipeline_latency_cycles = 0;
  int s_pim_movement_cycles = 1;
  int s_pim_writeback_cycles = 0;
  int s_pim_completion_latency_cycles = 0;
  int s_pim_slots_per_request = 1;
  int s_pim_banks_per_mpu = 1;
  int s_pim_mpu_group_count = 0;
  int s_total_banks = 0;
  int s_effective_mpu_groups = 0;
  int s_pim_mac_execution_model = 0;
  double s_pim_compute_energy_pJ_per_mac = 0.0;
  double s_pim_array_local_energy_pJ = 0.0;
  double s_pim_cell_to_pim_energy_pJ_per_256b = 0.0;
  double s_pim_interconnect_energy_pJ_per_256b = 0.0;
  double s_pim_vrf_access_energy_pJ = 0.0;
  double s_pim_srf_access_energy_pJ = 0.0;
  double s_pim_mode_switch_energy_pJ = 0.0;
  static constexpr int kObservedPimBanks = 4;
  std::vector<size_t> s_pim_launches_per_bank;
  std::vector<size_t> s_pim_inflight_peak_per_bank;

  void update_pim_observability();

  bool is_access_cmd(int cmd) const;
  bool is_read_cmd(int cmd) const;
  void extend_wck_expiry(int cmd);

  bool cas_would_block_deadline() const;
  bool would_block_activating(int cmd, const AddrVec_t& addr_vec) const;
  bool would_block_pim_launch(const Request& req);
  int effective_pim_slot_capacity() const;
  bool mpu_group_has_inflight_pim(int flat_bank_id, bool exclude_target_bank) const;
  bool has_queued_pim_request();
  void account_blocked_pim_cycle();
  bool would_block_host_request(const Request& req) const;
  bool is_owned_act2_candidate(const Request& req) const;
  uint64_t get_pim_dependency_id(const Request& req) const;
  void complete_pim_if_ready();
  void launch_inflight_pim(Candidate cand, int flat_bank_id);
  void handle_mode_or_bcast_completion(Request& req);
  void launch_inflight_pim_ab(Candidate cand);

  Candidate select_normal_candidate();
  Candidate pick_urgent_act2();
  Candidate pick_deferred_act2();

  void issue_owned_act2(Candidate cand, Act2IssueKind kind);
  bool try_issue_cas_sync(Candidate& cand);
  void issue_standard_candidate(Candidate cand);
  void move_to_activating(ReqBuffer::iterator& req_it, ReqBuffer& buffer);
  void promote_from_activating(ReqBuffer::iterator& req_it, ReqBuffer& buffer);
};

void LPDDR5PIMController::init() {
  init_base();

  const auto& spec = *m_device.m_spec;
  m_cmd_act1 = spec.get_command_id("ACT1");
  m_cmd_act2 = spec.get_command_id("ACT2");
  m_cmd_cas_rd = spec.get_command_id("CAS_RD");
  m_cmd_cas_wr = spec.get_command_id("CAS_WR");
  m_cmd_rd = spec.get_command_id("RD");
  m_cmd_wr = spec.get_command_id("WR");
  m_cmd_rda = spec.get_command_id("RDA");
  m_cmd_wra = spec.get_command_id("WRA");
  m_cmd_sb = spec.get_command_id("SB");
  m_cmd_hab = spec.get_command_id("HAB");
  m_cmd_hab_pim = spec.get_command_id("HAB_PIM");
  m_cmd_pim_bcast = spec.get_command_id("PIM_BCAST");
  m_cmd_pim_mac_ab = spec.get_command_id("PIM_MAC_AB");

  m_nAAD = spec.get_timing_value("nAAD");
  m_nCL = spec.get_timing_value("nCL");
  m_nCWL = spec.get_timing_value("nCWL");
  m_nBL = spec.get_timing_value("nBL");
  m_nWCKPST = spec.get_timing_value("nWCKPST");

  m_activating_buffer.max_size = m_device.m_bank_nodes.size();
  m_act2_owner_valid.assign(m_device.m_bank_nodes.size(), false);
  m_act2_deadline.assign(m_device.m_bank_nodes.size(), -1);
  m_pim_blocks_per_bank = spec.pim_blocks_per_bank > 0 ? spec.pim_blocks_per_bank : 1;
  m_pim_banks_per_mpu = spec.pim_banks_per_mpu;
  if (m_pim_banks_per_mpu <= 0) {
    throw std::runtime_error("LPDDR5PIMController: pim_banks_per_mpu must be positive");
  }
  m_rank_level = spec.get_level_id("Rank");
  m_bank_level = spec.get_level_id("Bank");
  m_pim_banks_per_rank = 1;
  for (int level = m_rank_level + 1; level <= m_bank_level; level++) {
    m_pim_banks_per_rank *= spec.organization.level_sizes[level];
  }
  if (m_pim_banks_per_mpu > m_pim_banks_per_rank) {
    throw std::runtime_error(fmt::format(
        "LPDDR5PIMController: pim_banks_per_mpu {} exceeds banks per rank {}",
        m_pim_banks_per_mpu,
        m_pim_banks_per_rank));
  }
  if (m_pim_banks_per_rank % m_pim_banks_per_mpu != 0) {
    throw std::runtime_error(fmt::format(
        "LPDDR5PIMController: banks per rank {} is not divisible by pim_banks_per_mpu {}",
        m_pim_banks_per_rank,
        m_pim_banks_per_mpu));
  }
  if (static_cast<int>(m_device.m_bank_nodes.size()) % m_pim_banks_per_mpu != 0) {
    throw std::runtime_error(fmt::format(
        "LPDDR5PIMController: bank count {} is not divisible by pim_banks_per_mpu {}",
        m_device.m_bank_nodes.size(),
        m_pim_banks_per_mpu));
  }
  m_pim_mpu_group_count = static_cast<int>(m_device.m_bank_nodes.size()) / m_pim_banks_per_mpu;
  m_pim_slots_per_request = spec.pim_slots_per_request > 0 ? spec.pim_slots_per_request : 1;
  if (spec.pim_mac_execution_model == "shared_mpu_serial") {
    m_pim_mac_execution_model = PIMMACExecutionModel::SharedMPUSerial;
  } else if (spec.pim_mac_execution_model == "subbank_overlap_experimental") {
    m_pim_mac_execution_model = PIMMACExecutionModel::SubbankOverlapExperimental;
  } else {
    throw std::runtime_error(fmt::format(
        "LPDDR5PIMController: unknown pim_mac_execution_model '{}'",
        spec.pim_mac_execution_model));
  }
  m_pim_slots_in_use.assign(m_device.m_bank_nodes.size(), 0);
  m_inflight_pim.assign(m_device.m_bank_nodes.size(), {});
  s_pim_datatype_behavior_enabled = spec.pim_datatype_behavior_enabled ? 1 : 0;
  s_pim_datatype_bits = spec.pim_datatype_bits;
  s_pim_simd_width_bits = spec.pim_simd_width_bits;
  s_pim_lanes = spec.pim_lanes;
  s_pim_ops_per_mac = spec.pim_ops_per_mac;
  s_pim_ops_per_block_issue = spec.pim_ops_per_block_issue;
  s_pim_ops_per_request = spec.pim_ops_per_request;
  m_pim_mac_issue_interval_cycles = spec.pim_mac_issue_interval_cycles;
  m_pim_mac_pipeline_latency_cycles = spec.pim_mac_pipeline_latency_cycles;
  m_pim_movement_cycles = spec.pim_movement_cycles;
  m_pim_writeback_cycles = spec.pim_writeback_cycles;
  m_pim_completion_latency_cycles = m_pim_mac_pipeline_latency_cycles + m_pim_movement_cycles + m_pim_writeback_cycles;
  s_pim_mac_issue_interval_cycles = m_pim_mac_issue_interval_cycles;
  s_pim_mac_pipeline_latency_cycles = m_pim_mac_pipeline_latency_cycles;
  s_pim_movement_cycles = m_pim_movement_cycles;
  s_pim_writeback_cycles = m_pim_writeback_cycles;
  s_pim_completion_latency_cycles = m_pim_completion_latency_cycles;
  s_pim_slots_per_request = m_pim_slots_per_request;
  s_pim_banks_per_mpu = m_pim_banks_per_mpu;
  s_pim_mpu_group_count = m_pim_mpu_group_count;
  s_total_banks = static_cast<int>(m_device.m_bank_nodes.size());
  s_effective_mpu_groups = m_pim_mpu_group_count;
  s_pim_mac_execution_model =
      m_pim_mac_execution_model == PIMMACExecutionModel::SharedMPUSerial ? 0 : 1;
  s_pim_compute_energy_pJ_per_mac = spec.pim_compute_energy_pJ_per_mac;
  s_pim_array_local_energy_pJ = spec.pim_array_local_energy_pJ;
  s_pim_cell_to_pim_energy_pJ_per_256b = spec.pim_cell_to_pim_energy_pJ_per_256b;
  s_pim_interconnect_energy_pJ_per_256b = spec.pim_interconnect_energy_pJ_per_256b;
  s_pim_vrf_access_energy_pJ = spec.pim_vrf_access_energy_pJ;
  s_pim_srf_access_energy_pJ = spec.pim_srf_access_energy_pJ;
  s_pim_mode_switch_energy_pJ = spec.pim_mode_switch_energy_pJ;
  s_pim_launches_per_bank.assign(kObservedPimBanks, 0);
  s_pim_inflight_peak_per_bank.assign(kObservedPimBanks, 0);
}

void LPDDR5PIMController::setup(IFrontEnd* frontend, IMemorySystem* memory_system) {
  setup_base(frontend, memory_system);

  m_cmd_pim_mac = m_device.m_spec->get_command_id("PIM_MAC");
  m_row_level = m_device.m_spec->get_level_id("Row");
  m_column_level = m_device.m_spec->get_level_id("Column");

  m_stats.add("cas_issued", s_cas_issued);
  m_stats.add("cas_skipped", s_cas_skipped);
  m_stats.add("act2_deadline_forced", s_act2_deadline_forced);
  m_stats.add("act2_deferred", s_act2_deferred);
  m_stats.add("pim_capacity_stalls", s_pim_capacity_stalls);
  m_stats.add("pim_mpu_group_stalls", s_pim_mpu_group_stalls);
  m_stats.add("pim_dependency_stalls", s_pim_dependency_stalls);
  m_stats.add("pim_mode_stalls", s_pim_mode_stalls);
  m_stats.add("pim_load_stalls", s_pim_load_stalls);
  m_stats.add("pim_inflight_peak", s_pim_inflight_peak);
  m_stats.add("pim_simultaneous_active_banks_peak", s_pim_simultaneous_active_banks_peak);
  m_stats.add("num_pim_reqs_served", s_num_pim_reqs_served);
  m_stats.add("num_pim_ab_reqs_served", s_num_pim_ab_reqs_served);
  m_stats.add("pim_ab_inflight_peak", s_pim_ab_inflight_peak);
  m_stats.add("pim_latency", s_pim_latency);
  m_stats.add("avg_pim_latency", s_avg_pim_latency);
  m_stats.add("pim_service_latency", s_pim_service_latency);
  m_stats.add("avg_pim_service_latency", s_avg_pim_service_latency);
  m_stats.add("pim_launch_wait", s_pim_launch_wait);
  m_stats.add("avg_pim_launch_wait", s_avg_pim_launch_wait);
  m_stats.add("pim_response_latency", s_pim_response_latency);
  m_stats.add("avg_pim_response_latency", s_avg_pim_response_latency);
  m_stats.add("pim_datatype_behavior_enabled", s_pim_datatype_behavior_enabled);
  m_stats.add("pim_datatype_bits", s_pim_datatype_bits);
  m_stats.add("pim_simd_width_bits", s_pim_simd_width_bits);
  m_stats.add("pim_lanes", s_pim_lanes);
  m_stats.add("pim_ops_per_mac", s_pim_ops_per_mac);
  m_stats.add("pim_ops_per_block_issue", s_pim_ops_per_block_issue);
  m_stats.add("pim_ops_per_request", s_pim_ops_per_request);
  m_stats.add("pim_mac_issue_interval_cycles", s_pim_mac_issue_interval_cycles);
  m_stats.add("pim_mac_pipeline_latency_cycles", s_pim_mac_pipeline_latency_cycles);
  m_stats.add("pim_movement_cycles", s_pim_movement_cycles);
  m_stats.add("pim_writeback_cycles", s_pim_writeback_cycles);
  m_stats.add("pim_completion_latency_cycles", s_pim_completion_latency_cycles);
  m_stats.add("pim_slots_per_request", s_pim_slots_per_request);
  m_stats.add("pim_slot_cost", s_pim_slots_per_request);
  m_stats.add("pim_banks_per_mpu", s_pim_banks_per_mpu);
  m_stats.add("pim_mpu_group_count", s_pim_mpu_group_count);
  m_stats.add("total_banks", s_total_banks);
  m_stats.add("effective_mpu_groups", s_effective_mpu_groups);
  m_stats.add("pim_mac_execution_model", s_pim_mac_execution_model);
  m_stats.add("num_global_issue_blocked_cycles", s_num_global_issue_blocked_cycles);
  m_stats.add("num_mpu_group_busy_blocked_cycles", s_num_mpu_group_busy_blocked_cycles);
  m_stats.add("num_bank_timing_blocked_cycles", s_num_bank_timing_blocked_cycles);
  m_stats.add("num_rr_head_of_line_blocked_cycles", s_num_rr_head_of_line_blocked_cycles);
  m_stats.add("num_queue_empty_cycles", s_num_queue_empty_cycles);
  m_stats.add("num_issued_pim_mac", s_num_issued_pim_mac);
  m_stats.add("pim_compute_energy_pJ_per_mac", s_pim_compute_energy_pJ_per_mac);
  m_stats.add("pim_array_local_energy_pJ", s_pim_array_local_energy_pJ);
  m_stats.add("pim_cell_to_pim_energy_pJ_per_256b", s_pim_cell_to_pim_energy_pJ_per_256b);
  m_stats.add("pim_interconnect_energy_pJ_per_256b", s_pim_interconnect_energy_pJ_per_256b);
  m_stats.add("pim_vrf_access_energy_pJ", s_pim_vrf_access_energy_pJ);
  m_stats.add("pim_srf_access_energy_pJ", s_pim_srf_access_energy_pJ);
  m_stats.add("pim_mode_switch_energy_pJ", s_pim_mode_switch_energy_pJ);
  for (int bank_id = 0; bank_id < kObservedPimBanks; bank_id++) {
    m_stats.add(fmt::format("pim_launches_bank_{}", bank_id), s_pim_launches_per_bank[bank_id]);
    m_stats.add(fmt::format("pim_inflight_peak_bank_{}", bank_id), s_pim_inflight_peak_per_bank[bank_id]);
  }
}

bool LPDDR5PIMController::send(Request& req) {
  m_addr_mapper->apply(req);
  req.addr_vec[0] = m_channel_id;

  if (req.type_id == -1) {
    if (req.final_command != m_cmd_sb && req.final_command != m_cmd_hab && req.final_command != m_cmd_hab_pim) {
      throw std::runtime_error(fmt::format(
          "LPDDR5PIM concrete direct-command replay supports SB/HAB/HAB_PIM only; got final_command {}",
          req.final_command));
    }
    req.arrive = m_clk;
    bool is_success = m_read_buffer.enqueue(req);
    if (!is_success) {
      req.arrive = -1;
    }
    return is_success;
  }

  if (req.type_id < 0 || req.type_id >= static_cast<int>(m_device.m_spec->supported_requests.size())) {
    throw std::runtime_error(fmt::format(
        "LPDDR5PIM supports request type ids in [0, {}), got {}",
        m_device.m_spec->supported_requests.size(),
        req.type_id));
  }
  req.final_command = m_device.m_spec->supported_requests[req.type_id];

  if (req.type_id == Request::Type::Read) {
    if (m_buffered_write_addrs.count(req.addr)) {
      req.arrive = m_clk;
      req.depart = m_clk + 1;
      m_pending.push_back(req);
      s_num_read_reqs++;
      return true;
    }
  }

  bool is_success = false;
  req.arrive = m_clk;
  if (req.type_id == Request::Type::Read || req.final_command == m_cmd_pim_mac ||
      req.final_command == m_cmd_pim_bcast || req.final_command == m_cmd_pim_mac_ab ||
      req.final_command == m_cmd_sb || req.final_command == m_cmd_hab ||
      req.final_command == m_cmd_hab_pim) {
    is_success = m_read_buffer.enqueue(req);
  } else if (req.type_id == Request::Type::Write) {
    if (m_buffered_write_addrs.count(req.addr)) {
      if (req.callback) {
        req.callback(req);
      }
      s_num_write_reqs++;
      s_num_write_reqs_served++;
      return true;
    }
    is_success = m_write_buffer.enqueue(req);
    if (is_success) {
      m_buffered_write_addrs.insert(req.addr);
    }
  } else {
    throw std::runtime_error(fmt::format(
        "LPDDR5PIM supports Read/Write, PIMCompute, PIMLoadAll, and PIMComputeAll; got type_id {}",
        req.type_id));
  }

  if (!is_success) {
    req.arrive = -1;
    return false;
  }

  if (req.type_id == Request::Type::Read) {
    s_num_read_reqs++;
  } else if (req.type_id == Request::Type::Write) {
    s_num_write_reqs++;
  }

  return true;
}

void LPDDR5PIMController::tick() {
  tick_prologue();
  complete_pim_if_ready();
  m_refresh->tick();

  m_rowpolicy->pre_schedule();
  for (auto* p : m_plugins) {
    p->pre_schedule();
  }

  Candidate urgent_act2 = pick_urgent_act2();

  if (m_cas_issued) {
    assert(!urgent_act2.valid);
    assert(m_cas_buffer != nullptr);

    if (check_timing(m_cas_req_it->command, m_cas_req_it->addr_vec)) {
      m_cas_issued = false;
      extend_wck_expiry(m_cas_req_it->command);
      s_cas_issued++;

      m_device.issue_command(m_cas_req_it->command, m_cas_req_it->addr_vec, m_clk);

      if (!m_cas_req_it->is_stat_updated) {
        update_request_stats(m_cas_req_it);
      }

      m_rowpolicy->on_issue(*m_cas_req_it);
      for (auto* p : m_plugins) {
        p->on_issue(*m_cas_req_it);
      }

      if (m_cas_req_it->command == m_cas_req_it->final_command) {
        retire_request(m_cas_req_it, *m_cas_buffer);
      }
    }

    m_rowpolicy->post_schedule();
    for (auto* p : m_plugins) {
      p->post_schedule();
    }
    return;
  }

  Candidate cand = urgent_act2.valid ? urgent_act2 : select_normal_candidate();
  if (!cand.valid) {
    Candidate deferred = pick_deferred_act2();
    if (deferred.valid) {
      issue_owned_act2(deferred, Act2IssueKind::Deferred);
    }
    m_rowpolicy->post_schedule();
    for (auto* p : m_plugins) {
      p->post_schedule();
    }
    return;
  }

  if (cand.buffer == &m_activating_buffer) {
    issue_owned_act2(cand, Act2IssueKind::Urgent);
  } else if (cand.it->final_command == m_cmd_pim_mac || cand.it->final_command == m_cmd_pim_bcast ||
             cand.it->final_command == m_cmd_pim_mac_ab || cand.it->final_command == m_cmd_sb ||
             cand.it->final_command == m_cmd_hab || cand.it->final_command == m_cmd_hab_pim) {
    issue_standard_candidate(cand);
  } else if (!try_issue_cas_sync(cand)) {
    issue_standard_candidate(cand);
  }

  m_rowpolicy->post_schedule();
  for (auto* p : m_plugins) {
    p->post_schedule();
  }

  if (s_num_pim_reqs_served > 0) {
    s_avg_pim_latency = static_cast<float>(s_pim_latency) / s_num_pim_reqs_served;
    s_avg_pim_service_latency = static_cast<float>(s_pim_service_latency) / s_num_pim_reqs_served;
    s_avg_pim_launch_wait = static_cast<float>(s_pim_launch_wait) / s_num_pim_reqs_served;
    s_avg_pim_response_latency = static_cast<float>(s_pim_response_latency) / s_num_pim_reqs_served;
  }
}

bool LPDDR5PIMController::is_access_cmd(int cmd) const {
  return cmd == m_cmd_rd || cmd == m_cmd_wr || cmd == m_cmd_rda || cmd == m_cmd_wra;
}

bool LPDDR5PIMController::is_read_cmd(int cmd) const {
  return cmd == m_cmd_rd || cmd == m_cmd_rda;
}

bool LPDDR5PIMController::would_block_host_request(const Request& req) const {
  if (req.type_id != Request::Type::Read && req.type_id != Request::Type::Write) {
    return false;
  }
  return m_pim_rank_mode != PIMRankMode::SingleBank;
}

void LPDDR5PIMController::extend_wck_expiry(int cmd) {
  int lat = is_read_cmd(cmd) ? m_nCL : m_nCWL;
  Clk_t exp = m_clk + lat + m_nBL + m_nWCKPST;
  if (exp > m_wck_expiry) {
    m_wck_expiry = exp;
  }
}

bool LPDDR5PIMController::cas_would_block_deadline() const {
  for (int bank_id = 0; bank_id < static_cast<int>(m_act2_owner_valid.size()); bank_id++) {
    if (!m_act2_owner_valid[bank_id]) {
      continue;
    }
    if (m_act2_deadline[bank_id] <= m_clk + 1) {
      return true;
    }
  }
  return false;
}

bool LPDDR5PIMController::would_block_activating(int cmd, const AddrVec_t& addr_vec) const {
  const auto& meta = m_device.m_spec->command_meta[cmd];
  if (!meta.is_closing && !meta.is_refreshing) return false;

  if (m_pim_ab_inflight) {
    return true;
  }

  bool blocked = false;
  m_device.for_each_target_bank_while(cmd, addr_vec, [&](int bank_id) {
    if (m_act2_owner_valid[bank_id] || !m_inflight_pim[bank_id].empty()) {
      blocked = true;
      return false;
    }
    return true;
  });
  return blocked;
}

int LPDDR5PIMController::effective_pim_slot_capacity() const {
  return std::max(m_pim_blocks_per_bank, m_pim_slots_per_request);
}

bool LPDDR5PIMController::mpu_group_has_inflight_pim(int flat_bank_id, bool exclude_target_bank) const {
  int rank_begin = (flat_bank_id / m_pim_banks_per_rank) * m_pim_banks_per_rank;
  int rank_local_bank_id = flat_bank_id - rank_begin;
  int group_begin = rank_begin + (rank_local_bank_id / m_pim_banks_per_mpu) * m_pim_banks_per_mpu;
  int group_end = group_begin + m_pim_banks_per_mpu;

  for (int bank_id = group_begin; bank_id < group_end; bank_id++) {
    if (exclude_target_bank && bank_id == flat_bank_id) {
      continue;
    }
    if (!m_inflight_pim[bank_id].empty()) {
      return true;
    }
  }
  return false;
}

bool LPDDR5PIMController::has_queued_pim_request() {
  auto contains_pim = [&](ReqBuffer& buffer) {
    for (const auto& req : buffer) {
      if (req.final_command == m_cmd_pim_mac || req.final_command == m_cmd_pim_bcast ||
          req.final_command == m_cmd_pim_mac_ab) {
        return true;
      }
    }
    return false;
  };

  return contains_pim(m_active_buffer) || contains_pim(m_priority_buffer) || contains_pim(m_read_buffer);
}

void LPDDR5PIMController::account_blocked_pim_cycle() {
  auto account_from_buffer = [&](ReqBuffer& buffer) {
    for (auto& req : buffer) {
      if (req.final_command != m_cmd_pim_mac) {
        continue;
      }

      req.command = get_preq_command(req.final_command, req.addr_vec);
      if (req.command != m_cmd_pim_mac) {
        if (!check_timing(req.command, req.addr_vec)) {
          s_num_bank_timing_blocked_cycles++;
        } else {
          s_num_rr_head_of_line_blocked_cycles++;
        }
        return true;
      }

      int flat_bank_id = m_device.get_flat_bank_id(req.addr_vec);
      if (m_pim_mac_execution_model == PIMMACExecutionModel::SharedMPUSerial && !m_inflight_pim[flat_bank_id].empty()) {
        s_num_bank_timing_blocked_cycles++;
      } else if (m_pim_mac_execution_model == PIMMACExecutionModel::SharedMPUSerial &&
          mpu_group_has_inflight_pim(flat_bank_id, true)) {
        s_num_mpu_group_busy_blocked_cycles++;
      } else if (!check_timing(req.command, req.addr_vec)) {
        s_num_global_issue_blocked_cycles++;
      } else {
        s_num_rr_head_of_line_blocked_cycles++;
      }
      return true;
    }
    return false;
  };

  if (!has_queued_pim_request()) {
    s_num_queue_empty_cycles++;
    return;
  }

  if (account_from_buffer(m_active_buffer) || account_from_buffer(m_priority_buffer) || account_from_buffer(m_read_buffer)) {
    return;
  }

  s_num_rr_head_of_line_blocked_cycles++;
}

bool LPDDR5PIMController::would_block_pim_launch(const Request& req) {
  if (req.final_command == m_cmd_pim_bcast) {
    if (req.command == m_cmd_hab) {
      return false;
    }
    if (m_pim_rank_mode != PIMRankMode::HostAllBank) {
      s_pim_mode_stalls++;
      return true;
    }
    return false;
  }

  if (req.final_command == m_cmd_pim_mac_ab) {
    if (req.command == m_cmd_hab_pim) {
      return false;
    }
    if (m_pim_rank_mode != PIMRankMode::PIMAllBank) {
      s_pim_mode_stalls++;
      return true;
    }
    if (!m_pim_all_bank_load_ready) {
      s_pim_load_stalls++;
      return true;
    }
    if (m_pim_ab_inflight) {
      s_pim_capacity_stalls++;
      return true;
    }
    return false;
  }

  if (req.final_command != m_cmd_pim_mac || req.command != m_cmd_pim_mac) {
    return false;
  }

  int flat_bank_id = m_device.get_flat_bank_id(req.addr_vec);
  uint64_t dependency_id = get_pim_dependency_id(req);
  for (const auto& inflight : m_inflight_pim[flat_bank_id]) {
    if (inflight.dependency_id == dependency_id) {
      s_pim_dependency_stalls++;
      return true;
    }
  }

  if (m_pim_mac_execution_model == PIMMACExecutionModel::SharedMPUSerial && !m_inflight_pim[flat_bank_id].empty()) {
    s_num_bank_timing_blocked_cycles++;
    return true;
  }

  if (m_pim_mac_execution_model == PIMMACExecutionModel::SharedMPUSerial &&
      mpu_group_has_inflight_pim(flat_bank_id, true)) {
    s_pim_mpu_group_stalls++;
    s_num_mpu_group_busy_blocked_cycles++;
    return true;
  }

  if (m_pim_slots_in_use[flat_bank_id] + m_pim_slots_per_request > effective_pim_slot_capacity()) {
    s_pim_capacity_stalls++;
    return true;
  }

  return false;
}

bool LPDDR5PIMController::is_owned_act2_candidate(const Request& req) const {
  if (req.command != m_cmd_act2) {
    return true;
  }

  int flat_bank_id = m_device.get_flat_bank_id(req.addr_vec);
  assert(m_act2_owner_valid[flat_bank_id]);

  return false;
}

ControllerBase::Candidate LPDDR5PIMController::select_normal_candidate() {
  Candidate cand = pick_best_ready_from(m_active_buffer, [&](const Request& req) {
    return !would_block_host_request(req) && !would_block_pim_launch(req);
  });
  if (!cand.valid) {
    cand = pick_priority_if([&](const Request& req) {
      return !would_block_host_request(req) && !would_block_pim_launch(req) && is_owned_act2_candidate(req) &&
             !would_block_activating(req.command, req.addr_vec);
    });
  }
  if (!cand.valid && m_priority_buffer.size() == 0) {
    cand = pick_rw_if([&](const Request& req) {
      return !would_block_host_request(req) && !would_block_pim_launch(req) && is_owned_act2_candidate(req) &&
             !would_block_activating(req.command, req.addr_vec);
    });
  }
  if (!cand.valid) {
    account_blocked_pim_cycle();
  }
  return cand;
}

uint64_t LPDDR5PIMController::get_pim_dependency_id(const Request& req) const {
  uint64_t row = static_cast<uint64_t>(req.addr_vec[m_row_level]);
  uint64_t column = static_cast<uint64_t>(req.addr_vec[m_column_level]);
  return (row << 32) | column;
}

void LPDDR5PIMController::update_pim_observability() {
  size_t simultaneous_active_banks = 0;
  for (int flat_bank_id = 0; flat_bank_id < static_cast<int>(m_pim_slots_in_use.size()); flat_bank_id++) {
    size_t inflight = static_cast<size_t>(m_pim_slots_in_use[flat_bank_id]);
    if (inflight > 0) {
      simultaneous_active_banks++;
    }
    if (flat_bank_id < kObservedPimBanks && inflight > s_pim_inflight_peak_per_bank[flat_bank_id]) {
      s_pim_inflight_peak_per_bank[flat_bank_id] = inflight;
    }
  }
  if (simultaneous_active_banks > s_pim_simultaneous_active_banks_peak) {
    s_pim_simultaneous_active_banks_peak = simultaneous_active_banks;
  }
}

void LPDDR5PIMController::complete_pim_if_ready() {
  if (m_pim_ab_inflight && m_pim_ab_done_clk <= m_clk) {
    m_pim_ab_request.depart = m_pim_ab_done_clk;
    const Clk_t service_latency = m_pim_ab_done_clk - m_pim_ab_start_clk;
    const Clk_t launch_wait = m_pim_ab_start_clk - m_pim_ab_request.arrive;
    const Clk_t response_latency = m_pim_ab_request.depart - m_pim_ab_request.arrive;
    s_num_pim_reqs_served++;
    s_num_pim_ab_reqs_served++;
    s_pim_latency += response_latency;
    s_pim_service_latency += service_latency;
    s_pim_launch_wait += launch_wait;
    s_pim_response_latency += response_latency;
    if (m_pim_ab_request.callback) {
      m_pim_ab_request.callback(m_pim_ab_request);
    }
    m_pim_ab_inflight = false;
    m_pim_ab_start_clk = -1;
    m_pim_ab_done_clk = -1;
    s_pim_simultaneous_active_banks_peak = std::max(
        s_pim_simultaneous_active_banks_peak,
        static_cast<size_t>(m_device.m_bank_nodes.size()));
  }

  for (int flat_bank_id = 0; flat_bank_id < static_cast<int>(m_inflight_pim.size()); flat_bank_id++) {
    auto& inflight_bank = m_inflight_pim[flat_bank_id];
    for (auto it = inflight_bank.begin(); it != inflight_bank.end();) {
      if (it->done_clk > m_clk) {
        ++it;
        continue;
      }

      assert(m_pim_slots_in_use[flat_bank_id] > 0);
      it->request.depart = it->done_clk;
      const Clk_t service_latency = it->done_clk - it->start_clk;
      const Clk_t launch_wait = it->start_clk - it->request.arrive;
      const Clk_t response_latency = it->request.depart - it->request.arrive;
      s_num_pim_reqs_served++;
      s_pim_latency += response_latency;
      s_pim_service_latency += service_latency;
      s_pim_launch_wait += launch_wait;
      s_pim_response_latency += response_latency;
      if (it->request.callback) {
        it->request.callback(it->request);
      }

      assert(m_pim_slots_in_use[flat_bank_id] >= it->slot_cost);
      m_pim_slots_in_use[flat_bank_id] -= it->slot_cost;
      update_pim_observability();
      it = inflight_bank.erase(it);
    }
  }
}

void LPDDR5PIMController::handle_mode_or_bcast_completion(Request& req) {
  if (req.command == m_cmd_sb) {
    m_pim_rank_mode = PIMRankMode::SingleBank;
  } else if (req.command == m_cmd_hab) {
    m_pim_rank_mode = PIMRankMode::HostAllBank;
    m_pim_all_bank_load_ready = false;
  } else if (req.command == m_cmd_hab_pim) {
    m_pim_rank_mode = PIMRankMode::PIMAllBank;
  } else if (req.command == m_cmd_pim_bcast) {
    // Bounded sequencing token: PIM_BCAST represents a completed all-bank
    // setup/broadcast in this backend, not a silicon-faithful proof of a
    // distinct LPDDR5 command or exact payload-source/timing path.
    m_pim_all_bank_load_ready = true;
  }
}

void LPDDR5PIMController::launch_inflight_pim_ab(Candidate cand) {
  assert(cand.it->final_command == m_cmd_pim_mac_ab);
  assert(!m_pim_ab_inflight);
  assert(m_pim_all_bank_load_ready);

  m_pim_ab_request = *cand.it;
  m_pim_ab_request.depart = -1;
  m_pim_ab_start_clk = m_clk;
  m_pim_ab_done_clk = m_clk + m_pim_completion_latency_cycles;
  m_pim_ab_inflight = true;
  m_pim_all_bank_load_ready = false;
  s_pim_ab_inflight_peak = std::max(s_pim_ab_inflight_peak, static_cast<size_t>(1));
  s_pim_inflight_peak = std::max(s_pim_inflight_peak, static_cast<size_t>(m_device.m_bank_nodes.size()));
  s_pim_simultaneous_active_banks_peak = std::max(
      s_pim_simultaneous_active_banks_peak,
      static_cast<size_t>(m_device.m_bank_nodes.size()));
  for (int flat_bank_id = 0; flat_bank_id < kObservedPimBanks; flat_bank_id++) {
    s_pim_launches_per_bank[flat_bank_id]++;
    s_pim_inflight_peak_per_bank[flat_bank_id] = std::max(
        s_pim_inflight_peak_per_bank[flat_bank_id],
        static_cast<size_t>(1));
  }

  cand.buffer->remove(cand.it);
}

void LPDDR5PIMController::launch_inflight_pim(Candidate cand, int flat_bank_id) {
  assert(cand.it->final_command == m_cmd_pim_mac);
  assert(m_pim_slots_in_use[flat_bank_id] + m_pim_slots_per_request <= effective_pim_slot_capacity());

  Request launched_req = *cand.it;
  launched_req.depart = -1;

  if (cand.buffer == &m_active_buffer) {
    assert(m_active_per_bank[flat_bank_id] > 0);
    m_active_per_bank[flat_bank_id]--;
  }

  m_pim_slots_in_use[flat_bank_id] += m_pim_slots_per_request;
  s_num_issued_pim_mac++;
  if (flat_bank_id < kObservedPimBanks) {
    s_pim_launches_per_bank[flat_bank_id]++;
  }
  m_inflight_pim[flat_bank_id].push_back(InflightPIM{
      .dependency_id = get_pim_dependency_id(launched_req),
      .slot_cost = m_pim_slots_per_request,
      .start_clk = m_clk,
      .done_clk = m_clk + m_pim_completion_latency_cycles,
      .request = std::move(launched_req),
  });
  if (static_cast<size_t>(m_pim_slots_in_use[flat_bank_id]) > s_pim_inflight_peak) {
    s_pim_inflight_peak = m_pim_slots_in_use[flat_bank_id];
  }
  update_pim_observability();

  cand.buffer->remove(cand.it);
}

ControllerBase::Candidate LPDDR5PIMController::pick_urgent_act2() {
  Candidate best;
  Clk_t best_deadline = -1;

  for (auto it = m_activating_buffer.begin(); it != m_activating_buffer.end(); it++) {
    int flat_bank_id = m_device.get_flat_bank_id(it->addr_vec);
    assert(m_act2_owner_valid[flat_bank_id]);

    Clk_t deadline = m_act2_deadline[flat_bank_id];
    assert(deadline >= 0);

    it->command = m_cmd_act2;

    bool timing_ok = check_timing(it->command, it->addr_vec);
    assert(deadline > m_clk || timing_ok);
    if (deadline > m_clk || !timing_ok) {
      continue;
    }

    if (!best.valid || deadline < best_deadline || (deadline == best_deadline && it->arrive < best.it->arrive)) {
      best.valid = true;
      best.it = it;
      best.buffer = &m_activating_buffer;
      best_deadline = deadline;
    }
  }

  return best;
}

ControllerBase::Candidate LPDDR5PIMController::pick_deferred_act2() {
  Candidate best;
  Clk_t best_deadline = -1;

  for (auto it = m_activating_buffer.begin(); it != m_activating_buffer.end(); it++) {
    int flat_bank_id = m_device.get_flat_bank_id(it->addr_vec);
    assert(m_act2_owner_valid[flat_bank_id]);

    Clk_t deadline = m_act2_deadline[flat_bank_id];
    assert(deadline >= 0);

    it->command = m_cmd_act2;
    if (!check_timing(it->command, it->addr_vec)) {
      continue;
    }

    if (!best.valid || deadline < best_deadline || (deadline == best_deadline && it->arrive < best.it->arrive)) {
      best.valid = true;
      best.it = it;
      best.buffer = &m_activating_buffer;
      best_deadline = deadline;
    }
  }

  return best;
}

void LPDDR5PIMController::issue_owned_act2(Candidate cand, Act2IssueKind kind) {
  assert(cand.valid);
  assert(cand.buffer == &m_activating_buffer);
  assert(cand.it->command == m_cmd_act2);

  int flat_bank_id = m_device.get_flat_bank_id(cand.it->addr_vec);
  assert(m_act2_owner_valid[flat_bank_id]);
  assert(m_act2_deadline[flat_bank_id] >= 0);
  assert(m_clk <= m_act2_deadline[flat_bank_id]);

  m_device.issue_command(m_cmd_act2, cand.it->addr_vec, m_clk);

  if (!cand.it->is_stat_updated) {
    update_request_stats(cand.it);
  }

  m_rowpolicy->on_issue(*cand.it);
  for (auto* p : m_plugins) {
    p->on_issue(*cand.it);
  }
  promote_from_activating(cand.it, *cand.buffer);

  if (kind == Act2IssueKind::Urgent) {
    s_act2_deadline_forced++;
  } else {
    s_act2_deferred++;
  }
}

bool LPDDR5PIMController::try_issue_cas_sync(Candidate& cand) {
  assert(cand.valid);
  assert(cand.buffer != &m_activating_buffer);

  int cmd = cand.it->command;
  if (!is_access_cmd(cmd) || m_clk < m_wck_expiry) {
    return false;
  }

  if (cas_would_block_deadline()) {
    Candidate deferred = pick_deferred_act2();
    assert(deferred.valid);
    issue_owned_act2(deferred, Act2IssueKind::Deferred);
    return true;
  }

  int cas = is_read_cmd(cmd) ? m_cmd_cas_rd : m_cmd_cas_wr;
  if (check_timing(cas, cand.it->addr_vec)) {
    int saved = cand.it->command;
    cand.it->command = cas;

    m_device.issue_command(cas, cand.it->addr_vec, m_clk);
    m_rowpolicy->on_issue(*cand.it);
    for (auto* p : m_plugins) {
      p->on_issue(*cand.it);
    }

    cand.it->command = saved;
    m_cas_issued = true;
    m_cas_req_it = cand.it;
    m_cas_buffer = cand.buffer;
  }

  return true;
}

void LPDDR5PIMController::issue_standard_candidate(Candidate cand) {
  assert(cand.valid);
  assert(cand.buffer != &m_activating_buffer);

  int saved_cmd = cand.it->command;
  int saved_final_command = cand.it->final_command;
  m_rowpolicy->try_upgrade_command(*cand.it);
  if (would_block_activating(cand.it->command, cand.it->addr_vec)) {
    cand.it->command = saved_cmd;
    cand.it->final_command = saved_final_command;
  }
  int cmd = cand.it->command;
  int flat_bank_id = -1;
  if (m_device.m_spec->bank_targets[cmd] == BankTarget::Single) {
    flat_bank_id = m_device.get_flat_bank_id(cand.it->addr_vec);
  }

  if (cmd == m_cmd_act1) {
    assert(flat_bank_id >= 0);
    assert(!m_act2_owner_valid[flat_bank_id]);
  }

  if (is_access_cmd(cmd)) {
    extend_wck_expiry(cmd);
    s_cas_skipped++;
  }

  m_device.issue_command(cmd, cand.it->addr_vec, m_clk);

  if (!cand.it->is_stat_updated) {
    update_request_stats(cand.it);
  }

  m_rowpolicy->on_issue(*cand.it);
  for (auto* p : m_plugins) {
    p->on_issue(*cand.it);
  }

  if (cmd == m_cmd_sb || cmd == m_cmd_hab || cmd == m_cmd_hab_pim || cmd == m_cmd_pim_bcast) {
    handle_mode_or_bcast_completion(*cand.it);
  }

  if (cmd == m_cmd_act1) {
    move_to_activating(cand.it, *cand.buffer);
  } else if (cand.it->command == cand.it->final_command) {
    if (cand.it->final_command == m_cmd_pim_mac) {
      launch_inflight_pim(cand, flat_bank_id);
      return;
    } else if (cand.it->final_command == m_cmd_pim_mac_ab) {
      launch_inflight_pim_ab(cand);
      return;
    }
    retire_request(cand.it, *cand.buffer);
  } else if (m_device.m_spec->command_meta[cand.it->command].is_opening) {
    promote_to_active(cand.it, *cand.buffer);
  }
}

void LPDDR5PIMController::move_to_activating(ReqBuffer::iterator& req_it, ReqBuffer& buffer) {
  int flat_bank_id = m_device.get_flat_bank_id(req_it->addr_vec);
  assert(!m_act2_owner_valid[flat_bank_id]);

  bool enqueued = m_activating_buffer.enqueue(*req_it);
  assert(enqueued);

  if (&buffer == &m_write_buffer) {
    m_buffered_write_addrs.erase(req_it->addr);
  }
  buffer.remove(req_it);

  m_act2_owner_valid[flat_bank_id] = true;
  m_act2_deadline[flat_bank_id] = m_clk + m_nAAD;
}

void LPDDR5PIMController::promote_from_activating(ReqBuffer::iterator& req_it, ReqBuffer& buffer) {
  int flat_bank_id = m_device.get_flat_bank_id(req_it->addr_vec);
  assert(m_act2_owner_valid[flat_bank_id]);
  assert(m_clk <= m_act2_deadline[flat_bank_id]);

  size_t old_size = m_active_buffer.size();
  promote_to_active(req_it, buffer);
  assert(m_active_buffer.size() == old_size + 1);

  m_act2_owner_valid[flat_bank_id] = false;
  m_act2_deadline[flat_bank_id] = -1;
}

}  // namespace Ramulator
