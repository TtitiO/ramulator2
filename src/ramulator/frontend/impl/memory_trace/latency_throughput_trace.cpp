#include <algorithm>

#include <fmt/format.h>
#include <optional>
#include <random>
#include <sstream>
#include <string>

#include "ramulator/base/param.h"
#include "ramulator/frontend/i_frontend.h"

namespace Ramulator {

class LatencyThroughputTrace : public IFrontEnd, public Implementation {
  // Latency-throughput evaluation frontend with streaming load and pointer-chasing probe.
  RAMULATOR_REGISTER_IMPLEMENTATION(IFrontEnd, LatencyThroughputTrace, "LatencyThroughputTrace")

 private:
  // Address generation recipe (pre-computed by Python, zero DRAM semantics)
  int m_addr_vec_size;
  std::vector<int> m_bank_positions;  // which addr_vec slots to decompose flat bank into
  std::vector<int> m_bank_counts;     // count per bank slot
  int m_total_bank_units;
  int m_row_pos;  // addr_vec slot index
  int m_col_pos;  // addr_vec slot index
  int m_num_rows;
  int m_num_cols;
  int m_stream_cols;  // columns accessed per row (to avoid starving probes, do not set this too high)

  // Streaming state (MESS-style alternation with NOP-based rate control)
  int m_nop_counter;  // NOP: interval between consequtive streaming requests to adjust load
  int m_curr_nop = 0;
  size_t m_streaming_idx = 0;  // Flat index: bank(fast) -> col -> row(slow)
  int m_read_ratio = 100;      // Percentage of streaming requests that are reads (0-100)
  bool m_issue_probe = false;  // Alternating flag: false=stream turn, true=probe turn

  // Streaming-only mode: disables probes entirely, fires streaming requests
  // as fast as the memory system can accept them.
  bool m_streaming_only = false;
  int m_num_streaming_requests = 0;

  bool m_pim_mode = false;
  int m_num_pim_requests = 0;
  std::string m_pim_distribution_mode = "same_bank";
  bool m_pim_same_bank = true;
  int m_pim_bank_group_size = 0;
  std::string m_pim_bank_sequence = "";
  std::string m_pim_bank_sequence_order = "frontend";
  std::vector<int> m_pim_bank_sequence_values;
  int m_pim_burst_length = 1;
  int m_pim_dependency_count = 1;
  int m_pim_row_start = 0;
  int m_pim_row_count = 1;
  int m_pim_request_type_id = -1;
  int m_pim_load_request_type_id = -1;
  int m_pim_compute_all_request_type_id = -1;
  bool m_pim_split_all_bank = false;
  int m_pim_split_phase = 0;
  bool m_pim_split_waiting_completion = false;

  // Pointer-chasing state
  int m_num_probe_requests;
  int m_warmup_cycles;
  bool m_probe_inflight = false;

  // Retry state (MESS-style: retry same request on backpressure)
  std::optional<Request> m_retry_stream_req;
  std::optional<Request> m_retry_probe_req;
  std::optional<Request> m_retry_pim_req;

  // PRNG for probe addresses and read/write selection
  std::mt19937_64 m_rng;
  std::uniform_int_distribution<int> m_ratio_dist{0, 99};
  uint64_t m_seed = 12345ULL;

  // Stats
  size_t s_streaming_sent = 0;
  int s_probes_completed = 0;
  int64_t s_total_probe_latency = 0;
  float s_avg_probe_latency = 0.0f;
  size_t s_pim_sent = 0;
  int s_pim_completed = 0;
  int64_t s_total_pim_latency = 0;
  float s_avg_pim_latency = 0.0f;

 public:
  void init() override {
    RAMULATOR_PARSE_PARAM(m_clock_ratio, unsigned int, "clock_ratio").required();
    RAMULATOR_PARSE_PARAM(m_nop_counter, int, "nop_counter").required();
    RAMULATOR_PARSE_PARAM(m_num_probe_requests, int, "num_probe_requests").required();
    RAMULATOR_PARSE_PARAM(m_streaming_only, bool, "streaming_only").default_val(false);
    RAMULATOR_PARSE_PARAM(m_num_streaming_requests, int, "num_streaming_requests").default_val(0);
    RAMULATOR_PARSE_PARAM(m_pim_mode, bool, "pim_mode").default_val(false);
    RAMULATOR_PARSE_PARAM(m_num_pim_requests, int, "num_pim_requests").default_val(0);
    RAMULATOR_PARSE_PARAM(m_pim_distribution_mode, std::string, "pim_distribution_mode").default_val("same_bank");
    RAMULATOR_PARSE_PARAM(m_pim_same_bank, bool, "pim_same_bank").default_val(true);
    RAMULATOR_PARSE_PARAM(m_pim_bank_group_size, int, "pim_bank_group_size").default_val(0);
    RAMULATOR_PARSE_PARAM(m_pim_bank_sequence, std::string, "pim_bank_sequence").default_val("");
    RAMULATOR_PARSE_PARAM(m_pim_bank_sequence_order, std::string, "pim_bank_sequence_order").default_val("frontend");
    RAMULATOR_PARSE_PARAM(m_pim_burst_length, int, "pim_burst_length").default_val(1);
    RAMULATOR_PARSE_PARAM(m_pim_dependency_count, int, "pim_dependency_count").default_val(1);
    RAMULATOR_PARSE_PARAM(m_pim_row_start, int, "pim_row_start").default_val(0);
    RAMULATOR_PARSE_PARAM(m_pim_row_count, int, "pim_row_count").default_val(1);
    RAMULATOR_PARSE_PARAM(m_pim_request_type_id, int, "pim_request_type_id").default_val(-1);
    RAMULATOR_PARSE_PARAM(m_pim_load_request_type_id, int, "pim_load_request_type_id").default_val(-1);
    RAMULATOR_PARSE_PARAM(m_pim_compute_all_request_type_id, int, "pim_compute_all_request_type_id").default_val(-1);
    RAMULATOR_PARSE_PARAM(m_pim_split_all_bank, bool, "pim_split_all_bank").default_val(false);
    RAMULATOR_PARSE_PARAM(m_stream_cols, int, "stream_cols").default_val(8);
    RAMULATOR_PARSE_PARAM(m_warmup_cycles, int, "warmup_cycles").default_val(10000);
    RAMULATOR_PARSE_PARAM(m_read_ratio, int, "read_ratio").default_val(100);
    RAMULATOR_PARSE_PARAM(m_seed, uint64_t, "seed").default_val(12345ULL);

    // Address generation recipe (injected by Python from DRAM object)
    RAMULATOR_PARSE_PARAM(m_addr_vec_size, int, "addr_vec_size").required();
    RAMULATOR_PARSE_PARAM(m_total_bank_units, int, "total_bank_units").required();
    RAMULATOR_PARSE_PARAM(m_row_pos, int, "row_pos").required();
    RAMULATOR_PARSE_PARAM(m_col_pos, int, "col_pos").required();
    RAMULATOR_PARSE_PARAM(m_num_rows, int, "num_rows").required();
    RAMULATOR_PARSE_PARAM(m_num_cols, int, "num_cols").required();

    RAMULATOR_PARSE_PARAM(m_bank_positions, std::vector<int>, "bank_positions").required();
    RAMULATOR_PARSE_PARAM(m_bank_counts, std::vector<int>, "bank_counts").required();

    m_rng.seed(m_seed);

    // Validate streaming_only + num_streaming_requests coupling
    if (m_streaming_only && m_num_streaming_requests <= 0) {
      throw std::runtime_error(
          "LatencyThroughputTrace: num_streaming_requests must be set when streaming_only=true");
    }
    if (m_pim_mode && m_num_pim_requests <= 0) {
      throw std::runtime_error(
          "LatencyThroughputTrace: num_pim_requests must be set when pim_mode=true");
    }
    if (m_pim_mode && m_pim_request_type_id < 0) {
      throw std::runtime_error(
          "LatencyThroughputTrace: pim_request_type_id must be set when pim_mode=true");
    }
    if (m_pim_split_all_bank && (m_pim_load_request_type_id < 0 || m_pim_compute_all_request_type_id < 0)) {
      throw std::runtime_error(
          "LatencyThroughputTrace: all-bank split mode requires load and compute-all request type ids");
    }
    if (m_pim_burst_length <= 0) {
      throw std::runtime_error(
          "LatencyThroughputTrace: pim_burst_length must be positive");
    }
    if (m_pim_dependency_count <= 0) {
      throw std::runtime_error(
          "LatencyThroughputTrace: pim_dependency_count must be positive");
    }
    if (m_pim_dependency_count > m_num_cols) {
      throw std::runtime_error(
          "LatencyThroughputTrace: pim_dependency_count cannot exceed num_cols");
    }
    if (m_pim_row_count <= 0) {
      throw std::runtime_error(
          "LatencyThroughputTrace: pim_row_count must be positive");
    }
    if (m_pim_row_start < 0 || m_pim_row_start >= m_num_rows) {
      throw std::runtime_error(
          "LatencyThroughputTrace: pim_row_start must be within [0, num_rows)");
    }
    if (m_pim_row_start + m_pim_row_count > m_num_rows) {
      throw std::runtime_error(
          "LatencyThroughputTrace: pim_row window must fit within num_rows");
    }

    if (m_pim_distribution_mode != "same_bank" && m_pim_distribution_mode != "bank_sequence") {
      throw std::runtime_error(
          "LatencyThroughputTrace: pim_distribution_mode must be 'same_bank' or 'bank_sequence'");
    }
    if (m_pim_bank_sequence_order != "frontend" && m_pim_bank_sequence_order != "controller") {
      throw std::runtime_error(
          "LatencyThroughputTrace: pim_bank_sequence_order must be 'frontend' or 'controller'");
    }
    if (m_pim_distribution_mode == "same_bank") {
      if (!m_pim_bank_sequence.empty()) {
        throw std::runtime_error(
            "LatencyThroughputTrace: pim_bank_sequence must be empty in same_bank mode");
      }
    } else {
      if (m_pim_bank_sequence.empty()) {
        throw std::runtime_error(
            "LatencyThroughputTrace: pim_bank_sequence must be set in bank_sequence mode");
      }
      m_pim_bank_sequence_values = parse_bank_sequence(m_pim_bank_sequence);
      if (m_pim_bank_group_size < 0 || m_pim_bank_group_size > m_total_bank_units) {
        throw std::runtime_error(
            "LatencyThroughputTrace: pim_bank_group_size must be within [0, total_bank_units]");
      }
      int bounded_bank_units = m_total_bank_units;
      if (m_pim_bank_group_size > 0) {
        bounded_bank_units = m_pim_bank_group_size;
      }
      for (int bank : m_pim_bank_sequence_values) {
        if (bank < 0 || bank >= bounded_bank_units) {
          throw std::runtime_error(
              "LatencyThroughputTrace: pim_bank_sequence entries must be within the bounded bank group");
        }
      }
    }

    m_stats.add("streaming_requests_sent", s_streaming_sent);
    m_stats.add("probe_requests_completed", s_probes_completed);
    m_stats.add("total_probe_latency", s_total_probe_latency);
    m_stats.add("avg_probe_latency", s_avg_probe_latency);
    m_stats.add("pim_requests_sent", s_pim_sent);
    m_stats.add("pim_requests_completed", s_pim_completed);
    m_stats.add("total_pim_latency", s_total_pim_latency);
    m_stats.add("avg_pim_latency", s_avg_pim_latency);

    m_logger.info(fmt::format("LatencyThroughputTrace: nop_counter={}, probes={}, warmup={}, bank_units={}",
                              m_nop_counter, m_num_probe_requests, m_warmup_cycles, m_total_bank_units));
  }

  int get_num_cores() override {
    return 2;
  }  // source_id 0=streaming, 1=probe

  void tick() override {
    m_clk++;

    if (m_pim_mode) {
      tick_pim();
      return;
    }

    if (m_streaming_only) {
      // Streaming-only mode: no NOP rate-limiting, no probes.
      // Fire streaming requests as fast as the memory system can accept them.
      tick_stream_only();
      return;
    }

    // Normal probe+stream interleaved mode
    if (tick_nop()) {
      return;
    }
    if (m_issue_probe && m_probe_inflight) {
      m_issue_probe = false;
      return;
    }
    if (m_issue_probe) {
      tick_probe();
    } else {
      tick_stream();
    }
  }

  bool is_finished() override {
    if (m_pim_mode) {
      if (m_pim_split_all_bank) {
        return s_pim_completed >= m_num_pim_requests && !m_pim_split_waiting_completion && !m_retry_pim_req;
      }
      return s_pim_completed >= m_num_pim_requests;
    }
    if (m_streaming_only) {
      return static_cast<int>(s_streaming_sent) >= m_num_streaming_requests;
    }
    return s_probes_completed >= m_num_probe_requests;
  }

  void finalize() override {
    if (s_probes_completed > 0) {
      s_avg_probe_latency = static_cast<float>(s_total_probe_latency) / s_probes_completed;
    }
    if (s_pim_completed > 0) {
      s_avg_pim_latency = static_cast<float>(s_total_pim_latency) / s_pim_completed;
    }
  }

 private:
  // Returns true if this tick should be skipped (NOP rate-limiting).
  // NOP skips only apply to stream turns: skip N-1 out of every N stream turns.
  bool tick_nop() {
    if (m_issue_probe) {
      return false;
    }
    bool is_nop = (m_nop_counter > 1 && m_curr_nop != 0);
    m_curr_nop = (m_curr_nop + 1) % m_nop_counter;
    if (is_nop) {
      m_issue_probe = !m_issue_probe;
    }
    return is_nop;
  }

  // Handle probe turn: issue a random-address read to measure latency under load.
  // On backpressure, the request is held in m_retry_probe_req for the next attempt.
  void tick_probe() {
    bool want_probe = (m_clk > m_warmup_cycles && s_probes_completed < m_num_probe_requests);
    if (!want_probe) {
      m_issue_probe = false;
      return;
    }

    if (!m_retry_probe_req) {
      Request req = make_request(random_addr_vec(), Request::Type::Read, 1);
      req.callback = [this](Request& completed) {
        s_total_probe_latency += (completed.depart - completed.arrive);
        s_probes_completed++;
        m_probe_inflight = false;
      };
      m_retry_probe_req = req;
    }
    if (m_memory_system->send(*m_retry_probe_req)) {
      m_probe_inflight = true;
      m_issue_probe = false;
      m_retry_probe_req.reset();
    }
  }

  // Streaming-only mode: issue sequential reads/writes at maximum rate.
  // No NOP rate-limiting, no probe alternation.
  void tick_stream_only() {
    if (!m_retry_stream_req) {
      int type = Request::Type::Read;
      if (m_read_ratio < 100) {
        type = (m_ratio_dist(m_rng) < m_read_ratio) ? Request::Type::Read : Request::Type::Write;
      }
      m_retry_stream_req = make_request(streaming_addr_vec(m_streaming_idx), type, 0);
    }
    if (m_memory_system->send(*m_retry_stream_req)) {
      s_streaming_sent++;
      m_streaming_idx++;
      m_retry_stream_req.reset();
    }
  }

  void tick_pim() {
    if (tick_pim_nop()) {
      return;
    }

    if (m_pim_split_all_bank && m_pim_split_waiting_completion) {
      return;
    }

    if (static_cast<int>(s_pim_sent) >= m_num_pim_requests) {
      return;
    }

    if (!m_retry_pim_req) {
      int type = m_pim_request_type_id;
      if (m_pim_split_all_bank) {
        type = (m_pim_split_phase == 0) ? m_pim_load_request_type_id : m_pim_compute_all_request_type_id;
      }
      Request req = make_request(pim_addr_vec(s_pim_sent), type, 0);
      req.callback = [this](Request& completed) {
        s_total_pim_latency += (completed.depart - completed.arrive);
        s_pim_completed++;
        if (m_pim_split_all_bank) {
          m_pim_split_phase = 1 - m_pim_split_phase;
          m_pim_split_waiting_completion = false;
        }
      };
      m_retry_pim_req = req;
    }

    if (m_memory_system->send(*m_retry_pim_req)) {
      s_pim_sent++;
      if (m_pim_split_all_bank) {
        m_pim_split_waiting_completion = true;
      }
      m_retry_pim_req.reset();
    }
  }

  bool tick_pim_nop() {
    if (m_nop_counter <= 1) {
      return false;
    }
    bool is_nop = (m_curr_nop != 0);
    m_curr_nop = (m_curr_nop + 1) % m_nop_counter;
    return is_nop;
  }

  // Handle stream turn: issue sequential accesses with configurable read/write mix.
  // On backpressure, the request is held in m_retry_stream_req for the next attempt.
  void tick_stream() {
    if (!m_retry_stream_req) {
      int type = Request::Type::Read;
      if (m_read_ratio < 100) {
        type = (m_ratio_dist(m_rng) < m_read_ratio) ? Request::Type::Read : Request::Type::Write;
      }
      m_retry_stream_req = make_request(streaming_addr_vec(m_streaming_idx), type, 0);
    }
    if (m_memory_system->send(*m_retry_stream_req)) {
      s_streaming_sent++;
      m_streaming_idx++;
      m_issue_probe = true;
      m_retry_stream_req.reset();
    }
  }

  // Build a Request from an address vector, setting the flat address for write-forwarding.
  Request make_request(const AddrVec_t& av, int type, int source_id) {
    Request req(av, type);
    int bank_flat = 0;
    for (size_t i = 0; i < m_bank_positions.size(); i++) {
      bank_flat = bank_flat * m_bank_counts[i] + av[m_bank_positions[i]];
    }
    req.addr = static_cast<Addr_t>(bank_flat * m_num_rows * m_num_cols + av[m_row_pos] * m_num_cols + av[m_col_pos]);
    req.source_id = source_id;
    req.size_bytes = m_memory_system->get_tx_bytes();
    return req;
  }

  // Streaming pattern: bank(fastest) -> column -> row(slowest).
  // Uses m_stream_cols (not full m_num_cols) to limit row-hit sequence length,
  // giving probe reads more scheduling opportunities during row transitions.
  AddrVec_t streaming_addr_vec(size_t idx) {
    AddrVec_t av(m_addr_vec_size, 0);
    int flat_bank = static_cast<int>(idx % m_total_bank_units);
    int col = static_cast<int>((idx / m_total_bank_units) % m_stream_cols);
    int row = static_cast<int>((idx / m_total_bank_units / m_stream_cols) % m_num_rows);
    decompose_bank(flat_bank, av);
    av[m_row_pos] = row;
    av[m_col_pos] = col;
    return av;
  }

  // Random address for pointer-chasing probes (targets random rows for row-buffer misses).
  AddrVec_t random_addr_vec() {
    AddrVec_t av(m_addr_vec_size, 0);
    for (size_t i = 0; i < m_bank_positions.size(); i++) {
      std::uniform_int_distribution<int> dist(0, m_bank_counts[i] - 1);
      av[m_bank_positions[i]] = dist(m_rng);
    }
    std::uniform_int_distribution<int> row_dist(0, m_num_rows - 1);
    std::uniform_int_distribution<int> col_dist(0, m_num_cols - 1);
    av[m_row_pos] = row_dist(m_rng);
    av[m_col_pos] = col_dist(m_rng);
    return av;
  }

  AddrVec_t pim_addr_vec(size_t idx) {
    AddrVec_t av(m_addr_vec_size, 0);
    int flat_bank = 0;
    size_t distribution_span = 1;
    if (m_pim_distribution_mode == "bank_sequence") {
      distribution_span = m_pim_bank_sequence_values.size();
      flat_bank = m_pim_bank_sequence_values[(idx / m_pim_burst_length) % distribution_span];
    } else if (!m_pim_same_bank && m_total_bank_units > 0) {
      int group_size = m_total_bank_units;
      if (m_pim_bank_group_size > 0 && m_pim_bank_group_size < m_total_bank_units) {
        group_size = m_pim_bank_group_size;
      }
      distribution_span = static_cast<size_t>(group_size);
      flat_bank = static_cast<int>((idx / m_pim_burst_length) % distribution_span);
    }

    int dep_ctx = 0;
    if (m_pim_dependency_count > 1) {
      dep_ctx = static_cast<int>(((idx / m_pim_burst_length) / distribution_span) % m_pim_dependency_count);
    }

    int row_offset = static_cast<int>(((idx / m_pim_burst_length) / distribution_span / m_pim_dependency_count) % m_pim_row_count);
    decompose_pim_bank(flat_bank, av);
    av[m_row_pos] = m_pim_row_start + row_offset;
    av[m_col_pos] = dep_ctx;
    return av;
  }

  void decompose_pim_bank(int flat, AddrVec_t& av) {
    if (m_pim_bank_sequence_order == "controller") {
      decompose_bank_controller_order(flat, av);
      return;
    }
    decompose_bank(flat, av);
  }

  // Mixed-radix decomposition: map flat bank index into per-slot addr_vec values.
  // Last entry in m_bank_positions cycles fastest.
  void decompose_bank(int flat, AddrVec_t& av) {
    for (int i = static_cast<int>(m_bank_positions.size()) - 1; i >= 0; i--) {
      av[m_bank_positions[i]] = flat % m_bank_counts[i];
      flat /= m_bank_counts[i];
    }
  }

  // Natural controller order follows the DRAM address-vector hierarchy instead of
  // the frontend throughput-optimized order.  The highest address-vector bank
  // position cycles slowest, and the lowest bank position before Row cycles fastest.
  void decompose_bank_controller_order(int flat, AddrVec_t& av) {
    std::vector<size_t> order(m_bank_positions.size());
    for (size_t i = 0; i < order.size(); i++) {
      order[i] = i;
    }
    std::sort(order.begin(), order.end(), [this](size_t lhs, size_t rhs) {
      return m_bank_positions[lhs] < m_bank_positions[rhs];
    });

    for (int i = static_cast<int>(order.size()) - 1; i >= 0; i--) {
      size_t bank_idx = order[i];
      av[m_bank_positions[bank_idx]] = flat % m_bank_counts[bank_idx];
      flat /= m_bank_counts[bank_idx];
    }
  }

  std::vector<int> parse_bank_sequence(const std::string& spec) {
    std::vector<int> values;
    std::stringstream ss(spec);
    std::string token;
    while (std::getline(ss, token, ',')) {
      if (token.empty()) {
        throw std::runtime_error(
            "LatencyThroughputTrace: pim_bank_sequence contains an empty entry");
      }
      values.push_back(std::stoi(token));
    }
    if (values.empty()) {
      throw std::runtime_error(
          "LatencyThroughputTrace: pim_bank_sequence must contain at least one bank id");
    }
    return values;
  }
};

}  // namespace Ramulator
