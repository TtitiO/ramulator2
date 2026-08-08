#include <cstdint>
#include <filesystem>
#include <fmt/format.h>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "ramulator/base/param.h"
#include "ramulator/frontend/i_frontend.h"

namespace Ramulator {

namespace fs = std::filesystem;

class LPDDR5PIMConcreteTrace : public IFrontEnd, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IFrontEnd, LPDDR5PIMConcreteTrace, "LPDDR5PIMConcreteTrace")

 private:
  struct OpcodeRecord {
    std::string opcode;
    AddrVec_t addr_vec;
    int request_type_id;
    int command_id;
    int repeat;
    int64_t addr_byte;
    int64_t addr_byte_stride;
    // In-memory bank interleaving fields (optional; only active when
    // bank_sequence is non-empty on a PIM_MAC record).  One compact record
    // expands into N interleaved issues at replay time with zero file growth.
    std::vector<int> bank_sequence;
    std::vector<int> bank_positions;
    std::vector<int> bank_counts;
    int dependency_count = 0;
    int row_count = 1;
    int row_start = 0;
    int column_start = 0;
    int resolved_row_offset = 0;
    int resolved_col_offset = 0;
    int interleave_depth = 4;
    int interleave_start_idx = 0;
    int bank_level = 3;
    int row_level = 4;
    int col_level = 5;
  };

  std::vector<OpcodeRecord> m_records;
  std::optional<Request> m_retry_req;

  std::string m_trace_path;
  int m_pim_compute_request_type_id = -1;
  int m_pim_load_all_request_type_id = -1;
  int m_pim_compute_all_request_type_id = -1;
  int m_cmd_sb = -1;
  int m_cmd_hab = -1;
  int m_cmd_hab_pim = -1;
  int m_addr_vec_size = 0;
  int m_address_mapping_version = 1;
  std::vector<std::string> m_level_names;
  std::vector<int> m_level_sizes;
  std::vector<int> m_address_level_sizes;
  int m_internal_prefetch_size = 0;
  int m_tx_bytes = 0;
  int m_bank_level = -1;
  int m_row_level = -1;
  int m_col_level = -1;
  uint64_t m_addressable_transactions = 0;
  uint64_t m_addressable_capacity_bytes = 0;
  int64_t m_max_trace_bytes = 1073741824;
  int m_max_records = 1000000;
  int m_max_repeat = 1000000;
  int64_t m_max_expanded_records = 1000000000;
  int m_max_inflight_requests = 1;

  size_t m_curr_record_idx = 0;
  int m_curr_repeat_idx = 0;
  int64_t m_inflight_requests = 0;

  size_t s_records_loaded = 0;
  size_t s_records_expanded = 0;
  int64_t s_opcode_requests_sent = 0;
  int64_t s_opcode_requests_completed = 0;
  size_t s_sb_records = 0;
  size_t s_hab_records = 0;
  size_t s_hab_pim_records = 0;
  size_t s_pim_bcast_records = 0;
  size_t s_pim_mac_records = 0;
  size_t s_pim_mac_ab_records = 0;
  size_t s_read_records = 0;
  size_t s_write_records = 0;

 public:
  void init() override {
    RAMULATOR_PARSE_PARAM(m_clock_ratio, unsigned int, "clock_ratio").required();
    RAMULATOR_PARSE_PARAM(m_trace_path, std::string, "path").required();
    RAMULATOR_PARSE_PARAM(m_pim_compute_request_type_id, int, "pim_compute_request_type_id").required();
    RAMULATOR_PARSE_PARAM(m_pim_load_all_request_type_id, int, "pim_load_all_request_type_id").required();
    RAMULATOR_PARSE_PARAM(m_pim_compute_all_request_type_id, int, "pim_compute_all_request_type_id").required();
    RAMULATOR_PARSE_PARAM(m_cmd_sb, int, "sb_command_id").required();
    RAMULATOR_PARSE_PARAM(m_cmd_hab, int, "hab_command_id").required();
    RAMULATOR_PARSE_PARAM(m_cmd_hab_pim, int, "hab_pim_command_id").required();
    RAMULATOR_PARSE_PARAM(m_addr_vec_size, int, "addr_vec_size").required();
    RAMULATOR_PARSE_PARAM(m_address_mapping_version, int, "address_mapping_version").required();
    RAMULATOR_PARSE_PARAM(m_level_names, std::vector<std::string>, "level_names").required();
    RAMULATOR_PARSE_PARAM(m_level_sizes, std::vector<int>, "level_sizes").required();
    RAMULATOR_PARSE_PARAM(m_address_level_sizes, std::vector<int>, "address_level_sizes").required();
    RAMULATOR_PARSE_PARAM(m_internal_prefetch_size, int, "internal_prefetch_size").required();
    RAMULATOR_PARSE_PARAM(m_tx_bytes, int, "tx_bytes").required();
    RAMULATOR_PARSE_PARAM(m_max_trace_bytes, int64_t, "max_trace_bytes").default_val(1073741824);
    RAMULATOR_PARSE_PARAM(m_max_records, int, "max_records").default_val(1000000);
    RAMULATOR_PARSE_PARAM(m_max_repeat, int, "max_repeat").default_val(1000000);
    RAMULATOR_PARSE_PARAM(m_max_expanded_records, int64_t, "max_expanded_records").default_val(1000000000);
    RAMULATOR_PARSE_PARAM(m_max_inflight_requests, int, "max_inflight_requests").default_val(1);

    validate_address_layout();
    if (m_max_trace_bytes <= 0 || m_max_records <= 0 || m_max_repeat <= 0 || m_max_expanded_records <= 0) {
      throw std::runtime_error("LPDDR5PIMConcreteTrace: max trace limits must be positive");
    }
    if (m_max_inflight_requests <= 0) {
      throw std::runtime_error("LPDDR5PIMConcreteTrace: max_inflight_requests must be positive");
    }
    load_trace(m_trace_path);
    validate_sequence();

    m_stats.add("records_loaded", s_records_loaded);
    m_stats.add("records_expanded", s_records_expanded);
    m_stats.add("opcode_requests_sent", s_opcode_requests_sent);
    m_stats.add("opcode_requests_completed", s_opcode_requests_completed);
    m_stats.add("sb_records", s_sb_records);
    m_stats.add("hab_records", s_hab_records);
    m_stats.add("hab_pim_records", s_hab_pim_records);
    m_stats.add("pim_bcast_records", s_pim_bcast_records);
    m_stats.add("pim_mac_records", s_pim_mac_records);
    m_stats.add("pim_mac_ab_records", s_pim_mac_ab_records);
    m_stats.add("read_records", s_read_records);
    m_stats.add("write_records", s_write_records);
  }

  int get_num_cores() override { return 1; }

  void tick() override {
    m_clk++;
    if (m_retry_req) {
      try_send_retry();
      return;
    }
    if (m_curr_record_idx >= m_records.size()) {
      return;
    }

    const OpcodeRecord& record = m_records[m_curr_record_idx];

    // Per-record inflight cap.  Only a bank-rotating per-bank PIM_MAC record
    // (non-empty bank_sequence) may have multiple issues outstanding at once:
    // consecutive issues target different banks (bank_sequence rotation) and are
    // genuinely independent, so non-interfering banks (different shared-block groups,
    // no shared dependency) execute in parallel.  The controller still serializes
    // same-bank and same-shared-block ops, so this exposes real inter-bank
    // parallelism without overstating it.
    //
    // Everything else (all-bank PIM_MAC_AB, mode switches SB/HAB/HAB_PIM/BCAST,
    // host READ/WRITE, and plain non-rotating PIM_MAC) stays strictly serial
    // (cap 1).  This preserves the HAB→BCAST→HAB_PIM→PIM_MAC_AB→SB group
    // ordering required by the all-bank FFN path — a wide window there would
    // interleave records from different groups and corrupt the rank-mode state.
    const bool parallelizable =
        (record.opcode == "PIM_MAC") && !record.bank_sequence.empty();
    const int64_t record_inflight_cap =
        parallelizable ? m_max_inflight_requests : 1;
    if (m_inflight_requests >= record_inflight_cap) {
      return;
    }

    if (m_curr_repeat_idx >= record.repeat) {
      m_curr_record_idx++;
      m_curr_repeat_idx = 0;
      return;
    }

    m_retry_req = make_request(record);
    try_send_retry();
  }

  bool is_finished() override {
    return m_curr_record_idx >= m_records.size() && !m_retry_req && m_inflight_requests == 0;
  }

 private:
  void validate_address_layout() {
    if (m_address_mapping_version != 1) {
      throw std::runtime_error(fmt::format(
          "LPDDR5PIMConcreteTrace: unsupported address_mapping_version {}; expected 1",
          m_address_mapping_version));
    }
    if (m_addr_vec_size <= 0) {
      throw std::runtime_error("LPDDR5PIMConcreteTrace: addr_vec_size must be positive");
    }
    if (static_cast<int>(m_level_names.size()) != m_addr_vec_size ||
        static_cast<int>(m_level_sizes.size()) != m_addr_vec_size ||
        static_cast<int>(m_address_level_sizes.size()) != m_addr_vec_size) {
      throw std::runtime_error(
          "LPDDR5PIMConcreteTrace: level_names, level_sizes, and address_level_sizes must match addr_vec_size");
    }
    if (m_level_names.empty() || m_level_names.front() != "Channel") {
      throw std::runtime_error(
          "LPDDR5PIMConcreteTrace: address layout must begin with Channel");
    }
    if (m_level_names.back() != "Column") {
      throw std::runtime_error(
          "LPDDR5PIMConcreteTrace: address layout must end with Column");
    }
    for (size_t index = 0; index < m_level_sizes.size(); index++) {
      if (m_level_sizes[index] <= 0 || m_address_level_sizes[index] <= 0 ||
          m_address_level_sizes[index] > m_level_sizes[index]) {
        throw std::runtime_error(fmt::format(
            "LPDDR5PIMConcreteTrace: level '{}' sizes must satisfy 0 < address size <= physical size",
            m_level_names[index]));
      }
      if (m_level_names[index] == "Bank") {
        m_bank_level = static_cast<int>(index);
      }
      if (m_level_names[index] == "Row") {
        m_row_level = static_cast<int>(index);
      }
      if (m_level_names[index] == "Column") {
        m_col_level = static_cast<int>(index);
      }
    }
    if (m_bank_level < 0 || m_row_level < 0 || m_col_level != m_addr_vec_size - 1) {
      throw std::runtime_error(
          "LPDDR5PIMConcreteTrace: address layout must contain Bank and Row, with Column final");
    }
    if (m_internal_prefetch_size <= 0 ||
        m_level_sizes[m_col_level] % m_internal_prefetch_size != 0 ||
        m_address_level_sizes[m_col_level] !=
            m_level_sizes[m_col_level] / m_internal_prefetch_size) {
      throw std::runtime_error(
          "LPDDR5PIMConcreteTrace: address-level Column size must equal physical Column size / internal_prefetch_size");
    }
    for (int index = 0; index < m_addr_vec_size - 1; index++) {
      if (m_address_level_sizes[index] != m_level_sizes[index]) {
        throw std::runtime_error(fmt::format(
            "LPDDR5PIMConcreteTrace: address-level size for '{}' must equal its physical size",
            m_level_names[index]));
      }
    }
    if (m_tx_bytes <= 0) {
      throw std::runtime_error("LPDDR5PIMConcreteTrace: tx_bytes must be positive");
    }

    uint64_t transactions = 1;
    for (int index = 0; index < m_addr_vec_size; index++) {
      uint64_t radix = static_cast<uint64_t>(m_address_level_sizes[index]);
      if (radix == 0 || transactions > std::numeric_limits<uint64_t>::max() / radix) {
        throw std::runtime_error(
            "LPDDR5PIMConcreteTrace: address layout transaction capacity overflows uint64_t");
      }
      transactions *= radix;
    }
    if (transactions > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(m_tx_bytes)) {
      throw std::runtime_error(
          "LPDDR5PIMConcreteTrace: address layout byte capacity overflows uint64_t");
    }
    m_addressable_transactions = transactions;
    m_addressable_capacity_bytes = transactions * static_cast<uint64_t>(m_tx_bytes);
  }

  void validate_addr_vec(const AddrVec_t& addr_vec, const std::string& path, int line_num) const {
    if (static_cast<int>(addr_vec.size()) != m_addr_vec_size) {
      throw std::runtime_error(fmt::format(
          "LPDDR5PIMConcreteTrace: {} line {} addr_vec size {} must equal addr_vec_size {}",
          path, line_num, addr_vec.size(), m_addr_vec_size));
    }
    for (int index = 0; index < m_addr_vec_size; index++) {
      int value = addr_vec[index];
      if (value < 0 || value >= m_level_sizes[index]) {
        throw std::runtime_error(fmt::format(
            "LPDDR5PIMConcreteTrace: {} line {} addr_vec[{}] ({})={} must be in [0, {})",
            path, line_num, index, m_level_names[index], value, m_level_sizes[index]));
      }
    }
  }

  void load_trace(const std::string& file_path_str) {
    fs::path trace_path(file_path_str);
    if (!fs::exists(trace_path)) {
      throw std::runtime_error(fmt::format("LPDDR5PIMConcreteTrace: trace {} does not exist", file_path_str));
    }
    if (fs::file_size(trace_path) > static_cast<uint64_t>(m_max_trace_bytes)) {
      throw std::runtime_error(fmt::format("LPDDR5PIMConcreteTrace: trace {} exceeds max_trace_bytes {}", file_path_str, m_max_trace_bytes));
    }
    std::ifstream trace_file(trace_path);
    if (!trace_file.is_open()) {
      throw std::runtime_error(fmt::format("LPDDR5PIMConcreteTrace: trace {} cannot be opened", file_path_str));
    }

    std::string line;
    int line_num = 0;
    bool header_parsed = false;
    while (std::getline(trace_file, line)) {
      line_num++;
      if (line.empty()) {
        continue;
      }
      YAML::Node node;
      try {
        node = YAML::Load(line);
      } catch (const YAML::Exception& exc) {
        throw std::runtime_error(fmt::format("LPDDR5PIMConcreteTrace: {} line {} parse error: {}", file_path_str, line_num, exc.what()));
      }
      // The first non-empty line is the v0.2 header envelope: it carries the
      // file-level constants (schema_version + provenance) asserted once here
      // instead of being repeated on every record.
      if (!header_parsed) {
        parse_header(node, file_path_str, line_num);
        header_parsed = true;
        continue;
      }
      if (static_cast<int>(m_records.size()) >= m_max_records) {
        throw std::runtime_error(fmt::format("LPDDR5PIMConcreteTrace: {} exceeds max_records {}", file_path_str, m_max_records));
      }
      m_records.push_back(parse_record(node, file_path_str, line_num));
      s_records_loaded++;
      s_records_expanded += m_records.back().repeat;
      if (static_cast<int64_t>(s_records_expanded) > m_max_expanded_records) {
        throw std::runtime_error(fmt::format("LPDDR5PIMConcreteTrace: {} exceeds max_expanded_records {}", file_path_str, m_max_expanded_records));
      }
    }
    if (!header_parsed) {
      throw std::runtime_error(fmt::format("LPDDR5PIMConcreteTrace: {} is missing the v0.2 header line", file_path_str));
    }
  }

  void parse_header(const YAML::Node& node, const std::string& path, int line_num) {
    if (node["opcode"]) {
      throw std::runtime_error(fmt::format("LPDDR5PIMConcreteTrace: {} line {} expected a v0.2 header (schema_version) but found a record; regenerate the trace as lpddr5-pim-opcode-v0.2", path, line_num));
    }
    require_string(node, "schema_version", path, line_num, "lpddr5-pim-opcode-v0.2");
  }

  OpcodeRecord parse_record(const YAML::Node& node, const std::string& path, int line_num) {
    // v0.2 records carry only what varies; file-level constants live in the
    // header.  Reject stale v0.1 records that still embed those fields.
    if (node["schema_version"] || node["provenance"]) {
      throw std::runtime_error(fmt::format("LPDDR5PIMConcreteTrace: {} line {} record must not carry schema_version/provenance; regenerate the trace as lpddr5-pim-opcode-v0.2", path, line_num));
    }
    const std::string opcode = require_string(node, "opcode", path, line_num);
    const int repeat = require_int(node, "repeat", path, line_num);
    if (repeat <= 0) {
      throw std::runtime_error(fmt::format("LPDDR5PIMConcreteTrace: {} line {} repeat must be positive", path, line_num));
    }
    if (repeat > m_max_repeat) {
      throw std::runtime_error(fmt::format("LPDDR5PIMConcreteTrace: {} line {} repeat exceeds max_repeat {}", path, line_num, m_max_repeat));
    }
    AddrVec_t addr_vec = require_addr_vec(node, path, line_num);
    int64_t addr_byte = -1;
    int64_t addr_byte_stride = 0;
    if (node["addr_byte"]) {
      addr_byte = node["addr_byte"].as<int64_t>();
      if (addr_byte < 0) {
        throw std::runtime_error(fmt::format("LPDDR5PIMConcreteTrace: {} line {} addr_byte must be non-negative", path, line_num));
      }
    }
    if (node["addr_byte_stride"]) {
      addr_byte_stride = node["addr_byte_stride"].as<int64_t>();
      if (addr_byte_stride <= 0) {
        throw std::runtime_error(fmt::format("LPDDR5PIMConcreteTrace: {} line {} addr_byte_stride must be positive", path, line_num));
      }
      if (addr_byte < 0) {
        throw std::runtime_error(fmt::format("LPDDR5PIMConcreteTrace: {} line {} addr_byte_stride requires addr_byte", path, line_num));
      }
    }

    int request_type_id = -1;
    int command_id = -1;
    if (opcode == "READ") {
      request_type_id = Request::Type::Read;
      s_read_records++;
    } else if (opcode == "WRITE") {
      request_type_id = Request::Type::Write;
      s_write_records++;
    } else if (opcode == "SB") {
      command_id = m_cmd_sb;
      s_sb_records++;
    } else if (opcode == "HAB") {
      command_id = m_cmd_hab;
      s_hab_records++;
    } else if (opcode == "HAB_PIM") {
      command_id = m_cmd_hab_pim;
      s_hab_pim_records++;
    } else if (opcode == "PIM_BCAST") {
      request_type_id = m_pim_load_all_request_type_id;
      s_pim_bcast_records++;
    } else if (opcode == "PIM_MAC") {
      request_type_id = m_pim_compute_request_type_id;
      s_pim_mac_records++;
    } else if (opcode == "PIM_MAC_AB") {
      request_type_id = m_pim_compute_all_request_type_id;
      s_pim_mac_ab_records++;
    } else {
      throw std::runtime_error(fmt::format("LPDDR5PIMConcreteTrace: {} line {} unsupported opcode '{}'", path, line_num, opcode));
    }
    if (addr_byte >= 0 && opcode != "READ" && opcode != "WRITE") {
      throw std::runtime_error(fmt::format("LPDDR5PIMConcreteTrace: {} line {} addr_byte is only valid for READ/WRITE", path, line_num));
    }
    if ((opcode == "READ" || opcode == "WRITE") && addr_byte < 0) {
      throw std::runtime_error(fmt::format("LPDDR5PIMConcreteTrace: {} line {} READ/WRITE records require addr_byte", path, line_num));
    }
    if (opcode == "READ" || opcode == "WRITE") {
      AddrVec_t expected_addr_vec = addr_vec_from_byte_address(addr_byte);
      if (addr_vec != expected_addr_vec) {
        throw std::runtime_error(fmt::format("LPDDR5PIMConcreteTrace: {} line {} READ/WRITE addr_vec must match decomposed addr_byte", path, line_num));
      }
      if (addr_byte_stride > 0) {
        if (repeat - 1 > (std::numeric_limits<int64_t>::max() - addr_byte) / addr_byte_stride) {
          throw std::runtime_error(fmt::format("LPDDR5PIMConcreteTrace: {} line {} repeated host byte address overflows int64_t", path, line_num));
        }
        addr_vec_from_byte_address(addr_byte + static_cast<int64_t>(repeat - 1) * addr_byte_stride);
      }
    }

    // In-memory bank interleaving fields (PIM_MAC only; compact expansion).
    std::vector<int> bank_sequence;
    std::vector<int> bank_positions;
    std::vector<int> bank_counts;
    int dependency_count = 0;
    int row_count = 1;
    int row_start = 0;
    int column_start = 0;
    int resolved_row_offset = 0;
    int resolved_col_offset = 0;
    int interleave_depth = 4;
    int interleave_start_idx = 0;
    int bank_level = m_bank_level;
    int row_level = m_row_level;
    int col_level = m_col_level;

    if (node["interleave_depth"] || node["bank_sequence"]) {
      if (opcode != "PIM_MAC") {
        throw std::runtime_error(fmt::format("LPDDR5PIMConcreteTrace: {} line {} bank interleaving fields are only valid for PIM_MAC", path, line_num));
      }
      if (!node["bank_sequence"] || !node["bank_sequence"].IsSequence()) {
        throw std::runtime_error(fmt::format("LPDDR5PIMConcreteTrace: {} line {} PIM_MAC interleaving requires a non-empty bank_sequence", path, line_num));
      }
      for (const auto& entry : node["bank_sequence"]) {
        int bank = entry.as<int>();
        if (bank < 0) {
          throw std::runtime_error(fmt::format("LPDDR5PIMConcreteTrace: {} line {} bank_sequence entries must be non-negative", path, line_num));
        }
        bank_sequence.push_back(bank);
      }
      if (bank_sequence.empty()) {
        throw std::runtime_error(fmt::format("LPDDR5PIMConcreteTrace: {} line {} bank_sequence must not be empty", path, line_num));
      }

      if (node["interleave_depth"]) {
        interleave_depth = node["interleave_depth"].as<int>();
        if (interleave_depth < 1) {
          throw std::runtime_error(fmt::format("LPDDR5PIMConcreteTrace: {} line {} interleave_depth must be >= 1", path, line_num));
        }
      }

      dependency_count = require_int(node, "dependency_count", path, line_num);
      if (dependency_count < 1) {
        throw std::runtime_error(fmt::format("LPDDR5PIMConcreteTrace: {} line {} dependency_count must be >= 1", path, line_num));
      }

      if (node["row_count"]) {
        row_count = node["row_count"].as<int>();
        if (row_count < 1) {
          throw std::runtime_error(fmt::format("LPDDR5PIMConcreteTrace: {} line {} row_count must be >= 1", path, line_num));
        }
      }
      if (node["row_start"]) {
        row_start = node["row_start"].as<int>();
      }
      if (node["column_start"]) {
        column_start = node["column_start"].as<int>();
      }
      if (node["resolved_row_offset"]) {
        resolved_row_offset = node["resolved_row_offset"].as<int>();
        if (resolved_row_offset < 0 || resolved_row_offset >= row_count) {
          throw std::runtime_error(fmt::format("LPDDR5PIMConcreteTrace: {} line {} resolved_row_offset must be in [0, row_count)", path, line_num));
        }
      }
      if (node["resolved_col_offset"]) {
        resolved_col_offset = node["resolved_col_offset"].as<int>();
        if (resolved_col_offset < 0 || resolved_col_offset >= dependency_count) {
          throw std::runtime_error(fmt::format("LPDDR5PIMConcreteTrace: {} line {} resolved_col_offset must be in [0, dependency_count)", path, line_num));
        }
      }

      if (node["interleave_start_idx"]) {
        interleave_start_idx = node["interleave_start_idx"].as<int>();
        if (interleave_start_idx < 0) {
          throw std::runtime_error(fmt::format("LPDDR5PIMConcreteTrace: {} line {} interleave_start_idx must be non-negative", path, line_num));
        }
      }

      if (node["row_level"]) {
        row_level = node["row_level"].as<int>();
      }
      if (node["col_level"]) {
        col_level = node["col_level"].as<int>();
      }
      if (node["bank_level"]) {
        bank_level = node["bank_level"].as<int>();
      }

      if (node["bank_positions"] || node["bank_counts"]) {
        if (!node["bank_positions"] || !node["bank_counts"]) {
          throw std::runtime_error(fmt::format("LPDDR5PIMConcreteTrace: {} line {} bank_positions and bank_counts must be provided together", path, line_num));
        }
        bank_positions = node["bank_positions"].as<std::vector<int>>();
        bank_counts = node["bank_counts"].as<std::vector<int>>();
        if (bank_positions.size() != bank_counts.size() || bank_positions.empty()) {
          throw std::runtime_error(fmt::format("LPDDR5PIMConcreteTrace: {} line {} bank_positions and bank_counts must be non-empty lists of equal length", path, line_num));
        }
        std::vector<bool> seen(m_addr_vec_size, false);
        uint64_t total_banks = 1;
        for (size_t index = 0; index < bank_positions.size(); index++) {
          int pos = bank_positions[index];
          int cnt = bank_counts[index];
          if (pos < 0 || pos >= m_addr_vec_size) {
            throw std::runtime_error(fmt::format("LPDDR5PIMConcreteTrace: {} line {} bank_positions entries must fit within addr_vec_size", path, line_num));
          }
          if (seen[pos] || pos == row_level || pos == col_level) {
            throw std::runtime_error(fmt::format("LPDDR5PIMConcreteTrace: {} line {} bank_positions entries must be unique and must not overlap row_level/col_level", path, line_num));
          }
          seen[pos] = true;
          if (cnt <= 0 || cnt != m_level_sizes[pos]) {
            throw std::runtime_error(fmt::format(
                "LPDDR5PIMConcreteTrace: {} line {} bank_counts[{}]={} must equal configured level '{}' size {}",
                path, line_num, index, cnt, m_level_names[pos], m_level_sizes[pos]));
          }
          if (total_banks > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(cnt)) {
            throw std::runtime_error(fmt::format("LPDDR5PIMConcreteTrace: {} line {} bank count product overflows", path, line_num));
          }
          total_banks *= static_cast<uint64_t>(cnt);
        }
        for (int bank : bank_sequence) {
          if (static_cast<uint64_t>(bank) >= total_banks) {
            throw std::runtime_error(fmt::format(
                "LPDDR5PIMConcreteTrace: {} line {} bank_sequence entry {} must be in [0, {})",
                path, line_num, bank, total_banks));
          }
        }
      } else {
        if (bank_level < 0 || bank_level >= m_addr_vec_size || bank_level != m_bank_level) {
          throw std::runtime_error(fmt::format("LPDDR5PIMConcreteTrace: {} line {} bank_level must refer to the resolved Bank level", path, line_num));
        }
        for (int bank : bank_sequence) {
          if (bank >= m_level_sizes[bank_level]) {
            throw std::runtime_error(fmt::format(
                "LPDDR5PIMConcreteTrace: {} line {} bank_sequence entry {} must fit configured level '{}' size {}",
                path, line_num, bank, m_level_names[bank_level], m_level_sizes[bank_level]));
          }
        }
      }

      if (row_level < 0 || row_level >= m_addr_vec_size || col_level < 0 || col_level >= m_addr_vec_size ||
          row_level != m_row_level || col_level != m_col_level) {
        throw std::runtime_error(fmt::format("LPDDR5PIMConcreteTrace: {} line {} row_level/col_level must refer to the resolved Row/Column levels", path, line_num));
      }
      if (row_start < 0 || row_start >= m_level_sizes[row_level] ||
          row_count > m_level_sizes[row_level] - row_start) {
        throw std::runtime_error(fmt::format(
            "LPDDR5PIMConcreteTrace: {} line {} row range [{}, {}) exceeds configured level '{}' size {}",
            path, line_num, row_start, row_start + row_count, m_level_names[row_level], m_level_sizes[row_level]));
      }
      if (column_start < 0 || column_start >= m_level_sizes[col_level] ||
          dependency_count > m_level_sizes[col_level] - column_start) {
        throw std::runtime_error(fmt::format(
            "LPDDR5PIMConcreteTrace: {} line {} column range [{}, {}) exceeds configured level '{}' size {}",
            path, line_num, column_start, column_start + dependency_count, m_level_names[col_level], m_level_sizes[col_level]));
      }
    }

    validate_addr_vec(addr_vec, path, line_num);

    return OpcodeRecord{
        .opcode = opcode,
        .addr_vec = std::move(addr_vec),
        .request_type_id = request_type_id,
        .command_id = command_id,
        .repeat = repeat,
        .addr_byte = addr_byte,
        .addr_byte_stride = addr_byte_stride,
        .bank_sequence = std::move(bank_sequence),
        .bank_positions = std::move(bank_positions),
        .bank_counts = std::move(bank_counts),
        .dependency_count = dependency_count,
        .row_count = row_count,
        .row_start = row_start,
        .column_start = column_start,
        .resolved_row_offset = resolved_row_offset,
        .resolved_col_offset = resolved_col_offset,
        .interleave_depth = interleave_depth,
        .interleave_start_idx = interleave_start_idx,
        .bank_level = bank_level,
        .row_level = row_level,
        .col_level = col_level,
    };
  }

  void validate_sequence() const {
    enum class Mode { SB, HAB, HAB_PIM };
    Mode mode = Mode::SB;
    bool saw_bcast_since_hab = false;
    for (size_t i = 0; i < m_records.size(); i++) {
      const std::string& opcode = m_records[i].opcode;
      if (opcode == "READ" || opcode == "WRITE") {
        if (mode != Mode::SB) {
          throw std::runtime_error(fmt::format("LPDDR5PIMConcreteTrace: record {} {} requires SB mode", i, opcode));
        }
        continue;
      }
      if (opcode == "SB") {
        mode = Mode::SB;
        saw_bcast_since_hab = false;
      } else if (opcode == "HAB") {
        mode = Mode::HAB;
        saw_bcast_since_hab = false;
      } else if (opcode == "HAB_PIM") {
        if (!saw_bcast_since_hab) {
          throw std::runtime_error(fmt::format("LPDDR5PIMConcreteTrace: record {} HAB_PIM requires a preceding PIM_BCAST in HAB mode", i));
        }
        mode = Mode::HAB_PIM;
      } else if (opcode == "PIM_BCAST") {
        if (mode != Mode::HAB) {
          throw std::runtime_error(fmt::format("LPDDR5PIMConcreteTrace: record {} PIM_BCAST requires HAB mode", i));
        }
        saw_bcast_since_hab = true;
      } else if (opcode == "PIM_MAC_AB") {
        if (mode != Mode::HAB_PIM || !saw_bcast_since_hab) {
          throw std::runtime_error(fmt::format("LPDDR5PIMConcreteTrace: record {} PIM_MAC_AB requires HAB_PIM mode after PIM_BCAST", i));
        }
      } else if (opcode == "PIM_MAC") {
        if (mode != Mode::SB) {
          throw std::runtime_error(fmt::format("LPDDR5PIMConcreteTrace: record {} PIM_MAC requires SB mode", i));
        }
      }
    }
  }

  Request make_request(const OpcodeRecord& record) {
    Request req;
    AddrVec_t addr_vec = record.addr_vec;
    int64_t host_addr = -1;
    if ((record.opcode == "READ" || record.opcode == "WRITE") && record.addr_byte >= 0) {
      if (record.addr_byte_stride > 0 && m_curr_repeat_idx > (std::numeric_limits<int64_t>::max() - record.addr_byte) / record.addr_byte_stride) {
        throw std::runtime_error("LPDDR5PIMConcreteTrace: repeated host byte address overflows int64_t");
      }
      host_addr = record.addr_byte + static_cast<int64_t>(m_curr_repeat_idx) * record.addr_byte_stride;
      addr_vec = addr_vec_from_byte_address(host_addr);
    }

    // In-memory bank interleaving: expand one compact PIM_MAC record into N
    // interleaved issues.  Mirror the validated synthetic frontend's
    // pim_addr_vec(record, idx) rotation, with interleave_depth > 1 keeping
    // D ops on the same bank before switching.
    if ((record.opcode == "PIM_MAC") && !record.bank_sequence.empty()) {
      const int idx = record.interleave_start_idx + m_curr_repeat_idx;
      const int D = record.interleave_depth;
      const int span = static_cast<int>(record.bank_sequence.size());
      const int group = idx / D;
      const int flat_bank = record.bank_sequence[group % span];
      const int cycle = group / span;

      const int col = record.column_start
          + ((record.resolved_col_offset + cycle) % record.dependency_count);
      const int row = record.row_start
          + ((record.resolved_row_offset + cycle / record.dependency_count) % record.row_count);

      if (record.col_level < 0 || record.col_level >= m_addr_vec_size ||
          record.row_level < 0 || record.row_level >= m_addr_vec_size) {
        throw std::runtime_error("LPDDR5PIMConcreteTrace: interleaving row/column level is outside the resolved hierarchy");
      }
      if (col < 0 || col >= m_level_sizes[record.col_level] ||
          row < 0 || row >= m_level_sizes[record.row_level]) {
        throw std::runtime_error(fmt::format(
            "LPDDR5PIMConcreteTrace: interleaved row/column coordinate out of range (row {} / {}, column {} / {})",
            row, m_level_sizes[record.row_level], col, m_level_sizes[record.col_level]));
      }
      addr_vec[record.col_level] = col;
      addr_vec[record.row_level] = row;

      if (!record.bank_positions.empty()) {
        int remaining = flat_bank;
        for (int i = static_cast<int>(record.bank_positions.size()) - 1; i >= 0; i--) {
          addr_vec[record.bank_positions[i]] = remaining % record.bank_counts[i];
          remaining /= record.bank_counts[i];
        }
      } else {
        if (record.bank_level < 0 || record.bank_level >= m_addr_vec_size ||
            flat_bank >= m_level_sizes[record.bank_level]) {
          throw std::runtime_error("LPDDR5PIMConcreteTrace: interleaved bank coordinate is outside the resolved hierarchy");
        }
        addr_vec[record.bank_level] = flat_bank;
      }
      for (int index = 0; index < m_addr_vec_size; index++) {
        if (addr_vec[index] < 0 || addr_vec[index] >= m_level_sizes[index]) {
          throw std::runtime_error(fmt::format(
              "LPDDR5PIMConcreteTrace: interleaved addr_vec[{}] ({})={} must be in [0, {})",
              index, m_level_names[index], addr_vec[index], m_level_sizes[index]));
        }
      }
    }

    if (record.command_id >= 0) {
      req = Request(addr_vec, Request::Cmd, record.command_id);
    } else {
      req = Request(addr_vec, record.request_type_id);
    }
    req.source_id = 0;
    req.size_bytes = m_memory_system->get_tx_bytes();
    req.addr = host_addr >= 0 ? static_cast<Addr_t>(host_addr) : flatten_addr(addr_vec);
    req.callback = [this](Request&) {
      m_inflight_requests--;
      s_opcode_requests_completed++;
    };
    return req;
  }

  void try_send_retry() {
    if (!m_retry_req) {
      return;
    }
    m_inflight_requests++;
    bool accepted = false;
    try {
      accepted = m_memory_system->send(*m_retry_req);
    } catch (...) {
      m_inflight_requests--;
      throw;
    }
    if (!accepted) {
      m_inflight_requests--;
      return;
    }
    m_retry_req.reset();
    m_curr_repeat_idx++;
    s_opcode_requests_sent++;
  }

  Addr_t flatten_addr(const AddrVec_t& av) const {
    // Synthetic, deterministic flattening for trace bookkeeping. Scheduling and
    // command legality use addr_vec directly; this is not the structured
    // frontend's bank/row/column physical flattening model.
    Addr_t result = 0;
    for (int value : av) {
      result = result * 4096 + static_cast<Addr_t>(value + 1);
    }
    return result;
  }

  AddrVec_t addr_vec_from_byte_address(int64_t address) const {
    if (address < 0) {
      throw std::runtime_error("LPDDR5PIMConcreteTrace: host byte address must be non-negative");
    }
    const uint64_t byte_address = static_cast<uint64_t>(address);
    if (byte_address >= m_addressable_capacity_bytes) {
      throw std::runtime_error(fmt::format(
          "LPDDR5PIMConcreteTrace: host byte address {} exceeds configured addressable capacity {} bytes",
          address, m_addressable_capacity_bytes));
    }
    uint64_t transaction = byte_address / static_cast<uint64_t>(m_tx_bytes);
    AddrVec_t addr_vec(m_addr_vec_size, 0);
    for (int index = m_addr_vec_size - 1; index >= 0; index--) {
      uint64_t radix = static_cast<uint64_t>(m_address_level_sizes[index]);
      addr_vec[index] = static_cast<int>(transaction % radix);
      transaction /= radix;
    }
    if (transaction != 0) {
      throw std::runtime_error(fmt::format(
          "LPDDR5PIMConcreteTrace: host byte address {} exceeds configured addressable capacity {} bytes",
          address, m_addressable_capacity_bytes));
    }
    return addr_vec;
  }

  static void require_present(const YAML::Node& node, const std::string& key, const std::string& path, int line_num) {
    if (!node[key]) {
      throw std::runtime_error(fmt::format("LPDDR5PIMConcreteTrace: {} line {} missing required field '{}'", path, line_num, key));
    }
  }

  static std::string require_string(const YAML::Node& node, const std::string& key, const std::string& path, int line_num, const std::string& exact = "") {
    require_present(node, key, path, line_num);
    std::string value = node[key].as<std::string>();
    if (!exact.empty() && value != exact) {
      throw std::runtime_error(fmt::format("LPDDR5PIMConcreteTrace: {} line {} field '{}' must equal '{}'", path, line_num, key, exact));
    }
    return value;
  }

  static int require_int(const YAML::Node& node, const std::string& key, const std::string& path, int line_num) {
    require_present(node, key, path, line_num);
    return node[key].as<int>();
  }

  AddrVec_t require_addr_vec(const YAML::Node& node, const std::string& path, int line_num) const {
    require_present(node, "addr_vec", path, line_num);
    if (!node["addr_vec"].IsSequence()) {
      throw std::runtime_error(fmt::format("LPDDR5PIMConcreteTrace: {} line {} addr_vec must be a sequence", path, line_num));
    }
    AddrVec_t addr_vec = node["addr_vec"].as<std::vector<int>>();
    if (static_cast<int>(addr_vec.size()) != m_addr_vec_size) {
      throw std::runtime_error(fmt::format("LPDDR5PIMConcreteTrace: {} line {} addr_vec size must equal addr_vec_size", path, line_num));
    }
    validate_addr_vec(addr_vec, path, line_num);
    return addr_vec;
  }
};

}  // namespace Ramulator
