#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fmt/format.h>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "ramulator/base/param.h"
#include "ramulator/frontend/i_frontend.h"

namespace Ramulator {

namespace fs = std::filesystem;

class StructuredWorkloadSurrogateTrace : public IFrontEnd, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(
      IFrontEnd,
      StructuredWorkloadSurrogateTrace,
      "StructuredWorkloadSurrogateTrace")

 private:
  struct HostRecord {
    bool is_write;
    Addr_t base_byte;
    Addr_t stride_bytes;
    int count;
  };

  struct PIMRecord {
    int request_type_id;
    bool all_bank;
    int num_requests;
    std::vector<int> bank_sequence;
    int burst_length;
    int dependency_count;
    int dependency_offset;
    int row_start;
    int row_count;
    int row_offset;
    int column_start;
    int column_offset;
    bool controller_bank_order;
  };

  struct BarrierRecord {};
  struct DrainRecord {};

  using Record = std::variant<HostRecord, PIMRecord, BarrierRecord, DrainRecord>;

  std::vector<Record> m_records;
  std::optional<Request> m_retry_req;

  std::string m_trace_path;

  int m_pim_request_type_id = -1;
  int m_pim_load_request_type_id = -1;
  int m_pim_compute_all_request_type_id = -1;

  int m_addr_vec_size = 0;
  std::vector<int> m_bank_positions;
  std::vector<int> m_bank_counts;
  int m_total_bank_units = 0;
  int m_row_pos = 0;
  int m_col_pos = 0;
  int m_num_rows = 0;
  int m_num_cols = 0;

  size_t m_curr_record_idx = 0;
  int m_curr_record_req_idx = 0;
  int64_t m_inflight_requests = 0;

  size_t s_records_loaded = 0;
  size_t s_host_records = 0;
  size_t s_pim_records = 0;
  size_t s_barrier_records = 0;
  size_t s_drain_records = 0;
  int64_t s_host_requests_sent = 0;
  int64_t s_host_requests_completed = 0;
  int64_t s_pim_requests_sent = 0;
  int64_t s_pim_requests_completed = 0;
  int64_t s_barriers_retired = 0;
  int64_t s_drains_retired = 0;

 public:
  void init() override {
    RAMULATOR_PARSE_PARAM(m_clock_ratio, unsigned int, "clock_ratio").required();
    RAMULATOR_PARSE_PARAM(m_trace_path, std::string, "path").required();
    RAMULATOR_PARSE_PARAM(m_pim_request_type_id, int, "pim_request_type_id").required();
    RAMULATOR_PARSE_PARAM(m_pim_load_request_type_id, int, "pim_load_request_type_id").default_val(-1);
    RAMULATOR_PARSE_PARAM(m_pim_compute_all_request_type_id, int, "pim_compute_all_request_type_id").default_val(-1);
    RAMULATOR_PARSE_PARAM(m_addr_vec_size, int, "addr_vec_size").required();
    RAMULATOR_PARSE_PARAM(m_total_bank_units, int, "total_bank_units").required();
    RAMULATOR_PARSE_PARAM(m_row_pos, int, "row_pos").required();
    RAMULATOR_PARSE_PARAM(m_col_pos, int, "col_pos").required();
    RAMULATOR_PARSE_PARAM(m_num_rows, int, "num_rows").required();
    RAMULATOR_PARSE_PARAM(m_num_cols, int, "num_cols").required();
    RAMULATOR_PARSE_PARAM(m_bank_positions, std::vector<int>, "bank_positions").required();
    RAMULATOR_PARSE_PARAM(m_bank_counts, std::vector<int>, "bank_counts").required();

    validate_layout();
    load_trace(m_trace_path);

    m_stats.add("records_loaded", s_records_loaded);
    m_stats.add("host_records", s_host_records);
    m_stats.add("pim_records", s_pim_records);
    m_stats.add("barrier_records", s_barrier_records);
    m_stats.add("drain_records", s_drain_records);
    m_stats.add("host_requests_sent", s_host_requests_sent);
    m_stats.add("host_requests_completed", s_host_requests_completed);
    m_stats.add("pim_requests_sent", s_pim_requests_sent);
    m_stats.add("pim_requests_completed", s_pim_requests_completed);
    m_stats.add("barriers_retired", s_barriers_retired);
    m_stats.add("drains_retired", s_drains_retired);

    m_logger.info(
        fmt::format(
            "StructuredWorkloadSurrogateTrace: loaded {} expanded records from {}",
            m_records.size(),
            m_trace_path));
  }

  int get_num_cores() override {
    return 1;
  }

  void tick() override {
    m_clk++;

    if (m_retry_req) {
      try_send_retry();
      return;
    }

    if (m_curr_record_idx >= m_records.size()) {
      return;
    }

    Record& record = m_records[m_curr_record_idx];
    if (std::holds_alternative<HostRecord>(record)) {
      tick_host(std::get<HostRecord>(record));
      return;
    }
    if (std::holds_alternative<PIMRecord>(record)) {
      tick_pim(std::get<PIMRecord>(record));
      return;
    }
    if (std::holds_alternative<BarrierRecord>(record)) {
      if (m_inflight_requests == 0) {
        s_barriers_retired++;
        m_curr_record_idx++;
        m_curr_record_req_idx = 0;
      }
      return;
    }
    if (m_inflight_requests == 0) {
      s_drains_retired++;
      m_curr_record_idx++;
      m_curr_record_req_idx = 0;
    }
  }

  bool is_finished() override {
    return m_curr_record_idx >= m_records.size() && !m_retry_req && m_inflight_requests == 0;
  }

 private:
  void validate_layout() {
    if (m_bank_positions.empty()) {
      throw std::runtime_error(
          "StructuredWorkloadSurrogateTrace: bank_positions must not be empty");
    }
    if (m_bank_positions.size() != m_bank_counts.size()) {
      throw std::runtime_error(
          "StructuredWorkloadSurrogateTrace: bank_positions and bank_counts size mismatch");
    }
    if (m_addr_vec_size <= 0 || m_total_bank_units <= 0 || m_num_rows <= 0 || m_num_cols <= 0) {
      throw std::runtime_error(
          "StructuredWorkloadSurrogateTrace: address-vector layout parameters must be positive");
    }
    if (m_row_pos < 0 || m_row_pos >= m_addr_vec_size || m_col_pos < 0 || m_col_pos >= m_addr_vec_size) {
      throw std::runtime_error(
          "StructuredWorkloadSurrogateTrace: row_pos and col_pos must be within addr_vec_size");
    }
    for (size_t i = 0; i < m_bank_positions.size(); i++) {
      if (m_bank_positions[i] < 0 || m_bank_positions[i] >= m_addr_vec_size) {
        throw std::runtime_error(
            "StructuredWorkloadSurrogateTrace: bank_positions entries must be within addr_vec_size");
      }
      if (m_bank_counts[i] <= 0) {
        throw std::runtime_error(
            "StructuredWorkloadSurrogateTrace: bank_counts entries must be positive");
      }
    }
  }

  void load_trace(const std::string& file_path_str) {
    fs::path trace_path(file_path_str);
    if (!fs::exists(trace_path)) {
      throw std::runtime_error(
          fmt::format("StructuredWorkloadSurrogateTrace: trace {} does not exist", file_path_str));
    }

    std::ifstream trace_file(trace_path);
    if (!trace_file.is_open()) {
      throw std::runtime_error(
          fmt::format("StructuredWorkloadSurrogateTrace: trace {} cannot be opened", file_path_str));
    }

    std::string line;
    int line_num = 0;
    while (std::getline(trace_file, line)) {
      line_num++;
      if (line.empty()) {
        continue;
      }

      YAML::Node node;
      try {
        node = YAML::Load(line);
      } catch (const YAML::Exception& exc) {
        throw std::runtime_error(fmt::format(
            "StructuredWorkloadSurrogateTrace: {} line {} parse error: {}",
            file_path_str,
            line_num,
            exc.what()));
      }

      int repeat = require_int(node, "repeat", file_path_str, line_num);
      if (repeat <= 0) {
        throw std::runtime_error(fmt::format(
            "StructuredWorkloadSurrogateTrace: {} line {} repeat must be positive",
            file_path_str,
            line_num));
      }

      Record parsed = parse_record(node, file_path_str, line_num);
      for (int i = 0; i < repeat; i++) {
        m_records.push_back(parsed);
      }
    }

    s_records_loaded = m_records.size();
  }

  Record parse_record(const YAML::Node& node, const std::string& path, int line_num) {
    require_string(node, "schema_version", path, line_num, "v0.1");
    require_present(node, "record_id", path, line_num);
    const std::string kind = require_string(node, "kind", path, line_num);
    require_present(node, "phase", path, line_num);
    require_present(node, "layer", path, line_num);
    require_present(node, "op", path, line_num);
    YAML::Node provenance = require_map(node, "provenance", path, line_num);
    YAML::Node mapping_policy = require_map(node, "mapping_policy", path, line_num);
    validate_provenance(provenance, path, line_num);
    validate_mapping_policy(mapping_policy, path, line_num);

    if (kind == "HostRead" || kind == "HostWrite") {
      s_host_records++;
      YAML::Node address_policy = require_map(node, "address_policy", path, line_num);
      const int count = require_int(address_policy, "count", path, line_num);
      const int64_t base = require_i64(address_policy, "base_byte", path, line_num);
      const int64_t stride = require_i64(address_policy, "stride_bytes", path, line_num);
      if (count <= 0 || stride <= 0) {
        throw std::runtime_error(fmt::format(
            "StructuredWorkloadSurrogateTrace: {} line {} host address_policy count/stride must be positive",
            path,
            line_num));
      }
      const int bytes = require_int(node, "bytes", path, line_num);
      if (bytes <= 0) {
        throw std::runtime_error(fmt::format(
            "StructuredWorkloadSurrogateTrace: {} line {} host bytes must be positive",
            path,
            line_num));
      }
      return HostRecord{
          .is_write = (kind == "HostWrite"),
          .base_byte = static_cast<Addr_t>(base),
          .stride_bytes = static_cast<Addr_t>(stride),
          .count = count,
      };
    }

    if (kind == "PIMCompute" || kind == "PIMLoadAll" || kind == "PIMComputeAll") {
      s_pim_records++;
      const bool all_bank = (kind == "PIMLoadAll" || kind == "PIMComputeAll");
      int request_type_id = m_pim_request_type_id;
      if (kind == "PIMLoadAll") {
        if (m_pim_load_request_type_id < 0) {
          throw std::runtime_error(fmt::format(
              "StructuredWorkloadSurrogateTrace: {} line {} PIMLoadAll requires pim_load_request_type_id",
              path,
              line_num));
        }
        request_type_id = m_pim_load_request_type_id;
      } else if (kind == "PIMComputeAll") {
        if (m_pim_compute_all_request_type_id < 0) {
          throw std::runtime_error(fmt::format(
              "StructuredWorkloadSurrogateTrace: {} line {} PIMComputeAll requires pim_compute_all_request_type_id",
              path,
              line_num));
        }
        request_type_id = m_pim_compute_all_request_type_id;
      }

      const int num_requests = require_int(node, "num_requests", path, line_num);
      if (num_requests <= 0) {
        throw std::runtime_error(fmt::format(
            "StructuredWorkloadSurrogateTrace: {} line {} {} num_requests must be positive",
            path,
            line_num,
            kind));
      }

      std::vector<int> bank_sequence;
      if (all_bank) {
        const std::string scope = require_string(node, "all_bank_scope", path, line_num);
        if (scope != "rank") {
          throw std::runtime_error(fmt::format(
              "StructuredWorkloadSurrogateTrace: {} line {} all_bank_scope must be 'rank'",
              path,
              line_num));
        }
        bank_sequence = {0};
      } else {
        bank_sequence = require_int_vector(node, "bank_sequence", path, line_num);
        if (bank_sequence.empty()) {
          throw std::runtime_error(fmt::format(
              "StructuredWorkloadSurrogateTrace: {} line {} PIMCompute bank_sequence must not be empty",
              path,
              line_num));
        }
        for (int bank : bank_sequence) {
          if (bank < 0 || bank >= m_total_bank_units) {
            throw std::runtime_error(fmt::format(
                "StructuredWorkloadSurrogateTrace: {} line {} bank_sequence entry {} out of range [0, {})",
                path,
                line_num,
                bank,
                m_total_bank_units));
          }
        }
      }

      YAML::Node dep = require_map(node, "dependency_context", path, line_num);
      YAML::Node row = require_map(node, "row_policy", path, line_num);
      YAML::Node column = require_map(node, "column_policy", path, line_num);
      require_present(node, "datatype_metadata", path, line_num);

      const int burst_length = all_bank ? optional_int(node, "burst_length", 1) : require_int(node, "burst_length", path, line_num);
      const int dependency_count = require_int(dep, "dependency_count", path, line_num);
      const int dependency_offset = require_int(dep, "dependency_id", path, line_num);
      const int row_start = require_int(row, "row_start", path, line_num);
      const int row_count = require_int(row, "row_count", path, line_num);
      const int row_resolved = require_int(row, "resolved_row", path, line_num);
      const int column_start = require_int(column, "column_start", path, line_num);
      const int column_resolved = require_int(column, "resolved_column", path, line_num);

      if (burst_length <= 0 || dependency_count <= 0 || row_count <= 0) {
        throw std::runtime_error(fmt::format(
            "StructuredWorkloadSurrogateTrace: {} line {} PIMCompute burst/dependency/row counts must be positive",
            path,
            line_num));
      }
      if (row_start < 0 || row_start >= m_num_rows || row_start + row_count > m_num_rows) {
        throw std::runtime_error(fmt::format(
            "StructuredWorkloadSurrogateTrace: {} line {} PIMCompute row window must fit within num_rows",
            path,
            line_num));
      }
      if (row_resolved < row_start || row_resolved >= row_start + row_count) {
        throw std::runtime_error(fmt::format(
            "StructuredWorkloadSurrogateTrace: {} line {} PIMCompute resolved_row must lie within the row window",
            path,
            line_num));
      }
      if (column_resolved < 0 || column_resolved >= m_num_cols) {
        throw std::runtime_error(fmt::format(
            "StructuredWorkloadSurrogateTrace: {} line {} PIMCompute resolved_column must lie within [0, num_cols)",
            path,
            line_num));
      }
      if (dependency_offset < 0 || dependency_offset >= dependency_count) {
        throw std::runtime_error(fmt::format(
            "StructuredWorkloadSurrogateTrace: {} line {} PIMCompute dependency_id must lie within [0, dependency_count)",
            path,
            line_num));
      }

      bool controller_bank_order = true;
      if (mapping_policy["controller_bank_order"]) {
        const std::string order = mapping_policy["controller_bank_order"].as<std::string>();
        if (order != "controller" && order != "frontend") {
          throw std::runtime_error(fmt::format(
              "StructuredWorkloadSurrogateTrace: {} line {} mapping_policy.controller_bank_order must be 'controller' or 'frontend'",
              path,
              line_num));
        }
        controller_bank_order = (order == "controller");
      }

      return PIMRecord{
          .request_type_id = request_type_id,
          .all_bank = all_bank,
          .num_requests = num_requests,
          .bank_sequence = std::move(bank_sequence),
          .burst_length = burst_length,
          .dependency_count = dependency_count,
          .dependency_offset = dependency_offset,
          .row_start = row_start,
          .row_count = row_count,
          .row_offset = row_resolved - row_start,
          .column_start = column_start,
          .column_offset = column_resolved - column_start,
          .controller_bank_order = controller_bank_order,
      };
    }

    if (kind == "Barrier") {
      s_barrier_records++;
      require_present(node, "barrier_scope", path, line_num);
      return BarrierRecord{};
    }

    if (kind == "Drain") {
      s_drain_records++;
      require_present(node, "drain_scope", path, line_num);
      return DrainRecord{};
    }

    throw std::runtime_error(fmt::format(
        "StructuredWorkloadSurrogateTrace: {} line {} unsupported kind '{}'",
        path,
        line_num,
        kind));
  }

  void tick_host(const HostRecord& record) {
    if (m_curr_record_req_idx >= record.count) {
      m_curr_record_idx++;
      m_curr_record_req_idx = 0;
      return;
    }

    if (!m_retry_req) {
      m_retry_req = make_host_request(record, m_curr_record_req_idx);
    }
    try_send_retry();
  }

  void tick_pim(const PIMRecord& record) {
    if (m_curr_record_req_idx >= record.num_requests) {
      m_curr_record_idx++;
      m_curr_record_req_idx = 0;
      return;
    }

    if (!m_retry_req) {
      m_retry_req = make_pim_request(record, m_curr_record_req_idx);
    }
    try_send_retry();
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
    m_curr_record_req_idx++;
    if (m_curr_record_idx >= m_records.size()) {
      return;
    }
    Record& record = m_records[m_curr_record_idx];
    if (std::holds_alternative<HostRecord>(record)) {
      s_host_requests_sent++;
      if (m_curr_record_req_idx >= std::get<HostRecord>(record).count) {
        m_curr_record_idx++;
        m_curr_record_req_idx = 0;
      }
    } else if (std::holds_alternative<PIMRecord>(record)) {
      s_pim_requests_sent++;
      if (m_curr_record_req_idx >= std::get<PIMRecord>(record).num_requests) {
        m_curr_record_idx++;
        m_curr_record_req_idx = 0;
      }
    }
  }

  Request make_host_request(const HostRecord& record, int idx) {
    Addr_t addr = record.base_byte + static_cast<Addr_t>(idx) * record.stride_bytes;
    Request req(addr, record.is_write ? Request::Type::Write : Request::Type::Read);
    req.source_id = 0;
    req.size_bytes = m_memory_system->get_tx_bytes();
    req.callback = [this](Request&) {
      m_inflight_requests--;
      s_host_requests_completed++;
    };
    return req;
  }

  Request make_pim_request(const PIMRecord& record, int idx) {
    Request req(pim_addr_vec(record, idx), record.request_type_id);
    req.source_id = 0;
    req.size_bytes = m_memory_system->get_tx_bytes();
    req.callback = [this](Request&) {
      m_inflight_requests--;
      s_pim_requests_completed++;
    };
    req.addr = flatten_addr(req.addr_vec);
    return req;
  }

  AddrVec_t pim_addr_vec(const PIMRecord& record, int idx) {
    AddrVec_t av(m_addr_vec_size, 0);
    const int distribution_span = static_cast<int>(record.bank_sequence.size());
    const int phase = idx / record.burst_length;
    const int flat_bank = record.bank_sequence[phase % distribution_span];
    const int dep_phase = phase / distribution_span;
    const int dependency_offset = (record.dependency_offset + dep_phase) % record.dependency_count;
    const int row_phase = dep_phase / record.dependency_count;
    const int row_offset = (record.row_offset + row_phase) % record.row_count;

    if (!record.all_bank) {
      if (record.controller_bank_order) {
        decompose_bank_controller_order(flat_bank, av);
      } else {
        decompose_bank(flat_bank, av);
      }
    }

    av[m_row_pos] = record.row_start + row_offset;
    av[m_col_pos] = record.column_start + ((record.column_offset + dep_phase) % record.dependency_count);
    return av;
  }

  Addr_t flatten_addr(const AddrVec_t& av) const {
    int bank_flat = 0;
    for (size_t i = 0; i < m_bank_positions.size(); i++) {
      bank_flat = bank_flat * m_bank_counts[i] + av[m_bank_positions[i]];
    }
    return static_cast<Addr_t>(
        bank_flat * m_num_rows * m_num_cols + av[m_row_pos] * m_num_cols + av[m_col_pos]);
  }

  void decompose_bank(int flat, AddrVec_t& av) const {
    for (int i = static_cast<int>(m_bank_positions.size()) - 1; i >= 0; i--) {
      av[m_bank_positions[i]] = flat % m_bank_counts[i];
      flat /= m_bank_counts[i];
    }
  }

  void decompose_bank_controller_order(int flat, AddrVec_t& av) const {
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

  static void require_present(const YAML::Node& node, const std::string& key, const std::string& path, int line_num) {
    if (!node[key]) {
      throw std::runtime_error(fmt::format(
          "StructuredWorkloadSurrogateTrace: {} line {} missing required field '{}'",
          path,
          line_num,
          key));
    }
  }

  static YAML::Node require_map(const YAML::Node& node, const std::string& key, const std::string& path, int line_num) {
    require_present(node, key, path, line_num);
    if (!node[key].IsMap()) {
      throw std::runtime_error(fmt::format(
          "StructuredWorkloadSurrogateTrace: {} line {} field '{}' must be a map",
          path,
          line_num,
          key));
    }
    return node[key];
  }

  static std::string require_string(
      const YAML::Node& node,
      const std::string& key,
      const std::string& path,
      int line_num,
      const std::string& exact = "") {
    require_present(node, key, path, line_num);
    std::string value = node[key].as<std::string>();
    if (!exact.empty() && value != exact) {
      throw std::runtime_error(fmt::format(
          "StructuredWorkloadSurrogateTrace: {} line {} field '{}' must equal '{}'",
          path,
          line_num,
          key,
          exact));
    }
    return value;
  }

  static int require_int(const YAML::Node& node, const std::string& key, const std::string& path, int line_num) {
    require_present(node, key, path, line_num);
    return node[key].as<int>();
  }

  static int64_t require_i64(const YAML::Node& node, const std::string& key, const std::string& path, int line_num) {
    require_present(node, key, path, line_num);
    return node[key].as<int64_t>();
  }

  static int optional_int(const YAML::Node& node, const std::string& key, int default_value) {
    if (!node[key]) {
      return default_value;
    }
    return node[key].as<int>();
  }

  static std::vector<int> require_int_vector(
      const YAML::Node& node,
      const std::string& key,
      const std::string& path,
      int line_num) {
    require_present(node, key, path, line_num);
    if (!node[key].IsSequence()) {
      throw std::runtime_error(fmt::format(
          "StructuredWorkloadSurrogateTrace: {} line {} field '{}' must be a sequence",
          path,
          line_num,
          key));
    }
    return node[key].as<std::vector<int>>();
  }

  static void validate_provenance(const YAML::Node& provenance, const std::string& path, int line_num) {
    const std::string source_kind = require_string(provenance, "source_kind", path, line_num);
    if (source_kind != "generated" && source_kind != "handwritten") {
      throw std::runtime_error(fmt::format(
          "StructuredWorkloadSurrogateTrace: {} line {} provenance.source_kind must be 'generated' or 'handwritten'",
          path,
          line_num));
    }
    require_present(provenance, "tuple_manifest", path, line_num);
    require_present(provenance, "generator_version", path, line_num);
    require_present(provenance, "literature_anchor", path, line_num);
    require_present(provenance, "claim_boundary", path, line_num);
    require_present(provenance, "non_claims", path, line_num);
    if (!provenance["literature_anchor"].IsSequence() || provenance["literature_anchor"].size() == 0) {
      throw std::runtime_error(fmt::format(
          "StructuredWorkloadSurrogateTrace: {} line {} provenance.literature_anchor must be a non-empty sequence",
          path,
          line_num));
    }
    if (!provenance["claim_boundary"].IsSequence() || provenance["claim_boundary"].size() == 0) {
      throw std::runtime_error(fmt::format(
          "StructuredWorkloadSurrogateTrace: {} line {} provenance.claim_boundary must be a non-empty sequence",
          path,
          line_num));
    }
    if (!provenance["non_claims"].IsSequence() || provenance["non_claims"].size() == 0) {
      throw std::runtime_error(fmt::format(
          "StructuredWorkloadSurrogateTrace: {} line {} provenance.non_claims must be a non-empty sequence",
          path,
          line_num));
    }

    bool saw_structured_surrogate = false;
    bool saw_simulator_diagnostic = false;
    bool saw_non_silicon = false;
    bool saw_decode_only_first = false;
    for (const YAML::Node& entry : provenance["claim_boundary"]) {
      const std::string value = entry.as<std::string>();
      saw_structured_surrogate = saw_structured_surrogate || value == "structured workload-surrogate";
      saw_simulator_diagnostic = saw_simulator_diagnostic || value == "simulator-diagnostic";
      saw_non_silicon = saw_non_silicon || value == "non-silicon-calibrated";
      saw_decode_only_first = saw_decode_only_first || value == "decode-only-first";
    }
    if (!saw_structured_surrogate || !saw_simulator_diagnostic || !saw_non_silicon || !saw_decode_only_first) {
      throw std::runtime_error(fmt::format(
          "StructuredWorkloadSurrogateTrace: {} line {} provenance.claim_boundary must include 'structured workload-surrogate', 'simulator-diagnostic', 'non-silicon-calibrated', and 'decode-only-first'",
          path,
          line_num));
    }

    bool saw_not_runtime = false;
    bool saw_not_vllm = false;
    bool saw_not_mixed = false;
    bool saw_not_all_bank = false;
    for (const YAML::Node& entry : provenance["non_claims"]) {
      const std::string value = entry.as<std::string>();
      saw_not_runtime = saw_not_runtime || value == "not_runtime_replay";
      saw_not_vllm = saw_not_vllm || value == "not_vllm_replay";
      saw_not_mixed = saw_not_mixed || value == "not_mixed_prefill_decode";
      saw_not_all_bank = saw_not_all_bank || value == "not_all_bank_fidelity";
    }
    if (!saw_not_runtime || !saw_not_vllm || !saw_not_mixed || !saw_not_all_bank) {
      throw std::runtime_error(fmt::format(
          "StructuredWorkloadSurrogateTrace: {} line {} provenance.non_claims must include 'not_runtime_replay', 'not_vllm_replay', 'not_mixed_prefill_decode', and 'not_all_bank_fidelity'",
          path,
          line_num));
    }
  }

  static void validate_mapping_policy(const YAML::Node& mapping_policy, const std::string& path, int line_num) {
    require_present(mapping_policy, "host_policy", path, line_num);
    require_present(mapping_policy, "pim_policy", path, line_num);
    require_present(mapping_policy, "bank_sequence_policy", path, line_num);
    require_present(mapping_policy, "mpu_grouping_policy", path, line_num);
  }
};

}  // namespace Ramulator
