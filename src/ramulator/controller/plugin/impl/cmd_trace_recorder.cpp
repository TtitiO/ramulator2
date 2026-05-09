#include <cstdint>
#include <fmt/format.h>
#include <fstream>
#include <string>
#include <vector>

#include "ramulator/base/base.h"
#include "ramulator/controller/controller_base.h"
#include "ramulator/controller/plugin/i_controller_plugin.h"
#include "ramulator/dram/dram_spec.h"

namespace Ramulator {

/// Records every DRAM command issued by the controller to a trace file.
///
/// Two output modes:
///   - **Text** (default): CSV with a self-documenting header line.
///     Each row is: clock, command_name, addr_vec..., type_id, source_id
///   - **Binary**: compact fixed-size records with a self-describing header
///     containing level/command name tables for offline decoding.
///
/// Output files are per-channel: path is suffixed with ".ch0", ".ch1", etc.
///
/// Example config (Python):
///   ramulator.ControllerPlugin.CmdTraceRecorder(path="trace.csv")
///   ramulator.ControllerPlugin.CmdTraceRecorder(path="trace.bin", binary=True)
class CmdTraceRecorder : public IControllerPlugin, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IControllerPlugin, CmdTraceRecorder, "CmdTraceRecorder")

 public:
  void init() override {
    RAMULATOR_PARSE_PARAM(m_path, std::string, "path").required();
    RAMULATOR_PARSE_PARAM(m_binary, bool, "binary").default_val(false);
  }

  void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {
    m_ctrl = cast_parent<ControllerBase>();
    const auto& spec = *m_ctrl->m_device.m_spec;
    m_level_count = spec.level_count;

    std::string filepath = fmt::format("{}.ch{}", m_path, m_ctrl->m_channel_id);

    if (m_binary) {
      m_file.open(filepath, std::ios::binary);
      write_binary_header(spec);
    } else {
      m_file.open(filepath);
      write_text_header(spec);
    }
  }

  void on_issue(const Request& req) override {
    if (m_binary) {
      write_binary_record(req);
    } else {
      write_text_record(req);
    }
  }

  void finalize() override {
    if (m_file.is_open()) {
      m_file.close();
    }
  }

 private:
  ControllerBase* m_ctrl = nullptr;
  std::string m_path;
  bool m_binary = false;
  int m_level_count = 0;
  int m_rank_level = -1;
  int m_bank_group_level = -1;
  int m_bank_level = -1;
  int m_banks_per_rank = 1;
  int m_banks_per_bank_group = 1;
  std::ofstream m_file;

  // ── Text mode ──────────────────────────────────────────────────────

  void write_text_header(const DRAMSpec& spec) {
    if (spec.has_level("Rank")) {
      m_rank_level = spec.get_level_id("Rank");
    }
    if (spec.has_level("BankGroup")) {
      m_bank_group_level = spec.get_level_id("BankGroup");
    }
    if (spec.has_level("Bank")) {
      m_bank_level = spec.get_level_id("Bank");
      m_banks_per_bank_group = spec.organization.level_sizes[m_bank_level];
      if (m_rank_level >= 0) {
        m_banks_per_rank = 1;
        for (int level = m_rank_level + 1; level <= m_bank_level; level++) {
          m_banks_per_rank *= spec.organization.level_sizes[level];
        }
      }
    }
    m_file << "clock,command";
    for (const auto& name : spec.level_names) {
      m_file << "," << name;
    }
    m_file << ",type,source,bank_id,mpu_group_id,pim_banks_per_mpu,global_ready,bank_ready,mpu_ready,issue_or_stall_reason\n";
  }

  void write_text_record(const Request& req) {
    const auto& cmd_names = m_ctrl->m_device.m_spec->command_names;
    m_file << m_ctrl->m_clk << "," << cmd_names[req.command];
    for (int i = 0; i < m_level_count; i++) {
      m_file << "," << req.addr_vec[i];
    }
    int bank_id = -1;
    if (m_bank_level >= 0 && req.addr_vec[m_bank_level] >= 0) {
      bank_id = req.addr_vec[m_bank_level];
      if (m_bank_group_level >= 0 && req.addr_vec[m_bank_group_level] >= 0) {
        bank_id = req.addr_vec[m_bank_group_level] * m_banks_per_bank_group + req.addr_vec[m_bank_level];
      }
      if (m_rank_level >= 0 && req.addr_vec[m_rank_level] >= 0) {
        bank_id += req.addr_vec[m_rank_level] * m_banks_per_rank;
      }
    }
    int pim_banks_per_mpu = m_ctrl->m_device.m_spec->pim_banks_per_mpu;
    int mpu_group = (bank_id >= 0 && pim_banks_per_mpu > 0) ? bank_id / pim_banks_per_mpu : -1;
    int is_pim_mac = cmd_names[req.command] == "PIM_MAC" ? 1 : 0;
    m_file << "," << req.type_id << "," << req.source_id << "," << bank_id << "," << mpu_group << ","
           << pim_banks_per_mpu << "," << is_pim_mac << "," << is_pim_mac << "," << is_pim_mac << ","
           << (is_pim_mac ? "issued" : "not_pim_mac") << "\n";
    m_file.flush();
  }

  // ── Binary mode ────────────────────────────────────────────────────
  //
  // File layout:
  //   Header:
  //     uint32_t level_count
  //     uint32_t command_count
  //     [level_count null-terminated strings]    (level names)
  //     [command_count null-terminated strings]  (command names)
  //   Records (repeated):
  //     uint64_t clock
  //     int32_t  command_id
  //     int32_t  type_id
  //     int32_t  source_id
  //     int32_t  addr_vec[level_count]

  void write_binary_header(const DRAMSpec& spec) {
    auto write_u32 = [&](uint32_t v) { m_file.write(reinterpret_cast<const char*>(&v), sizeof(v)); };

    write_u32(static_cast<uint32_t>(spec.level_count));
    write_u32(static_cast<uint32_t>(spec.command_count));
    for (const auto& name : spec.level_names) {
      m_file.write(name.c_str(), name.size() + 1);
    }
    for (const auto& name : spec.command_names) {
      m_file.write(name.c_str(), name.size() + 1);
    }
  }

  void write_binary_record(const Request& req) {
    uint64_t clk = m_ctrl->m_clk;
    int32_t cmd = req.command;
    int32_t type = req.type_id;
    int32_t src = req.source_id;

    m_file.write(reinterpret_cast<const char*>(&clk), sizeof(clk));
    m_file.write(reinterpret_cast<const char*>(&cmd), sizeof(cmd));
    m_file.write(reinterpret_cast<const char*>(&type), sizeof(type));
    m_file.write(reinterpret_cast<const char*>(&src), sizeof(src));
    m_file.write(reinterpret_cast<const char*>(req.addr_vec.data()), m_level_count * sizeof(int32_t));
  }
};

}  // namespace Ramulator
