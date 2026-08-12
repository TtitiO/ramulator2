#include "ramulator/dram/device.h"

#include <stdexcept>

#include <fmt/format.h>

namespace Ramulator {

void DRAMDevice::init(std::unique_ptr<DRAMSpec> spec) {
  m_spec_owner = std::move(spec);
  m_spec = m_spec_owner.get();
  m_bank_level = m_spec->get_level_id("Bank");
  m_root = std::make_unique<DRAMNode>(m_spec, nullptr, 0, 0);
  m_root->for_each_at_level(m_bank_level, [&](DRAMNode* bank) { m_bank_nodes.push_back(bank); });
}

void DRAMDevice::set_channel_id(int channel_id) {
  m_root->m_node_id = channel_id;
}

void DRAMDevice::issue_command(int command, const AddrVec_t& addr_vec, Clk_t clk) {
  m_root->update_timing(command, addr_vec, clk);
  m_root->update_powers(command, addr_vec, clk);
  apply_action(command, addr_vec, clk);
}

bool DRAMDevice::check_timing(int command, const AddrVec_t& addr_vec, Clk_t clk) {
  return m_root->check_timing(command, addr_vec, clk);
}

int DRAMDevice::get_preq_command(int command, const AddrVec_t& addr_vec, Clk_t clk) {
  auto preq_fn = m_spec->funcs.preqs[command];
  if (!preq_fn) return command;

  int resolved = command;
  for_each_target_bank_while(command, addr_vec, [&](int flat_bank_id) {
    int preq = preq_fn(m_bank_nodes[flat_bank_id], command, addr_vec, clk);
    if (preq != command) { resolved = preq; return false; }
    return true;
  });
  return resolved;
}

bool DRAMDevice::check_rowbuffer_hit(int command, const AddrVec_t& addr_vec, Clk_t clk) {
  auto rowhit_fn = m_spec->funcs.rowhits[command];
  if (!rowhit_fn) {
    return false;
  }
  int flat_bank_id = get_flat_bank_id(addr_vec);
  return rowhit_fn(m_bank_nodes[flat_bank_id], command, addr_vec, clk);
}

bool DRAMDevice::check_node_open(int command, const AddrVec_t& addr_vec, Clk_t clk) {
  auto rowopen_fn = m_spec->funcs.rowopens[command];
  if (!rowopen_fn) {
    return false;
  }
  int flat_bank_id = get_flat_bank_id(addr_vec);
  return rowopen_fn(m_bank_nodes[flat_bank_id], command, addr_vec, clk);
}

int DRAMDevice::get_flat_bank_id(const AddrVec_t& addr_vec) const {
  int id = 0;
  for (int lvl = 1; lvl <= m_bank_level; lvl++) {
    id = id * m_spec->organization.level_sizes[lvl] + addr_vec[lvl];
  }
  return id;
}

int DRAMDevice::get_banks_per_rank() const {
  const int rank_level = m_spec->get_level_id("Rank");
  int banks = 1;
  for (int level = rank_level + 1; level <= m_bank_level; level++) {
    banks *= m_spec->organization.level_sizes[level];
  }
  return banks;
}

int DRAMDevice::get_rank_id_for_flat_bank(int flat_bank_id) const {
  const int banks_per_rank = get_banks_per_rank();
  if (flat_bank_id < 0 || flat_bank_id >= static_cast<int>(m_bank_nodes.size())) {
    throw std::runtime_error(fmt::format(
        "DRAMDevice: flat bank {} is outside [0, {})", flat_bank_id, m_bank_nodes.size()));
  }
  return flat_bank_id / banks_per_rank;
}

int DRAMDevice::get_rank_local_bank_id(int flat_bank_id) const {
  get_rank_id_for_flat_bank(flat_bank_id);
  return flat_bank_id % get_banks_per_rank();
}

int DRAMDevice::get_rank_local_group_id(int flat_bank_id, int banks_per_group) const {
  const int banks_per_rank = get_banks_per_rank();
  if (banks_per_group <= 0 || banks_per_group > banks_per_rank ||
      banks_per_rank % banks_per_group != 0) {
    throw std::runtime_error(fmt::format(
        "DRAMDevice: banks_per_group {} must be positive, no greater than banks per rank {}, and divide it exactly",
        banks_per_group,
        banks_per_rank));
  }
  get_rank_id_for_flat_bank(flat_bank_id);
  return get_rank_local_bank_id(flat_bank_id) / banks_per_group;
}

int DRAMDevice::get_global_rank_local_group_id(int flat_bank_id, int banks_per_group) const {
  const int rank_local_group = get_rank_local_group_id(flat_bank_id, banks_per_group);
  const int groups_per_rank = get_banks_per_rank() / banks_per_group;
  return get_rank_id_for_flat_bank(flat_bank_id) * groups_per_rank + rank_local_group;
}

std::vector<int> DRAMDevice::get_rank_local_group_banks(int flat_bank_id, int banks_per_group) const {
  const int rank = get_rank_id_for_flat_bank(flat_bank_id);
  const int group = get_rank_local_group_id(flat_bank_id, banks_per_group);
  const int begin = rank * get_banks_per_rank() + group * banks_per_group;
  std::vector<int> banks;
  banks.reserve(banks_per_group);
  for (int bank = begin; bank < begin + banks_per_group; bank++) {
    banks.push_back(bank);
  }
  return banks;
}

bool DRAMDevice::bank_matches(DRAMNode* bank, const AddrVec_t& addr_vec) {
  for (auto* n = bank; n != nullptr; n = n->m_parent_node) {
    if (addr_vec[n->m_level] != -1 && addr_vec[n->m_level] != n->m_node_id) {
      return false;
    }
  }
  return true;
}

std::vector<int> DRAMDevice::get_target_banks(int command, const AddrVec_t& addr_vec) const {
  std::vector<int> ids;
  for_each_target_bank(command, addr_vec, [&](int id) { ids.push_back(id); });
  return ids;
}

void DRAMDevice::finalize_power(Clk_t clk) {
  if (m_spec && m_root) {
    m_spec->finalize_power(clk, m_root.get());
  }
}

void DRAMDevice::apply_action(int command, const AddrVec_t& addr_vec, Clk_t clk) {
  auto action_fn = m_spec->funcs.actions[command];
  if (!action_fn) return;
  for_each_target_bank(command, addr_vec, [&](int flat_bank_id) {
    action_fn(m_bank_nodes[flat_bank_id], command, addr_vec, clk);
  });
}

}  // namespace Ramulator
