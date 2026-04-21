#ifndef RAMULATOR_DRAM_COMMANDS_SB_H
#define RAMULATOR_DRAM_COMMANDS_SB_H

#include "ramulator/dram/node.h"

namespace Ramulator::Cmd {

template <class T>
inline DRAMNode* pim_rank_node(DRAMNode* bank) {
  auto* bankgroup = bank->m_parent_node;
  return bankgroup ? bankgroup->m_parent_node : nullptr;
}

template <class T>
struct SB {
  static constexpr DRAMCommandMeta meta = {};
  static constexpr BankTarget bank_target = BankTarget::All;

  static void action(DRAMNode* bank, int cmd, const AddrVec_t& addr_vec, Clk_t clk) {
    if (auto* rank = pim_rank_node<T>(bank); rank != nullptr) {
      rank->m_state = T::State::PIM_SB;
    }
  }
};

}  // namespace Ramulator::Cmd

#endif  // RAMULATOR_DRAM_COMMANDS_SB_H
