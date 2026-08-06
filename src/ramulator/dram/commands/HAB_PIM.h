#ifndef RAMULATOR_DRAM_COMMANDS_HAB_PIM_H
#define RAMULATOR_DRAM_COMMANDS_HAB_PIM_H

#include "ramulator/dram/commands/SB.h"

namespace Ramulator::Cmd {

template <class T>
struct HAB_PIM {
  static constexpr DRAMCommandMeta meta = {};
  static constexpr BankTarget bank_target = BankTarget::All;

  static void action(DRAMNode* bank, int cmd, const AddrVec_t& addr_vec, Clk_t clk) {
    if (auto* rank = pim_rank_node<T>(bank); rank != nullptr) {
      rank->m_state = T::State::PIM_HAB_PIM;
    }
  }
};

}  // namespace Ramulator::Cmd

#endif  // RAMULATOR_DRAM_COMMANDS_HAB_PIM_H
