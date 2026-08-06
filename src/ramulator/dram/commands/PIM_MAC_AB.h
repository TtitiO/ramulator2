#ifndef RAMULATOR_DRAM_COMMANDS_PIM_MAC_AB_H
#define RAMULATOR_DRAM_COMMANDS_PIM_MAC_AB_H

#include <stdexcept>

#include "ramulator/dram/commands/SB.h"

namespace Ramulator::Cmd {

template <class T>
struct PIM_MAC_AB {
  static constexpr DRAMCommandMeta meta = {.is_accessing = true};
  static constexpr BankTarget bank_target = BankTarget::All;

  static int preq(DRAMNode* bank, int cmd, const AddrVec_t& addr_vec, Clk_t clk) {
    auto* rank = pim_rank_node<T>(bank);
    if (rank == nullptr) {
      throw std::runtime_error("[PIM_MAC_AB] Missing rank node!");
    }
    return rank->m_state == T::State::PIM_HAB_PIM ? cmd : T::Command::HAB_PIM;
  }
};

}  // namespace Ramulator::Cmd

#endif  // RAMULATOR_DRAM_COMMANDS_PIM_MAC_AB_H
