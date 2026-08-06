#ifndef RAMULATOR_DRAM_COMMANDS_PIM_BCAST_H
#define RAMULATOR_DRAM_COMMANDS_PIM_BCAST_H

#include <stdexcept>

#include "ramulator/dram/commands/SB.h"

namespace Ramulator::Cmd {

template <class T>
struct PIM_BCAST {
  // Bounded backend abstraction: public Samsung-style PIM sources describe the
  // corresponding setup/load as HAB/all-bank WR-like broadcast behavior, not a
  // literal LPDDR5 command named PIM_BCAST with vendor-calibrated timing.
  static constexpr DRAMCommandMeta meta = {.is_accessing = true};
  static constexpr BankTarget bank_target = BankTarget::All;

  static int preq(DRAMNode* bank, int cmd, const AddrVec_t& addr_vec, Clk_t clk) {
    auto* rank = pim_rank_node<T>(bank);
    if (rank == nullptr) {
      throw std::runtime_error("[PIM_BCAST] Missing rank node!");
    }
    return rank->m_state == T::State::PIM_HAB ? cmd : T::Command::HAB;
  }
};

}  // namespace Ramulator::Cmd

#endif  // RAMULATOR_DRAM_COMMANDS_PIM_BCAST_H
