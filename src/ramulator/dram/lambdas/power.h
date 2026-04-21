#ifndef RAMULATOR_DRAM_LAMBDAS_POWER_H
#define RAMULATOR_DRAM_LAMBDAS_POWER_H

namespace Ramulator::Lambdas::Power {

namespace Bank {

template <class T>
int get_flat_rank_id(DRAMNode* node) {
  if constexpr (T::Level::Bank - T::Level::Rank == 1) {
    return node->m_parent_node->m_node_id;
  } else {
    return node->m_parent_node->m_parent_node->m_node_id;
  }
}

template <class T>
void ACT(DRAMNode* node, int cmd, const AddrVec_t&, Clk_t) {
  node->m_spec->power_stats[get_flat_rank_id<T>(node)].command_counters[T::PowerCommand::ACT]++;
}

template <class T>
void PRE(DRAMNode* node, int cmd, const AddrVec_t&, Clk_t) {
  node->m_spec->power_stats[get_flat_rank_id<T>(node)].command_counters[T::PowerCommand::PRE]++;
}

template <class T>
void RD(DRAMNode* node, int cmd, const AddrVec_t&, Clk_t) {
  node->m_spec->power_stats[get_flat_rank_id<T>(node)].command_counters[T::PowerCommand::RD]++;
}

template <class T>
void WR(DRAMNode* node, int cmd, const AddrVec_t&, Clk_t) {
  node->m_spec->power_stats[get_flat_rank_id<T>(node)].command_counters[T::PowerCommand::WR]++;
}

}  // namespace Bank

namespace Rank {

template <class T>
int get_flat_rank_id(DRAMNode* node) {
  return node->m_node_id;
}

template <class T>
int get_open_bank_count(DRAMNode* node) {
  int count = 0;
  if constexpr (T::Level::Bank - T::Level::Rank == 1) {
    for (auto& bank : node->m_child_nodes) {
      if (bank->m_state == T::State::Opened || bank->m_state == T::State::Activating) {
        count++;
      }
    }
  } else {
    for (auto& bank_group : node->m_child_nodes) {
      for (auto& bank : bank_group->m_child_nodes) {
        if (bank->m_state == T::State::Opened || bank->m_state == T::State::Activating) {
          count++;
        }
      }
    }
  }
  return count;
}

template <class T>
void ACT(DRAMNode* node, int, const AddrVec_t&, Clk_t clk) {
  auto& rank = node->m_spec->power_stats[get_flat_rank_id<T>(node)];
  if (rank.current_state == DRAMPowerStats::PowerState::Idle && get_open_bank_count<T>(node) == 0) {
    rank.idle_cycles += clk - rank.last_update_clk;
    rank.last_update_clk = clk;
    rank.current_state = DRAMPowerStats::PowerState::Active;
  }
}

template <class T>
void PRE(DRAMNode* node, int, const AddrVec_t&, Clk_t clk) {
  auto& rank = node->m_spec->power_stats[get_flat_rank_id<T>(node)];
  if (rank.current_state == DRAMPowerStats::PowerState::Active && get_open_bank_count<T>(node) == 1) {
    rank.active_cycles += clk - rank.last_update_clk;
    rank.last_update_clk = clk;
    rank.current_state = DRAMPowerStats::PowerState::Idle;
  }
}

template <class T>
void PREA(DRAMNode* node, int, const AddrVec_t&, Clk_t clk) {
  auto& rank = node->m_spec->power_stats[get_flat_rank_id<T>(node)];
  int open_banks = get_open_bank_count<T>(node);
  rank.command_counters[T::PowerCommand::PRE] += open_banks;
  if (rank.current_state == DRAMPowerStats::PowerState::Active && open_banks > 0) {
    rank.active_cycles += clk - rank.last_update_clk;
    rank.last_update_clk = clk;
    rank.current_state = DRAMPowerStats::PowerState::Idle;
  }
}

template <class T>
void REFab(DRAMNode* node, int, const AddrVec_t&, Clk_t) {
  auto& rank = node->m_spec->power_stats[get_flat_rank_id<T>(node)];
  rank.command_counters[T::PowerCommand::REF]++;
}

template <class T>
void COUNT_PIM_INCREMENTAL_ENERGY(DRAMNode* node, int cmd, const AddrVec_t&, Clk_t) {
  auto& rank = node->m_spec->power_stats[get_flat_rank_id<T>(node)];
  int cmd_id = cmd;
  if (cmd_id < 0 || cmd_id >= static_cast<int>(rank.incremental_command_counters.size())) {
    return;
  }
  rank.incremental_command_counters[cmd_id]++;
}

template <class T>
void finalize_rank(DRAMNode* node, Clk_t clk) {
  auto& rank = node->m_spec->power_stats[get_flat_rank_id<T>(node)];
  if (rank.current_state == DRAMPowerStats::PowerState::Active) {
    rank.active_cycles += clk - rank.last_update_clk;
  } else {
    rank.idle_cycles += clk - rank.last_update_clk;
  }
  rank.last_update_clk = clk;
}

}  // namespace Rank

}  // namespace Ramulator::Lambdas::Power

#endif  // RAMULATOR_DRAM_LAMBDAS_POWER_H
