"""Canonical hierarchy-aware address layout for PIMScope frontends.

The concrete trace path supplies a complete ``addr_vec`` to Ramulator's
``PassThroughAddrMapper``.  This module derives that vector from the resolved
DRAM organization instead of assuming one fixed LPDDR5 geometry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import prod
from typing import Any


def extract_dram_layout(dram: Any) -> dict[str, Any]:
    """Resolve hierarchy names, dimensions, and byte-address mapping metadata."""
    cls = type(dram)
    level_names = list(cls.levels.keys())
    organization, _ = dram.resolve()
    level_sizes = [1 if name == "Channel" else int(organization[name.lower()]) for name in level_names]

    if not level_names or level_names[0] != "Channel":
        raise ValueError("DRAM hierarchy must begin with Channel")
    if "Rank" not in level_names or "Bank" not in level_names:
        raise ValueError("DRAM hierarchy must contain Rank and Bank levels")
    if "Row" not in level_names or "Column" not in level_names:
        raise ValueError("DRAM hierarchy must contain Row and Column levels")
    if any(count <= 0 for count in level_sizes):
        raise ValueError("DRAM hierarchy level sizes must be positive")

    channel_pos = level_names.index("Channel")
    rank_pos = level_names.index("Rank")
    bank_pos = level_names.index("Bank")
    row_pos = level_names.index("Row")
    col_pos = level_names.index("Column")
    if channel_pos != 0 or not (rank_pos < bank_pos < row_pos < col_pos):
        raise ValueError(
            "PIMScope concrete traces require Channel first and ordered "
            "Rank/.../Bank/Row/Column hierarchy levels"
        )

    internal_prefetch_size = int(cls.internal_prefetch_size)
    num_cols = level_sizes[col_pos]
    if internal_prefetch_size <= 0 or num_cols % internal_prefetch_size != 0:
        raise ValueError(
            "DRAM Column count must be divisible by internal_prefetch_size"
        )
    tx_bytes = (
        int(cls.data_payload_bytes)
        if cls.data_payload_bytes is not None
        else internal_prefetch_size * int(organization["channel_width"]) // 8
    )
    if tx_bytes <= 0:
        raise ValueError("DRAM transaction size must be positive")

    # Controller-native bank flattening follows hierarchy order from Rank
    # through Bank.  Keep a separate interleave order for synthetic streaming
    # frontends that intentionally rotate BankGroup/PseudoChannel fastest.
    controller_bank_positions = list(range(rank_pos, bank_pos + 1))
    controller_bank_counts = [level_sizes[index] for index in controller_bank_positions]
    bank_positions = list(controller_bank_positions)
    bank_counts = list(controller_bank_counts)
    for name in ("BankGroup", "PseudoChannel"):
        if name in level_names:
            pos = level_names.index(name)
            order_index = bank_positions.index(pos)
            bank_positions.append(bank_positions.pop(order_index))
            bank_counts.append(bank_counts.pop(order_index))

    address_level_sizes = list(level_sizes)
    address_level_sizes[col_pos] //= internal_prefetch_size
    capacity_bytes = tx_bytes * prod(address_level_sizes)
    total_bank_units = prod(controller_bank_counts)

    return {
        "dram_class": cls.name,
        "mapping_version": 1,
        "level_names": level_names,
        "level_sizes": level_sizes,
        "address_level_sizes": address_level_sizes,
        "addr_vec_size": len(level_names),
        "channel_pos": channel_pos,
        "rank_pos": rank_pos,
        "bank_positions": bank_positions,
        "bank_counts": bank_counts,
        "controller_bank_positions": controller_bank_positions,
        "controller_bank_counts": controller_bank_counts,
        "total_bank_units": total_bank_units,
        "row_pos": row_pos,
        "col_pos": col_pos,
        "num_rows": level_sizes[row_pos],
        "num_cols": num_cols,
        "internal_prefetch_size": internal_prefetch_size,
        "num_cls": num_cols // internal_prefetch_size,
        "tx_bytes": tx_bytes,
        "capacity_bytes": capacity_bytes,
    }


def concrete_address_layout(layout: Mapping[str, Any]) -> dict[str, Any]:
    """Return the subset serialized into the native concrete frontend config."""
    return {
        "address_mapping_version": int(layout["mapping_version"]),
        "level_names": list(layout["level_names"]),
        "level_sizes": list(layout["level_sizes"]),
        "address_level_sizes": list(layout["address_level_sizes"]),
        "internal_prefetch_size": int(layout["internal_prefetch_size"]),
        "tx_bytes": int(layout["tx_bytes"]),
    }


def validate_addr_vec(
    addr_vec: Sequence[int],
    *,
    level_names: Sequence[str],
    level_sizes: Sequence[int],
    allow_wildcards: bool = False,
    context: str = "addr_vec",
) -> None:
    """Validate a complete hierarchy coordinate with field-specific errors."""
    if len(level_names) != len(level_sizes):
        raise ValueError("address layout level_names and level_sizes must have equal length")
    if len(addr_vec) != len(level_names):
        raise ValueError(
            f"{context} length {len(addr_vec)} must equal hierarchy level count {len(level_names)}"
        )
    for index, (name, size, value) in enumerate(zip(level_names, level_sizes, addr_vec, strict=True)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{context}[{index}] ({name}) must be an integer")
        if allow_wildcards and value == -1:
            continue
        if value < 0 or value >= int(size):
            raise ValueError(
                f"{context}[{index}] ({name})={value} must be in [0, {int(size)})"
            )


def addr_vec_from_byte_address(
    address: int,
    *,
    level_names: Sequence[str],
    level_sizes: Sequence[int],
    internal_prefetch_size: int,
    tx_bytes: int,
) -> list[int]:
    """Map a byte address to a complete hierarchy vector using mixed radix.

    The last hierarchy level cycles fastest.  Column coordinates represent
    internal-prefetch units, matching Ramulator's flat-address mappers.  Byte
    addresses within the same transaction therefore map to the same vector.
    """
    if isinstance(address, bool) or not isinstance(address, int) or address < 0:
        raise ValueError("host byte address must be a non-negative integer")
    if len(level_names) != len(level_sizes) or not level_names:
        raise ValueError("address layout level_names and level_sizes must be non-empty and equal length")
    if level_names[-1] != "Column":
        raise ValueError("address layout requires Column as the final hierarchy level")
    if any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in level_sizes):
        raise ValueError("address layout level_sizes must contain positive integers")
    if internal_prefetch_size <= 0 or level_sizes[-1] % internal_prefetch_size != 0:
        raise ValueError("Column size must be divisible by internal_prefetch_size")
    if tx_bytes <= 0:
        raise ValueError("tx_bytes must be positive")

    radices = list(level_sizes)
    radices[-1] //= internal_prefetch_size
    capacity_bytes = tx_bytes * prod(radices)
    if address >= capacity_bytes:
        raise ValueError(
            f"host byte address {address} exceeds configured addressable capacity "
            f"{capacity_bytes} bytes"
        )

    value = address // tx_bytes
    addr_vec = [0] * len(radices)
    for index in range(len(radices) - 1, -1, -1):
        addr_vec[index] = value % radices[index]
        value //= radices[index]
    if value:
        raise ValueError(
            f"host byte address {address} exceeds configured addressable capacity "
            f"{capacity_bytes} bytes"
        )
    validate_addr_vec(
        addr_vec,
        level_names=level_names,
        level_sizes=level_sizes,
        context="mapped addr_vec",
    )
    return addr_vec
