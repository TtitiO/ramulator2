"""Native LPDDR5-PIM concrete opcode trace helpers.

This module is intentionally separate from ``structured_trace.py``.  The
structured trace surface preserves semantic workload-surrogate records, while
this surface is backend-specific command replay for LPDDR5-PIM validation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from ramulator.dram.addressing import addr_vec_from_byte_address as _map_byte_address
from ramulator.dram.addressing import validate_addr_vec

CONCRETE_SCHEMA_VERSION = "lpddr5-pim-opcode-v0.2"
CONCRETE_GENERATOR_VERSION = "lpddr5-pim-opcode-generator-v0.1"
CONCRETE_OPCODES = {"READ", "WRITE", "SB", "HAB", "HAB_PIM", "PIM_BCAST", "PIM_MAC", "PIM_MAC_AB"}
MODE_OPCODES = {"SB", "HAB", "HAB_PIM"}
REQUEST_OPCODES = {"READ", "WRITE", "PIM_BCAST", "PIM_MAC", "PIM_MAC_AB"}
MAX_REPEAT = 1_000_000
MAX_EXPANDED_RECORDS = 1_000_000_000
MAX_EXPANDED_RECORDS_ENV = "RAMULATOR_MAX_EXPANDED_RECORDS"
FORBIDDEN_RAW_ATTACC_OPCODES = {
    "PIM_WR_GB",
    "PIM_MV_BA",
    "PIM_MV_BF",
    "PIM_SFM",
    "PIM_SET_CONFIG",
    "PIM_SET",
    "PIM_ACT_AB",
}
REQUIRED_BOUNDARY_CLAIMS = [
    "native-lpddr5-pim-concrete-opcode-replay",
    "backend-specific-command-validation",
    "simulator-diagnostic",
    "non-silicon-calibrated",
]
REQUIRED_NON_CLAIMS = [
    "not_semantic_workload_replay",
    "not_runtime_replay",
    "not_vllm_replay",
    "not_raw_attacc_schema",
]
PIM_BCAST_BOUNDARY_NON_CLAIM = "not_silicon_faithful_pim_bcast_source_or_timing"
DEFAULT_NON_CLAIMS = [*REQUIRED_NON_CLAIMS, PIM_BCAST_BOUNDARY_NON_CLAIM]


def stable_json_dumps(data: object) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def stable_json_pretty(data: object) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def addr_vec_from_byte_address(
    address: int,
    *,
    addr_vec_size: int | None = None,
    address_layout: Mapping[str, Any] | None = None,
) -> list[int]:
    """Map a byte address using the shared organization-derived mapper."""
    if address_layout is None:
        raise ValueError("address_layout is required for host byte-address mapping")
    level_names = list(address_layout["level_names"])
    if addr_vec_size is not None and len(level_names) != addr_vec_size:
        raise ValueError("address_layout level count must equal addr_vec_size")
    return _map_byte_address(
        address,
        level_names=level_names,
        level_sizes=list(address_layout["level_sizes"]),
        internal_prefetch_size=int(address_layout["internal_prefetch_size"]),
        tx_bytes=int(address_layout["tx_bytes"]),
    )


def concrete_provenance(
    *,
    source_kind: str = "generated",
    manifest_name: str = "lpddr5_pim_concrete_minimal",
) -> dict:
    return {
        "source_kind": source_kind,
        "manifest": manifest_name,
        "generator_version": CONCRETE_GENERATOR_VERSION if source_kind == "generated" else "manual",
        "claim_boundary": list(REQUIRED_BOUNDARY_CLAIMS),
        "non_claims": list(DEFAULT_NON_CLAIMS),
        "notes": (
            "backend-specific native LPDDR5-PIM opcode replay; PIM_BCAST is a bounded all-bank "
            "setup abstraction rather than a vendor-faithful payload-source or timing model; "
            "semantic JSONL remains separate"
        ),
    }


def build_header() -> dict:
    """Build the v0.2 trace header envelope (asserted once per file)."""
    header = {"schema_version": CONCRETE_SCHEMA_VERSION}
    validate_header(header)
    return header


def validate_header(header: dict) -> None:
    if header.get("schema_version") != CONCRETE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported concrete opcode schema_version: "
            f"{header.get('schema_version')}"
        )


def validate_record(record: dict, *, address_layout: Mapping[str, Any] | None = None) -> None:
    required = {"opcode", "repeat", "addr_vec"}
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"Concrete opcode record missing required fields: {missing}")
    # v0.2 records must not carry file-level constants (those live in the header).
    forbidden = {"schema_version", "provenance", "record_id"} & set(record)
    if forbidden:
        raise ValueError(
            "Concrete opcode record must not carry header/constant fields: "
            f"{sorted(forbidden)}"
        )
    opcode = record["opcode"]
    if opcode in FORBIDDEN_RAW_ATTACC_OPCODES:
        raise ValueError(
            "Raw AttAcc opcode is not part of the LPDDR5-PIM concrete schema: "
            f"{opcode}"
        )
    if opcode not in CONCRETE_OPCODES:
        raise ValueError(f"Unsupported LPDDR5-PIM concrete opcode: {opcode}")
    repeat = record["repeat"]
    if isinstance(repeat, bool) or not isinstance(repeat, int):
        raise ValueError("Concrete opcode repeat must be an integer")
    if repeat <= 0 or repeat > MAX_REPEAT:
        raise ValueError(f"Concrete opcode repeat must be in [1, {MAX_REPEAT}]")
    if not isinstance(record["addr_vec"], list) or not record["addr_vec"]:
        raise ValueError("Concrete opcode addr_vec must be a non-empty list")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in record["addr_vec"]):
        raise ValueError("Concrete opcode addr_vec entries must be integers")
    resolved_layout = address_layout
    if resolved_layout is not None:
        validate_addr_vec(
            record["addr_vec"],
            level_names=list(resolved_layout["level_names"]),
            level_sizes=list(resolved_layout["level_sizes"]),
            context="Concrete opcode addr_vec",
        )
    elif any(value < 0 for value in record["addr_vec"]):
        raise ValueError("Concrete opcode addr_vec entries must be non-negative")
    is_host_opcode = opcode in {"READ", "WRITE"}
    if is_host_opcode and resolved_layout is None:
        raise ValueError("Concrete READ/WRITE validation requires address_layout")
    if {"addr_byte", "addr_byte_stride"} & set(record) and not is_host_opcode:
        raise ValueError("Concrete opcode addr_byte fields are only valid for READ/WRITE")
    if is_host_opcode and "addr_byte" not in record:
        raise ValueError("Concrete READ/WRITE records require addr_byte")
    if "addr_byte" in record:
        addr_byte = record["addr_byte"]
        if isinstance(addr_byte, bool) or not isinstance(addr_byte, int) or addr_byte < 0:
            raise ValueError("Concrete opcode addr_byte must be a non-negative integer")
        if is_host_opcode:
            expected_addr_vec = addr_vec_from_byte_address(
                addr_byte, addr_vec_size=len(record["addr_vec"]), address_layout=resolved_layout
            )
            if record["addr_vec"] != expected_addr_vec:
                raise ValueError("Concrete READ/WRITE addr_vec must match decomposed addr_byte")
    if "addr_byte_stride" in record:
        if "addr_byte" not in record:
            raise ValueError("Concrete opcode addr_byte_stride requires addr_byte")
        addr_byte_stride = record["addr_byte_stride"]
        if (
            isinstance(addr_byte_stride, bool)
            or not isinstance(addr_byte_stride, int)
            or addr_byte_stride <= 0
        ):
            raise ValueError("Concrete opcode addr_byte_stride must be a positive integer")
        addr_vec_from_byte_address(
            record["addr_byte"] + (repeat - 1) * addr_byte_stride,
            addr_vec_size=len(record["addr_vec"]),
            address_layout=resolved_layout,
        )

    # In-memory bank interleaving fields (PIM_MAC only; compact expansion).
    _INTERLEAVE_FIELDS = {
        "bank_sequence", "dependency_count", "row_count", "row_start",
        "column_start", "resolved_row_offset", "resolved_col_offset",
        "interleave_depth", "interleave_start_idx",
        "bank_positions", "bank_counts", "bank_level", "row_level", "col_level",
    }
    if _INTERLEAVE_FIELDS & set(record) and opcode != "PIM_MAC":
        raise ValueError("Concrete opcode bank interleaving fields are only valid for PIM_MAC")
    if "bank_sequence" in record:
        bank_seq = record["bank_sequence"]
        if not isinstance(bank_seq, list) or not bank_seq:
            raise ValueError("Concrete opcode bank_sequence must be a non-empty list")
        if any(not isinstance(b, int) or b < 0 for b in bank_seq):
            raise ValueError("Concrete opcode bank_sequence entries must be non-negative integers")
    if "dependency_count" in record:
        dc = record["dependency_count"]
        if isinstance(dc, bool) or not isinstance(dc, int) or dc < 1:
            raise ValueError("Concrete opcode dependency_count must be a positive integer")
    if "row_count" in record:
        rc = record["row_count"]
        if isinstance(rc, bool) or not isinstance(rc, int) or rc < 1:
            raise ValueError("Concrete opcode row_count must be a positive integer")
    if "interleave_depth" in record:
        idep = record["interleave_depth"]
        if isinstance(idep, bool) or not isinstance(idep, int) or idep < 1:
            raise ValueError("Concrete opcode interleave_depth must be a positive integer")
    if "interleave_start_idx" in record:
        isi = record["interleave_start_idx"]
        if isinstance(isi, bool) or not isinstance(isi, int) or isi < 0:
            raise ValueError("Concrete opcode interleave_start_idx must be a non-negative integer")
    if "bank_sequence" in record:
        if resolved_layout is None:
            raise ValueError("Concrete bank interleaving validation requires address_layout")
        level_names = list(resolved_layout["level_names"])
        level_sizes = [int(value) for value in resolved_layout["level_sizes"]]
        row_level = int(record.get("row_level", level_names.index("Row")))
        col_level = int(record.get("col_level", level_names.index("Column")))
        for field, level in (("row_level", row_level), ("col_level", col_level)):
            if level < 0 or level >= len(level_names):
                raise ValueError(f"Concrete opcode {field} must fit within addr_vec")
        if level_names[row_level] != "Row" or level_names[col_level] != "Column":
            raise ValueError(
                "Concrete opcode row_level/col_level must refer to the resolved "
                "Row/Column levels"
            )
        row_start = int(record.get("row_start", 0))
        row_count = int(record.get("row_count", 1))
        column_start = int(record.get("column_start", 0))
        dependency_count = int(record.get("dependency_count", 0))
        if row_start < 0 or row_count < 1 or row_start + row_count > level_sizes[row_level]:
            raise ValueError("Concrete opcode row range exceeds configured hierarchy")
        if (
            column_start < 0
            or dependency_count < 1
            or column_start + dependency_count > level_sizes[col_level]
        ):
            raise ValueError("Concrete opcode column range exceeds configured hierarchy")

        if "bank_positions" in record or "bank_counts" in record:
            if "bank_positions" not in record or "bank_counts" not in record:
                raise ValueError(
                    "Concrete opcode bank_positions and bank_counts must be "
                    "provided together"
                )
            bp = record["bank_positions"]
            bc = record["bank_counts"]
            if (
                not isinstance(bp, list)
                or not isinstance(bc, list)
                or len(bp) != len(bc)
                or not bp
            ):
                raise ValueError(
                    "Concrete opcode bank_positions and bank_counts must be "
                    "non-empty lists of equal length"
                )
            if any(isinstance(v, bool) or not isinstance(v, int) for v in bp + bc):
                raise ValueError(
                    "Concrete opcode bank_positions and bank_counts entries "
                    "must be integers"
                )
            if len(set(bp)) != len(bp) or row_level in bp or col_level in bp:
                raise ValueError(
                    "Concrete opcode bank_positions must be unique and not "
                    "overlap row/column"
                )
            total_banks = 1
            for index, (position, count) in enumerate(zip(bp, bc, strict=True)):
                if position < 0 or position >= len(level_names):
                    raise ValueError(
                        "Concrete opcode bank_positions entries must fit within "
                        "addr_vec"
                    )
                if count != level_sizes[position]:
                    raise ValueError(
                        f"Concrete opcode bank_counts[{index}]={count} must equal configured "
                        f"level {level_names[position]} size {level_sizes[position]}"
                    )
                total_banks *= count
        else:
            bank_level = int(record.get("bank_level", level_names.index("Bank")))
            if bank_level < 0 or bank_level >= len(level_names):
                raise ValueError("Concrete opcode bank_level must fit within addr_vec")
            if level_names[bank_level] != "Bank":
                raise ValueError("Concrete opcode bank_level must refer to the resolved Bank level")
            total_banks = level_sizes[bank_level]
        if any(bank >= total_banks for bank in record["bank_sequence"]):
            raise ValueError(
                f"Concrete opcode bank_sequence entries must be in [0, {total_banks})"
            )

    # Optional compact semantic-source traceability (lowered traces only).
    if "sem" in record:
        sem = record["sem"]
        if not isinstance(sem, dict):
            raise ValueError("Concrete opcode sem must be a map")
        if "id" not in sem:
            raise ValueError("Concrete opcode sem must contain an 'id'")



def validate_sequence(
    records: list[dict],
    *,
    address_layout: Mapping[str, Any] | None = None,
    max_expanded_records: int | None = None,
) -> None:
    mode = "SB"
    saw_bcast_since_hab = False
    expanded_records = 0
    if max_expanded_records is None:
        max_expanded_records = int(os.environ.get(MAX_EXPANDED_RECORDS_ENV, MAX_EXPANDED_RECORDS))
    if max_expanded_records <= 0:
        raise ValueError(f"{MAX_EXPANDED_RECORDS_ENV} must be positive when set")
    for index, record in enumerate(records):
        # Accept both rich in-memory records and already-slim v0.2 lines.
        record = record if "provenance" not in record else slim_record(record)
        validate_record(record, address_layout=address_layout)
        expanded_records += record["repeat"]
        if expanded_records > max_expanded_records:
            raise ValueError(
                "Concrete opcode trace exceeds max expanded records "
                f"{max_expanded_records}"
            )
        opcode = record["opcode"]
        if opcode in {"READ", "WRITE"}:
            if mode != "SB":
                raise ValueError(f"Concrete opcode record {index} {opcode} requires SB mode")
            continue
        if opcode == "SB":
            mode = "SB"
            saw_bcast_since_hab = False
        elif opcode == "HAB":
            mode = "HAB"
            saw_bcast_since_hab = False
        elif opcode == "HAB_PIM":
            if not saw_bcast_since_hab:
                raise ValueError(
                    f"Concrete opcode record {index} HAB_PIM requires a preceding "
                    "PIM_BCAST in HAB mode"
                )
            mode = "HAB_PIM"
        elif opcode == "PIM_BCAST":
            if mode != "HAB":
                raise ValueError(f"Concrete opcode record {index} PIM_BCAST requires HAB mode")
            saw_bcast_since_hab = True
        elif opcode == "PIM_MAC_AB":
            if mode != "HAB_PIM" or not saw_bcast_since_hab:
                raise ValueError(
                    f"Concrete opcode record {index} PIM_MAC_AB requires HAB_PIM "
                    "mode after PIM_BCAST"
                )
        elif opcode == "PIM_MAC":
            if mode != "SB":
                raise ValueError(f"Concrete opcode record {index} PIM_MAC requires SB mode")


def expanded_record_count(records: list[dict]) -> int:
    return sum(int(record["repeat"]) for record in records)


# Fields that are file-level constants (header) or human prose; never per-record in v0.2.
_SLIM_DROP = {"schema_version", "record_id", "notes", "provenance"}


def slim_record(record: dict) -> dict:
    """Strip header constants/prose and compact semantic_source -> sem (v0.2 line)."""
    slim = {key: value for key, value in record.items() if key not in _SLIM_DROP}
    semantic_source = (record.get("provenance") or {}).get("semantic_source")
    if isinstance(semantic_source, dict):
        sem = {
            short: semantic_source[full]
            for short, full in (
                ("id", "record_id"),
                ("kind", "kind"),
                ("layer", "layer"),
                ("op", "op"),
            )
            if semantic_source.get(full) is not None
        }
        if sem:
            slim["sem"] = sem
    return slim


def write_jsonl(
    records: list[dict],
    output_path: Path,
    *,
    address_layout: Mapping[str, Any] | None = None,
    max_expanded_records: int | None = None,
) -> None:
    header = build_header()
    slim_records = [slim_record(record) for record in records]
    validate_sequence(
        slim_records,
        address_layout=address_layout,
        max_expanded_records=max_expanded_records,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(stable_json_dumps(header) + "\n")
        for record in slim_records:
            handle.write(stable_json_dumps(record) + "\n")


def read_jsonl(
    input_path: Path,
    *,
    address_layout: Mapping[str, Any] | None = None,
    max_expanded_records: int | None = None,
) -> tuple[dict, list[dict]]:
    """Read a v0.2 trace: returns (validated header, validated slim records)."""
    lines = [
        line
        for line in Path(input_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        raise ValueError("Concrete opcode trace is empty")
    header = json.loads(lines[0])
    validate_header(header)
    records = [json.loads(line) for line in lines[1:]]
    validate_sequence(
        records,
        address_layout=address_layout,
        max_expanded_records=max_expanded_records,
    )
    return header, records


def write_json(data: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(stable_json_pretty(data), encoding="utf-8")
