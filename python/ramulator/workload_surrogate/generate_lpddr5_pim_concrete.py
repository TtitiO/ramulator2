"""Deterministic native LPDDR5-PIM concrete opcode trace generator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ramulator.workload_surrogate.lpddr5_pim_concrete_trace import (
    CONCRETE_GENERATOR_VERSION,
    CONCRETE_SCHEMA_VERSION,
    DEFAULT_NON_CLAIMS,
    MAX_REPEAT,
    REQUIRED_BOUNDARY_CLAIMS,
    addr_vec_from_byte_address,
    concrete_provenance,
    expanded_record_count,
    validate_sequence,
    write_json,
    write_jsonl,
)


DEFAULT_OUTPUT_DIR = Path("ramulator2/tests/data/lpddr5_pim_concrete_opcode/minimal_v0_1")
DEFAULT_SEMANTIC_OUTPUT_DIR = Path("ramulator2/tests/data/lpddr5_pim_concrete_opcode/from_semantic_v0_1")
PER_BANK_COMPUTE_SEMANTIC_KINDS = {"PIMCompute", "AttentionScore", "AttentionContext", "FFNProjection", "MoERouter", "MoEExpertFFN"}
ALL_BANK_LOAD_SEMANTIC_KINDS = {"PIMLoadAll", "PIMDataMove"}
ALL_BANK_COMPUTE_SEMANTIC_KINDS = {"PIMComputeAll"}
HOST_SEMANTIC_KINDS = {"HostRead", "HostWrite"}
SEMANTIC_ONLY_PIM_DATA_MOVE_KINDS = {"dynamic_activation_tile", "bank_local_tile_activation", "cross_bank_operand_shuffle_accounting"}


def _split_repeat(repeat: int, max_repeat: int = MAX_REPEAT) -> list[int]:
    if repeat <= 0:
        raise ValueError("repeat must be positive")
    if max_repeat <= 0:
        raise ValueError("max_repeat must be positive")
    chunks: list[int] = []
    remaining = repeat
    while remaining > 0:
        chunk = min(max_repeat, remaining)
        chunks.append(chunk)
        remaining -= chunk
    return chunks


def _record(
    record_id: str,
    opcode: str,
    addr_vec: list[int],
    *,
    repeat: int = 1,
    notes: str = "",
    provenance: dict | None = None,
    extra_fields: dict | None = None,
) -> dict:
    record = {
        "schema_version": CONCRETE_SCHEMA_VERSION,
        "record_id": record_id,
        "opcode": opcode,
        "repeat": repeat,
        "addr_vec": list(addr_vec),
        "provenance": concrete_provenance() if provenance is None else provenance,
        "notes": notes,
    }
    if extra_fields:
        record.update(extra_fields)
    return record


def load_semantic_records(trace_path: Path | str) -> list[dict]:
    """Load Phase 2 semantic JSONL records for offline concrete lowering."""
    path = Path(trace_path)
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _semantic_provenance(record: dict, *, manifest_name: str) -> dict:
    semantic_provenance = dict(record.get("provenance", {}))
    return concrete_provenance(manifest_name=manifest_name) | {
        "semantic_source": {
            "schema_version": record.get("schema_version"),
            "record_id": record.get("record_id"),
            "kind": record.get("kind"),
            "phase": record.get("phase"),
            "layer": record.get("layer"),
            "op": record.get("op"),
            "tuple_manifest": semantic_provenance.get("tuple_manifest"),
            "mapping_policy": dict(record.get("mapping_policy", {})),
        },
        "notes": (
            "lowered from Phase 2 semantic workload-surrogate record; native LPDDR5-PIM concrete "
            "opcode replay only; PIM_BCAST remains a bounded all-bank setup abstraction, not a "
            "silicon-faithful source/timing claim"
        ),
    }


def _addr_vec_from_semantic(
    record: dict,
    request_index: int,
    *,
    addr_vec_size: int,
    bank_level: int,
    bank_positions: list[int] | None,
    bank_counts: list[int] | None,
    row_level: int,
    col_level: int,
) -> list[int]:
    if addr_vec_size <= 0:
        raise ValueError("addr_vec_size must be positive")
    for level_name, level in {"bank_level": bank_level, "row_level": row_level, "col_level": col_level}.items():
        if level < 0 or level >= addr_vec_size:
            raise ValueError(f"{level_name} must fit within addr_vec_size")
    if (bank_positions is None) != (bank_counts is None):
        raise ValueError("bank_positions and bank_counts must be provided together")
    if bank_positions is not None and bank_counts is not None:
        if len(bank_positions) != len(bank_counts) or not bank_positions:
            raise ValueError("bank_positions and bank_counts must be non-empty lists of equal length")
        if len(set(bank_positions)) != len(bank_positions):
            raise ValueError("bank_positions entries must be unique")
        if row_level in bank_positions or col_level in bank_positions:
            raise ValueError("bank_positions must not overlap row_level or col_level")
        for level in bank_positions:
            if level < 0 or level >= addr_vec_size:
                raise ValueError("bank_positions entries must fit within addr_vec_size")
        if any(int(count) <= 0 for count in bank_counts):
            raise ValueError("bank_counts entries must be positive")

    av = [0] * addr_vec_size
    kind = record["kind"]
    row_policy = dict(record.get("row_policy", {}))
    column_policy = dict(record.get("column_policy", {}))
    dep = dict(record.get("dependency_context", {}))
    burst_length = int(record.get("burst_length", 1))
    dependency_count = max(1, int(dep.get("dependency_count", 1)))
    dependency_id = int(dep.get("dependency_id", 0)) % dependency_count
    row_start = int(row_policy.get("row_start", 0))
    row_count = max(1, int(row_policy.get("row_count", 1)))
    resolved_row = int(row_policy.get("resolved_row", row_start))
    column_start = int(column_policy.get("column_start", 0))
    resolved_column = int(column_policy.get("resolved_column", column_start))

    phase = request_index // max(1, burst_length)
    distribution_span = 1
    if kind in PER_BANK_COMPUTE_SEMANTIC_KINDS:
        distribution_span = max(1, len(record.get("bank_sequence", [])))
    dep_phase = phase // distribution_span
    row_offset = ((resolved_row - row_start) + dep_phase // dependency_count) % row_count
    col_offset = ((resolved_column - column_start) + dep_phase) % dependency_count
    av[row_level] = row_start + row_offset
    av[col_level] = column_start + col_offset

    if kind in PER_BANK_COMPUTE_SEMANTIC_KINDS:
        bank_sequence = list(record.get("bank_sequence", []))
        if not bank_sequence:
            raise ValueError(f"Semantic record {record.get('record_id')} {kind} missing bank_sequence")
        flat_bank = int(bank_sequence[phase % len(bank_sequence)])
        if flat_bank < 0:
            raise ValueError(f"Semantic record {record.get('record_id')} bank_sequence entries must be non-negative")
        if bank_positions is not None and bank_counts is not None:
            _decompose_flat_bank(
                flat_bank,
                av,
                bank_positions=bank_positions,
                bank_counts=bank_counts,
                controller_order=dict(record.get("mapping_policy", {})).get("controller_bank_order") == "controller",
            )
        else:
            av[bank_level] = flat_bank
    else:
        if bank_positions is None:
            av[bank_level] = 0
    return av


def _addr_vec_from_byte_address(address: int, *, addr_vec_size: int) -> list[int]:
    return addr_vec_from_byte_address(address, addr_vec_size=addr_vec_size)


def _semantic_datatype_bytes(record: dict) -> int:
    datatype = dict(record.get("datatype_metadata", {})).get("datatype")
    if datatype is None:
        datatype = dict(record.get("compute_shape", {})).get("datatype", "int8")
    return 1 if datatype == "int8" else 2


def _pim_data_move_tx_bytes(record: dict) -> int:
    tx_bytes = int(dict(record.get("movement_policy", {})).get("materialization_tx_bytes", 64))
    if tx_bytes <= 0:
        raise ValueError(f"Semantic record {record.get('record_id')} materialization_tx_bytes must be positive")
    return tx_bytes


def _pim_data_move_materialization_bytes(record: dict) -> int:
    movement_policy = dict(record.get("movement_policy", {}))
    movement_elements = int(movement_policy.get("movement_elements", 0))
    if movement_elements <= 0:
        raise ValueError(f"Semantic record {record.get('record_id')} weight materialization missing positive movement_elements")
    return max(1, movement_elements * _semantic_datatype_bytes(record))


def _context_int(record: dict, field: str, default: int = 0) -> int:
    value = dict(record.get("operator_context", {})).get(field, default)
    if value is None:
        return default
    return int(value)


def _pim_data_move_materialization_base_byte(record: dict, *, tx_bytes: int) -> int:
    """Derive a deterministic host byte range for synthetic cold-start preloads."""
    context = dict(record.get("operator_context", {}))
    if "layer_id" in context:
        layer_index = int(context["layer_id"])
    else:
        layer_index = int(str(record.get("layer", "layer_00")).split("_")[-1])
    stage_index = _context_int(record, "stage_index", 0)
    tile_index = _context_int(record, "tile_id", 0)
    expert_value = context.get("expert_id")
    expert_index = 999 if expert_value is None else int(expert_value)
    op = str(record.get("op", ""))
    op_offset = sum((index + 1) * ord(char) for index, char in enumerate(op)) % 1000
    request_index = 1_000_000_000 + layer_index * 1_000_000 + stage_index * 100_000 + expert_index * 1_000 + tile_index * 10 + op_offset
    return request_index * tx_bytes


def _decompose_flat_bank(
    flat_bank: int,
    addr_vec: list[int],
    *,
    bank_positions: list[int],
    bank_counts: list[int],
    controller_order: bool,
) -> None:
    total_banks = 1
    for count in bank_counts:
        total_banks *= int(count)
    if flat_bank < 0 or flat_bank >= total_banks:
        raise ValueError(f"flat bank index {flat_bank} must be in [0, {total_banks})")
    order = list(range(len(bank_positions)))
    if controller_order:
        order.sort(key=lambda index: bank_positions[index])
    for bank_index in reversed(order):
        addr_vec[bank_positions[bank_index]] = flat_bank % int(bank_counts[bank_index])
        flat_bank //= int(bank_counts[bank_index])


def lower_semantic_records_to_concrete(
    semantic_records: list[dict],
    *,
    addr_vec_size: int = 6,
    bank_level: int = 3,
    bank_positions: list[int] | None = None,
    bank_counts: list[int] | None = None,
    row_level: int = 4,
    col_level: int = 5,
    manifest_name: str = "phase2_semantic_lowered_manifest",
    materialize_weights: bool = False,
) -> list[dict]:
    """Lower Phase 2 semantic records into native LPDDR5-PIM concrete opcodes.

    This is an offline compiler stage. HostRead/HostWrite lower to concrete READ
    and WRITE requests. Barrier/Drain become provenance/order annotations only;
    concrete ordering is expressed by legal host requests and LPDDR5-PIM
    mode/control opcodes. Phase 4 semantic-only accounting stages such as
    AttentionSoftmax, PIMElementwise, TopK, dispatch, and combine intentionally do
    not lower to fake hardware commands.

    By default, weight PIMDataMove records (operand_role="weight") are NOT lowered
    to concrete opcodes (weights are resident/preloaded_stationary in steady-state
    inference).  Pass materialize_weights=True for cold-start preload or frontend
    stress-test replay; materialized weights lower to regular concrete WRITE
    records, not PIM_BCAST, because PIM_BCAST is reserved for common all-bank
    setup/broadcast payloads rather than per-bank resident weight preload.
    """
    records: list[dict] = []
    next_id = 0
    mode = "SB"
    all_bank_load_ready = False

    def append(
        opcode: str,
        semantic_record: dict,
        addr_vec: list[int],
        *,
        repeat: int = 1,
        notes: str = "",
        extra_fields: dict | None = None,
    ) -> None:
        nonlocal next_id
        records.append(
            _record(
                f"op_{next_id:04d}",
                opcode,
                addr_vec,
                repeat=repeat,
                notes=notes,
                provenance=_semantic_provenance(semantic_record, manifest_name=manifest_name),
                extra_fields=extra_fields,
            )
        )
        next_id += 1

    for semantic in semantic_records:
        kind = semantic.get("kind")
        if kind not in HOST_SEMANTIC_KINDS | PER_BANK_COMPUTE_SEMANTIC_KINDS | ALL_BANK_LOAD_SEMANTIC_KINDS | ALL_BANK_COMPUTE_SEMANTIC_KINDS:
            continue
        if semantic.get("accounting_only"):
            continue
        if kind in HOST_SEMANTIC_KINDS:
            if mode != "SB":
                append("SB", semantic, [0] * addr_vec_size, notes=f"return to single-bank mode before semantic {kind} lowering")
                mode = "SB"
                all_bank_load_ready = False
            address_policy = dict(semantic.get("address_policy", {}))
            missing_policy = sorted({"base_byte", "stride_bytes", "count"} - set(address_policy))
            if missing_policy:
                raise ValueError(f"Semantic record {semantic.get('record_id')} address_policy missing required fields: {missing_policy}")
            base_byte = int(address_policy.get("base_byte", 0))
            stride_bytes = int(address_policy.get("stride_bytes", 0))
            count = int(address_policy.get("count", 0))
            semantic_repeat = int(semantic.get("repeat", 1))
            if base_byte < 0:
                raise ValueError(f"Semantic record {semantic.get('record_id')} base_byte must be non-negative")
            if stride_bytes <= 0:
                raise ValueError(f"Semantic record {semantic.get('record_id')} stride_bytes must be positive")
            if count <= 0:
                raise ValueError(f"Semantic record {semantic.get('record_id')} address_policy.count must be positive")
            if semantic_repeat <= 0:
                raise ValueError(f"Semantic record {semantic.get('record_id')} repeat must be positive")
            opcode = "WRITE" if kind == "HostWrite" else "READ"
            for repeat_index in range(semantic_repeat):
                remaining = count
                chunk_start = 0
                split_index = 0
                while remaining > 0:
                    repeat_chunk = min(MAX_REPEAT, remaining)
                    chunk_base_byte = base_byte + chunk_start * stride_bytes
                    av = _addr_vec_from_byte_address(chunk_base_byte, addr_vec_size=addr_vec_size)
                    extra_fields = {"addr_byte": chunk_base_byte}
                    if repeat_chunk > 1:
                        extra_fields["addr_byte_stride"] = stride_bytes
                    notes = (
                        f"semantic {kind} lowered to concrete {opcode}"
                        if semantic_repeat == 1 and count <= MAX_REPEAT
                        else f"semantic {kind} lowered to concrete {opcode} repeat {repeat_index + 1}/{semantic_repeat} split {split_index + 1}"
                    )
                    append(opcode, semantic, av, repeat=repeat_chunk, notes=notes, extra_fields=extra_fields)
                    remaining -= repeat_chunk
                    chunk_start += repeat_chunk
                    split_index += 1
            continue
        # Weight records: skip lowering in steady-state inference (weights are resident).
        # Use materialize_weights=True for cold-start preload or frontend stress testing;
        # weights materialize through ordinary host WRITE traffic, not all-bank PIM_BCAST.
        if kind == "PIMDataMove":
            movement_policy = dict(semantic.get("movement_policy", {}))
            operand_role = movement_policy.get("operand_role")
            if operand_role == "weight":
                if not materialize_weights:
                    continue
                if mode != "SB":
                    append("SB", semantic, [0] * addr_vec_size, notes="return to single-bank mode before host WRITE weight preload")
                    mode = "SB"
                    all_bank_load_ready = False
                total_bytes = _pim_data_move_materialization_bytes(semantic)
                tx_bytes = _pim_data_move_tx_bytes(semantic)
                count = max(1, (total_bytes + tx_bytes - 1) // tx_bytes)
                semantic_repeat = int(semantic.get("repeat", 1))
                if semantic_repeat <= 0:
                    raise ValueError(f"Semantic record {semantic.get('record_id')} repeat must be positive")
                base_byte = _pim_data_move_materialization_base_byte(semantic, tx_bytes=tx_bytes)
                for repeat_index in range(semantic_repeat):
                    remaining = count
                    chunk_start = 0
                    split_index = 0
                    while remaining > 0:
                        repeat_chunk = min(MAX_REPEAT, remaining)
                        chunk_base_byte = base_byte + (repeat_index * count + chunk_start) * tx_bytes
                        av = _addr_vec_from_byte_address(chunk_base_byte, addr_vec_size=addr_vec_size)
                        extra_fields = {"addr_byte": chunk_base_byte}
                        if repeat_chunk > 1:
                            extra_fields["addr_byte_stride"] = tx_bytes
                        notes = (
                            "semantic PIMDataMove weight residency materialized as concrete WRITE preload"
                            if semantic_repeat == 1 and count <= MAX_REPEAT
                            else "semantic PIMDataMove weight residency materialized as concrete WRITE preload "
                            f"repeat {repeat_index + 1}/{semantic_repeat} split {split_index + 1}"
                        )
                        append("WRITE", semantic, av, repeat=repeat_chunk, notes=notes, extra_fields=extra_fields)
                        remaining -= repeat_chunk
                        chunk_start += repeat_chunk
                        split_index += 1
                continue
        num_requests = int(semantic.get("num_requests", 0))
        if num_requests <= 0:
            raise ValueError(f"Semantic record {semantic.get('record_id')} num_requests must be positive")
        base_av = _addr_vec_from_semantic(
            semantic,
            0,
            addr_vec_size=addr_vec_size,
            bank_level=bank_level,
            bank_positions=bank_positions,
            bank_counts=bank_counts,
            row_level=row_level,
            col_level=col_level,
        )

        semantic_repeat = int(semantic.get("repeat", 1))
        if semantic_repeat <= 0:
            raise ValueError(f"Semantic record {semantic.get('record_id')} repeat must be positive")

        if kind in PER_BANK_COMPUTE_SEMANTIC_KINDS:
            if mode != "SB":
                append("SB", semantic, base_av, notes="return to single-bank mode before per-bank PIM_MAC lowering")
                mode = "SB"
                all_bank_load_ready = False
            elif not records:
                append("SB", semantic, base_av, notes="enter single-bank mode for semantic PIMCompute lowering")
            total_per_request = num_requests * semantic_repeat
            bank_seq = list(semantic.get("bank_sequence", [0, 1, 2, 3]))
            bank_len = max(1, len(bank_seq))
            full_cycles = total_per_request // bank_len
            remainder = total_per_request % bank_len
            for bank_index in range(bank_len):
                per_bank_total = full_cycles + (1 if bank_index < remainder else 0)
                if per_bank_total == 0:
                    continue
                av = _addr_vec_from_semantic(
                    semantic,
                    bank_index,
                    addr_vec_size=addr_vec_size,
                    bank_level=bank_level,
                    bank_positions=bank_positions,
                    bank_counts=bank_counts,
                    row_level=row_level,
                    col_level=col_level,
                )
                repeat_chunks = _split_repeat(per_bank_total)
                for split_index, repeat_chunk in enumerate(repeat_chunks):
                    notes = (
                        f"semantic {kind} lowered to concrete PIM_MAC"
                        if len(repeat_chunks) == 1
                        else f"semantic {kind} lowered to concrete PIM_MAC split {split_index + 1}/{len(repeat_chunks)}"
                    )
                    append("PIM_MAC", semantic, av, repeat=repeat_chunk, notes=notes)
            mode = "SB"
        elif kind in ALL_BANK_LOAD_SEMANTIC_KINDS:
            if kind == "PIMDataMove":
                movement_kind = dict(semantic.get("movement_policy", {})).get("movement_kind")
                if movement_kind in SEMANTIC_ONLY_PIM_DATA_MOVE_KINDS:
                    continue  # recognized semantic/accounting-only movement kind - no concrete opcode emitted
                if movement_kind != "broadcast_or_accounted_tile_load":
                    raise ValueError(
                        f"Semantic record {semantic.get('record_id')} PIMDataMove movement_kind {movement_kind!r} "
                        "is not supported by native PIM_BCAST lowering"
                    )
            if mode != "HAB":
                append("HAB", semantic, base_av, notes=f"enter host all-bank mode for semantic {kind} lowering")
                mode = "HAB"
            total_repeat = num_requests * semantic_repeat
            repeat_chunks = _split_repeat(total_repeat)
            for split_index, repeat_chunk in enumerate(repeat_chunks):
                notes = (
                    f"semantic {kind} lowered to concrete PIM_BCAST"
                    if len(repeat_chunks) == 1
                    else f"semantic {kind} lowered to concrete PIM_BCAST split {split_index + 1}/{len(repeat_chunks)}"
                )
                append("PIM_BCAST", semantic, base_av, repeat=repeat_chunk, notes=notes)
            all_bank_load_ready = True
        elif kind in ALL_BANK_COMPUTE_SEMANTIC_KINDS:
            for request_index in range(num_requests * semantic_repeat):
                av = _addr_vec_from_semantic(
                    semantic,
                    request_index % num_requests,
                    addr_vec_size=addr_vec_size,
                    bank_level=bank_level,
                    bank_positions=bank_positions,
                    bank_counts=bank_counts,
                    row_level=row_level,
                    col_level=col_level,
                )
                if mode != "HAB":
                    append("HAB", semantic, av, notes="enter host all-bank mode before all-bank compute lowering")
                    mode = "HAB"
                    all_bank_load_ready = False
                if not all_bank_load_ready:
                    append("PIM_BCAST", semantic, av, notes="bounded all-bank load before semantic PIMComputeAll lowering")
                    all_bank_load_ready = True
                append("HAB_PIM", semantic, av, notes="enter PIM all-bank mode for semantic PIMComputeAll lowering")
                append("PIM_MAC_AB", semantic, av, notes="semantic PIMComputeAll lowered to concrete PIM_MAC_AB")
                append("SB", semantic, av, notes="return to single-bank mode after all-bank compute lowering")
                mode = "SB"
                all_bank_load_ready = False

    validate_sequence(records)
    return records


def generate_minimal_concrete_records(
    *,
    addr_vec_size: int = 6,
    bank_level: int = 3,
    row_level: int = 4,
    col_level: int = 5,
    single_bank_mac_repeats: int = 2,
    post_all_bank_mac_repeats: int = 1,
    include_all_bank_sequence: bool = True,
) -> list[dict]:
    if addr_vec_size <= 0:
        raise ValueError("addr_vec_size must be positive")
    if not (0 <= bank_level < addr_vec_size and 0 <= row_level < addr_vec_size and 0 <= col_level < addr_vec_size):
        raise ValueError("bank_level, row_level, and col_level must fit within addr_vec_size")
    if single_bank_mac_repeats <= 0:
        raise ValueError("single_bank_mac_repeats must be positive")
    if post_all_bank_mac_repeats <= 0:
        raise ValueError("post_all_bank_mac_repeats must be positive")

    base = [0] * addr_vec_size
    bank0 = list(base)
    bank0[bank_level] = 0
    bank0[row_level] = 0
    bank0[col_level] = 0
    bank1 = list(bank0)
    bank1[bank_level] = 1

    records = [
        _record("op_0000", "SB", base, notes="enter single-bank LPDDR5-PIM mode"),
        _record("op_0001", "PIM_MAC", bank0, repeat=single_bank_mac_repeats, notes="single-bank concrete MAC replay"),
    ]
    next_id = 2
    if include_all_bank_sequence:
        records.extend(
            [
                _record(f"op_{next_id:04d}", "HAB", base, notes="enter host all-bank mode for broadcast load"),
                _record(f"op_{next_id + 1:04d}", "PIM_BCAST", base, notes="all-bank broadcast load before all-bank compute"),
                _record(f"op_{next_id + 2:04d}", "HAB_PIM", base, notes="enter all-bank PIM mode after broadcast"),
                _record(f"op_{next_id + 3:04d}", "PIM_MAC_AB", base, notes="all-bank concrete MAC replay"),
                _record(f"op_{next_id + 4:04d}", "SB", base, notes="return to single-bank mode"),
            ]
        )
        next_id += 5
    records.append(_record(f"op_{next_id:04d}", "PIM_MAC", bank1, repeat=post_all_bank_mac_repeats, notes="single-bank MAC after bounded mode sequence"))
    validate_sequence(records)
    return records


def build_provenance_summary(records: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for record in records:
        counts[record["opcode"]] = counts.get(record["opcode"], 0) + int(record["repeat"])
    return {
        "schema_version": CONCRETE_SCHEMA_VERSION,
        "generator_version": CONCRETE_GENERATOR_VERSION,
        "record_counts_by_opcode": counts,
        "total_logical_records": len(records),
        "total_expanded_records": expanded_record_count(records),
        "claim_boundary": list(REQUIRED_BOUNDARY_CLAIMS),
        "non_claims": list(DEFAULT_NON_CLAIMS),
        "semantic_sources": [
            record["provenance"].get("semantic_source")
            for record in records
            if record.get("provenance", {}).get("semantic_source")
        ],
        "notes": "deterministic native LPDDR5-PIM concrete opcode trace for backend-specific command validation",
    }


def generate_concrete_artifacts(
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    single_bank_mac_repeats: int = 2,
    post_all_bank_mac_repeats: int = 1,
    include_all_bank_sequence: bool = True,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    records = generate_minimal_concrete_records(
        single_bank_mac_repeats=single_bank_mac_repeats,
        post_all_bank_mac_repeats=post_all_bank_mac_repeats,
        include_all_bank_sequence=include_all_bank_sequence,
    )
    summary = build_provenance_summary(records)
    trace_path = output_dir / "concrete_opcode_trace.jsonl"
    summary_path = output_dir / "provenance_summary.json"
    write_jsonl(records, trace_path)
    write_json(summary, summary_path)
    return trace_path, summary_path


def generate_concrete_artifacts_from_semantic(
    semantic_trace_path: Path | str,
    output_dir: Path | str = DEFAULT_SEMANTIC_OUTPUT_DIR,
    *,
    addr_vec_size: int = 6,
    bank_level: int = 3,
    row_level: int = 4,
    col_level: int = 5,
    manifest_name: str = "phase2_semantic_lowered_manifest",
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    semantic_records = load_semantic_records(semantic_trace_path)
    records = lower_semantic_records_to_concrete(
        semantic_records,
        addr_vec_size=addr_vec_size,
        bank_level=bank_level,
        row_level=row_level,
        col_level=col_level,
        manifest_name=manifest_name,
    )
    summary = build_provenance_summary(records)
    summary["semantic_trace_path"] = str(semantic_trace_path)
    summary["lowering_mode"] = "phase2-semantic-jsonl-to-native-lpddr5-pim-opcodes"
    trace_path = output_dir / "concrete_opcode_trace.jsonl"
    summary_path = output_dir / "provenance_summary.json"
    write_jsonl(records, trace_path)
    write_json(summary, summary_path)
    return trace_path, summary_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate deterministic native LPDDR5-PIM concrete opcode trace artifacts")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--single-bank-mac-repeats", type=int, default=2)
    parser.add_argument("--post-all-bank-mac-repeats", type=int, default=1)
    parser.add_argument("--skip-all-bank-sequence", action="store_true")
    parser.add_argument("--semantic-trace", type=Path, help="Lower a Phase 2 semantic JSONL trace into native LPDDR5-PIM concrete opcodes")
    parser.add_argument("--addr-vec-size", type=int, default=6)
    parser.add_argument("--bank-level", type=int, default=3)
    parser.add_argument("--row-level", type=int, default=4)
    parser.add_argument("--col-level", type=int, default=5)
    return parser


def main() -> int:
    opts = _build_arg_parser().parse_args()
    if opts.semantic_trace is not None:
        trace_path, summary_path = generate_concrete_artifacts_from_semantic(
            opts.semantic_trace,
            opts.output_dir,
            addr_vec_size=opts.addr_vec_size,
            bank_level=opts.bank_level,
            row_level=opts.row_level,
            col_level=opts.col_level,
        )
    else:
        trace_path, summary_path = generate_concrete_artifacts(
            opts.output_dir,
            single_bank_mac_repeats=opts.single_bank_mac_repeats,
            post_all_bank_mac_repeats=opts.post_all_bank_mac_repeats,
            include_all_bank_sequence=not opts.skip_all_bank_sequence,
        )
    print(f"Generated: {trace_path}")
    print(f"Generated: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
