"""Command-line interface for the simulator-owned PIMScope experiment API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ramulator.pimscope.config import (
    apply_overrides,
    load_raw_manifest,
    resolve_experiment_manifest,
)
from ramulator.pimscope.experiment import run_experiment, validate_backend
from ramulator.pimscope.schema import (
    load_json_object,
    validate_aggregate,
    validate_result,
    validate_trace_file,
)


def _dump_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _load_resolved(path: Path, overrides: list[str]):
    raw = apply_overrides(load_raw_manifest(path), overrides)
    return resolve_experiment_manifest(raw, source=str(path.resolve()))


def _cmd_validate(args: argparse.Namespace) -> int:
    resolved = _load_resolved(args.config, args.set)
    payload: dict[str, Any] = {
        "valid": True,
        "manifest_fingerprint": resolved.fingerprint,
        "resolved_manifest": resolved.manifest,
    }
    if not args.no_backend:
        payload["resolved_backend"] = validate_backend(resolved)
    print(_dump_json(payload), end="")
    return 0


def _cmd_validate_result(args: argparse.Namespace) -> int:
    result = validate_result(load_json_object(args.result))
    print(_dump_json({"valid": True, "schema_version": result["schema_version"]}), end="")
    return 0


def _cmd_validate_aggregate(args: argparse.Namespace) -> int:
    aggregate = validate_aggregate(load_json_object(args.aggregate), kind=args.kind)
    print(_dump_json({"valid": True, "rows": len(aggregate["rows"])}), end="")
    return 0


def _cmd_validate_trace(args: argparse.Namespace) -> int:
    layout_payload = validate_backend(_load_resolved(args.config, []))["address_layout"]
    summary = validate_trace_file(
        args.trace,
        address_layout={
            "mapping_version": layout_payload["mapping_version"],
            "level_names": layout_payload["level_names"],
            "level_sizes": layout_payload["level_sizes"],
            "internal_prefetch_size": layout_payload["internal_prefetch_size"],
            "tx_bytes": layout_payload["tx_bytes"],
            "capacity_bytes": layout_payload["capacity_bytes"],
        },
        max_expanded_records=args.max_expanded_records,
    )
    print(_dump_json(summary), end="")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    resolved = _load_resolved(args.config, args.set)
    result = run_experiment(resolved, provenance=args.provenance)
    output = Path(args.output or resolved.manifest["output"]["path"])
    if not output.is_absolute():
        output = args.output_base / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_dump_json(result), encoding="utf-8")
    print(f"result: {output}")
    print(f"status: {result['status']}")
    print(f"cycles: {result['simulation']['cycles']}")
    print(f"manifest fingerprint: {resolved.fingerprint}")
    return 0 if result["status"] == "PASS" else 1


def build_parser(*, prog: str = "ramulator-pimscope") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Validate and run configurable LPDDR5-PIM workload-surrogate experiments",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate", help="validate a JSON/YAML manifest and print the resolved configuration"
    )
    validate.add_argument("config", type=Path)
    validate.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="override a manifest field; VALUE accepts JSON syntax",
    )
    validate.add_argument(
        "--no-backend",
        action="store_true",
        help="perform schema validation without importing the compiled Ramulator backend",
    )
    validate.set_defaults(func=_cmd_validate)

    result = subparsers.add_parser(
        "validate-result", help="validate a JSON result against the result schema"
    )
    result.add_argument("result", type=Path)
    result.set_defaults(func=_cmd_validate_result)

    aggregate = subparsers.add_parser(
        "validate-aggregate", help="validate a paper-artifact aggregate JSON file"
    )
    aggregate.add_argument("aggregate", type=Path)
    aggregate.add_argument(
        "--kind",
        choices=("decode_cycles", "prefill_cycles", "pim_sharing_comparison"),
        required=True,
    )
    aggregate.set_defaults(func=_cmd_validate_aggregate)

    trace = subparsers.add_parser(
        "validate-trace", help="validate a concrete opcode JSONL trace against a manifest"
    )
    trace.add_argument("trace", type=Path)
    trace.add_argument("--config", type=Path, required=True)
    trace.add_argument("--max-expanded-records", type=int)
    trace.set_defaults(func=_cmd_validate_trace)

    run = subparsers.add_parser(
        "run", help="generate, lower, and replay one validated workload experiment"
    )
    run.add_argument("config", type=Path)
    run.add_argument("--output", type=Path, help="override output.path")
    run.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="override a manifest field; VALUE accepts JSON syntax",
    )
    run.set_defaults(func=_cmd_run)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    prog: str = "ramulator-pimscope",
    output_base: Path | None = None,
    provenance: dict[str, Any] | None = None,
) -> int:
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)
    args.output_base = Path.cwd() if output_base is None else Path(output_base)
    args.provenance = provenance
    try:
        return args.func(args)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"{prog}: error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
