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
