"""Command-line interface for the simulator-owned PIMScope experiment API."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from ramulator.pimscope.capabilities import pim_backend_capabilities
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


def _git_revision(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _find_ramulator_root(package_path: Path) -> Path | None:
    for candidate in (package_path.parent.parent.parent, *package_path.parents):
        if (candidate / "CMakeLists.txt").exists() and (
            candidate / "python" / "ramulator"
        ).exists():
            return candidate
    return None


def _doctor_check(name: str, check) -> dict[str, Any]:
    try:
        details = check()
    except Exception as exc:  # doctor must report all failures in one invocation
        return {"name": name, "status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
    return {"name": name, "status": "PASS", **(details or {})}


def _cmd_doctor(args: argparse.Namespace) -> int:
    checks: list[dict[str, Any]] = []

    def python_check() -> dict[str, Any]:
        if sys.version_info < (3, 10):
            raise RuntimeError("Python >= 3.10 is required")
        return {"version": platform.python_version(), "supported": True}

    checks.append(_doctor_check("python", python_check))

    def package_check() -> dict[str, Any]:
        import ramulator

        package_path = Path(ramulator.__file__).resolve()
        root = _find_ramulator_root(package_path)
        details = {"package": str(package_path)}
        if root is not None:
            details.update({"repository": str(root), "commit": _git_revision(root)})
        return details

    checks.append(_doctor_check("ramulator-package", package_check))

    def native_check() -> dict[str, Any]:
        import ramulator._ramulator as native

        return {"extension": str(Path(native.__file__).resolve())}

    checks.append(_doctor_check("native-extension", native_check))

    def component_check() -> dict[str, Any]:
        import ramulator

        resolved = []
        for dram_class, org_preset, timing_preset in (
            ("LPDDR5PIM", "LPDDR5_8Gb_x16", "LPDDR5_6400"),
            ("LPDDR6PIM", "LPDDR6_16Gb_x12", "LPDDR6_10667_BL24"),
        ):
            dram = getattr(ramulator.dram, dram_class)(
                org_preset=org_preset,
                timing_preset=timing_preset,
                pim_datatype="int8",
                pim_banks_per_block=2,
                pim_mac_execution_model="shared_block_serial",
            )
            organization, timing = dram.resolve()
            resolved.append(
                {
                    "dram_class": type(dram).__name__,
                    "rank": organization.get("rank"),
                    "bank": organization.get("bank"),
                    "timing_tCK_ps": timing.get("tCK_ps"),
                }
            )
        return {"standards": resolved}

    checks.append(_doctor_check("pim-dram-components", component_check))
    checks.append(
        _doctor_check(
            "pim-backend-capabilities",
            lambda: {"backends": pim_backend_capabilities()},
        )
    )

    if args.config is not None:
        checks.append(
            _doctor_check(
                "manifest-backend",
                lambda: {
                    "manifest": str(args.config.resolve()),
                    "fingerprint": (_resolved := _load_resolved(args.config, [])).fingerprint,
                    "address_layout": validate_backend(_resolved)["address_layout"],
                },
            )
        )

    valid = all(check["status"] == "PASS" for check in checks)
    payload = {
        "doctor": "pimscope-doctor-v1",
        "valid": valid,
        "checks": checks,
    }
    print(_dump_json(payload), end="")
    return 0 if valid else 1


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
        description=(
            "Validate and run configurable LPDDR5-PIM workload-surrogate "
            "experiments (LPDDR5PIM supported; LPDDR6PIM experimental)"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser(
        "doctor", help="check the Python package, native extension, and optional manifest backend"
    )
    doctor.add_argument(
        "--config",
        type=Path,
        help="also validate this JSON/YAML manifest and its resolved backend",
    )
    doctor.set_defaults(func=_cmd_doctor)

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
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"{prog}: error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
