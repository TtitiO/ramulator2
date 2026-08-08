"""Run one validated LPDDR5-PIM workload-surrogate experiment."""

from pathlib import Path

from ramulator.pimscope import load_experiment_manifest, run_experiment

HERE = Path(__file__).resolve().parent
manifest = load_experiment_manifest(HERE / "pimscope_custom_model.json")
result = run_experiment(manifest)

print(f"Status:             {result['status']}")
print(f"Controller cycles:  {result['simulation']['cycles']}")
print(f"Concrete records:   {result['workload_summary']['concrete_records']}")
print(f"Manifest SHA-256:   {result['manifest_fingerprint']}")

if result["status"] != "PASS":
    raise SystemExit(1)
