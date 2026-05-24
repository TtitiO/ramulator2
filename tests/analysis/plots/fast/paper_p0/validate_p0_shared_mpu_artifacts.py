#!/usr/bin/env python3
"""P0 Paper-facing artifact validator for LPDDR5-PIM Shared-MPU validation sweep.

Validates that generated artifacts meet all requirements and constraints.
"""

import csv
import re
from pathlib import Path


SOURCE_CSV = Path("ramulator2/tests/analysis/plots/fast/lpddr5_pim_shared_mpu_validation_sweep.csv")
OUTPUT_DIR = Path("ramulator2/tests/analysis/plots/fast/paper_p0")

REQUIRED_FILES = [
    "gen_p0_shared_mpu_artifacts.py",
    "validate_p0_shared_mpu_artifacts.py",
    "p0_shared_mpu_throughput_vs_active_banks.pdf",
    "p0_shared_mpu_throughput_vs_active_banks.png",
    "p0_nop1_edp_energy_summary.tex",
    "p0_nop1_edp_energy_summary.csv",
    "p0_latex_includes.tex",
]

FORBIDDEN_PATTERNS = [
    (r"(?<!non-)silicon-calibrated", "silicon-calibrated (without 'non-' prefix)"),
    (r"(?<!non-)silicon calibrated", "silicon calibrated (without 'non-' prefix)"),
    (r"speedup", "speedup"),
    (r"full workload-trace validation", "full workload-trace validation"),
    (r"per-command PIM energy", "per-command PIM energy"),
    (r"PIM MAC command energy", "PIM MAC command energy"),
]


def check_source_csv():
    """Validate source CSV has expected structure."""
    print("Checking source CSV...")
    
    with open(SOURCE_CSV, 'r', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    total_rows = len(rows)
    print(f"  Total rows: {total_rows}")
    
    if total_rows != 60:
        return False, f"Expected 60 source rows, got {total_rows}"
    
    nop1_rows = [r for r in rows if int(r['NOP']) == 1]
    print(f"  NOP=1 rows: {len(nop1_rows)}")
    
    if len(nop1_rows) != 12:
        return False, f"Expected 12 NOP=1 rows, got {len(nop1_rows)}"
    
    active_banks_set = set(int(r['active_banks']) for r in nop1_rows)
    expected_active = {4, 8, 16, 32}
    if active_banks_set != expected_active:
        return False, f"Expected active_banks {expected_active}, got {active_banks_set}"
    print(f"  Active banks: {sorted(active_banks_set)}")
    
    bpm_set = set(int(r['pim_banks_per_mpu']) for r in nop1_rows)
    expected_bpm = {1, 2, 4}
    if bpm_set != expected_bpm:
        return False, f"Expected pim_banks_per_mpu {expected_bpm}, got {bpm_set}"
    print(f"  Banks/MPU: {sorted(bpm_set)}")
    
    completed = set(int(r['completed_pim_requests']) for r in nop1_rows)
    if completed != {4096}:
        return False, f"Expected all completed_requests=4096, got {completed}"
    print(f"  All completed_requests: 4096")
    
    direct_valid = all(r['direct_total_energy_comparison_valid'] == 'True' for r in nop1_rows)
    if not direct_valid:
        return False, "Not all rows have direct_total_energy_comparison_valid=True"
    print(f"  All direct comparison valid: True")
    
    builtin_total = all(r['energy_attribution_mode'] == 'builtin_total' for r in nop1_rows)
    if not builtin_total:
        return False, "Not all rows have energy_attribution_mode=builtin_total"
    print(f"  All energy_attribution_mode: builtin_total")
    
    for r in nop1_rows:
        tp = float(r['pim_throughput'])
        edp = float(r['edp_pJ_ns'])
        energy = float(r['total_energy_pJ'])
        if not (0 < tp < float('inf')):
            return False, f"Invalid throughput: {tp}"
        if not (0 < edp < float('inf')):
            return False, f"Invalid EDP: {edp}"
        if not (0 < energy < float('inf')):
            return False, f"Invalid energy: {energy}"
    print(f"  All metrics positive and finite: True")
    
    return True, "Source CSV validation passed"


def check_output_files():
    """Check all required output files exist and are non-empty."""
    print("\nChecking output files...")
    
    for fname in REQUIRED_FILES:
        fpath = OUTPUT_DIR / fname
        if not fpath.exists():
            return False, f"Missing required file: {fname}"
        if fpath.stat().st_size == 0:
            return False, f"Empty file: {fname}"
        print(f"  {fname}: exists ({fpath.stat().st_size} bytes)")
    
    return True, "All output files present and non-empty"


def check_forbidden_words():
    """Check that no forbidden wording appears in generated files."""
    import re
    
    print("\nChecking for forbidden wording...")
    
    tex_files = [
        OUTPUT_DIR / "p0_nop1_edp_energy_summary.tex",
        OUTPUT_DIR / "p0_latex_includes.tex",
    ]
    
    for fpath in tex_files:
        content = fpath.read_text()
        for pattern, description in FORBIDDEN_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                return False, f"Forbidden wording '{description}' found in {fpath.name}"
    
    print(f"  No forbidden wording found")
    return True, "No forbidden wording detected"


def check_csv_content():
    """Validate generated CSV has correct structure and data."""
    print("\nChecking generated CSV content...")
    
    csv_path = OUTPUT_DIR / "p0_nop1_edp_energy_summary.csv"
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if len(rows) != 12:
        return False, f"Expected 12 rows in CSV, got {len(rows)}"
    
    required_cols = [
        'active_banks', 'banks_per_mpu', 'throughput_req_ns',
        'energy_per_req_pJ', 'edp_pJ_ns', 'completed_requests'
    ]
    header = rows[0].keys()
    for col in required_cols:
        if col not in header:
            return False, f"Missing column in CSV: {col}"
    
    for r in rows:
        if int(r['completed_requests']) != 4096:
            return False, f"Expected completed_requests=4096, got {r['completed_requests']}"
    
    print(f"  CSV rows: {len(rows)}")
    print(f"  All required columns present: True")
    print(f"  All completed_requests = 4096: True")
    
    return True, "CSV content valid"


def check_latex_content():
    """Validate generated LaTeX has required disclaimers."""
    print("\nChecking LaTeX content...")
    
    tex_path = OUTPUT_DIR / "p0_nop1_edp_energy_summary.tex"
    content = tex_path.read_text()
    
    required_disclaimers = [
        "simulator-internal",
        "non-silicon-calibrated",
        "4096-request",
    ]
    
    for disclaimer in required_disclaimers:
        if disclaimer not in content:
            return False, f"Missing required disclaimer: {disclaimer}"
    
    print(f"  Required disclaimers present: True")
    print(f"  Contains 'simulator-internal': True")
    print(f"  Contains 'non-silicon-calibrated': True")
    print(f"  Contains '4096-request': True")
    
    return True, "LaTeX content valid"


def check_latex_includes():
    """Validate LaTeX includes file has proper structure."""
    print("\nChecking LaTeX includes file...")
    
    includes_path = OUTPUT_DIR / "p0_latex_includes.tex"
    content = includes_path.read_text()
    
    if "simulator-internal" not in content:
        return False, "Missing 'simulator-internal' in includes file"
    
    if "non-silicon-calibrated" not in content:
        return False, "Missing 'non-silicon-calibrated' in includes file"
    
    if "4096-request" not in content:
        return False, "Missing '4096-request' in includes file"
    
    if "\\begin{figure}" not in content:
        return False, "Missing figure environment in includes file"
    
    if "\\caption" not in content:
        return False, "Missing caption in includes file"
    
    print(f"  Required disclaimers present: True")
    print(f"  Figure environment present: True")
    print(f"  Caption with disclaimers: True")
    
    return True, "LaTeX includes file valid"


def main():
    """Run all validation checks."""
    print("=" * 60)
    print("P0 Paper-Facing Artifact Validator")
    print("=" * 60)
    
    checks = [
        ("Source CSV", check_source_csv),
        ("Output Files", check_output_files),
        ("Forbidden Words", check_forbidden_words),
        ("CSV Content", check_csv_content),
        ("LaTeX Content", check_latex_content),
        ("LaTeX Includes", check_latex_includes),
    ]
    
    all_passed = True
    results = []
    
    for name, check_fn in checks:
        try:
            passed, msg = check_fn()
            results.append((name, passed, msg))
            if not passed:
                all_passed = False
        except Exception as e:
            results.append((name, False, f"Exception: {e}"))
            all_passed = False
    
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    for name, passed, msg in results:
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {name}: {status} - {msg}")
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ ALL VALIDATION CHECKS PASSED")
        return 0
    else:
        print("\n✗ SOME VALIDATION CHECKS FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
