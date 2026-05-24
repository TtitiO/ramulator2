#!/usr/bin/env python3
"""P0 Paper-facing artifact generator for LPDDR5-PIM Shared-MPU validation sweep.

Generates publication-quality figures and tables from the normalized sweep data.
Source: ramulator2/tests/analysis/plots/fast/lpddr5_pim_shared_mpu_validation_sweep.csv

Artifacts produced:
- p0_shared_mpu_throughput_vs_active_banks.pdf (vector, publication-quality)
- p0_shared_mpu_throughput_vs_active_banks.png (PNG fallback)
- p0_nop1_edp_energy_summary.tex (LaTeX table)
- p0_nop1_edp_energy_summary.csv (CSV table)
- p0_latex_includes.tex (LaTeX include snippets)
"""

import os
import sys
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SOURCE_CSV = Path("ramulator2/tests/analysis/plots/fast/lpddr5_pim_shared_mpu_validation_sweep.csv")
OUTPUT_DIR = Path("ramulator2/tests/analysis/plots/fast/paper_p0")
DPI = 300
FONT_SIZE = 10


def setup_publication_style():
    """Configure matplotlib for publication-quality output."""
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times', 'serif'],
        'axes.labelsize': FONT_SIZE,
        'axes.titlesize': FONT_SIZE + 1,
        'xtick.labelsize': FONT_SIZE - 1,
        'ytick.labelsize': FONT_SIZE - 1,
        'legend.fontsize': FONT_SIZE - 1,
        'figure.dpi': DPI,
        'savefig.dpi': DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'text.usetex': False,
        'mathtext.fontset': 'stix',
    })


def load_and_filter_data():
    """Load CSV and filter for NOP=1 rows only.
    
    Returns list of dicts with 12 rows (4 active_banks × 3 pim_banks_per_mpu).
    """
    data = []
    with open(SOURCE_CSV, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            nop = int(row['NOP'])
            if nop == 1:
                data.append({
                    'active_banks': int(row['active_banks']),
                    'pim_banks_per_mpu': int(row['pim_banks_per_mpu']),
                    'throughput': float(row['pim_throughput']),
                    'energy_per_req': float(row['energy_per_completed_pim_request_pJ']),
                    'edp': float(row['edp_pJ_ns']),
                    'completed_requests': int(row['completed_pim_requests']),
                    'avg_latency_ns': float(row['avg_pim_latency_ns']),
                    'total_energy_pJ': float(row['total_energy_pJ']),
                    'energy_attribution_mode': row['energy_attribution_mode'],
                    'direct_comparison_valid': row['direct_total_energy_comparison_valid'],
                })
    data.sort(key=lambda x: (x['active_banks'], x['pim_banks_per_mpu']))
    return data


def generate_throughput_plot(data):
    """Generate throughput vs active banks line plot.
    
    Three lines for pim_banks_per_mpu = 1, 2, 4.
    """
    setup_publication_style()
    banks_per_mpu_data = {1: [], 2: [], 4: []}
    for row in data:
        bpm = row['pim_banks_per_mpu']
        banks_per_mpu_data[bpm].append((row['active_banks'], row['throughput']))
    for bpm in banks_per_mpu_data:
        banks_per_mpu_data[bpm].sort(key=lambda x: x[0])
    
    fig, ax = plt.subplots(figsize=(5, 3.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    
    colors = {1: '#0072B2', 2: '#D55E00', 4: '#009E73'}
    markers = {1: 'o', 2: 's', 4: '^'}
    linestyles = {1: '-', 2: '--', 4: ':'}
    labels = {1: 'banks/MPU = 1', 2: 'banks/MPU = 2', 4: 'banks/MPU = 4'}
    
    for bpm in [1, 2, 4]:
        x_vals = [p[0] for p in banks_per_mpu_data[bpm]]
        y_vals = [p[1] for p in banks_per_mpu_data[bpm]]
        ax.plot(x_vals, y_vals, marker=markers[bpm], color=colors[bpm], linestyle=linestyles[bpm],
                linewidth=2.0, markersize=6, label=labels[bpm], zorder=3)
    
    ax.set_xlabel('Active Banks', fontsize=FONT_SIZE, labelpad=6)
    ax.set_ylabel('Throughput (requests/ns)', fontsize=FONT_SIZE, labelpad=6)
    ax.set_title('PIM Throughput vs Active Banks (NOP=1)', fontsize=FONT_SIZE + 1, pad=10)
    
    ax.set_xscale('log', base=2)
    ax.set_xticks([4, 8, 16, 32])
    ax.set_xticklabels(['4', '8', '16', '32'])
    ax.set_xlim(3.5, 36)
    ax.set_ylim(bottom=0)
    
    ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.35)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE - 1)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    
    legend = ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=FONT_SIZE - 1, frameon=True,
                       framealpha=0.95, edgecolor='#888')
    legend.get_frame().set_linewidth(0.9)
    pdf_path = OUTPUT_DIR / 'p0_shared_mpu_throughput_vs_active_banks.pdf'
    png_path = OUTPUT_DIR / 'p0_shared_mpu_throughput_vs_active_banks.png'
    
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.08)
    fig.savefig(png_path, format='png', dpi=DPI, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
    
    print(f"Generated: {pdf_path}")
    print(f"Generated: {png_path}")
    return pdf_path, png_path


def generate_edp_table(data):
    """Generate EDP/energy summary table in CSV and LaTeX formats."""
    csv_path = OUTPUT_DIR / 'p0_nop1_edp_energy_summary.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'active_banks', 'banks_per_mpu', 'throughput_req_ns',
            'energy_per_req_pJ', 'edp_pJ_ns', 'completed_requests'
        ])
        for row in data:
            writer.writerow([
                row['active_banks'],
                row['pim_banks_per_mpu'],
                f"{row['throughput']:.6f}",
                f"{row['energy_per_req']:.6f}",
                f"{row['edp']:.2f}",
                row['completed_requests']
            ])
    print(f"Generated: {csv_path}")
    tex_path = OUTPUT_DIR / 'p0_nop1_edp_energy_summary.tex'
    with open(tex_path, 'w') as f:
        f.write("% P0 NOP=1 EDP and Energy Summary Table\n")
        f.write("% Source: LPDDR5-PIM Shared-MPU Validation Sweep (NOP=1 subset)\n")
        f.write("% NOTE: All values are simulator-internal estimates (non-silicon-calibrated).\n")
        f.write("% The workload is a same-NOP synthetic 4096-request PIMCompute stream.\n")
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{LPDDR5-PIM Shared-MPU Energy and EDP Summary (NOP=1). "
                "All values are simulator-internal diagnostics (non-silicon-calibrated) "
                "for a same-NOP synthetic 4096-request PIMCompute stream; the summary "
                "excludes unmodeled PIM compute-array energy.}\n")
        f.write("\\label{tab:p0_nop1_edp_summary}\n")
        f.write("\\begin{tabular}{rrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Active Banks & Banks/MPU & Throughput & Energy/Req & EDP & Completed \\\n")
        f.write(" &  & (req/ns) & (pJ) & (pJ$\\cdot$ns) & Requests \\\n")
        f.write("\\midrule\n")
        
        for row in data:
            f.write(f"{row['active_banks']} & {row['pim_banks_per_mpu']} & "
                    f"{row['throughput']:.4f} & {row['energy_per_req']:.4f} & "
                    f"{row['edp']:.2f} & {row['completed_requests']} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"Generated: {tex_path}")
    return csv_path, tex_path


def generate_latex_includes(data):
    """Generate LaTeX include snippets for easy integration."""
    latex_path = OUTPUT_DIR / 'p0_latex_includes.tex'
    with open(latex_path, 'w') as f:
        f.write("% P0 Paper-Facing LaTeX Includes\n")
        f.write("% Generated from LPDDR5-PIM Shared-MPU Validation Sweep (NOP=1 subset)\n")
        f.write("%\n")
        f.write("% IMPORTANT DISCLAIMERS:\n")
        f.write("% - The 32-bank point uses the 32-bank diagnostic configuration.\n")
        f.write("% - All values are simulator-internal diagnostics (non-silicon-calibrated).\n")
        f.write("% - Workload: same-NOP synthetic 4096-request PIMCompute stream.\n")
        f.write("% - Energy attribution mode: builtin_total (direct simulator accounting).\n")
        f.write("%\n\n")
        f.write("% === Figure: Throughput vs Active Banks ===\n")
        f.write("\\begin{figure}[t]\n")
        f.write("\\centering\n")
        f.write("\\includegraphics[width=0.48\\textwidth]{figures/p0_shared_mpu_throughput_vs_active_banks.pdf}\n")
        f.write("\\caption{PIM throughput vs. active banks at NOP=1 (minimum inter-PIM gap). "
                "Lines show different banks-per-MPU configurations. "
                "When multiple banks share one MPU group, throughput degrades in the "
                "bank-limited regime ($\\leq$8 banks). The 32-bank point uses the 32-bank "
                "diagnostic configuration and should not be mixed with lower-bank points as "
                "a same-total-bank scaling claim. At 16--32 active banks, banks/MPU=2 remains "
                "close to the dedicated-MPU diagnostic baseline while banks/MPU=4 retains "
                "a visible gap. The EDP/energy summary excludes unmodeled PIM compute-array energy. "
                "\\textbf{Simulator-internal diagnostics (non-silicon-calibrated)} for a "
                "same-NOP synthetic 4096-request PIMCompute stream.}\n")
        f.write("\\label{fig:p0_throughput_vs_banks}\n")
        f.write("\\end{figure}\n\n")
        f.write("% === Table: EDP and Energy Summary ===\n")
        f.write("\\input{tables/p0_nop1_edp_energy_summary.tex}\n\n")
        f.write("% === Methodology Snippet ===\n")
        f.write("% Total energy is computed directly by the simulator's built-in power model "
                "(energy\\_attribution\\_mode = builtin\\_total). "
                "Comparisons between configurations with the same NOP value are quantitatively sound. "
                "All values are simulator-internal and non-silicon-calibrated; EDP/energy "
                "excludes unmodeled PIM compute-array energy. The 32-bank point uses the "
                "32-bank diagnostic configuration.\n")
    
    print(f"Generated: {latex_path}")
    return latex_path


def main():
    """Main entry point."""
    print("=" * 60)
    print("P0 Paper-Facing Artifact Generator")
    print("=" * 60)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Source CSV: {SOURCE_CSV}")
    if not SOURCE_CSV.exists():
        print(f"ERROR: Source CSV not found: {SOURCE_CSV}")
        sys.exit(1)
    print("\nLoading data and filtering NOP=1...")
    data = load_and_filter_data()
    print(f"Loaded {len(data)} rows (NOP=1 subset of 60 total rows)")
    
    if len(data) != 12:
        print(f"WARNING: Expected 12 NOP=1 rows (4 banks × 3 configs), got {len(data)}")
    print("\nGenerating artifacts...")
    generate_throughput_plot(data)
    generate_edp_table(data)
    generate_latex_includes(data)
    
    print("\n" + "=" * 60)
    print("Artifact generation complete!")
    print("=" * 60)
    print(f"\nOutput files in: {OUTPUT_DIR}")
    print("  - p0_shared_mpu_throughput_vs_active_banks.pdf")
    print("  - p0_shared_mpu_throughput_vs_active_banks.png")
    print("  - p0_nop1_edp_energy_summary.tex")
    print("  - p0_nop1_edp_energy_summary.csv")
    print("  - p0_latex_includes.tex")


if __name__ == "__main__":
    main()
