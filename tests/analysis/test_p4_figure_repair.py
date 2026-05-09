from __future__ import annotations

from pathlib import Path


def test_p4_figure_generators_do_not_embed_latex_figure_numbers() -> None:
    root = Path(__file__).resolve().parent / "figures"
    for rel in ["gen_p4_figures.py", "gen_p4_backend_figures.py"]:
        text = (root / rel).read_text()
        for forbidden in ["Figure 4:", "Figure 5:", "Figure 6:", "Figure 7:", "Figure 8:", "Figure 9:"]:
            assert forbidden not in text


def test_replay_validation_is_exported_as_latex_table_not_raster_figure(tmp_path: Path) -> None:
    from tests.analysis.figures import gen_p4_figures

    rows = [
        {
            "trace_name": "attention_serialized",
            "semantic_records": 8,
            "concrete_records": 12,
            "replay_status": "PASS",
            "pim_mac_issued": 4,
            "runtime_ns": 42.5,
        }
    ]
    gen_p4_figures.write_replay_validation_table(rows, tmp_path)

    table = tmp_path / "p4_replay_validation.tex"
    assert table.exists()
    text = table.read_text()
    assert "attention_serialized" in text
    assert "PASS" in text
    assert "tCK" in text
    assert not (tmp_path / "fig6_replay_validation.png").exists()
    assert not (tmp_path / "fig6_replay_validation.pdf").exists()


def test_backend_stall_summary_omits_figure_when_stalls_are_all_zero(tmp_path: Path) -> None:
    from tests.analysis.figures import gen_p4_backend_figures

    stats = {
        name: {
            "pim_dependency_stalls": 0,
            "pim_capacity_stalls": 0,
            "mpu_group_busy_cycles": 0,
            "pim_mac_issued": 4,
        }
        for name in [
            "attention_serialized",
            "attention_overlapped",
            "ffn_swiglu",
            "moe_top2",
            "combined_layer",
        ]
    }

    gen_p4_backend_figures.gen_stall_summary_or_figure(tmp_path, stats)

    assert not (tmp_path / "fig9_stall_breakdown.png").exists()
    assert not (tmp_path / "fig9_stall_breakdown.pdf").exists()
    summary = tmp_path / "p4_stall_summary.tex"
    assert summary.exists()
    assert "zero" in summary.read_text().lower()


def test_system_architecture_svg_uses_publication_facing_language() -> None:
    svg = Path("paper/figures/fig1_system_architecture.svg").read_text()
    for forbidden in ["(P3/4)", "(P4)", "fake hardware commands", "Explicit non-claim"]:
        assert forbidden not in svg
    assert "not emitted as hardware commands" in svg


def test_p0_caption_discloses_32_bank_context_without_overstating_convergence() -> None:
    text = Path(
        "ramulator2/tests/analysis/plots/fast/paper_p0/gen_p0_shared_mpu_artifacts.py"
    ).read_text()
    assert "converges with the dedicated baseline" not in text
    assert "32-bank point uses the 32-bank diagnostic configuration" in text
    assert "excludes unmodeled PIM compute-array energy" in text
