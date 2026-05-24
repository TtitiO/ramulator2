from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


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
            "command_counts": {"PIM_MAC": 4, "PIM_BCAST": 6},
            "runtime_ns": 42.5,
        }
    ]
    gen_p4_figures.write_replay_validation_table(rows, tmp_path)

    table = tmp_path / "p4_replay_validation.tex"
    assert table.exists()
    text = table.read_text()
    assert "attention_serialized" in text
    assert "PASS" in text
    assert r"PIM\_BCAST" in text
    assert "tiny surrogate rows" in text
    assert "Llama2-7B rows use full-depth surrogate dimensions" in text
    assert "32 scalar MACs" in text
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
    assert "not emitted" in svg  # publication-facing phrasing


def test_p0_caption_discloses_32_bank_context_without_overstating_convergence() -> None:
    text = Path(
        "ramulator2/tests/analysis/plots/fast/paper_p0/gen_p0_shared_mpu_artifacts.py"
    ).read_text()
    assert "converges with the dedicated baseline" not in text
    assert "32-bank point uses the 32-bank diagnostic configuration" in text
    assert "excludes unmodeled PIM compute-array energy" in text


def test_llama2_replay_stats_delegate_to_backend_simulation(monkeypatch) -> None:
    from tests.analysis.figures import p4_figure_data

    p4_figure_data._collect_llama2_7b_backend_stats_cached.cache_clear()
    calls = []

    def fake_backend_stats():
        calls.append("backend")
        return {
            "llama2_7b_32_layer_steady_state": {
                "manifest_name": "llama2_7b_32_layer_steady_state",
                "semantic_records": 10,
                "concrete_records": 20,
                "replay_ok": True,
                "frontend_stats": {"opcode_requests_sent": 7, "opcode_requests_completed": 7},
                "pim_mac_issued": 5,
                "runtime_ns": 12.5,
                "cycles": 20,
                "command_counts": {"PIM_MAC": 100, "PIM_BCAST": 3},
            },
            "llama2_7b_32_layer_cold_start": {
                "manifest_name": "llama2_7b_32_layer_cold_start",
                "semantic_records": 10,
                "concrete_records": 30,
                "replay_ok": True,
                "frontend_stats": {"opcode_requests_sent": 9, "opcode_requests_completed": 9},
                "pim_mac_issued": 5,
                "runtime_ns": 15.0,
                "cycles": 24,
                "command_counts": {"PIM_MAC": 100, "PIM_BCAST": 9},
            },
        }

    monkeypatch.setattr(
        "tests.analysis.figures.p4_backend_data.collect_all_backend_stats_llama2_7b",
        fake_backend_stats,
    )

    rows = p4_figure_data.collect_llama2_7b_replay_stats()

    assert calls == ["backend"]
    assert [row["trace_name"] for row in rows] == [
        "llama2_7b_32_layer_steady_state",
        "llama2_7b_32_layer_cold_start",
    ]
    assert all(row["data_source"] == "real_backend_simulation" for row in rows)
    assert rows[0]["replay_status"] == "PASS"
    assert rows[1]["command_counts"]["PIM_BCAST"] > rows[0]["command_counts"]["PIM_BCAST"]
    p4_figure_data._collect_llama2_7b_backend_stats_cached.cache_clear()


def test_llama2_replay_stats_reuse_backend_simulation_within_process(monkeypatch) -> None:
    from tests.analysis.figures import p4_figure_data

    p4_figure_data._collect_llama2_7b_backend_stats_cached.cache_clear()
    calls = []

    def fake_backend_stats():
        calls.append("backend")
        return {
            "llama2_7b_32_layer_steady_state": {
                "semantic_records": 10,
                "concrete_records": 20,
                "replay_ok": True,
                "frontend_stats": {"opcode_requests_sent": 7, "opcode_requests_completed": 7},
                "pim_mac_issued": 5,
                "runtime_ns": 12.5,
                "cycles": 20,
                "command_counts": {"PIM_MAC": 100, "PIM_BCAST": 3},
            },
            "llama2_7b_32_layer_cold_start": {
                "semantic_records": 10,
                "concrete_records": 30,
                "replay_ok": True,
                "frontend_stats": {"opcode_requests_sent": 9, "opcode_requests_completed": 9},
                "pim_mac_issued": 5,
                "runtime_ns": 15.0,
                "cycles": 24,
                "command_counts": {"PIM_MAC": 100, "PIM_BCAST": 9},
            },
        }

    monkeypatch.setattr(
        "tests.analysis.figures.p4_backend_data.collect_all_backend_stats_llama2_7b",
        fake_backend_stats,
    )

    first = p4_figure_data.collect_llama2_7b_replay_stats()
    second = p4_figure_data.collect_llama2_7b_replay_stats()

    assert calls == ["backend"]
    assert first == second
    p4_figure_data._collect_llama2_7b_backend_stats_cached.cache_clear()


def test_p4_figures_use_simplified_current_text() -> None:
    text = Path("ramulator2/tests/analysis/figures/gen_p4_figures.py").read_text()

    assert "Backend replay" in text
    assert "Steady-state" in text
    assert "Cold-start" in text
    assert "Decode-block v2 scope" in text
    assert "Q/K/V/O projections plus QK^T + PV attention core" in text
    assert "Semantic-only:" not in text
    assert "Tiny deterministic routing" not in text
    assert "SB/HAB" not in text


def test_llama2_dense_decoder_data_cross_checks_formula_against_concrete(monkeypatch) -> None:
    from tests.analysis.figures import p4_figure_data

    monkeypatch.setattr(p4_figure_data, "collect_llama2_7b_replay_stats", lambda: [])

    data = p4_figure_data.collect_llama2_7b_dense_decoder_data()

    assert data["pim_mac_lanes"] == 32
    assert data["qkvo_projection_pim_mac_per_layer"] == 2_097_152
    assert data["attention_pim_mac_per_layer"] == 262_144
    assert data["ffn_pim_mac_per_layer"] == 4_227_072
    assert data["expected_pim_mac_repeats"] == 210_763_776
    assert data["total_pim_mac_repeats"] == data["expected_pim_mac_repeats"]
    assert data["concrete_per_layer"]["PIM_MAC"] == 6_586_368


def test_llama2_dense_decoder_data_accepts_nondefault_past_len(monkeypatch) -> None:
    from tests.analysis.figures import p4_figure_data

    monkeypatch.setattr(p4_figure_data, "collect_llama2_7b_replay_stats", lambda: [])

    short_context = p4_figure_data.collect_llama2_7b_dense_decoder_data(past_len=128)
    default_context = p4_figure_data.collect_llama2_7b_dense_decoder_data(past_len=1024)

    assert short_context["past_len"] == 128
    assert default_context["past_len"] == 1024
    assert short_context["qkvo_projection_pim_mac_per_layer"] == default_context["qkvo_projection_pim_mac_per_layer"]
    assert short_context["ffn_pim_mac_per_layer"] == default_context["ffn_pim_mac_per_layer"]
    assert short_context["attention_pim_mac_per_layer"] < default_context["attention_pim_mac_per_layer"]
    assert short_context["total_pim_mac_repeats"] < default_context["total_pim_mac_repeats"]


def test_llama2_13b_dense_decoder_data_cross_checks_formula_against_concrete(monkeypatch) -> None:
    from tests.analysis.figures import p4_figure_data

    monkeypatch.setattr(p4_figure_data, "collect_llama2_13b_replay_stats", lambda: [])

    data = p4_figure_data.collect_llama2_13b_dense_decoder_data()

    assert data["num_layers"] == 40
    assert data["hidden_size"] == 5120
    assert data["num_heads"] == 40
    assert data["head_dim"] == 128
    assert data["ffn_hidden_size"] == 13824
    assert data["past_len"] == 1024
    assert data["pim_mac_lanes"] == 32
    assert data["qkvo_projection_pim_mac_per_layer"] == 3_276_800
    assert data["attention_pim_mac_per_layer"] == 327_680
    assert data["ffn_pim_mac_per_layer"] == 6_635_520
    assert data["expected_pim_mac_repeats"] == 409_600_000
    assert data["total_pim_mac_repeats"] == data["expected_pim_mac_repeats"]
    assert data["concrete_per_layer"]["PIM_MAC"] == 10_240_000


def test_llama2_13b_replay_stats_delegate_to_backend_simulation(monkeypatch) -> None:
    from tests.analysis.figures import p4_figure_data

    p4_figure_data._collect_llama2_13b_backend_stats_cached.cache_clear()
    calls = []

    def fake_backend_stats():
        calls.append("backend")
        return {
            "llama2_13b_40_layer_steady_state": {
                "semantic_records": 10,
                "concrete_records": 20,
                "replay_ok": True,
                "frontend_stats": {"opcode_requests_sent": 7, "opcode_requests_completed": 7},
                "pim_mac_issued": 5,
                "runtime_ns": 12.5,
                "cycles": 20,
                "command_counts": {"PIM_MAC": 100, "PIM_BCAST": 3},
            },
            "llama2_13b_40_layer_cold_start": {
                "semantic_records": 10,
                "concrete_records": 30,
                "replay_ok": True,
                "frontend_stats": {"opcode_requests_sent": 9, "opcode_requests_completed": 9},
                "pim_mac_issued": 5,
                "runtime_ns": 15.0,
                "cycles": 24,
                "command_counts": {"PIM_MAC": 100, "PIM_BCAST": 9},
            },
        }

    monkeypatch.setattr(
        "tests.analysis.figures.p4_backend_data.collect_all_backend_stats_llama2_13b",
        fake_backend_stats,
    )

    rows = p4_figure_data.collect_llama2_13b_replay_stats()

    assert calls == ["backend"]
    assert [row["trace_name"] for row in rows] == [
        "llama2_13b_40_layer_steady_state",
        "llama2_13b_40_layer_cold_start",
    ]
    assert all(row["data_source"] == "real_backend_simulation" for row in rows)
    assert rows[0]["replay_status"] == "PASS"
    assert all(row["runtime_ns"] > 0 for row in rows)
    assert all(row["cycles"] > 0 for row in rows)
    p4_figure_data._collect_llama2_13b_backend_stats_cached.cache_clear()


def test_gen_figure_6_includes_llama2_13b_replay_rows(monkeypatch, tmp_path: Path) -> None:
    from tests.analysis.figures import gen_p4_figures
    from tests.analysis.figures import p4_figure_data

    captured_rows = []

    monkeypatch.setattr(p4_figure_data, "collect_replay_stats", lambda: [])
    monkeypatch.setattr(p4_figure_data, "collect_llama2_7b_replay_stats", lambda: [])
    monkeypatch.setattr(
        p4_figure_data,
        "collect_llama2_13b_replay_stats",
        lambda: [
            {
                "trace_name": "llama2_13b_40_layer_steady_state",
                "semantic_records": 10,
                "concrete_records": 20,
                "replay_status": "PASS",
                "pim_mac_issued": 100,
                "runtime_ns": 1.25,
                "cycles": 2,
                "command_counts": {"PIM_MAC": 100},
            }
        ],
    )
    monkeypatch.setattr(gen_p4_figures, "write_replay_validation_table", lambda rows, output_dir: captured_rows.extend(rows))

    gen_p4_figures.gen_figure_6(tmp_path)

    assert [row["trace_name"] for row in captured_rows] == ["llama2_13b_40_layer_steady_state"]


def test_gen_multi_model_llama2_latency_breakdown_writes_model_named_file(monkeypatch, tmp_path: Path) -> None:
    from tests.analysis.figures import gen_p4_figures

    saved = []

    monkeypatch.setattr(
        gen_p4_figures,
        "_save",
        lambda fig, output_dir, name: saved.append((output_dir, name)),
    )
    monkeypatch.setattr(
        "tests.analysis.figures.p4_figure_data.collect_llama2_7b_dense_decoder_data",
        lambda: {
            "model_name": "Llama2-7B",
            "backend_replay_stats": [],
            "qkvo_projection_pim_mac_per_layer": 1,
            "attention_pim_mac_per_layer": 1,
            "ffn_pim_mac_per_layer": 1,
        },
    )
    monkeypatch.setattr(
        "tests.analysis.figures.p4_figure_data.collect_llama2_13b_dense_decoder_data",
        lambda: {
            "model_name": "Llama2-13B",
            "backend_replay_stats": [],
            "qkvo_projection_pim_mac_per_layer": 2,
            "attention_pim_mac_per_layer": 2,
            "ffn_pim_mac_per_layer": 2,
        },
    )

    gen_p4_figures.gen_multi_model_llama2_latency_breakdown(tmp_path)

    assert saved
    assert any("llama2" in name and ("models" in name or "7b_13b" in name) for _, name in saved)


def test_gen_multi_model_llama2_latency_breakdown_places_model_specs_in_same_figure(monkeypatch, tmp_path: Path) -> None:
    from tests.analysis.figures import gen_p4_figures

    saved_figures = []

    monkeypatch.setattr(
        gen_p4_figures,
        "_save",
        lambda fig, output_dir, name: saved_figures.append(fig),
    )
    monkeypatch.setattr(
        "tests.analysis.figures.p4_figure_data.collect_llama2_7b_dense_decoder_data",
        lambda: {
            "model_name": "Llama2-7B",
            "num_layers": 32,
            "hidden_size": 4096,
            "num_heads": 32,
            "head_dim": 128,
            "backend_replay_stats": [{"runtime_ns": 10.0, "cycles": 16}],
            "qkvo_projection_pim_mac_per_layer": 1,
            "attention_pim_mac_per_layer": 1,
            "ffn_pim_mac_per_layer": 1,
        },
    )
    monkeypatch.setattr(
        "tests.analysis.figures.p4_figure_data.collect_llama2_13b_dense_decoder_data",
        lambda: {
            "model_name": "Llama2-13B",
            "num_layers": 40,
            "hidden_size": 5120,
            "num_heads": 40,
            "head_dim": 128,
            "backend_replay_stats": [{"runtime_ns": 20.0, "cycles": 32}],
            "qkvo_projection_pim_mac_per_layer": 2,
            "attention_pim_mac_per_layer": 2,
            "ffn_pim_mac_per_layer": 2,
        },
    )

    gen_p4_figures.gen_multi_model_llama2_latency_breakdown(tmp_path)

    assert saved_figures
    fig = saved_figures[0]
    assert len(fig.axes) >= 3
    figure_text = "\n".join(
        [axis.get_ylabel() for axis in fig.axes]
        + [text.get_text() for axis in fig.axes for text in axis.texts]
    )
    assert "Approx. FP16 weight size" in figure_text
    assert "~14 GB" in figure_text
    assert "~26 GB" in figure_text
    assert "32L" in figure_text
    assert "40L" in figure_text


def _fake_llama2_scaling_model(model_name: str, num_layers: int, hidden_size: int) -> dict[str, object]:
    return {
        "model_name": model_name,
        "dimensions": {
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "num_heads": 32 if "7B" in model_name else 40,
            "head_dim": 128,
            "ffn_hidden_size": 11008 if "7B" in model_name else 13824,
            "past_len": 1024,
        },
        "per_layer_pim_mac_buckets": {
            "qkvo_projection": hidden_size,
            "attention": hidden_size // 4,
            "ffn": hidden_size * 2,
        },
        "backend_replay_stats": {
            "steady_state": {
                "runtime_ns": float(hidden_size),
                "cycles": hidden_size // 2,
                "command_counts": {"PIM_MAC": hidden_size * 10, "PIM_BCAST": num_layers},
            },
            "cold_start": {
                "runtime_ns": float(hidden_size + 100),
                "cycles": hidden_size // 2 + 10,
                "command_counts": {"PIM_MAC": hidden_size * 10, "PIM_BCAST": num_layers * 2},
            },
        },
    }


def test_write_llama2_scaling_cache_writes_requested_json_schema(monkeypatch, tmp_path: Path) -> None:
    from tests.analysis.figures import gen_p4_figures
    from tests.analysis.figures import p4_figure_data

    cache_path = tmp_path / "fig9_llama2_7b_13b_models_latency_breakdown.json"
    monkeypatch.setattr(
        p4_figure_data,
        "collect_llama2_7b_dense_decoder_data",
        lambda: _fake_llama2_scaling_model("Llama2-7B", 32, 4096),
    )
    monkeypatch.setattr(
        p4_figure_data,
        "collect_llama2_13b_dense_decoder_data",
        lambda: _fake_llama2_scaling_model("Llama2-13B", 40, 5120),
    )

    written_path = gen_p4_figures.write_llama2_scaling_cache(cache_path, collect_backend=True)

    assert written_path == cache_path
    payload = json.loads(cache_path.read_text())
    assert set(payload) >= {"schema_version", "figure_id", "models", "provenance"}
    assert payload["figure_id"] == "fig9_llama2_7b_13b_models_latency_breakdown"
    assert len(payload["models"]) == 2
    assert set(payload["provenance"]) >= {"date", "generator_version", "commit", "replay_mode"}
    for model in payload["models"]:
        assert set(model) >= {"model_name", "dimensions", "per_layer_pim_mac_buckets", "replay_stats"}
        assert set(model["dimensions"]) >= {
            "num_layers",
            "hidden_size",
            "num_heads",
            "head_dim",
            "ffn_hidden_size",
            "past_len",
        }
        assert set(model["per_layer_pim_mac_buckets"]) == {"qkvo_projection", "attention", "ffn"}
        assert set(model["replay_stats"]) >= {"steady_state", "cold_start"}
        for mode in ("steady_state", "cold_start"):
            assert set(model["replay_stats"][mode]) >= {"runtime_ns", "cycles", "command_counts"}


def test_render_llama2_scaling_figure_from_cache_is_render_only(monkeypatch, tmp_path: Path) -> None:
    from tests.analysis.figures import gen_p4_figures
    from tests.analysis.figures import p4_figure_data

    cache_path = tmp_path / "llama2_scaling_cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "figure_id": "fig9_llama2_7b_13b_models_latency_breakdown",
                "provenance": {
                    "date": "2026-05-10",
                    "generator_version": "test",
                    "commit": "test-commit",
                    "replay_mode": "backend",
                },
                "models": [
                    _fake_llama2_scaling_model("Llama2-7B", 32, 4096),
                    _fake_llama2_scaling_model("Llama2-13B", 40, 5120),
                ],
            }
        )
    )
    saved: list[tuple[Any, Path, str]] = []

    def fail_backend_collection() -> None:
        raise AssertionError("render-only path must not collect backend data")

    monkeypatch.setattr(p4_figure_data, "collect_llama2_7b_dense_decoder_data", fail_backend_collection)
    monkeypatch.setattr(p4_figure_data, "collect_llama2_13b_dense_decoder_data", fail_backend_collection)
    monkeypatch.setattr(gen_p4_figures, "_save", lambda fig, output_dir, name: saved.append((fig, output_dir, name)))

    gen_p4_figures.render_llama2_scaling_figure_from_cache(cache_path, tmp_path)

    assert saved
    fig, output_dir, name = saved[0]
    assert output_dir == tmp_path
    assert "fig9_llama2_7b_13b_models_latency_breakdown" in name
    assert len(fig.axes) >= 2
    figure_text = "\n".join(
        [axis.get_ylabel() for axis in fig.axes]
        + [axis.get_title() for axis in fig.axes]
        + [text.get_text() for axis in fig.axes for text in axis.texts]
    )
    assert "Backend runtime" in figure_text
    assert "Per-layer PIM_MAC repeats" in figure_text


def test_render_llama2_scaling_figure_from_cache_marks_missing_latency_not_collected(monkeypatch, tmp_path: Path) -> None:
    from tests.analysis.figures import gen_p4_figures

    seven_b = _fake_llama2_scaling_model("Llama2-7B", 32, 4096)
    thirteen_b = _fake_llama2_scaling_model("Llama2-13B", 40, 5120)
    for mode in ("steady_state", "cold_start"):
        thirteen_b["backend_replay_stats"][mode]["runtime_ns"] = None
        thirteen_b["backend_replay_stats"][mode]["cycles"] = None
    cache_path = tmp_path / "llama2_scaling_missing_latency.json"
    cache_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "figure_id": "fig9_llama2_7b_13b_models_latency_breakdown",
                "provenance": {"date": "2026-05-10", "generator_version": "test", "commit": "test", "replay_mode": "precomputed_partial"},
                "models": [seven_b, thirteen_b],
            }
        )
    )
    saved: list[tuple[Any, Path, str]] = []
    monkeypatch.setattr(gen_p4_figures, "_save", lambda fig, output_dir, name: saved.append((fig, output_dir, name)))

    gen_p4_figures.render_llama2_scaling_figure_from_cache(cache_path, tmp_path)

    assert saved
    fig = saved[0][0]
    runtime_axis_text = "\n".join(text.get_text() for text in fig.axes[0].texts)
    assert "not collected" in runtime_axis_text
    assert "0 ns" not in runtime_axis_text


def _fake_decode_context_sweep_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "figure_id": "fig11_llama2_7b_decode_context_length_sweep",
        "description": "Decode-only context-length backend sweep for bounded surrogate replay",
        "phase": "decode",
        "seq_len": 1,
        "model": "Llama2-7B",
        "provenance": {"date": "2026-05-10", "generator_version": "test", "commit": "test", "replay_mode": "backend"},
        "sweep": {"past_len_values": [128, 256], "score_tile_tokens": 256, "context_tile_tokens": 256},
        "rows": [
            {
                "past_len": 128,
                "mode": "steady_state",
                "status": "PASS",
                "runtime_ns": 100.0,
                "cycles": 160,
                "pim_mac": 1000,
                "pim_bcast": 32,
                "pim_mac_density": 6.25,
                "per_layer_pim_mac_buckets": {"qkvo_projection": 10, "attention": 1, "ffn": 20},
                "concrete_opcode_counts_replay_input": {"PIM_MAC": 1000, "PIM_BCAST": 32},
            },
            {
                "past_len": 256,
                "mode": "steady_state",
                "status": "PASS",
                "runtime_ns": 150.0,
                "cycles": 240,
                "pim_mac": 1200,
                "pim_bcast": 32,
                "pim_mac_density": 5.0,
                "per_layer_pim_mac_buckets": {"qkvo_projection": 10, "attention": 2, "ffn": 20},
                "concrete_opcode_counts_replay_input": {"PIM_MAC": 1200, "PIM_BCAST": 32},
            },
        ],
        "caveats": ["decode_only_seq_len_1", "bounded_surrogate_not_serving", "non_silicon_calibrated"],
    }


def test_write_decode_context_sweep_cache_writes_schema(monkeypatch, tmp_path: Path) -> None:
    from tests.analysis.figures import gen_p4_figures
    from tests.analysis.figures import p4_figure_data

    cache_path = tmp_path / "fig11_llama2_7b_decode_context_length_sweep.json"
    monkeypatch.setattr(
        p4_figure_data,
        "collect_llama2_7b_decode_context_length_sweep",
        lambda past_len_values=None, modes=None: _fake_decode_context_sweep_payload(),
    )

    written_path = gen_p4_figures.write_decode_context_sweep_cache(cache_path)

    assert written_path == cache_path
    payload = json.loads(cache_path.read_text())
    assert payload["figure_id"] == "fig11_llama2_7b_decode_context_length_sweep"
    assert payload["phase"] == "decode"
    assert payload["seq_len"] == 1
    assert payload["sweep"]["past_len_values"] == [128, 256]
    assert all(row["mode"] == "steady_state" for row in payload["rows"])


def test_render_decode_context_sweep_from_cache_is_render_only(monkeypatch, tmp_path: Path) -> None:
    from tests.analysis.figures import gen_p4_figures
    from tests.analysis.figures import p4_figure_data

    cache_path = tmp_path / "fig11_llama2_7b_decode_context_length_sweep.json"
    cache_path.write_text(json.dumps(_fake_decode_context_sweep_payload()))
    saved: list[tuple[Any, Path, str]] = []

    def fail_backend_collection(*args, **kwargs) -> None:
        raise AssertionError("render-only path must not collect backend data")

    monkeypatch.setattr(p4_figure_data, "collect_llama2_7b_decode_context_length_sweep", fail_backend_collection)
    monkeypatch.setattr(gen_p4_figures, "_save", lambda fig, output_dir, name: saved.append((fig, output_dir, name)))

    gen_p4_figures.render_decode_context_sweep_figure_from_cache(cache_path, tmp_path)

    assert saved
    fig, output_dir, name = saved[0]
    assert output_dir == tmp_path
    assert name == "fig11_llama2_7b_decode_context_length_sweep"
    assert len(fig.axes) >= 2
    figure_text = "\n".join(
        [axis.get_xlabel() for axis in fig.axes]
        + [axis.get_ylabel() for axis in fig.axes]
        + [axis.get_title() for axis in fig.axes]
        + [text.get_text() for axis in fig.axes for text in axis.texts]
    )
    assert "Decode context length" in figure_text
    assert "past_len" in figure_text
    assert "seq_len=1" in figure_text


def test_render_decode_context_sweep_rejects_missing_required_values(tmp_path: Path) -> None:
    from tests.analysis.figures import gen_p4_figures

    payload = _fake_decode_context_sweep_payload()
    del payload["rows"][0]["runtime_ns"]
    cache_path = tmp_path / "bad_decode_context_sweep.json"
    cache_path.write_text(json.dumps(payload))

    try:
        gen_p4_figures.render_decode_context_sweep_figure_from_cache(cache_path, tmp_path)
    except ValueError as exc:
        assert "runtime_ns" in str(exc)
    else:
        raise AssertionError("missing runtime_ns must fail schema validation")


def _fake_generated_token_sweep_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "figure_id": "fig12_llama2_7b_generated_token_sweep",
        "description": "Decode-only generated-token sweep using independent single-token backend replays",
        "phase": "decode",
        "seq_len_per_step": 1,
        "initial_past_len": 1024,
        "num_generated_tokens": 2,
        "model": "Llama2-7B",
        "provenance": {"date": "2026-05-10", "generator_version": "test", "commit": "test", "replay_mode": "backend"},
        "rows": [
            {
                "generated_token_index": 0,
                "generated_tokens_total": 1,
                "past_len": 1024,
                "mode": "steady_state",
                "status": "PASS",
                "runtime_ns": 100.0,
                "cumulative_runtime_ns": 100.0,
                "cycles": 160,
                "pim_mac": 1000,
                "pim_bcast": 32,
                "pim_mac_density": 6.25,
            },
            {
                "generated_token_index": 1,
                "generated_tokens_total": 2,
                "past_len": 1025,
                "mode": "steady_state",
                "status": "PASS",
                "runtime_ns": 101.0,
                "cumulative_runtime_ns": 201.0,
                "cycles": 162,
                "pim_mac": 1001,
                "pim_bcast": 32,
                "pim_mac_density": 6.18,
            },
        ],
        "caveats": ["independent_single_token_replays", "decode_only_seq_len_1", "bounded_surrogate_not_serving"],
    }


def test_generated_token_sweep_collects_incrementing_past_len(monkeypatch) -> None:
    from tests.analysis.figures import p4_figure_data

    calls: list[int] = []

    def fake_dense_data(*, past_len: int, replay_stats_fn=None) -> dict[str, object]:
        return {
            "num_layers": 32,
            "datatype": "int8",
            "qkvo_projection_pim_mac_per_layer": 10,
            "attention_pim_mac_per_layer": past_len,
            "ffn_pim_mac_per_layer": 20,
        }

    def fake_backend(past_len: int, *, modes=("steady_state",)) -> dict[str, dict[str, object]]:
        calls.append(past_len)
        return {
            f"llama2_7b_32_layer_past_len_{past_len}_steady_state": {
                "replay_ok": True,
                "runtime_ns": float(past_len),
                "cycles": past_len * 2,
                "pim_mac_issued": past_len * 10,
                "avg_pim_latency_cycles": 1,
                "command_counts": {"PIM_MAC": past_len * 10, "PIM_BCAST": 32},
            }
        }

    monkeypatch.setattr(p4_figure_data, "collect_llama2_7b_dense_decoder_data", fake_dense_data)
    monkeypatch.setattr(
        "tests.analysis.figures.p4_backend_data.collect_all_backend_stats_llama2_7b_past_len",
        fake_backend,
    )

    payload = p4_figure_data.collect_llama2_7b_generated_token_sweep(initial_past_len=100, num_generated_tokens=3)

    assert calls == [100, 101, 102]
    assert payload["initial_past_len"] == 100
    assert payload["num_generated_tokens"] == 3
    assert [row["generated_tokens_total"] for row in payload["rows"]] == [1, 2, 3]
    assert [row["cumulative_runtime_ns"] for row in payload["rows"]] == [100.0, 201.0, 303.0]


def test_write_generated_token_sweep_cache_writes_schema(monkeypatch, tmp_path: Path) -> None:
    from tests.analysis.figures import gen_p4_figures
    from tests.analysis.figures import p4_figure_data

    cache_path = tmp_path / "fig12_llama2_7b_generated_token_sweep.json"
    monkeypatch.setattr(
        p4_figure_data,
        "collect_llama2_7b_generated_token_sweep",
        lambda initial_past_len=1024, num_generated_tokens=4, modes=None: _fake_generated_token_sweep_payload(),
    )

    written_path = gen_p4_figures.write_generated_token_sweep_cache(cache_path)

    assert written_path == cache_path
    payload = json.loads(cache_path.read_text())
    assert payload["figure_id"] == "fig12_llama2_7b_generated_token_sweep"
    assert payload["phase"] == "decode"
    assert payload["seq_len_per_step"] == 1
    assert payload["initial_past_len"] == 1024
    assert payload["num_generated_tokens"] == 2


def test_render_generated_token_sweep_from_cache_is_render_only(monkeypatch, tmp_path: Path) -> None:
    from tests.analysis.figures import gen_p4_figures
    from tests.analysis.figures import p4_figure_data

    cache_path = tmp_path / "fig12_llama2_7b_generated_token_sweep.json"
    cache_path.write_text(json.dumps(_fake_generated_token_sweep_payload()))
    saved: list[tuple[Any, Path, str]] = []

    def fail_backend_collection(*args, **kwargs) -> None:
        raise AssertionError("render-only path must not collect backend data")

    monkeypatch.setattr(p4_figure_data, "collect_llama2_7b_generated_token_sweep", fail_backend_collection)
    monkeypatch.setattr(gen_p4_figures, "_save", lambda fig, output_dir, name: saved.append((fig, output_dir, name)))

    gen_p4_figures.render_generated_token_sweep_figure_from_cache(cache_path, tmp_path)

    assert saved
    fig, output_dir, name = saved[0]
    assert output_dir == tmp_path
    assert name == "fig12_llama2_7b_generated_token_sweep"
    figure_text = "\n".join(
        [axis.get_xlabel() for axis in fig.axes]
        + [axis.get_ylabel() for axis in fig.axes]
        + [axis.get_title() for axis in fig.axes]
        + [text.get_text() for axis in fig.axes for text in axis.texts]
    )
    assert "Generated tokens" in figure_text
    assert "independent single-token" in figure_text
    assert "seq_len=1" in figure_text


def test_gen_p4_figures_cli_exposes_llama2_scaling_cache_modes() -> None:
    text = Path("ramulator2/tests/analysis/figures/gen_p4_figures.py").read_text()

    assert "--collect-llama2-scaling-cache" in text
    assert "--render-llama2-scaling-cache" in text
    assert "--collect-decode-context-sweep-cache" in text
    assert "--render-decode-context-sweep-cache" in text
    assert "--collect-generated-token-sweep-cache" in text
    assert "--render-generated-token-sweep-cache" in text


def test_llama2_backend_collection_avoids_full_command_trace_recorder() -> None:
    text = Path("ramulator2/tests/analysis/figures/p4_backend_data.py").read_text()
    llama_section = text.split("def collect_all_backend_stats_llama2_7b", maxsplit=1)[1]

    assert "_make_mem_with_plugins" not in llama_section
    assert "run_trace_through_backend" not in llama_section
    assert '"command_trace_total_commands": 0' in llama_section


def test_mixtral_expert_mac_diagnostic_ratios_are_explicit() -> None:
    from tests.analysis.figures import p4_figure_data

    fused_scaled = p4_figure_data._moe_expert_pim_mac_per_layer(
        hidden_size=512,
        expert_hidden_size=2048,
        top_k=2,
        lanes=32,
        projections=1,
    )
    real_scaled = p4_figure_data._moe_expert_pim_mac_per_layer(
        hidden_size=512,
        expert_hidden_size=2048,
        top_k=2,
        lanes=32,
        projections=3,
    )
    real_full_dim = p4_figure_data._moe_expert_pim_mac_per_layer(
        hidden_size=4096,
        expert_hidden_size=14336,
        top_k=2,
        lanes=32,
        projections=3,
    )

    assert fused_scaled == 65_536
    assert real_scaled == 196_608
    assert real_scaled == fused_scaled * 3
    assert real_full_dim == 11_010_048


def test_mixtral_decoder_data_rejects_formula_concrete_mismatch(monkeypatch) -> None:
    from ramulator.workload_surrogate import generate_full_transformer
    from tests.analysis.figures import p4_figure_data

    attention_manifest = {
        "num_layers": 1,
        "hidden_size": 32,
        "num_heads": 4,
        "num_kv_heads": 1,
        "head_dim": 8,
        "past_len": 1024,
        "datatype": "int8",
    }
    moe_manifest = {
        "num_layers": 1,
        "hidden_size": 32,
        "expert_hidden_size": 64,
        "num_experts": 4,
        "top_k": 2,
        "datatype": "int8",
        "real_model_dimensions": {
            "hidden_size": 4096,
            "expert_hidden_size": 14336,
        },
    }

    monkeypatch.setattr(
        generate_full_transformer,
        "get_mixtral_8x7b_moe_decoder_manifests",
        lambda past_len=1024: (attention_manifest, moe_manifest),
    )
    monkeypatch.setattr(
        generate_full_transformer,
        "generate_mixtral_8x7b_decoder_records",
        lambda attention_manifest, moe_manifest: [{"kind": "SyntheticMixtralRecord"}],
    )
    monkeypatch.setattr(
        p4_figure_data,
        "lower_semantic_records_to_concrete",
        lambda semantic, manifest_name: [{"opcode": "PIM_MAC", "repeat": 1}],
    )

    with pytest.raises(ValueError, match="Mixtral-8x7B PIM_MAC accounting mismatch"):
        p4_figure_data.collect_mixtral_8x7b_moe_decoder_data(replay_stats_fn=None)


def test_moe_figures_have_inline_scope_caveats() -> None:
    text = Path("ramulator2/tests/analysis/figures/gen_p4_figures.py").read_text()

    assert "Mixtral is scaled (H=512, expert=2048)" in text
    assert "Raw backend cycles exist in caches but are not fairness-normalized" in text
    assert "Generator-level sweep only" in text
    assert "Tiny manifest: H=32, expert=64" in text
    assert "Full-dim ×3 bar is analytical only, not backend-replayed" in text
