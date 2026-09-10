from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import pytest
import torch
import cv2

from tools.oct_denoise_benchmark.data import audit_protocol, load_protocol_manifest
from tools.oct_denoise_benchmark.evaluate import mark_recovered_failures
from tools.oct_denoise_benchmark.cli import _candidate_grid, _complete_summary, _evaluate_candidates, _expand_non_bm3d, _position_macro_candidates, _select_candidate
from tools.oct_denoise_benchmark.metrics import ms_ssim
from tools.oct_denoise_benchmark.methods import AdapterContext, denoise
from tools.oct_denoise_benchmark.methods.dncnn import DnCNN
from tools.oct_denoise_benchmark.methods.ksvd_adapter import _dct_dictionary, _ksvd, omp
from tools.oct_denoise_benchmark.methods.nafnet import NAFBlock, NAFNet
from tools.oct_denoise_benchmark.package_light import _ascii_stage_paths, materialize_fixed_atlas
from tools.oct_denoise_benchmark.statistics import bootstrap_confidence_intervals
from tools.oct_denoise_benchmark.table_store import merge_records
from tools.oct_denoise_benchmark.train_paired import PositionBalancedPairDataset, checkpoint_selection_reason, cpu_rng_state, validation_can_select_checkpoint
from tools.oct_denoise_benchmark.methods.deep_common import tiled_forward
from tools.oct_denoise_benchmark.inference import run as run_inference
from tools.oct_denoise_benchmark.registry import save_yaml
from tools.oct_denoise_benchmark.merge_tracks import merge_tracks


def test_nafblock_matches_official_operation_order_and_channels():
    block = NAFBlock(8)
    assert block.sca[1].in_channels == 8 and block.sca[1].out_channels == 8
    assert block.conv3.in_channels == 8 and block.conv3.out_channels == 8
    with torch.no_grad():
        block.beta.fill_(0.7); block.gamma.fill_(0.4)
    source = torch.randn(2, 8, 11, 13, requires_grad=True)
    actual = block(source)
    x = block.norm1(source)
    x = block.conv1(x)
    x = block.conv2(x)
    x = block.sg(x)
    x = x * block.sca(x)
    x = block.conv3(x)
    expected_y = source + block.dropout1(x) * block.beta
    x = block.conv4(block.norm2(expected_y))
    x = block.sg(x)
    x = block.conv5(x)
    expected = expected_y + block.dropout2(x) * block.gamma
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    actual.square().mean().backward()
    assert actual.shape == source.shape and torch.isfinite(actual).all()
    assert source.grad is not None and torch.isfinite(source.grad).all()


def test_nafnet_single_channel_padding_restores_source_shape():
    model = NAFNet(width=4, enc_blocks=(1, 1), middle_blocks=1, dec_blocks=(1, 1))
    source = torch.rand(1, 1, 31, 47, requires_grad=True)
    result = model(source)
    assert result.shape == source.shape and torch.isfinite(result).all()
    result.mean().backward()
    assert source.grad is not None and torch.isfinite(source.grad).all()


def test_position_balanced_dataset_is_resume_exact_and_position_balanced(monkeypatch):
    rows = pd.DataFrame([
        {"dataset": "PKU37", "split": "train", "position_id": "p1", "sample_id": "a", "image_path": "a", "clean_path": "a"},
        {"dataset": "PKU37", "split": "train", "position_id": "p2", "sample_id": "b", "image_path": "b", "clean_path": "b"},
        {"dataset": "PKU37", "split": "train", "position_id": "p2", "sample_id": "c", "image_path": "c", "clean_path": "c"},
    ])
    monkeypatch.setattr("tools.oct_denoise_benchmark.train_paired.read_image", lambda path: (np.full((12, 12), len(str(path)), np.float32) / 10, {}))
    whole = PositionBalancedPairDataset(rows, 8, 42, 2000)
    positions = [whole[index][2] for index in range(2000)]
    assert abs(positions.count("p1") - positions.count("p2")) < 120
    uninterrupted = [whole[index][0] for index in range(15, 25)]
    resumed = PositionBalancedPairDataset(rows, 8, 42, 10, start_index=15)
    for expected, actual in zip(uninterrupted, (resumed[index][0] for index in range(10))):
        torch.testing.assert_close(expected, actual)


def test_cosine_scheduler_state_restores_exactly():
    parameter_a = torch.nn.Parameter(torch.ones(())); optimizer_a = torch.optim.AdamW([parameter_a], lr=1e-3)
    scheduler_a = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_a, T_max=10, eta_min=1e-6)
    continuous = []
    saved = None
    for update in range(10):
        optimizer_a.step(); scheduler_a.step(); continuous.append(scheduler_a.get_last_lr()[0])
        if update == 4: saved = (optimizer_a.state_dict(), scheduler_a.state_dict())
    parameter_b = torch.nn.Parameter(torch.ones(())); optimizer_b = torch.optim.AdamW([parameter_b], lr=1e-3)
    scheduler_b = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_b, T_max=10, eta_min=1e-6)
    assert saved is not None
    optimizer_b.load_state_dict(saved[0]); scheduler_b.load_state_dict(saved[1])
    resumed = []
    for _ in range(5):
        optimizer_b.step(); scheduler_b.step(); resumed.append(scheduler_b.get_last_lr()[0])
    assert resumed == pytest.approx(continuous[5:])


def test_fast_validation_cannot_select_checkpoint():
    assert not validation_can_select_checkpoint("fixed_fast_subset_not_checkpoint_eligible")
    assert validation_can_select_checkpoint("full_277_checkpoint_eligible")


def test_cosine_tiling_preserves_identity_output():
    image = np.random.default_rng(2).random((37, 53), dtype=np.float32)
    result = tiled_forward(torch.nn.Identity(), image, AdapterContext(tile_size=16, tile_overlap=8))
    np.testing.assert_allclose(result, image, rtol=1e-6, atol=1e-7)


def test_package_stage_paths_are_ascii_and_manifest_safe(tmp_path: Path):
    nested = tmp_path / "预览图"
    nested.mkdir()
    (nested / "方法比较.png").write_bytes(b"png")
    _ascii_stage_paths(tmp_path)
    paths = [path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")]
    assert paths and all(path.isascii() for path in paths)


def test_single_file_inference_preserves_uint16_and_resumes_by_hash(tmp_path: Path):
    source = tmp_path / "input.tif"; output = tmp_path / "result.tif"; registry = tmp_path / "registry.yaml"
    raw = np.arange(20 * 24, dtype=np.uint16).reshape(20, 24)
    ok, encoded = cv2.imencode(".tif", raw); assert ok
    encoded.tofile(str(source))
    save_yaml(registry, {"status": "locked", "methods": {"noisy_identity": {"config": {"method_id": "noisy_identity"}, "seed": 0}}})
    args = argparse.Namespace(method="noisy_identity", input=source, output=output, registry=registry, checkpoint=None,
                              device="cpu", recursive=False, extensions=".tif", preserve_relative_path=False,
                              preserve_bit_depth=True, save_preview=False, overwrite=False, tile_size=None, tile_overlap=64)
    first = run_inference(args); second = run_inference(args)
    restored = cv2.imdecode(np.fromfile(str(output), np.uint8), cv2.IMREAD_UNCHANGED)
    assert first[0]["status"] == "success" and second[0]["status"] == "success"
    assert restored.dtype == np.uint16 and restored.shape == raw.shape
    np.testing.assert_array_equal(restored, raw)
    manifest = pd.read_csv(tmp_path / "inference_manifest.csv")
    assert len(manifest) == 1 and manifest.source_code_sha256.str.len().item() == 64


def test_classical_track_merge_is_idempotent_and_rejects_source_conflict(tmp_path: Path):
    run = tmp_path / "run"; classical = run / "tracks" / "classical"
    save_yaml(run / "configs" / "locked_classical_configs.yaml", {"status": "unlocked", "methods": {}})
    save_yaml(classical / "configs" / "locked_classical_configs.yaml",
              {"status": "locked_on_pku37_validation", "methods": {"tv_chambolle": {"method_id": "tv_chambolle", "weight": 0.1}}})
    row = {"method_id": "tv_chambolle", "search_phase": "full_validation", "search_round": 0,
           "candidate_uid": "x", "sample_id": "a", "candidate_json": "{}", "source_sha256": "source-a"}
    (classical / "metrics").mkdir(parents=True)
    pd.DataFrame([row]).to_csv(classical / "metrics" / "parameter_search_partial.csv", index=False)
    merge_tracks(run, classical); merge_tracks(run, classical)
    assert len(pd.read_csv(run / "metrics" / "parameter_search_partial.csv")) == 1
    pd.DataFrame([{**row, "source_sha256": "source-b"}]).to_csv(classical / "metrics" / "parameter_search_partial.csv", index=False)
    with pytest.raises(RuntimeError, match="source_sha256"):
        merge_tracks(run, classical)


def test_modelwhale_clean_gate_allows_runtime_untracked_files_but_not_source():
    script = (Path(__file__).parents[1] / "tools" / "oct_denoise_benchmark" / "scripts" / "run_modelwhale_protocol.sh").read_text(encoding="utf-8")
    assert "git status --porcelain --untracked-files=no" in script
    assert "git ls-files --others --exclude-standard -- configs docs sabids tests tools" in script


def test_rng_checkpoint_states_are_restored_as_cpu_byte_tensors():
    state = torch.get_rng_state()
    restored = cpu_rng_state(state, "test_rng")
    assert restored.device.type == "cpu" and restored.dtype == torch.uint8
    torch.set_rng_state(restored)
    with pytest.raises(TypeError, match="torch.uint8"):
        cpu_rng_state(torch.ones(2), "invalid_rng")


@pytest.mark.parametrize("config", [
    {"method_id": "noisy_identity"},
    {"method_id": "tv_chambolle", "weight": 0.03, "eps": 2e-4, "max_num_iter": 10},
    {"method_id": "nlm", "h_sigma_multiplier": 0.8, "patch_size": 3, "patch_distance": 2, "fast_mode": True, "provide_sigma": True},
])
def test_standard_adapter_contract(config):
    image = np.random.default_rng(42).random((31, 47), dtype=np.float32)
    first = denoise(image, config, AdapterContext())
    second = denoise(image, config, AdapterContext())
    assert first.shape == image.shape and first.dtype == np.float32
    assert np.isfinite(first).all() and 0 <= first.min() <= first.max() <= 1
    np.testing.assert_array_equal(first, second)


def test_bm3d_standard_rejects_lc_profile():
    image = np.zeros((8, 8), np.float32)
    with pytest.raises(ValueError, match="standard"):
        denoise(image, {"method_id": "bm3d_standard", "sigma_psd": 0.05, "profile": "lc"}, AdapterContext())


def test_ksvd_dictionary_sparse_svd_and_determinism():
    import tools.oct_denoise_benchmark.methods.ksvd_adapter as ksvd_module
    ksvd_module.SVD_UPDATE_CALLS = 0
    rng = np.random.default_rng(42)
    signals = rng.normal(size=(16, 30))
    dictionary = _dct_dictionary(4, 12)
    np.testing.assert_allclose(np.linalg.norm(dictionary, axis=0), 1, atol=1e-10)
    codes = omp(dictionary, signals, 3)
    assert np.max(np.count_nonzero(np.abs(codes) > 1e-12, axis=0)) <= 3
    first_dictionary, first_codes = _ksvd(signals, dictionary, 2, 3, 0, np.random.default_rng(7))
    second_dictionary, second_codes = _ksvd(signals, dictionary, 2, 3, 0, np.random.default_rng(7))
    np.testing.assert_allclose(np.linalg.norm(first_dictionary, axis=0), 1, atol=1e-10)
    np.testing.assert_array_equal(first_dictionary, second_dictionary)
    np.testing.assert_array_equal(first_codes, second_codes)
    assert ksvd_module.SVD_UPDATE_CALLS > 0


def test_ksvd_reconstructs_without_holes_and_has_no_clean_context():
    image = np.random.default_rng(42).random((19, 23), dtype=np.float32)
    config = {"method_id": "ksvd_self", "patch_size": 4, "dictionary_atoms": 12, "iterations": 1, "omp_max_nonzero": 2, "stride": 4, "max_training_patches": 32}
    output = denoise(image, config, AdapterContext(seed=42))
    assert output.shape == image.shape and np.isfinite(output).all()
    assert not hasattr(AdapterContext(), "clean") and not hasattr(AdapterContext(), "reference")


def test_ksvd_standard_grid_fixes_weights_and_noise_weight_is_effective():
    grid = _candidate_grid("ksvd_self")
    for key in ("patch_size", "dictionary_atoms", "iterations", "omp_max_nonzero", "stride"):
        assert len({item[key] for item in grid}) > 1
    assert {item["noise_weight"] for item in grid} == {1.0}
    assert {item["aggregation_weight"] for item in grid} == {0.0}
    image = np.random.default_rng(8).random((16, 18), dtype=np.float32)
    config = {"method_id": "ksvd_self", "patch_size": 4, "dictionary_atoms": 12, "iterations": 1,
              "omp_max_nonzero": 2, "stride": 4, "max_training_patches": 24, "aggregation_weight": 1.0}
    identity = denoise(image, {**config, "noise_weight": 0.0}, AdapterContext(seed=3))
    sparse = denoise(image, {**config, "noise_weight": 1.0}, AdapterContext(seed=3))
    np.testing.assert_allclose(identity, image, atol=2e-7)
    assert not np.allclose(sparse, image)


def test_calibration_partial_file_skips_completed_work(tmp_path: Path, monkeypatch):
    calls = {"count": 0}
    monkeypatch.setattr("tools.oct_denoise_benchmark.cli.read_image", lambda path: (np.zeros((16, 16), np.float32), {}))
    def fake_denoise(image, config, context):
        calls["count"] += 1
        return image
    monkeypatch.setattr("tools.oct_denoise_benchmark.cli.denoise", fake_denoise)
    monkeypatch.setattr("tools.oct_denoise_benchmark.cli.compute_metrics", lambda *args: {"psnr": 1.0, "ssim": 0.5})
    rows = pd.DataFrame([{"sample_id": "a", "position_id": "p1", "image_path": "n", "clean_path": "r"},
                         {"sample_id": "b", "position_id": "p2", "image_path": "n", "clean_path": "r"}])
    partial = tmp_path / "partial.csv"
    config = [{"method_id": "tv_chambolle", "weight": 0.1}]
    assert len(_evaluate_candidates("tv_chambolle", config, rows, 0, "full", partial)) == 2
    assert len(_evaluate_candidates("tv_chambolle", config, rows, 0, "full", partial)) == 2
    assert calls["count"] == 2


def test_incomplete_calibration_candidate_cannot_be_selected():
    expected = pd.DataFrame([{"sample_id": "a", "position_id": "p1"}, {"sample_id": "b", "position_id": "p2"}])
    rows = [{"candidate_uid": "x", "candidate_json": "{}", "sample_id": "a", "position_id": "p1", "psnr": 30.0, "ssim": 0.9}]
    with pytest.raises(RuntimeError, match="every registered validation sample"):
        _complete_summary(rows, expected)


def test_fixed_atlas_materializer_does_not_open_reference_before_lock(tmp_path: Path, monkeypatch):
    run = tmp_path / "run"; (run / "audit").mkdir(parents=True)
    monkeypatch.setattr("tools.oct_denoise_benchmark.package_light.read_image", lambda path: pytest.fail("opened sealed image"))
    result = materialize_fixed_atlas(tmp_path, run, pd.DataFrame())
    assert result["status"] == "not_materialized_test_sealed"


def test_best_psnr_keeps_same_checkpoint_ssim_and_resume_state(tmp_path: Path):
    tolerance = 1e-4
    initial = {"psnr": -float("inf"), "ssim": -float("inf"), "update": 0}
    epochs = [(10, 30.0, 0.70), (20, 29.0, 0.90)]

    def advance(state, candidates):
        state = dict(state)
        for update, psnr, ssim in candidates:
            reason = checkpoint_selection_reason(state["psnr"], state["ssim"], psnr, ssim, tolerance)
            if reason:
                state = {"psnr": psnr, "ssim": ssim, "update": update, "reason": reason}
        return state

    continuous = advance(initial, epochs)
    interrupted = advance(initial, epochs[:1])
    state_path = tmp_path / "selection.pth"; torch.save(interrupted, state_path)
    resumed = advance(torch.load(state_path, weights_only=False), epochs[1:])
    assert continuous == resumed
    assert resumed["update"] == 10 and resumed["psnr"] == 30.0 and resumed["ssim"] == 0.70


def test_transactional_result_union_dedup_conflict_nan_and_backup(tmp_path: Path):
    run = tmp_path / "run"; path = run / "metrics" / "records.csv"
    keys = ["dataset", "split", "sample_id", "method_id", "seed"]
    consistency = ["config_sha256", "checkpoint_sha256", "output_sha256"]
    first = pd.DataFrame([
        {"dataset": "PKU37", "split": "train", "sample_id": "p1", "method_id": "m", "seed": 0, "config_sha256": "c", "checkpoint_sha256": "", "output_sha256": "h1"},
        {"dataset": "PKU37", "split": "val", "sample_id": "p2", "method_id": "m", "seed": 0, "config_sha256": "c", "checkpoint_sha256": "", "output_sha256": "h2"},
    ])
    second = pd.DataFrame([
        {"dataset": "PKU37", "split": "test", "sample_id": "p3", "method_id": "m", "seed": 0, "config_sha256": "c", "checkpoint_sha256": "", "output_sha256": "h3"},
        {"dataset": "Duke17", "split": "external_test", "sample_id": "d1", "method_id": "m", "seed": 0, "config_sha256": "c", "checkpoint_sha256": "", "output_sha256": "h4"},
    ])
    merge_records(path, first, run, keys, consistency)
    assert len(merge_records(path, second, run, keys, consistency)) == 4
    assert len(merge_records(path, second, run, keys, consistency)) == 4
    assert list((run / "audit" / "backups").glob("records.csv.*.bak"))
    conflict = first.iloc[[0]].copy(); conflict["checkpoint_sha256"] = "different"
    with pytest.raises(RuntimeError, match="provenance conflict"):
        merge_records(path, conflict, run, keys, consistency)
    # CSV round-trip converts an empty checkpoint hash to NaN; normalization
    # must still identify the record as the same successful item.
    roundtrip = pd.read_csv(path); assert pd.isna(roundtrip.loc[roundtrip.sample_id == "p1", "checkpoint_sha256"]).all()
    assert len(merge_records(path, first.iloc[[0]], run, keys, consistency)) == 4
    extra_seed = first.iloc[[0]].copy(); extra_seed["seed"] = 42; extra_seed["checkpoint_sha256"] = "deep"
    assert len(merge_records(path, extra_seed, run, keys, consistency)) == 5


def test_failed_item_is_retained_and_marked_recovered_after_success():
    failures = pd.DataFrame([{"sample_id": "p1", "method_id": "nlm", "error": "old failure"}])
    result = mark_recovered_failures(failures, [{"sample_id": "p1", "method_id": "nlm"}], "2026-09-08T00:00:00Z")
    assert len(result) == 1 and result.iloc[0].error == "old failure"
    assert result.iloc[0].resolution_status == "recovered" and result.iloc[0].resolved_at_utc == "2026-09-08T00:00:00Z"


def test_deep_adapter_never_uses_random_weights(tmp_path: Path):
    image = np.zeros((16, 16), np.float32)
    with pytest.raises(ValueError, match="checkpoint"):
        denoise(image, {"method_id": "dncnn_paired", "depth": 3, "features": 4}, AdapterContext())
    model = DnCNN(depth=3, features=4)
    checkpoint = tmp_path / "model.pth"
    torch.save({"architecture": "dncnn_paired", "model": model.state_dict()}, checkpoint)
    context = AdapterContext(checkpoint=checkpoint)
    output = denoise(image, {"method_id": "dncnn_paired", "depth": 3, "features": 4}, context)
    assert output.shape == image.shape
    loaded = context.extras["loaded_model"]
    denoise(image, {"method_id": "dncnn_paired", "depth": 3, "features": 4}, context)
    assert context.extras["loaded_model"] is loaded


def test_external_datasets_are_never_development_splits(tmp_path: Path):
    frame = pd.DataFrame([
        {"sample_id": "p1", "group_id": "p1", "dataset": "PKU37", "split": "train", "image_path": "a", "clean_path": "b"},
        {"sample_id": "d1", "group_id": "d1", "dataset": "Duke17", "split": "train", "image_path": "a", "clean_path": "b"},
    ])
    path = tmp_path / "manifest.csv"; frame.to_csv(path, index=False)
    loaded = load_protocol_manifest(tmp_path, path)
    assert loaded.loc[loaded.dataset == "Duke17", "split"].item() == "external_test"


def test_parameter_selection_is_position_macro_with_ssim_tolerance_tiebreak():
    rows = []
    # Candidate A has many high-scoring frames in one position but is worse
    # when independent positions receive equal weight.
    for index in range(10):
        rows.append({"candidate_uid": "a", "candidate_json": '{"name":"a"}', "position_id": "p1", "sample_id": f"a{index}", "psnr": 30.0, "ssim": 0.8})
    rows.append({"candidate_uid": "a", "candidate_json": '{"name":"a"}', "position_id": "p2", "sample_id": "a10", "psnr": 10.0, "ssim": 0.8})
    rows.extend([
        {"candidate_uid": "b", "candidate_json": '{"name":"b"}', "position_id": "p1", "sample_id": "b1", "psnr": 21.0, "ssim": 0.7},
        {"candidate_uid": "b", "candidate_json": '{"name":"b"}', "position_id": "p2", "sample_id": "b2", "psnr": 21.0, "ssim": 0.7},
    ])
    summary = _position_macro_candidates(rows)
    assert _select_candidate(summary, 1e-4).candidate_uid == "b"
    tied = pd.DataFrame([
        {"candidate_uid": "x", "candidate_json": "{}", "position_macro_psnr": 25.00000, "position_macro_ssim": 0.70},
        {"candidate_uid": "y", "candidate_json": "{}", "position_macro_psnr": 24.99995, "position_macro_ssim": 0.80},
    ])
    assert _select_candidate(tied, 1e-4).candidate_uid == "y"


def test_bootstrap_contains_paired_method_difference_ci():
    rows = []
    for position, baseline, method in [("p1", 20.0, 22.0), ("p2", 21.0, 22.0), ("p3", 19.0, 22.0)]:
        rows.extend([
            {"dataset": "PKU37", "position_id": position, "method_id": "noisy_identity", "psnr": baseline},
            {"dataset": "PKU37", "position_id": position, "method_id": "nlm", "psnr": method},
        ])
    result = bootstrap_confidence_intervals(pd.DataFrame(rows), iterations=1000, seed=42)
    paired = result[(result.comparison_type == "paired_difference") & (result.baseline_method == "noisy_identity") & (result.method_id == "nlm") & (result.metric == "psnr")]
    assert len(paired) == 1 and paired.iloc[0].ci95_low <= paired.iloc[0]["mean"] <= paired.iloc[0].ci95_high


def test_ms_ssim_identity_and_degradation():
    image = np.random.default_rng(4).random((128, 128), dtype=np.float32)
    assert ms_ssim(image, image) == pytest.approx(1.0, abs=1e-6)
    degraded = np.clip(image + 0.2, 0, 1)
    assert 0 <= ms_ssim(degraded, image) < 1


def test_tv_and_nlm_boundary_search_expand_without_changing_other_parameters():
    tv = [{"method_id": "tv_chambolle", "weight": w, "eps": 1e-4, "max_num_iter": 300} for w in (0.01, 0.24)]
    expanded, boundaries = _expand_non_bm3d("tv_chambolle", tv[-1], tv)
    assert "weight" in boundaries and any(item["weight"] > 0.24 for item in expanded)
    assert all(item["method_id"] == "tv_chambolle" for item in expanded)
    nlm = [{"method_id": "nlm", "h_sigma_multiplier": h, "patch_size": 5, "patch_distance": 6, "fast_mode": True, "provide_sigma": True} for h in (0.5, 1.5)]
    expanded, boundaries = _expand_non_bm3d("nlm", nlm[-1], nlm)
    assert "h_sigma_multiplier" in boundaries and any(item["h_sigma_multiplier"] > 1.5 for item in expanded)
