import pandas as pd
import pytest
import torch

from compass import assign_regimes, mutual_knn, per_sample_overlap, score_alignment, score_merging


def _feats(n=200, d=32, seed=0):
    return torch.randn(n, d, generator=torch.Generator().manual_seed(seed))


def test_identical_features_overlap_one():
    x = _feats()
    res = mutual_knn(x, x, k=10, num_permutations=20, device="cpu")
    assert res.raw == pytest.approx(1.0)
    assert res.score == pytest.approx(1.0)


def test_invariant_to_rotation_and_scale():
    x = _feats()
    q, _ = torch.linalg.qr(torch.randn(32, 32))
    res = mutual_knn(x, 3.0 * x @ q, k=10, calibrate=False, q_outlier=None, device="cpu")
    assert res.raw == pytest.approx(1.0)


def test_independent_features_calibrate_to_zero():
    res = mutual_knn(_feats(seed=0), _feats(seed=1), k=10, num_permutations=50, device="cpu")
    assert res.raw < 0.2
    assert res.score == 0.0


def test_layer_max_and_best_layers():
    x = _feats()
    stack_a = torch.stack([_feats(seed=5), x, _feats(seed=6)], dim=1)
    stack_b = torch.stack([_feats(seed=7), x], dim=1)
    res = mutual_knn(stack_a, stack_b, k=10, calibrate=False, q_outlier=None, device="cpu")
    assert res.best_layers == (1, 1)
    assert res.raw == pytest.approx(1.0)


def test_per_sample_overlap_mean_matches():
    a, b = _feats(seed=0), _feats(seed=0) + 0.5 * _feats(seed=1)
    per = per_sample_overlap(a, b, k=10)
    raw = mutual_knn(a, b, k=10, calibrate=False, q_outlier=None, device="cpu").raw
    assert per.mean().item() == pytest.approx(raw, abs=1e-6)


def test_regimes_median_split():
    df = pd.DataFrame({"ease": [0.9, 0.1, 0.9, 0.1], "ceiling": [0.9, 0.9, 0.1, 0.1]})
    assert list(assign_regimes(df)) == ["act now", "budget-limited", "ceiling-limited", "incompatible"]


def test_score_alignment_ranks_by_product():
    base = _feats(seed=0)
    vision = {"close": base + 0.1 * _feats(seed=1), "far": _feats(seed=2)}
    text = {"t1": base + 0.2 * _feats(seed=3), "t2": base + 0.3 * _feats(seed=4)}
    df = score_alignment(vision, text, num_permutations=20, device="cpu")
    assert len(df) == 4
    assert (df["compass"] == df["ease"] * df["ceiling"]).all()
    assert df.iloc[0]["vision"] == "close"
    assert df["rank"].tolist() == [1, 2, 3, 4]


def test_score_merging_only_within_base():
    feats = {f"m{i}": _feats(seed=i) for i in range(4)}
    base = {"m0": "A", "m1": "A", "m2": "B", "m3": "B"}
    df = score_merging(feats, base, {"A": 0.5, "B": 0.7}, device="cpu")
    assert len(df) == 2
    assert set(df["base_model"]) == {"A", "B"}


def test_permuted_knn_graph_matches_knn_of_permuted_features():
    from compass.overlap import _knn_masks, knn_indices

    x, perm = _feats(), torch.randperm(200, generator=torch.Generator().manual_seed(1))
    mask = _knn_masks(knn_indices(x))
    assert torch.equal(mask[:, perm][:, :, perm], _knn_masks(knn_indices(x[perm])))


def test_calibration_matches_calibrated_similarity():
    cs = pytest.importorskip("calibrated_similarity")
    from compass.overlap import knn_indices

    a = torch.stack([_feats(seed=0), _feats(seed=1)], dim=1)
    b = torch.stack([_feats(seed=0) + 0.8 * _feats(seed=2), _feats(seed=3)], dim=1)

    def overlap(x, y):
        ix, iy = knn_indices(x)[0], knn_indices(y)[0]
        return (ix[:, :, None] == iy[:, None, :]).sum() / (x.shape[0] * 10)

    ref, p_ref, tau_ref = cs.calibrate_layers(list(a.unbind(1)), list(b.unbind(1)), overlap, K=50)
    res = mutual_knn(a, b, k=10, num_permutations=50, q_outlier=None, device="cpu")
    assert 0 < res.score < 1
    assert res.score == pytest.approx(float(ref), abs=0.02)
    assert res.tau == pytest.approx(float(tau_ref), abs=0.01)
    assert res.p_value == pytest.approx(float(p_ref))


def test_per_sample_overlap_rejects_layer_stacks():
    with pytest.raises(ValueError):
        per_sample_overlap(torch.stack([_feats(), _feats()], dim=1), _feats())


def test_load_vision_model_checkpoint(tmp_path):
    import timm

    from compass import load_vision_model

    finetuned = timm.create_model("vit_tiny_patch16_224", num_classes=10)
    torch.save(finetuned.state_dict(), tmp_path / "ckpt.pt")
    model = load_vision_model("vit_tiny_patch16_224", checkpoint=tmp_path / "ckpt.pt", device="cpu")
    assert torch.equal(model.patch_embed.proj.weight, finetuned.patch_embed.proj.weight)

    torch.save({f"module.{k}": v for k, v in finetuned.state_dict().items()}, tmp_path / "bad.pt")
    with pytest.raises(RuntimeError):
        load_vision_model("vit_tiny_patch16_224", checkpoint=tmp_path / "bad.pt", device="cpu")
