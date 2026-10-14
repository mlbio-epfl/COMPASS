"""Mutual k-NN overlap between two representation spaces.

Features are ``[N, D]`` or ``[N, L, D]`` for ``L`` layers. With layers, the
overlap is the maximum over all layer pairs.

Calibration follows ``calibrated_similarity.calibrate_layers(..., agg="max")`` of
the Aristotelian Representation Hypothesis: shuffle the sample order of ``b``,
and rescale the observed overlap ``O`` against the ``1 - alpha`` critical value
``tau`` of the shuffled overlaps as ``(O - tau) / (1 - tau)``, clipped at 0.
Shuffling the samples only relabels a k-NN graph, so the graphs are built once
and then permuted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class OverlapResult:
    score: float  # calibrated overlap (equal to `raw` if not calibrated)
    raw: float  # mutual k-NN overlap, max over layer pairs
    tau: float | None
    p_value: float | None
    best_layers: tuple[int, int]


def _as_stack(feats: torch.Tensor) -> torch.Tensor:
    feats = torch.as_tensor(feats)
    if feats.ndim == 2:
        feats = feats[:, None, :]
    if feats.ndim != 3:
        raise ValueError(f"expected [N, D] or [N, L, D] features, got {tuple(feats.shape)}")
    return feats.float()


def prepare_features(feats: torch.Tensor, q: float = 0.95) -> torch.Tensor:
    """Clip values to ``±t``, with ``t`` the per-sample ``q``-quantile of ``|feats|``
    averaged over samples."""
    feats = _as_stack(feats)
    t = torch.quantile(feats.abs().flatten(start_dim=1), q, dim=1).mean()
    return feats.clamp(-t, t)


def knn_indices(feats: torch.Tensor, k: int = 10) -> torch.Tensor:
    """Cosine k-NN indices per layer, excluding the sample itself: ``[L, N, k]``."""
    feats = F.normalize(_as_stack(feats), dim=-1)
    out = []
    for layer in feats.unbind(dim=1):
        sim = layer @ layer.T
        sim.fill_diagonal_(-torch.inf)
        out.append(sim.topk(k, dim=1).indices)
    return torch.stack(out)


def _knn_masks(idx: torch.Tensor) -> torch.Tensor:
    L, N, _ = idx.shape
    return torch.zeros(L, N, N, device=idx.device).scatter_(2, idx, 1.0)


def _layer_overlaps(mask_a: torch.Tensor, mask_b: torch.Tensor, k: int) -> torch.Tensor:
    N = mask_a.shape[1]
    return (mask_a.flatten(1) @ mask_b.flatten(1).T) / (N * k)


def per_sample_overlap(a: torch.Tensor, b: torch.Tensor, k: int = 10) -> torch.Tensor:
    """Per-sample overlap ``o_i`` (Eq. 1) for single-layer features ``[N, D]``."""
    ia, ib = knn_indices(a, k), knn_indices(b, k)
    if len(ia) > 1 or len(ib) > 1:
        raise ValueError("per_sample_overlap expects single-layer features [N, D]")
    return (ia[0, :, :, None] == ib[0, :, None, :]).any(dim=2).float().mean(dim=1)


@torch.no_grad()
def mutual_knn(
    a: torch.Tensor,
    b: torch.Tensor,
    k: int = 10,
    *,
    calibrate: bool = True,
    num_permutations: int = 200,
    alpha: float = 0.05,
    q_outlier: float | None = 0.95,
    seed: int | None = 0,
    device: str | torch.device | None = None,
) -> OverlapResult:
    """Mutual k-NN overlap between ``a`` and ``b``, whose row ``i`` is the same
    input (or image-caption pair). ``q_outlier`` clamps outliers before the k-NN
    search, ``None`` turns this off."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    a, b = _as_stack(a).to(device), _as_stack(b).to(device)
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"a and b have different numbers of samples: {a.shape[0]} vs {b.shape[0]}")
    if q_outlier is not None:
        a, b = prepare_features(a, q_outlier), prepare_features(b, q_outlier)

    mask_a = _knn_masks(knn_indices(a, k))
    mask_b = _knn_masks(knn_indices(b, k))
    S = _layer_overlaps(mask_a, mask_b, k)
    i, j = divmod(int(S.argmax()), S.shape[1])
    raw = float(S[i, j])
    if not calibrate:
        return OverlapResult(score=raw, raw=raw, tau=None, p_value=None, best_layers=(i, j))

    # shuffling the samples of `b` permutes the rows and columns of its k-NN graph
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)
    null = torch.empty(num_permutations)
    for t in range(num_permutations):
        perm = torch.randperm(a.shape[0], generator=gen, device=device)
        null[t] = _layer_overlaps(mask_a, mask_b[:, perm][:, :, perm], k).max()

    # critical value and p-value as in calibrated_similarity.calibrate_layers
    combined = torch.cat([torch.tensor([raw]), null]).sort().values
    tau = float(combined[math.ceil((1 - alpha) * (num_permutations + 1)) - 1])
    p_value = float((1 + (null >= raw).sum()) / (num_permutations + 1))
    score = max(raw - tau, 0.0) / max(1 - tau, 1e-12)
    return OverlapResult(score=score, raw=raw, tau=tau, p_value=p_value, best_layers=(i, j))
