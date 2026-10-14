"""Ease, ceiling and the COMPASS ranking.

All functions take probe features from :mod:`compass.features`, so each model is
embedded once, however many pairs it is in.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations

import numpy as np
import pandas as pd
import torch

from .overlap import mutual_knn

Features = Mapping[str, torch.Tensor]


def ease_alignment(vision: torch.Tensor, text: torch.Tensor, k: int = 10, **kw) -> float:
    """Calibrated overlap of a frozen vision and text encoder on paired probe data."""
    return mutual_knn(vision, text, k, **kw).score


def ease_merging(a: torch.Tensor, b: torch.Tensor, k: int = 10, **kw) -> float:
    """Raw overlap of the final-layer features of two fine-tuned checkpoints."""
    return mutual_knn(a, b, k, **{"calibrate": False, "q_outlier": None, **kw}).score


def ceiling_alignment(vision: torch.Tensor, text_library: Features, k: int = 10, **kw) -> float:
    """Mean calibrated overlap of a vision encoder with a library of text encoders."""
    return float(np.mean([mutual_knn(vision, t, k, **kw).score for t in text_library.values()]))


def ceiling_merging(
    base: torch.Tensor, vision_library: Features, k: int = 10, *, base_name: str | None = None, **kw
) -> float:
    """Specialization of a pretrained base model: one minus its mean calibrated
    overlap with a library of vision models (skipping ``base_name`` if present)."""
    overlaps = [mutual_knn(base, v, k, **kw).score for name, v in vision_library.items() if name != base_name]
    return 1 - float(np.mean(overlaps))


def assign_regimes(df: pd.DataFrame, ease: str = "ease", ceiling: str = "ceiling") -> pd.Series:
    """Split each axis at its median into the four regimes."""
    hi_e = df[ease] >= df[ease].median()
    hi_c = df[ceiling] >= df[ceiling].median()
    regime = np.select(
        [hi_e & hi_c, ~hi_e & hi_c, hi_e & ~hi_c],
        ["act now", "budget-limited", "ceiling-limited"],
        default="incompatible",
    )
    return pd.Series(regime, index=df.index, name="regime")


def _rank(df: pd.DataFrame) -> pd.DataFrame:
    df["compass"] = df["ease"] * df["ceiling"]
    df["regime"] = assign_regimes(df)
    df = df.sort_values("compass", ascending=False, ignore_index=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def score_alignment(
    vision: Features, text: Features, *, text_library: Features | None = None, k: int = 10, **kw
) -> pd.DataFrame:
    """Rank all vision x text pairs for cross-modal alignment.

    ``vision`` and ``text`` map model names to features on the same paired probe
    set. The ceiling of a vision model is its mean overlap with ``text_library``,
    which defaults to ``text``. Returns one row per pair with ``ease``,
    ``ceiling``, ``compass`` (their product), ``regime`` and ``rank``.
    """
    library = text if text_library is None else text_library
    overlaps = {
        (v, t): mutual_knn(fv, ft, k, **kw).score for v, fv in vision.items() for t, ft in {**text, **library}.items()
    }
    rows = [
        {
            "vision": v,
            "text": t,
            "ease": overlaps[v, t],
            "ceiling": float(np.mean([overlaps[v, name] for name in library])),
        }
        for v in vision
        for t in text
    ]
    return _rank(pd.DataFrame(rows))


def score_merging(
    checkpoints: Features, base_model: Mapping[str, str], ceilings: Mapping[str, float], *, k: int = 10, **kw
) -> pd.DataFrame:
    """Rank all pairs of fine-tuned checkpoints that share a base model.

    ``checkpoints`` maps names to final-layer features on the same probe images,
    ``base_model`` maps each checkpoint to its base model, and ``ceilings`` holds
    the merging ceiling of each base model (see :func:`ceiling_merging`).
    """
    rows = [
        {
            "base_model": base_model[a],
            "checkpoint_a": a,
            "checkpoint_b": b,
            "ease": ease_merging(checkpoints[a], checkpoints[b], k, **kw),
            "ceiling": ceilings[base_model[a]],
        }
        for a, b in combinations(checkpoints, 2)
        if base_model[a] == base_model[b]
    ]
    if not rows:
        raise ValueError("no two checkpoints share a base model")
    return _rank(pd.DataFrame(rows))
