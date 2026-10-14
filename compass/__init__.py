"""COMPASS: COMPatibility ASsessment Score for model pairs.

Predicts, before alignment or merging, whether two pretrained models are
compatible, from how much they agree on local neighborhoods of a probe set.
"""

from .data import load_wit_probe
from .features import extract_text_features, extract_vision_features, load_vision_model
from .overlap import OverlapResult, mutual_knn, per_sample_overlap
from .scores import (
    assign_regimes,
    ceiling_alignment,
    ceiling_merging,
    ease_alignment,
    ease_merging,
    score_alignment,
    score_merging,
)

__version__ = "0.1.0"

__all__ = [
    "OverlapResult",
    "assign_regimes",
    "ceiling_alignment",
    "ceiling_merging",
    "ease_alignment",
    "ease_merging",
    "extract_text_features",
    "extract_vision_features",
    "load_vision_model",
    "load_wit_probe",
    "mutual_knn",
    "per_sample_overlap",
    "score_alignment",
    "score_merging",
]
