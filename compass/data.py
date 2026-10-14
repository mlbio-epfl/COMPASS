"""Probe sets."""

from __future__ import annotations

from datasets import load_dataset
from PIL import Image


def load_wit_probe(n: int | None = None) -> tuple[list[Image.Image], list[str]]:
    """The 1024 WIT image-caption pairs of the Platonic Representation Hypothesis
    (``minhuh/prh``, revision ``wit_1024``), as ``(images, captions)``."""
    ds = load_dataset("minhuh/prh", revision="wit_1024", split="train")
    if n is not None:
        ds = ds.select(range(n))
    return list(ds["image"]), [t[0] for t in ds["text"]]
