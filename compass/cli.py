"""Rank vision x text model pairs for cross-modal alignment.

    compass align --vision vit_base_patch16_clip_224.laion2b vit_base_patch16_224.mae \
                  --text allenai/OLMo-1B-hf bigscience/bloomz-560m \
                  --out scores.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from .data import load_wit_probe
from .features import extract_text_features, extract_vision_features
from .scores import score_alignment


def _cached(cache: Path | None, name: str, n: int, fn):
    if cache is None:
        return fn()
    path = cache / f"{name.replace('/', '__')}_wit{n}.pt"
    if path.exists():
        return torch.load(path)
    feats = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(feats, path)
    return feats


def _align(args: argparse.Namespace) -> None:
    images, texts = load_wit_probe(args.n)
    cache = Path(args.cache) if args.cache else None
    vision = {
        m: _cached(cache, m, args.n, lambda m=m: extract_vision_features(m, images, batch_size=args.batch_size))
        for m in args.vision
    }
    text = {m: _cached(cache, m, args.n, lambda m=m: extract_text_features(m, texts)) for m in args.text}
    df = score_alignment(vision, text, k=args.k)

    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nsaved to {args.out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="compass", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("align", help="rank vision x text pairs for cross-modal alignment")
    p.add_argument("--vision", nargs="+", required=True, help="timm model names")
    p.add_argument("--text", nargs="+", required=True, help="Hugging Face model names")
    p.add_argument("--n", type=int, default=1024, help="number of WIT probe pairs (default: 1024)")
    p.add_argument("--k", type=int, default=10, help="neighborhood size (default: 10)")
    p.add_argument("--batch-size", type=int, default=32, help="vision batch size")
    p.add_argument("--cache", help="directory to cache extracted features")
    p.add_argument("--out", help="write the ranking to this CSV")
    p.set_defaults(func=_align)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
