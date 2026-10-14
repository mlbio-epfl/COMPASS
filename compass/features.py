"""Probe features of timm vision models and Hugging Face text models.

Both extractors return ``[N, L, D]``: one embedding per sample and layer.

* vision: the output of every transformer block at the CLS token, or the mean
  over tokens for models without one (e.g. SigLIP), with the model's own eval
  transform.
* text: every hidden state, averaged over the non-padding tokens.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import timm
import torch
from PIL import Image
from timm.data import create_transform, resolve_data_config
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, CLIPTextModel, SiglipTextModel


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _batches(items: Sequence, size: int) -> Iterator[Sequence]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def load_vision_model(name: str, checkpoint: str | None = None, device: str | None = None):
    """Create a timm model without its classifier head, optionally from a
    fine-tuned state dict with timm keys."""
    model = timm.create_model(name, pretrained=checkpoint is None, num_classes=0)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        state = {k: v for k, v in state.items() if not k.startswith("head.")}
        model.load_state_dict(state)
    return model.to(device or _default_device()).eval()


def _has_cls_token(model) -> bool:
    return getattr(model, "cls_token", None) is not None and getattr(model, "num_prefix_tokens", 1) > 0


@torch.no_grad()
def extract_vision_features(
    model: str | torch.nn.Module,
    images: Sequence[Image.Image],
    *,
    layers: str = "all",
    batch_size: int = 32,
    device: str | None = None,
) -> torch.Tensor:
    """Embed ``images`` with a timm ViT.

    Args:
        model: a timm model name (downloads the pretrained weights) or a timm
            model.
        images: PIL images.
        layers: ``"all"`` for every block output (alignment), or ``"last"`` for
            the final normalized features (merging).

    Returns:
        ``[N, L, D]`` tensor on CPU (``L = 1`` for ``layers="last"``).
    """
    device = device or _default_device()
    if isinstance(model, str):
        model = load_vision_model(model, device=device)
    model.eval().to(device)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    pool_cls = _has_cls_token(model)

    captured: list[torch.Tensor] = []
    hooks = []
    if layers == "all":
        if not hasattr(model, "blocks"):
            raise ValueError("layers='all' requires a timm ViT with a `.blocks` attribute")
        for block in model.blocks:
            hooks.append(block.register_forward_hook(lambda _m, _i, out: captured.append(out)))
    elif layers != "last":
        raise ValueError(f"layers must be 'all' or 'last', got {layers!r}")

    out = []
    try:
        for batch in tqdm(list(_batches(images, batch_size)), desc="vision", leave=False):
            x = torch.stack([transform(img.convert("RGB")) for img in batch]).to(device)
            captured.clear()
            final = model.forward_features(x)
            hidden = captured if layers == "all" else [final]
            if hidden[0].ndim != 3:
                raise ValueError(f"expected token features [B, T, D], got {tuple(hidden[0].shape)}")
            pooled = [h[:, 0] if pool_cls else h.mean(dim=1) for h in hidden]
            out.append(torch.stack(pooled, dim=1).float().cpu())
    finally:
        for h in hooks:
            h.remove()
    return torch.cat(out)


def _load_text_model(name: str, device: str):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # with right padding, pads keep positions unchanged and never reach real tokens
    tok.padding_side = "right"
    # the SigLIP tokenizer does not return an attention mask by default
    if "attention_mask" not in tok.model_input_names:
        tok.model_input_names = [*tok.model_input_names, "attention_mask"]

    n = name.lower()
    if "clip" in n or "siglip" in n:
        model = (SiglipTextModel if "siglip" in n else CLIPTextModel).from_pretrained(name)
        max_length = min(512, model.config.max_position_embeddings)
    else:
        bf16 = device != "cpu" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        model = AutoModel.from_pretrained(name, dtype=torch.bfloat16 if bf16 else torch.float32)
        max_length = 512
    return tok, model.to(device).eval(), max_length


@torch.no_grad()
def extract_text_features(
    model: str,
    texts: Sequence[str],
    *,
    batch_size: int = 4,
    device: str | None = None,
) -> torch.Tensor:
    """Embed ``texts`` with a Hugging Face text model.

    Returns:
        ``[N, L, D]`` tensor on CPU, with ``L`` the number of layers plus the
        embedding layer.
    """
    device = device or _default_device()
    tok, lm, max_length = _load_text_model(model, device)
    out = []
    for batch in tqdm(list(_batches(list(texts), batch_size)), desc="text", leave=False):
        enc = tok(list(batch), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        hidden = lm(**enc, output_hidden_states=True).hidden_states
        m = enc["attention_mask"].unsqueeze(-1).to(hidden[0].dtype)
        pooled = [(h * m).sum(dim=1) / (m.sum(dim=1) + 1e-8) for h in hidden]
        out.append(torch.stack(pooled, dim=1).float().cpu())
    del lm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(out)
