"""H-Box geometry-token prior: inject H connected-component boxes as geometry tokens.

IAD-Unify style (arXiv:2604.12440): a frozen region expert supplies anomaly
evidence, and its *geometry* (bounding-box centre + size) is projected into a
token that is injected via placeholder replacement — giving the VLM a region
prior WITHOUT dense-feature cross-attention. Its key finding ("region grounding
is the decisive mechanism; removing it degrades location accuracy by >76pp") is
exactly the gap our VPT dense-feature channel failed to fill: the model detects
"where" but not "how big", so it emits under-sized boxes (see decompose_loc_miss).

Here the region expert is the frozen anomaly-prior H map. Connected components of
the thresholded H give K candidate boxes — the *extent* signal that dense-feature
VPT could not convey (VPT's softmax projector also caused KL explosion in RL).

Pipeline:

    H [Ht, Wt]
        -> connected components (threshold)           (region_proposals in inputs.py)
        -> K candidate boxes -> geom [K, GEOM_DIM]     (geometry_from_proposals)
        -> HBoxProjector (linear, static) -> g_proj [K, hidden]
        -> scattered into K <|h_box|> placeholder slots inside [localize]

This is a STATIC, linear injection (no control-token cross-attention, no gate),
so it avoids VPT's softmax instability and gives the model a direct, interpretable
"look here, at this size" prior that it refines into precise bboxes_2d.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn as nn

from models.h_vpt import _embedding_consumer, _language_hidden_size
from models.qwen35 import unwrap_model

BOX_TOKEN = "<|h_box|>"
# x1, y1, x2, y2, cx, cy, w, h, area, peak, is_valid  (all [0,1] except is_valid)
GEOM_DIM = 11


class HBoxProjector(nn.Module):
    """geom [K, GEOM_DIM] -> g_proj [K, hidden] (a small static MLP, no attention)."""

    def __init__(self, geom_dim: int = GEOM_DIM, hidden_size: int = 256):
        super().__init__()
        self.geom_dim = int(geom_dim)
        self.hidden_size = int(hidden_size)
        self.proj = nn.Sequential(
            nn.Linear(self.geom_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

    def forward(self, geom: torch.Tensor) -> torch.Tensor:
        """geom: [K, GEOM_DIM] -> [K, hidden]."""
        return self.proj(geom)

    @property
    def config_dict(self) -> dict:
        return dict(geom_dim=self.geom_dim, hidden_size=self.hidden_size)

    def extra_repr(self) -> str:
        return f"geom_dim={self.geom_dim}, hidden={self.hidden_size}"


def geometry_from_proposals(proposals, K: int, h_min: float = 0.0,
                            h_max: float = 1.0) -> np.ndarray:
    """Convert ``region_proposals`` output into a [K, GEOM_DIM] geometry matrix.

    Each box is normalised to [0,1] (bbox / 1000, peak / (h_max-h_min)). Rows
    beyond ``len(proposals)`` are all-zero with ``is_valid=0`` (an explicit "no
    candidate here" token the model learns to ignore).
    """
    K = max(0, int(K))
    g = np.zeros((K, GEOM_DIM), dtype=np.float32)
    peak_span = (float(h_max) - float(h_min)) if h_max > h_min else 1.0
    for i, p in enumerate(list(proposals)[:K]):
        b = (p or {}).get('bbox_2d') or [0, 0, 0, 0]
        x1 = float(b[0]) / 1000.0
        y1 = float(b[1]) / 1000.0
        x2 = float(b[2]) / 1000.0
        y2 = float(b[3]) / 1000.0
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        area = w * h
        peak = (float((p or {}).get('raw_peak') or 0.0) - float(h_min)) / peak_span
        peak = max(0.0, min(1.0, peak))
        g[i] = [x1, y1, x2, y2, cx, cy, w, h, area, peak, 1.0]
    return g


def ensure_box_token(processor, model) -> int:
    """Register ``<|h_box|>`` and resize the embedding (idempotent)."""
    tok = getattr(processor, "tokenizer", processor)
    if BOX_TOKEN not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": [BOX_TOKEN]})
        model.resize_token_embeddings(len(tok))
    return int(tok.convert_tokens_to_ids(BOX_TOKEN))


def box_token_id_of(processor) -> int:
    tok = getattr(processor, "tokenizer", processor)
    return int(tok.convert_tokens_to_ids(BOX_TOKEN))


def build_h_box_prior(cfg: dict, hidden_size: int) -> HBoxProjector:
    hc = (cfg.get("outcome", {}) or {}).get("h_box_prior", {}) or {}
    return HBoxProjector(
        geom_dim=int(hc.get("geom_dim", GEOM_DIM)),
        hidden_size=int(hidden_size),
    )


def save_h_box_prior(proj: Optional[HBoxProjector], path) -> None:
    if proj is None:
        return
    torch.save(dict(state_dict=proj.state_dict(), config=proj.config_dict), str(path))


def load_h_box_prior(path, hidden_size: int) -> HBoxProjector:
    ckpt = torch.load(str(path), map_location="cpu")
    config = dict(ckpt.get("config") or {})
    config.setdefault("hidden_size", int(hidden_size))
    proj = HBoxProjector(**config)
    proj.load_state_dict(ckpt["state_dict"])
    return proj


def attach_h_box_prior(model, cfg, sft_dir) -> Optional[HBoxProjector]:
    """Load the (frozen) HBoxProjector from ``sft_dir`` when enabled; else None."""
    hc = (cfg.get("outcome", {}) or {}).get("h_box_prior", {}) or {}
    if not bool(hc.get("enabled", False)):
        return None
    if not sft_dir:
        raise ValueError("outcome.h_box_prior.enabled requires outcome.sft_adapter")
    ckpt = Path(sft_dir) / "h_box_prior.pt"
    if not ckpt.exists():
        raise ValueError(f"h_box_prior weights missing: {ckpt}. Run train_region_sft.py first.")
    hidden_size = _language_hidden_size(model)
    proj = load_h_box_prior(ckpt, hidden_size)
    for p in proj.parameters():
        p.requires_grad = False
    proj.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    proj.to(device=device, dtype=dtype)
    model.h_box_prior = proj
    return proj


def compute_g_proj(proj: HBoxProjector, geom: torch.Tensor) -> torch.Tensor:
    """geom [K, GEOM_DIM] -> g_proj [K, hidden]."""
    ref = next(proj.parameters())
    if geom.ndim == 1:
        geom = geom.unsqueeze(0)
    return proj(geom.to(device=ref.device, dtype=ref.dtype))


class _HBoxScatterEmbedding(nn.Module):
    """Wrap the token embedding so ``<|h_box|>`` positions receive g_proj."""

    def __init__(self, base: nn.Module, box_token_id: int, g_proj: torch.Tensor):
        super().__init__()
        self.base = base
        self.box_token_id = int(box_token_id)
        self.g_proj = g_proj  # [K, hidden]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.base(input_ids)
        if self.g_proj is None:
            return embeds
        mask = input_ids == self.box_token_id
        n = int(mask.sum())
        if n > 0:
            flat = self.g_proj
            if flat.shape[0] != n:
                if n % flat.shape[0] == 0:
                    flat = flat.repeat(n // flat.shape[0], 1)
                else:
                    raise RuntimeError(
                        f"g_proj has {flat.shape[0]} boxes, cannot tile to {n} <|h_box|> slots"
                    )
            flat = flat.to(device=embeds.device, dtype=embeds.dtype)
            embeds = embeds.masked_scatter(mask.unsqueeze(-1), flat)
        return embeds


@contextmanager
def bind_h_box(model, g_proj: Optional[torch.Tensor], box_token_id: int) -> Iterator[None]:
    """Replace ``<|h_box|>`` placeholder embeddings with the projected g_proj."""
    if g_proj is None:
        yield
        return
    consumer = _embedding_consumer(model)
    orig = consumer.get_input_embeddings
    wrapper = _HBoxScatterEmbedding(orig(), int(box_token_id), g_proj)
    consumer.get_input_embeddings = lambda: wrapper
    try:
        yield
    finally:
        consumer.get_input_embeddings = orig
