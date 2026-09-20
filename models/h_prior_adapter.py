"""Independent H-prior side channel fused into region tokens via a gated residual.

Unlike ``RegionAdapter`` (which projects test/ref/diff/geom/hstat branches), this
module only ever sees the *local shape* of the anomaly prior: a ``P x P`` patch
sampled from the sample-normalized H map around each region token's center. Its
output is a small additive correction to the region embedding, gated by a learned
scalar ``alpha``:

    R_i' = R_i + tanh(alpha) * HPriorAdapter(H_local_i)

At ``alpha == 0`` the correction is exactly zero, so a freshly initialized adapter
is bit-identical to the baseline (region embeddings pass through unchanged). This
keeps ``RegionAdapter`` and ``HPriorAdapter`` cleanly separable, and does not add
any token, prompt text, or decoder change — the same ``<|region|>`` placeholders
are injected into Qwen.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HPriorAdapter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        patch_size: int = 7,
        intermediate_dim: int = 256,
        init_gate: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.patch_size = int(patch_size)
        self.intermediate_dim = int(intermediate_dim)
        self.init_gate = float(init_gate)

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.proj = nn.Sequential(
            nn.Linear(32 * 2 * 2, self.intermediate_dim),
            nn.GELU(),
            nn.Linear(self.intermediate_dim, self.hidden_size),
        )
        # Learned gate; tanh keeps it bounded. init 0 -> exact baseline identity.
        self.alpha = nn.Parameter(torch.tensor(self.init_gate))

    def forward(self, hpatch, region_embed, valid=None):
        """hpatch: [B, N, 1, P, P]; region_embed: [B, N, hidden]; valid: [B, N] bool.

        Returns region_embed + tanh(alpha) * delta. Invalid cells (valid=0, e.g. the
        single "empty" region when no H candidate exists) get a zero delta so the
        learned ``empty`` embedding of the main RegionAdapter is preserved exactly.

        The raw conv+proj output is unbounded (empirically ~60x the region embedding
        norm), which would let a small alpha overwrite the region token entirely and
        collapse the [localize] stage. We therefore rescale delta to match the
        region embedding's per-token L2 norm so ``tanh(alpha)`` directly controls the
        mixing fraction (0 -> baseline, 1 -> full replacement), not the raw magnitude.
        """
        B, N = hpatch.shape[:2]
        x = hpatch.reshape(B * N, 1, self.patch_size, self.patch_size)
        x = self.encoder(x).flatten(1)
        delta = self.proj(x).reshape(B, N, self.hidden_size).to(region_embed.dtype)
        if valid is not None:
            mask = valid.unsqueeze(-1).to(delta.dtype)
            delta = delta * mask
        dn = delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        rn = region_embed.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        delta = delta * (rn / dn)
        return region_embed + torch.tanh(self.alpha.to(region_embed.dtype)) * delta

    @property
    def config_dict(self) -> dict:
        return dict(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            intermediate_dim=self.intermediate_dim,
            init_gate=self.init_gate,
        )

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, patch_size={self.patch_size}, "
            f"intermediate_dim={self.intermediate_dim}, init_gate={self.init_gate}"
        )


def build_h_prior_adapter(cfg: dict, hidden_size: int) -> HPriorAdapter:
    """Build an HPriorAdapter from ``outcome.h_prior_adapter`` config."""
    hc = (cfg.get("outcome", {}) or {}).get("h_prior_adapter", {}) or {}
    return HPriorAdapter(
        hidden_size=int(hidden_size),
        patch_size=int(hc.get("patch_size", 7)),
        intermediate_dim=int(hc.get("intermediate_dim", 256)),
        init_gate=float(hc.get("init_gate", 0.0)),
    )


def save_h_prior_adapter(adapter, path) -> None:
    """Persist a separately-mounted non-PEFT module (model.save_pretrained skips it)."""
    if adapter is None:
        return
    torch.save(dict(state_dict=adapter.state_dict(), config=adapter.config_dict), str(path))


def load_h_prior_adapter(cls, path, hidden_size: int) -> nn.Module:
    """Rebuild an HPriorAdapter from ``save_h_prior_adapter`` output."""
    ckpt = torch.load(str(path), map_location="cpu")
    config = dict(ckpt.get("config") or {})
    config.setdefault("hidden_size", int(hidden_size))
    adapter = cls(**config)
    adapter.load_state_dict(ckpt["state_dict"])
    return adapter
