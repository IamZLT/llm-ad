"""VPT-style H injection: control-token cross-attention over the H feature map.

Strict VPT (arXiv:2502.17425, "Introducing Visual Perception Token into MLLM"):
the LLM emits a *control* token whose hidden state is used as K/V in a
cross-attention that modulates the visual features; the modulated features are
then fed back into the LLM as extra visual tokens.

Concretely, adapted to the scalar anomaly-prior ``H``:

    H [Ht, Wt]
        -> deterministic post-processing (gradients, multi-scale, local contrast, PE)
        -> small CNN
        -> z_H [N, d]                      (N spatial H-feature tokens)

    control token hidden h_C [1, hidden]   (the LLM's own hidden state at a fixed
                                           position, e.g. the [localize] stage)

    h_H = CrossAttn(Q = z_H, K = h_C, V = h_C)   -> [N, hidden]

    h_H is then concatenated into the LLM input right after the control token.

The direction is the *opposite* of the rejected HMemory (which used Q=LLM hidden,
K/V=Z_H as a per-layer residual). Here the LLM's current intent (h_C) *retrieves
and modulates* the H feature map, and the modulated H is injected as input tokens —
the VPT "content-dependent retrieval" interface.

This is the ONLY H channel. Proposals / region tokens / hstat / peak text are all
removed; H reaches the LLM exclusively through this control-conditioned
cross-attention.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.qwen35 import unwrap_model, unwrap_qwen_core

CONTROL_TOKEN = "<|h_ctrl|>"
FEAT_TOKEN = "<|h_feat|>"  # N placeholder slots whose embeddings receive the modulated h_H
N_CHANNELS = 9  # H, H3, H5, grad_x, grad_y, |grad|, H-AvgPool, PE_x, PE_y


def build_h_channels(h: torch.Tensor) -> torch.Tensor:
    """Deterministic multi-channel post-processing of the scalar H map.

    ``h``: [H, W] float -> [9, H, W]. All channels are a function of H alone, plus
    a fixed 2-D positional embedding so each H-feature token knows where it is.
    (No pre-merger feature, no merger, no vision block is touched.)
    """
    H, W = int(h.shape[0]), int(h.shape[1])
    h = h.float()
    gy, gx = torch.gradient(h)
    gradmag = torch.sqrt(gy * gy + gx * gx + 1e-8)
    h3 = F.avg_pool2d(h[None, None], 3, stride=1, padding=1)[0, 0]
    h5 = F.avg_pool2d(h[None, None], 5, stride=1, padding=2)[0, 0]
    local = h - h3
    ys = torch.linspace(-1.0, 1.0, H, device=h.device, dtype=h.dtype)
    xs = torch.linspace(-1.0, 1.0, W, device=h.device, dtype=h.dtype)
    pe_y, pe_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([h, h3, h5, gx, gy, gradmag, local, pe_x, pe_y], dim=0)


class HFeatureEncoder(nn.Module):
    """H [H, W] -> z_H [N, mem_dim] spatial tokens (a small CNN, no global pooling)."""

    def __init__(self, n_tokens: int = 64, mem_dim: int = 256, grid: Optional[int] = None):
        super().__init__()
        self.n_tokens = int(n_tokens)
        self.mem_dim = int(mem_dim)
        g = int(grid) if grid is not None else int(round(self.n_tokens ** 0.5))
        if g * g != self.n_tokens:
            raise ValueError(f"n_tokens={self.n_tokens} must be a perfect square (8x8=64, ...)")
        self.grid = g
        self.stem = nn.Sequential(
            nn.Conv2d(N_CHANNELS, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.GELU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2), nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((self.grid, self.grid))
        self.proj = nn.Sequential(
            nn.Linear(128, self.mem_dim), nn.GELU(), nn.Linear(self.mem_dim, self.mem_dim),
        )
        self.pos = nn.Parameter(torch.zeros(self.n_tokens, self.mem_dim))

    def forward(self, h_map: torch.Tensor) -> torch.Tensor:
        """h_map: [H, W] -> [1, N, mem_dim]."""
        dtype = next(self.parameters()).dtype
        chans = build_h_channels(h_map).to(dtype=dtype)
        x = self.stem(chans.unsqueeze(0))          # [1, 128, H', W']
        x = self.pool(x)                           # [1, 128, grid, grid]
        x = x.flatten(2).permute(0, 2, 1)          # [1, N, 128]
        z = self.proj(x)                           # [1, N, mem_dim]
        return z + self.pos.unsqueeze(0)


class HControlProjector(nn.Module):
    """Cross-attn: h_H = Out(Attn(Q=z_H, K=[z_H; h_C], V=[z_H; h_C])).

    Each H-feature token attends over the *other* H tokens plus the control-token
    hidden ``h_C``, so the control token's current intent modulates the H features
    (the "content-dependent retrieval" interface) while H's own spatial structure
    still flows through Q/K/V.

    NOTE: the original form ``Q=z_H, K/V=h_C`` (a single key/value) softmax-ed over
    a size-1 axis to a constant 1.0, which discarded ``z_H`` entirely — the encoder
    got zero gradient and H never reached the LLM (real == shuffled == zero).
    """

    def __init__(self, mem_dim: int, hidden_size: int, num_heads: int = 8,
                 init_gate: float = 1e-3):
        super().__init__()
        if int(hidden_size) % int(num_heads) != 0:
            raise ValueError(f"hidden_size={hidden_size} not divisible by num_heads={num_heads}")
        self.mem_dim = int(mem_dim)
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = int(hidden_size) // int(num_heads)
        self.q_proj = nn.Linear(self.mem_dim, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.gate = nn.Parameter(torch.tensor(float(init_gate)))

    def forward(self, z_h: torch.Tensor, h_ctrl: torch.Tensor) -> torch.Tensor:
        """z_h: [B, N, mem_dim]; h_ctrl: [B, 1, hidden] -> h_H [B, N, hidden].

        ``h_ctrl`` MUST be 3-D [B, 1, hidden] (one control token per sample). The
        [B, hidden] -> [B, 1, hidden] normalization is done once in
        :func:`compute_h_H`, the single entry point, so this layer keeps one fixed
        contract instead of guessing shapes.
        """
        assert h_ctrl.dim() == 3 and h_ctrl.shape[1] == 1, (
            f"h_ctrl must be [B, 1, hidden], got {tuple(h_ctrl.shape)}")
        b, N, _ = z_h.shape
        zh = self.q_proj(z_h)                # [B, N, hidden]  (H features -> hidden)
        kc = self.k_proj(h_ctrl)             # [B, 1, hidden]  (control key)
        vc = self.v_proj(h_ctrl)             # [B, 1, hidden]  (control value)
        k = torch.cat([zh, kc], dim=1)       # [B, N+1, hidden]
        v = torch.cat([zh, vc], dim=1)       # [B, N+1, hidden]
        q = zh.view(b, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, N + 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, N + 1, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-1, -2)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)       # [B, heads, N, N+1]
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, N, self.hidden_size)
        out = self.out_proj(out)
        return torch.tanh(self.gate.to(out.dtype)) * out


class HVPT(nn.Module):
    """H -> z_H (encoder) ; (z_H, h_C) -> h_H (control projector). One module."""

    def __init__(self, hidden_size: int, mem_dim: int = 256, n_tokens: int = 64,
                 num_heads: int = 8, init_gate: float = 1e-3):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.mem_dim = int(mem_dim)
        self.n_tokens = int(n_tokens)
        self.num_heads = int(num_heads)
        self.init_gate = float(init_gate)
        self.encoder = HFeatureEncoder(self.n_tokens, self.mem_dim)
        self.projector = HControlProjector(self.mem_dim, self.hidden_size,
                                           self.num_heads, self.init_gate)

    def encode(self, h_map: torch.Tensor) -> torch.Tensor:
        """h_map: [H, W] -> z_H [1, N, mem_dim]."""
        return self.encoder(h_map)

    def project(self, z_h: torch.Tensor, h_ctrl: torch.Tensor) -> torch.Tensor:
        """z_h: [B, N, mem_dim]; h_ctrl: [B, 1, hidden] -> h_H [B, N, hidden]."""
        return self.projector(z_h, h_ctrl)

    @property
    def config_dict(self) -> dict:
        return dict(hidden_size=self.hidden_size, mem_dim=self.mem_dim,
                    n_tokens=self.n_tokens, num_heads=self.num_heads,
                    init_gate=self.init_gate)

    def extra_repr(self) -> str:
        return (f"hidden={self.hidden_size}, mem_dim={self.mem_dim}, N={self.n_tokens}, "
                f"heads={self.num_heads}, init_gate={self.init_gate}")


def ensure_control_token(processor, model) -> int:
    """Register ``<|h_ctrl|>`` + ``<|h_feat|>`` and resize the embedding.

    Must be called on the base model BEFORE LoRA wrapping. Idempotent.
    Returns the control token id.
    """
    tok = getattr(processor, "tokenizer", processor)
    missing = [t for t in (CONTROL_TOKEN, FEAT_TOKEN) if t not in tok.get_vocab()]
    if missing:
        tok.add_special_tokens({"additional_special_tokens": missing})
        model.resize_token_embeddings(len(tok))
    return int(tok.convert_tokens_to_ids(CONTROL_TOKEN))


def control_token_id_of(processor) -> int:
    tok = getattr(processor, "tokenizer", processor)
    return int(tok.convert_tokens_to_ids(CONTROL_TOKEN))


def feat_token_id_of(processor) -> int:
    tok = getattr(processor, "tokenizer", processor)
    return int(tok.convert_tokens_to_ids(FEAT_TOKEN))


def build_h_vpt(cfg: dict, hidden_size: int) -> HVPT:
    hc = (cfg.get("outcome", {}) or {}).get("h_vpt", {}) or {}
    return HVPT(
        hidden_size=int(hidden_size),
        mem_dim=int(hc.get("mem_dim", 256)),
        n_tokens=int(hc.get("n_tokens", 64)),
        num_heads=int(hc.get("num_heads", 8)),
        init_gate=float(hc.get("init_gate", 1e-3)),
    )


def save_h_vpt(adapter: Optional[HVPT], path) -> None:
    if adapter is None:
        return
    torch.save(dict(state_dict=adapter.state_dict(), config=adapter.config_dict), str(path))


def load_h_vpt(cls, path, hidden_size: int) -> nn.Module:
    ckpt = torch.load(str(path), map_location="cpu")
    config = dict(ckpt.get("config") or {})
    config.setdefault("hidden_size", int(hidden_size))
    adapter = cls(**config)
    adapter.load_state_dict(ckpt["state_dict"])
    return adapter


def _embedding_consumer(model) -> nn.Module:
    """The module whose ``get_input_embeddings()`` is actually called during forward.

    Mirrors ``models/region_injection._embedding_consumer``: patch the inner
    ``Qwen3_5Model`` so the feat-token scatter takes effect during the real forward.
    """
    core = unwrap_qwen_core(unwrap_model(model))
    inner = getattr(core, "model", None)
    if inner is not None and inner is not core and hasattr(inner, "get_input_embeddings"):
        return inner
    return core


class _HFeatScatterEmbedding(nn.Module):
    """Wrap the token embedding so ``<|h_feat|>`` positions receive the modulated h_H."""

    def __init__(self, base: nn.Module, feat_token_id: int, h_H: torch.Tensor):
        super().__init__()
        self.base = base
        self.feat_token_id = int(feat_token_id)
        self.h_H = h_H  # [N, hidden]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.base(input_ids)
        if self.h_H is None:
            return embeds
        mask = input_ids == self.feat_token_id
        n = int(mask.sum())
        if n > 0:
            flat = self.h_H
            if flat.shape[0] != n:
                # Group members share one H map (and one prompt), so a single
                # [N, hidden] h_H tiles across a batch of ``batch_size * N`` slots.
                if n % flat.shape[0] == 0:
                    flat = flat.repeat(n // flat.shape[0], 1)
                else:
                    raise RuntimeError(
                        f"h_H has {flat.shape[0]} tokens, cannot tile to {n} <|h_feat|> slots"
                    )
            flat = flat.to(device=embeds.device, dtype=embeds.dtype)
            embeds = embeds.masked_scatter(mask.unsqueeze(-1), flat)
        return embeds


def compute_h_H(h_vpt: HVPT, h_map: torch.Tensor, h_ctrl: torch.Tensor) -> torch.Tensor:
    """Full VPT pass: z_H = encode(h_map); h_H = project(z_H, h_ctrl). Returns [N, hidden].

    ``h_ctrl`` is the control-token hidden state, accepted as either ``[1, hidden]``
    or ``[1, 1, hidden]``; it is normalized to the projector's 3-D contract here so
    every caller shares one shape discipline. Returns the single-sample ``[N, hidden]``.
    """
    ref = next(h_vpt.parameters())
    device, dtype = ref.device, ref.dtype
    if h_map.dim() == 3:
        h_map = h_map[0]
    if h_ctrl.dim() == 2:
        h_ctrl = h_ctrl.unsqueeze(1)         # [B, hidden] -> [B, 1, hidden]
    elif h_ctrl.dim() != 3:
        raise ValueError(f"h_ctrl must be [B, hidden] or [B, 1, hidden], got {tuple(h_ctrl.shape)}")
    z_h = h_vpt.encode(h_map.to(device=device, dtype=dtype))     # [1, N, mem_dim]
    h_H = h_vpt.project(z_h, h_ctrl.to(device=device, dtype=dtype))  # [B, N, hidden]
    return h_H[0]


@contextmanager
def bind_h_vpt(model, h_H: Optional[torch.Tensor], feat_token_id: int) -> Iterator[None]:
    """Replace ``<|h_feat|>`` placeholder embeddings with the modulated h_H.

    ``h_H`` is precomputed by the caller via :func:`compute_h_H` (two-pass VPT: the
    caller first obtains the control-token hidden, then enters this context for the
    main forward). The scatter preserves gradients when the caller does NOT wrap the
    enclosing forward in ``torch.no_grad``.
    """
    if h_H is None:
        yield
        return
    consumer = _embedding_consumer(model)
    orig = consumer.get_input_embeddings
    wrapper = _HFeatScatterEmbedding(orig(), int(feat_token_id), h_H)
    consumer.get_input_embeddings = lambda: wrapper
    try:
        yield
    finally:
        consumer.get_input_embeddings = orig


def _language_hidden_size(model) -> int:
    cfg = getattr(model, "config", None)
    tc = getattr(cfg, "text_config", None)
    if getattr(tc, "hidden_size", None) is not None:
        return int(tc.hidden_size)
    if getattr(cfg, "hidden_size", None) is not None:
        return int(cfg.hidden_size)
    raise ValueError("cannot determine language hidden size")


def attach_h_vpt(model, cfg, sft_dir) -> Optional[HVPT]:
    """Load HVPT from ``sft_dir`` when enabled; else None.

    ``outcome.h_vpt.trainable`` controls which submodules the RL optimizer may
    update (the encoder is the frozen perception front-end by default):

    - ``none`` / ``frozen`` (default): freeze everything (SFT-only H channel).
    - ``gate_projector`` / ``projector``: train the control projector (retrieval +
      modulation + gate) so the reward can shape *how* H is used; encoder frozen.
    - ``gate_only``: train only the scalar gate.
    - ``all``: train encoder + projector.
    """
    hc = (cfg.get("outcome", {}) or {}).get("h_vpt", {}) or {}
    if not bool(hc.get("enabled", False)):
        return None
    if not sft_dir:
        raise ValueError("outcome.h_vpt.enabled requires outcome.sft_adapter to locate weights")
    ckpt = Path(sft_dir) / "h_vpt.pt"
    if not ckpt.exists():
        raise ValueError(f"h_vpt weights are missing: {ckpt}. Run train_region_sft.py first.")
    hidden_size = _language_hidden_size(model)
    adapter = load_h_vpt(HVPT, ckpt, hidden_size)
    for p in adapter.parameters():
        p.requires_grad = False
    adapter.eval()

    trainable = str(hc.get("trainable", "none")).lower()
    if trainable in ("all", "true"):
        adapter.train()
        for p in adapter.parameters():
            p.requires_grad = True
    elif trainable in ("gate_projector", "projector"):
        adapter.projector.train()
        for p in adapter.projector.parameters():
            p.requires_grad = True
    elif trainable in ("gate_only", "gate"):
        adapter.projector.gate.requires_grad = True
    elif trainable not in ("none", "false", "frozen", ""):
        raise ValueError(f"unknown h_vpt.trainable: {trainable!r}")

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    adapter.to(device=device, dtype=dtype)
    model.h_vpt = adapter
    return adapter
