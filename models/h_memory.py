"""Spatial H-Memory cross-attention: a content-addressable prior for Qwen.

The frozen vision-comparison module already produced the scalar anomaly-prior map
``H``. This module turns ``H`` into a *structured spatial memory* WITHOUT touching
pre-merger features, the merger, or any vision block:

    H in R^{Ht x Wt}
        -> deterministic post-processing  (gradients, multi-scale, local contrast, PE)
        -> small CNN
        -> Z_H in R^{K x d}          (K = 8x8 = 64 memory tokens, each 2D-positioned)

then injects it into Qwen via cross-attention on a few decoder layers:

    Y_l = QwenBlock_l(X_l)
    C_l = CrossAttn(Q = LN(Y_l), K = Z_H, V = Z_H)
    X_{l+1} = Y_l + tanh(g_l) * C_l

Unlike the failed HPriorAdapter (a one-shot additive residual on every region
embedding), cross-attention lets each language hidden state ``q_t`` *retrieve* the
H region it needs: ``c_t = sum_k softmax(q_t . k_k) v_k``. ``[localize]`` can attend
the upper-left H memory, ``[confirm]`` can attend broadly or not at all.

Causal ablation (the "life-or-death" test): proposals + RegionAdapter always use the
real H; only the memory fed to this module is perturbed (real / shuffled / zero).
Success requires ``real > shuffled`` and ``shuffled ~= zero``.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.qwen35 import unwrap_model, unwrap_qwen_core

N_CHANNELS = 9  # H, H3, H5, grad_x, grad_y, |grad|, H-AvgPool, PE_x, PE_y


def build_h_channels(h: torch.Tensor) -> torch.Tensor:
    """Deterministic multi-channel post-processing of the scalar H map (no new vision info).

    ``h``: [H, W] float -> [9, H, W]. All channels are a function of H alone, plus a
    fixed 2-D positional embedding (PE) so each memory token knows where it is.
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


class SpatialHMemory(nn.Module):
    """H -> Z_H in R^{K x mem_dim}, with a fixed 2-D positional bias per token."""

    def __init__(self, memory_size: int = 64, mem_dim: int = 256, grid: Optional[int] = None):
        super().__init__()
        self.memory_size = int(memory_size)
        self.mem_dim = int(mem_dim)
        g = int(grid) if grid is not None else int(round(self.memory_size ** 0.5))
        if g * g != self.memory_size:
            raise ValueError(f"memory_size={memory_size} must be a perfect square (8x8=64, ...)")
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
        self.pos = nn.Parameter(torch.zeros(self.grid * self.grid, self.mem_dim))

    def forward(self, h_map: torch.Tensor) -> torch.Tensor:
        """h_map: [H, W] -> [1, K, mem_dim]."""
        dtype = next(self.parameters()).dtype
        chans = build_h_channels(h_map).to(dtype=dtype)
        x = self.stem(chans.unsqueeze(0))          # [1, 128, H', W']
        x = self.pool(x)                           # [1, 128, grid, grid]
        x = x.flatten(2).permute(0, 2, 1)          # [1, K, 128]
        z = self.proj(x)                           # [1, K, mem_dim]
        return z + self.pos.unsqueeze(0)


class HCrossAttn(nn.Module):
    """One cross-attention adapter: C = tanh(g) * Out(Attn(Q=LN(Y), K=Z_H, V=Z_H))."""

    def __init__(self, hidden_size: int, mem_dim: int, num_heads: int = 8,
                 init_gate: float = 1e-3):
        super().__init__()
        if int(hidden_size) % int(num_heads) != 0:
            raise ValueError(f"hidden_size={hidden_size} not divisible by num_heads={num_heads}")
        self.hidden_size = int(hidden_size)
        self.mem_dim = int(mem_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(hidden_size) // int(num_heads)
        self.q_norm = nn.LayerNorm(self.hidden_size)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.mem_dim, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.mem_dim, self.hidden_size, bias=False)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.gate = nn.Parameter(torch.tensor(float(init_gate)))

    def forward(self, hidden: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """hidden: [B, L, hidden]; memory: [B, K, mem_dim] -> residual [B, L, hidden]."""
        b, L, _ = hidden.shape
        q = self.q_proj(self.q_norm(hidden))
        k = self.k_proj(memory)
        v = self.v_proj(memory)
        q = q.view(b, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-1, -2)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, L, self.hidden_size)
        out = self.out_proj(out)
        return torch.tanh(self.gate.to(out.dtype)) * out


class HMemory(nn.Module):
    """Encoder + per-layer cross-attention adapters (one module, saved to h_memory.pt)."""

    def __init__(self, hidden_size: int, mem_dim: int = 256, memory_size: int = 64,
                 layers: Sequence[int] = (11, 19), num_heads: int = 8,
                 init_gate: float = 1e-3):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.mem_dim = int(mem_dim)
        self.memory_size = int(memory_size)
        self.num_heads = int(num_heads)
        self.init_gate = float(init_gate)
        self.layers = tuple(int(l) for l in layers)
        self.encoder = SpatialHMemory(self.memory_size, self.mem_dim)
        self.cross_attns = nn.ModuleDict({
            str(l): HCrossAttn(self.hidden_size, self.mem_dim, self.num_heads, self.init_gate)
            for l in self.layers
        })

    def build_memory(self, h_map: torch.Tensor) -> torch.Tensor:
        """h_map: [H, W] (single sample) -> [1, K, mem_dim]."""
        return self.encoder(h_map)

    @property
    def config_dict(self) -> dict:
        return dict(hidden_size=self.hidden_size, mem_dim=self.mem_dim,
                    memory_size=self.memory_size, layers=list(self.layers),
                    num_heads=self.num_heads, init_gate=self.init_gate)

    def extra_repr(self) -> str:
        return (f"hidden={self.hidden_size}, mem_dim={self.mem_dim}, K={self.memory_size}, "
                f"layers={list(self.layers)}, heads={self.num_heads}, init_gate={self.init_gate}")


def _text_model(core) -> nn.Module:
    qwen = unwrap_qwen_core(core)
    mm_model = getattr(qwen, "model", None)
    if mm_model is None:
        raise RuntimeError("cannot find Qwen multimodal model")
    text_model = getattr(mm_model, "language_model", None)
    if text_model is None:
        raise RuntimeError("cannot find Qwen language_model")
    return text_model


def _make_hook(cross_attn: HCrossAttn, z_h: torch.Tensor):
    def hook(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        b = int(hidden.shape[0])
        mem = z_h
        if mem.shape[0] == 1 and b > 1:
            mem = mem.expand(b, -1, -1)
        elif mem.shape[0] != b:
            # Batch mismatch (should not happen); leave the layer output untouched.
            return output
        residual = cross_attn(hidden, mem)
        new_hidden = hidden + residual
        if isinstance(output, tuple):
            return (new_hidden,) + output[1:]
        return new_hidden

    return hook


@contextmanager
def bind_h_cross_attn(model, h_memory, h_maps, device=None, dtype=None):
    """Inject H-memory cross-attention into selected decoder layers for one forward.

    ``h_maps`` is a single [H,W] tensor (eval/RL single sample) or a list of them
    (SFT batch). The memory Z_H is built once here (under the caller's grad context,
    so the encoder receives gradients during training) and reused by every hook.
    """
    core = unwrap_model(model)
    h_mem = h_memory if h_memory is not None else getattr(core, "h_memory", None)
    if h_mem is None or h_maps is None:
        yield
        return
    if torch.is_tensor(h_maps):
        maps: List[torch.Tensor] = [h_maps]
    else:
        maps = [m for m in h_maps if m is not None]
    if not maps:
        yield
        return

    ref = next(h_mem.parameters())
    device = device or ref.device
    dtype = dtype or ref.dtype
    mems = []
    for m in maps:
        if m.dim() == 3:
            m = m[0]
        mems.append(h_mem.build_memory(m.to(device=device, dtype=dtype)))
    z_h = torch.cat(mems, dim=0)  # [B, K, mem_dim]

    text_model = _text_model(core)
    handles = []
    for l in h_mem.layers:
        layer = text_model.layers[int(l)]
        ca = h_mem.cross_attns[str(l)]
        handles.append(layer.register_forward_hook(_make_hook(ca, z_h)))
    try:
        yield
    finally:
        for hdl in handles:
            hdl.remove()


def build_h_memory(cfg: dict, hidden_size: int) -> HMemory:
    hc = (cfg.get("outcome", {}) or {}).get("h_memory", {}) or {}
    return HMemory(
        hidden_size=int(hidden_size),
        mem_dim=int(hc.get("mem_dim", 256)),
        memory_size=int(hc.get("memory_size", 64)),
        layers=[int(l) for l in hc.get("layers", [11, 19])],
        num_heads=int(hc.get("num_heads", 8)),
        init_gate=float(hc.get("init_gate", 1e-3)),
    )


def save_h_memory(adapter: Optional[HMemory], path) -> None:
    if adapter is None:
        return
    torch.save(dict(state_dict=adapter.state_dict(), config=adapter.config_dict), str(path))


def load_h_memory(cls, path, hidden_size: int) -> nn.Module:
    ckpt = torch.load(str(path), map_location="cpu")
    config = dict(ckpt.get("config") or {})
    config.setdefault("hidden_size", int(hidden_size))
    adapter = cls(**config)
    adapter.load_state_dict(ckpt["state_dict"])
    return adapter


def attach_h_memory(model, cfg, sft_dir) -> Optional[HMemory]:
    """Load the (frozen) HMemory from ``sft_dir`` when enabled; else None."""
    hc = (cfg.get("outcome", {}) or {}).get("h_memory", {}) or {}
    if not bool(hc.get("enabled", False)):
        return None
    if not sft_dir:
        raise ValueError("outcome.h_memory.enabled requires outcome.sft_adapter to locate weights")
    ckpt = Path(sft_dir) / "h_memory.pt"
    if not ckpt.exists():
        raise ValueError(f"h_memory weights are missing: {ckpt}. Run train_region_sft.py first.")
    hidden_size = _language_hidden_size(model)
    adapter = load_h_memory(HMemory, ckpt, hidden_size)
    for p in adapter.parameters():
        p.requires_grad = False
    adapter.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    adapter.to(device=device, dtype=dtype)
    return adapter


def _language_hidden_size(model) -> int:
    cfg = getattr(model, "config", None)
    tc = getattr(cfg, "text_config", None)
    if getattr(tc, "hidden_size", None) is not None:
        return int(tc.hidden_size)
    if getattr(cfg, "hidden_size", None) is not None:
        return int(cfg.hidden_size)
    raise ValueError("cannot determine language hidden size")
