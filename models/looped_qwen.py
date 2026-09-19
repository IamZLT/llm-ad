"""Switchable latent-space recurrent-depth mode for the Qwen3.5 language stack.

External-FSM (token-space)::

    X -> U -> C -> L -> V -> A

recurrence happens OUTSIDE the model by emitting stage tokens and re-prefilling.

Internal-Loop (latent-space)::

    H^(0) -> F_theta(H^(0)) -> F_theta(H^(1)) -> ... -> F_theta(H^(K-1)) -> A

recurrence happens INSIDE the Qwen decoder. ``F_theta`` is the *same* original
``Qwen3_5TextModel.forward`` every time — no new Transformer block is created,
no ``ModuleList`` is added, and all loop iterations (including LoRA) share one
set of parameters. This reuses Qwen's native full/linear attention mask and
M-RoPE construction, so the only thing we touch is the ``language_model.forward``
entry point.

Recurrent-depth cache::

    C^(1), C^(2), ..., C^(K)

Each recurrent depth owns an independent ``DynamicCache``. For the t-th token
``x_t -> F_theta(C^(1)) -> h_t^(1) -> F_theta(C^(2)) -> h_t^(2) -> ...`` the
k-th pass only ever processes ``[B, 1, D]`` against ``C^(k)`` (which already
holds the full history of depth k), instead of re-running the whole prefix.
This is the standard cached autoregressive decode for a recurrent-depth stack:
position ``t`` does not change across depths, so the M-RoPE / ``position_ids``
are identical for all K passes and are simply passed through.

The 2-D ``(position t, depth k)`` recurrence therefore needs one cache per depth
(``K_{l,k}(t), V_{l,k}(t)`` plus the linear-attention conv/recurrent state), not
a single cache. ``LoopDepthCache`` presents these K caches to ``generate()`` as
one ``Cache`` object while routing each depth pass to its own inner cache.
"""

from __future__ import annotations

from types import MethodType

import torch
from transformers.cache_utils import Cache, DynamicCache

from models.qwen35 import unwrap_model, unwrap_qwen_core


def loop_config(cfg: dict) -> dict:
    return (cfg.get("outcome") or {}).get("loop", {}) or {}


def loop_enabled(cfg: dict) -> bool:
    lc = loop_config(cfg)
    return bool(lc.get("enabled", False))


def loop_steps(cfg: dict) -> int:
    if not loop_enabled(cfg):
        return 1
    return max(1, int(loop_config(cfg).get("steps", 1)))


def get_qwen_text_model(model):
    """Return the ``Qwen3_5TextModel`` from a (possibly Peft/DDP-wrapped) Qwen3.5-VL.

    Qwen3.5-VL layout::

        Qwen3_5ForConditionalGeneration
          -> model: Qwen3_5Model
             -> language_model: Qwen3_5TextModel
    """
    core = unwrap_qwen_core(unwrap_model(model))

    mm_model = getattr(core, "model", None)
    if mm_model is None:
        raise RuntimeError("cannot find Qwen multimodal model")

    text_model = getattr(mm_model, "language_model", None)
    if text_model is None:
        raise RuntimeError("cannot find Qwen language_model")

    return text_model


def is_looped(model) -> bool:
    """True when ``enable_looped_qwen`` has patched the language stack."""
    try:
        tm = get_qwen_text_model(model)
    except RuntimeError:
        return False
    return hasattr(tm, "_loop_original_forward")


class LoopDepthCache(Cache):
    """K independent ``DynamicCache`` instances, one per recurrent depth.

    ``generate()`` only ever calls the ``Cache`` surface on this object (seq
    length, reorder/crop/batch ops, ...) and forwards it back unchanged; the
    actual K/V / conv-state / recurrent-state updates happen on the inner caches
    inside ``looped_forward``. Every read-only query delegates to depth 0 (all
    depths share the same sequence length); every mutating op is broadcast to
    all depths.
    """

    def __init__(self, config, K: int):
        K = max(int(K), 1)
        self.depth_caches = [DynamicCache(config=config) for _ in range(K)]
        self._depth_count = K
        # Fallback so base-class helpers that read `self.layers` still work.
        self.layers = self.depth_caches[0].layers
        self.offloading = False

    # ---- read-only queries (delegate to depth 0) ----
    def __len__(self):
        return len(self.depth_caches[0])

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.depth_caches[0].get_seq_length(layer_idx)

    def get_max_length(self, layer_idx: int | None = None) -> int:
        return self.depth_caches[0].get_max_length(layer_idx)

    def has_previous_state(self, layer_idx: int | None = None, state_idx: int | None = None) -> bool:
        return self.depth_caches[0].has_previous_state(layer_idx, state_idx)

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple[int, int]:
        return self.depth_caches[0].get_mask_sizes(query_length, layer_idx)

    def get_query_offset(self, layer_idx: int = 0) -> int:
        return self.depth_caches[0].get_query_offset(layer_idx)

    # ---- mutating ops (broadcast to all depths) ----
    def reset(self):
        for c in self.depth_caches:
            c.reset()

    def reorder_cache(self, beam_idx: torch.LongTensor):
        for c in self.depth_caches:
            c.reorder_cache(beam_idx)

    def crop(self, max_length: int):
        for c in self.depth_caches:
            c.crop(max_length)

    def batch_repeat_interleave(self, repeats: int):
        for c in self.depth_caches:
            for layer in c.layers:
                _repeat_cache_layer_batch(layer, repeats)

    def batch_select_indices(self, indices: torch.Tensor):
        for c in self.depth_caches:
            c.batch_select_indices(indices)

    def activate_past_recording(self):
        for c in self.depth_caches:
            c.activate_past_recording()

    # ---- properties ----
    @property
    def batch_size(self) -> int:
        return self.depth_caches[0].batch_size

    @property
    def is_compileable(self) -> bool:
        return self.depth_caches[0].is_compileable

    @property
    def is_initialized(self) -> bool:
        return self.depth_caches[0].is_initialized

    @property
    def is_sliding(self) -> list:
        return self.depth_caches[0].is_sliding

    @property
    def is_linear(self) -> list:
        return self.depth_caches[0].is_linear


def _repeat_cache_layer_batch(layer, repeats: int) -> None:
    """Repeat one cache layer along the batch dim (full-attn KV + GatedDeltaNet states).

    ``transformers`` 5.x does not implement ``batch_repeat_interleave`` on
    ``LinearAttentionLayer`` (the GatedDeltaNet cache), only on ``DynamicLayer``.
    For the shared-prefill rollout (B=1 prefill then decode at B=group) we must
    repeat every layer ourselves: full-attention ``keys/values`` and the linear
    layer's ``conv_states`` / ``recurrent_states`` all live on batch dim 0.
    """
    repeats = max(int(repeats), 1)
    keys = getattr(layer, "keys", None)
    values = getattr(layer, "values", None)
    if isinstance(keys, torch.Tensor) and keys.numel() > 0:
        layer.keys = keys.repeat_interleave(repeats, dim=0)
    if isinstance(values, torch.Tensor) and values.numel() > 0:
        layer.values = values.repeat_interleave(repeats, dim=0)
    for name in ("conv_states", "recurrent_states"):
        states = getattr(layer, name, None)
        if isinstance(states, dict):
            for i, st in states.items():
                if isinstance(st, torch.Tensor) and st.numel() > 0:
                    states[i] = st.repeat_interleave(repeats, dim=0)


def _last_hidden(out, return_dict: bool):
    if return_dict:
        return out.last_hidden_state
    return out[0]


def enable_looped_qwen(model, cfg: dict):
    """Patch ``language_model.forward`` to run the same stack ``steps`` times.

    No new Transformer block is created; all loop iterations share exactly the
    same parameters, including the same LoRA parameters. Calling this again only
    updates the step count (idempotent, no double patching).

    With ``use_cache=True`` (generation) each depth pass is routed to its own
    inner ``DynamicCache`` inside a ``LoopDepthCache``; with ``use_cache=False``
    (teacher-forcing) the full sequence is recomputed per depth with no cache.
    """
    steps = loop_steps(cfg)

    if steps <= 1:
        return model

    text_model = get_qwen_text_model(model)

    # Avoid double patching: capture the ORIGINAL forward only once.
    if hasattr(text_model, "_loop_original_forward"):
        text_model._loop_steps = steps
        return model

    original_forward = text_model.forward

    text_model._loop_original_forward = original_forward
    text_model._loop_steps = steps
    text_model._last_loop_stats = {}

    def looped_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        **kwargs,
    ):
        K = int(self._loop_steps)
        return_dict = bool(kwargs.get("return_dict", True))

        # Resolve the per-depth cache. On the prefill step generate() may hand us
        # a freshly-created (empty) DynamicCache; replace it with a LoopDepthCache
        # holding K inner caches. On decode steps generate() hands back the exact
        # LoopDepthCache we returned, so it is reused directly.
        if isinstance(past_key_values, LoopDepthCache):
            loop_cache = past_key_values
            depth_caches = past_key_values.depth_caches
        elif use_cache:
            loop_cache = LoopDepthCache(self.config, K)
            depth_caches = loop_cache.depth_caches
        else:
            loop_cache = None
            depth_caches = [None] * K

        # ----- depth 0: normal forward on the embedded input -----
        out = original_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=depth_caches[0],
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden = _last_hidden(out, return_dict)

        deltas, norms, cosines = [], [], []

        # ----- depth 1..K-1: feed last hidden state as next embeds -----
        for k in range(1, K):
            previous = hidden

            out = original_forward(
                input_ids=None,
                inputs_embeds=hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=depth_caches[k],
                use_cache=use_cache,
                **kwargs,
            )

            hidden = _last_hidden(out, return_dict)

            # Diagnostics only; no gradient flows through the statistics.
            with torch.no_grad():
                h = hidden.detach().float()
                p = previous.detach().float()
                num = (h - p).norm(dim=-1).mean()
                den = p.norm(dim=-1).mean().clamp(min=1e-6)
                deltas.append(float((num / den).item()))

                norms.append(float(h.norm(dim=-1).mean().item()))

                cos = (h * p).sum(dim=-1) / (h.norm(dim=-1) * p.norm(dim=-1)).clamp(min=1e-6)
                cosines.append(float(cos.mean().item()))

        self._last_loop_stats = {
            "steps": K,
            "relative_deltas": deltas,
            "norms": norms,
            "cosines": cosines,
        }

        if use_cache and loop_cache is not None:
            out.past_key_values = loop_cache

        return out

    text_model.forward = MethodType(looped_forward, text_model)
    return model


def loop_stats(model) -> dict:
    """Diagnostics from the most recent looped forward (``None`` when not looped)."""
    try:
        text_model = get_qwen_text_model(model)
    except RuntimeError:
        return None
    stats = getattr(text_model, "_last_loop_stats", None)
    return stats or None
