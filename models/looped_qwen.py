"""Switchable latent-space recurrent-depth mode for the Qwen3.5 language stack.

External-FSM (token-space)::

    X -> U -> C -> L -> V -> A

recurrence happens OUTSIDE the model by emitting stage tokens and re-prefilling.

Internal-Loop (latent-space)::

    H^(0) = E(X)
    H^(k+1) = F_theta(H^(k)),   k = 0, ..., K-1
    H_out   = Norm(H^(K))

where ``F_theta = L_N o ... o L_1`` contains ONLY the decoder layers (no final
RMSNorm between recurrent steps) and the final ``Norm`` is applied exactly once
after the K-th pass. All K passes share one set of parameters (including LoRA):
no new Transformer block is created and no ``ModuleList`` is added.

This is the "recurrent-depth Transformer" definition (weight-shared depth
recursion), as opposed to the naive ``(Norm o F)^K`` which the original
implementation produced by re-calling ``Qwen3_5TextModel.forward`` (that method
runs ``self.norm`` after every layer stack, so the loop also re-normalised the
hidden state at every depth).

Recurrent-depth cache::

    C^(1), C^(2), ..., C^(K)

Each recurrent depth owns an independent ``DynamicCache``. For the t-th token
``x_t -> F_theta(C^(1)) -> h_t^(1) -> F_theta(C^(2)) -> h_t^(2) -> ...`` the
k-th pass only ever processes ``[B, 1, D]`` against ``C^(k)`` (which already
holds the full history of depth k), instead of re-running the whole prefix.
Position ``t`` does not change across depths, so the M-RoPE / ``position_ids``
are identical for all K passes and are computed once. ``LoopDepthCache`` presents
these K caches to ``generate()`` as one ``Cache`` while routing each depth pass to
its own inner cache.
"""

from __future__ import annotations

from types import MethodType

import torch
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast

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


def set_loop_steps(model, steps: int) -> int:
    """Runtime change of the recurrent depth K (curriculum / diagnostics).

    Must be called after ``enable_looped_qwen`` (which installs the patch).
    Returns the effective (clamped) step count.
    """
    tm = get_qwen_text_model(model)
    if not hasattr(tm, "_loop_original_forward"):
        raise RuntimeError("looped Qwen is not enabled; call enable_looped_qwen first")
    tm._loop_steps = max(1, int(steps))
    return tm._loop_steps


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


def enable_looped_qwen(model, cfg: dict):
    """Patch ``language_model.forward`` to run the decoder layers ``steps`` times.

    ``F_theta = L_N o ... o L_1`` (decoder layers only); the final RMSNorm is
    applied ONCE after the K-th pass: ``H_out = Norm(H^(K))``. All K passes share
    one set of parameters (including LoRA). Re-entry only updates the step count
    (idempotent, no double patching). ``steps=1`` still installs the patch so the
    K=1 forward is bit-comparable to the original stack (a required unit test).

    With ``use_cache=True`` (generation) each depth pass is routed to its own
    inner ``DynamicCache`` inside a ``LoopDepthCache``; with ``use_cache=False``
    (teacher-forcing) the full sequence is recomputed per depth with no cache.
    """
    if not loop_enabled(cfg):
        return model

    steps = loop_steps(cfg)
    text_model = get_qwen_text_model(model)

    # Avoid double patching: capture the ORIGINAL forward only once.
    if hasattr(text_model, "_loop_original_forward"):
        text_model._loop_steps = steps
        return model

    original_forward = text_model.forward

    text_model._loop_original_forward = original_forward
    text_model._loop_steps = steps
    text_model._last_loop_stats = {}
    text_model._last_loop_states = None
    text_model._capture_loop_states = False

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

        # Exactly one of input_ids / inputs_embeds.
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Mirror merge_with_config_defaults: default use_cache from config, and
        # force it off under gradient checkpointing in training mode.
        if use_cache is None:
            use_cache = bool(getattr(self.config, "use_cache", False))
        if getattr(self, "gradient_checkpointing", False) and self.training and use_cache:
            use_cache = False

        # ---- 1. K independent caches ----
        if isinstance(past_key_values, LoopDepthCache) and past_key_values._depth_count == K:
            loop_cache = past_key_values
            depth_caches = past_key_values.depth_caches
        elif use_cache:
            loop_cache = LoopDepthCache(self.config, K)
            depth_caches = loop_cache.depth_caches
        else:
            loop_cache = None
            depth_caches = [None] * K

        # ---- 2. position ids (M-RoPE). Token-position, NOT depth-position, so a
        # single set is shared by all K passes. Depth-0 cache length == every
        # depth's length. ----
        if position_ids is None:
            past_seen_tokens = depth_caches[0].get_seq_length() if depth_caches[0] is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            rope_position_ids = position_ids[1:]
        else:
            text_position_ids = None
            rope_position_ids = position_ids

        # ---- 3. build masks once (all depths share the same sequence length) ----
        if not isinstance(attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": depth_caches[0],
                "position_ids": text_position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
            }
        else:
            causal_mask_mapping = attention_mask

        # ---- 4. RoPE is token-position dependent, shared across depths ----
        position_embeddings = self.rotary_emb(inputs_embeds, rope_position_ids)

        # ---- 5. TRUE recurrent depth: same layers, same weights, NO final norm
        # between recurrent steps. ----
        hidden = inputs_embeds
        capture = bool(getattr(self, "_capture_loop_states", False))
        states = [] if capture else None
        deltas, norms, cosines = [], [], []

        for depth in range(K):
            prev_hidden = hidden
            depth_cache = depth_caches[depth]

            for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
                hidden = decoder_layer(
                    hidden,
                    position_embeddings=position_embeddings,
                    attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                    position_ids=text_position_ids,
                    past_key_values=depth_cache,
                    use_cache=use_cache,
                    **kwargs,
                )

            if capture:
                states.append(hidden)

            # Diagnostics only; no gradient flows through the statistics.
            if depth > 0:
                with torch.no_grad():
                    h = hidden.detach().float()
                    p = prev_hidden.detach().float()
                    num = (h - p).norm(dim=-1).mean()
                    den = p.norm(dim=-1).mean().clamp(min=1e-6)
                    deltas.append(float((num / den).item()))
                    cos = torch.nn.functional.cosine_similarity(h, p, dim=-1).mean()
                    cosines.append(float(cos.item()))
                    norms.append(float(h.norm(dim=-1).mean().item()))

        # ---- 6. final RMSNorm ONCE ----
        hidden = self.norm(hidden)

        self._last_loop_stats = {
            "steps": K,
            "relative_deltas": deltas,
            "norms": norms,
            "cosines": cosines,
        }
        if capture:
            self._last_loop_states = states
        else:
            self._last_loop_states = None

        return BaseModelOutputWithPast(
            last_hidden_state=hidden,
            past_key_values=loop_cache if use_cache else None,
        )

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
