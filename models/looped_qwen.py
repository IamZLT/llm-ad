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

First version deliberately forbids KV cache: the 2-D ``(position t, depth k)``
recurrence would need ``K_{l,k}(t), V_{l,k}(t)`` states (plus the linear-attention
conv state), which the vanilla cache cannot express. ``use_cache`` must therefore
be ``False`` in both training and generation until a recurrent-aware cache exists.
"""

from __future__ import annotations

from types import MethodType

import torch

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


def _last_hidden(out, return_dict: bool):
    if return_dict:
        return out.last_hidden_state
    return out[0]


def enable_looped_qwen(model, cfg: dict):
    """Patch ``language_model.forward`` to run the same stack ``steps`` times.

    No new Transformer block is created; all loop iterations share exactly the
    same parameters, including the same LoRA parameters. Calling this again only
    updates the step count (idempotent, no double patching).
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
        # v1 does NOT support KV cache: the full prefix is recomputed at every
        # generation step (semantically exact, no recurrent-aware cache yet).
        if past_key_values is not None:
            raise RuntimeError(
                "looped Qwen v1 requires past_key_values=None; "
                "generation must run with use_cache=False"
            )

        K = int(self._loop_steps)
        return_dict = bool(kwargs.get("return_dict", True))

        # ----- pass 1: normal forward on the embedded input -----
        out = original_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            **kwargs,
        )

        hidden = _last_hidden(out, return_dict)

        deltas, norms, cosines = [], [], []

        # ----- recurrent passes 2..K: feed last hidden state as next embeds -----
        for _ in range(1, K):
            previous = hidden

            out = original_forward(
                input_ids=None,
                inputs_embeds=hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
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
