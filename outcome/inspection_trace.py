"""InspectionTrace: structural bookkeeping shared by SFT, RL, and eval.

The two-stage world-model flow is:

    full image + reference
        -> B0  (stage 1: [understand][compare][localize][imagine], pre-obs prediction)
        -> crop (padded window around the selected candidate, outline drawn)
        -> B1  (stage 2: [confirm] + <answer>, conditioned on ref + test + crop)

``InspectionTrace`` records *only* the observation bookkeeping (which candidate
was looked at, whether the crop ran and why, what the model predicted before and
decided after). GT, candidate-quality labels, and rewards live in the
training-supervision fields and are **never** passed into the policy prompt.

This object is the single source of truth for evaluation recording and is reused
by the (phase-3) two-stage GRPO optimizer, so ``B0`` and ``B1`` are kept strictly
separate — a final box must never overwrite the initial box, otherwise "did the
observation actually help" becomes unmeasurable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

# Zoom skip reasons. A skipped crop keeps the already-generated stage-1 prefix and
# runs a no-crop stage-2, instead of re-rolling a fresh single-pass trajectory (so
# the same candidate is still evaluated, only without the local observation).
SKIP_NO_CANDIDATE = 'no_candidate'
SKIP_INVALID_BOX = 'invalid_box'
SKIP_OVERSIZED_WINDOW = 'oversized_window'
SKIP_NO_ORIG_SIZE = 'no_orig_size'
SKIP_ZOOM_DISABLED = 'zoom_disabled'


@dataclass
class RolloutSegment:
    """One generation stage's conditioning and generated output.

    The stage-1 prefix is replayed as the stage-2 prompt, so each segment must
    carry its *own* full conditioning (input ids, vision cache, H injection ids,
    and the multimodal pixel/grid tensors needed to replay the forward for
    logprob accounting) — a group of trajectories cannot share a single batch when
    their stage-2 crops differ.
    """
    input_ids: Any = None            # torch.Tensor [1, L] (prompt + any prefill)
    prompt_len: int = 0              # index where newly-sampled tokens begin
    attention_mask: Any = None       # torch.Tensor [1, L]
    image_embeds: Any = None         # torch.Tensor vision cache for this stage
    # Multimodal inputs for logprob replay (mirrors ``model_inputs``).
    pixel_values: Any = None
    image_grid_thw: Any = None
    mm_token_type_ids: Any = None
    # Per-stage H injection (stage 1 only; stage 2 is H-free).
    box_token_id: int = -1
    h_box_geom: Any = None
    control_token_id: int = -1
    feat_token_id: int = -1
    h_map: Any = None
    # Generated output (Completion: ids/text/stop_reason/sampled_mask).
    completion: Any = None
    # Optional per-token bool mask marking which positions were actually sampled
    # (vs. replayed prefill / controller-inserted text), for logprob accounting.
    sampled_mask: Any = None

    def as_batch(self) -> dict:
        """Reconstruct a forward-ready dict (the subset ``model_inputs``/``_h_*`` need).

        ``image_embeds`` is bound separately via ``bind_cached_image_features``; the
        H fields are consumed by ``_h_box_ctx`` / ``_h_vpt_ctx``.
        """
        return dict(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            image_embeds=self.image_embeds,
            pixel_values=self.pixel_values,
            image_grid_thw=self.image_grid_thw,
            mm_token_type_ids=self.mm_token_type_ids,
            box_token_id=self.box_token_id,
            h_box_geom=self.h_box_geom,
            control_token_id=self.control_token_id,
            feat_token_id=self.feat_token_id,
            h_map=self.h_map,
        )


@dataclass
class InspectionTrace:
    """One inspection's full two-stage trajectory (B0 -> observe -> B1)."""
    initial_boxes: List[List[float]] = field(default_factory=list)   # B0 (0-1000)
    selected_box_index: int = -1                                      # candidate zoomed (-1 = none)
    pre_observation_prediction: str = ''                              # [imagine] body (pre-crop)

    crop_window_px: Optional[Tuple[int, int, int, int]] = None        # (x1,y1,x2,y2) orig px
    zoom_executed: bool = False
    zoom_skip_reason: Optional[str] = None                            # one of SKIP_* / None

    candidate_parse_state: Optional[str] = None                       # parse_boxes_list state
    observation_executed: bool = False                                # any stage-2 observation ran
    observations: List[dict] = field(default_factory=list)            # [{kind,window_px,candidate_index}]

    action_candidates: List[str] = field(default_factory=list)        # proposed actions (phase 2)
    selected_action: Optional[str] = None                             # keep/expand/.../reject
    predicted_effect: Optional[str] = None                            # improved/unchanged/degraded
    final_boxes: List[List[float]] = field(default_factory=list)      # B1 (0-1000)

    segments: List[RolloutSegment] = field(default_factory=list)      # stage 1, then stage 2
    stage1_early_end: Optional[str] = None                            # </think>/<answer>/eos/length

    def to_record(self) -> dict:
        """JSON-serializable snapshot for evaluation rows."""
        return dict(
            initial_boxes=self.initial_boxes,
            selected_box_index=self.selected_box_index,
            pre_observation_prediction=self.pre_observation_prediction,
            crop_window_px=list(self.crop_window_px) if self.crop_window_px else None,
            zoom_executed=self.zoom_executed,
            zoom_skip_reason=self.zoom_skip_reason,
            candidate_parse_state=self.candidate_parse_state,
            observation_executed=self.observation_executed,
            observations=self.observations,
            selected_action=self.selected_action,
            predicted_effect=self.predicted_effect,
            final_boxes=self.final_boxes,
            stage1_early_end=self.stage1_early_end,
        )
