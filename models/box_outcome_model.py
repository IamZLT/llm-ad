"""Action-outcome predictor: ``Q_hat(a) = f_phi(s_crop, B0, a)``.

Phase 2 of the world-model plan. After observing the crop, the detector proposes
a small finite set of geometric actions (``outcome/box_actions.py``); this module
predicts the detection quality *after* applying each action, and a controller
picks ``argmax`` (with ``keep`` always available). It is trained by supervised
regression against the realized quality label of each action's resulting box set,
computed from GT (``action_outcome_quality``) — it is **not** trained by RL and its
output is never used as a correctness reward directly (that would let the policy
exploit its error).

This is the "box-refinement action result model" that anchors the broader
world-model claim: whether it generalizes to a world model is judged by whether
predicting action outcomes beats directly fixing the box, not by this module's
existence alone.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from torch import nn

from outcome.box_actions import ALL_ACTIONS, apply_action
from outcome.protocol import to_pixels
from outcome.protocol_multibox import set_localization_reward

DEFAULT_FP_COST_PER_BOX = -0.5
DEFAULT_CORRECT_REJECT = 1.0
DEFAULT_MISS = 0.0


def action_outcome_quality(new_boxes: Sequence[Sequence[float]], meta: dict,
                           iou_threshold: float = 0.30, geometry_weight: float = 0.30,
                           fp_cost_per_box: float = DEFAULT_FP_COST_PER_BOX,
                           correct_reject: float = DEFAULT_CORRECT_REJECT,
                           miss: float = DEFAULT_MISS) -> float:
    """GT quality of a box *set* (0-1000 coords) against ``meta``.

    This is the supervised label for ``BoxOutcomeModel``. It is the analogue of
    ``score_output``'s ``q1``, plus an explicit false-positive cost so that normal
    images and ``reject`` are supervised correctly (a quality function that only
    measured anomaly IoU would give ``reject`` on a normal image a meaningless 0):

    * anomaly + non-empty boxes -> Hungarian set-localization reward (DCLR term);
    * anomaly + empty boxes     -> ``miss`` (0 by default);
    * normal  + empty boxes     -> ``correct_reject`` (1 by default);
    * normal  + K boxes         -> ``fp_cost_per_box * K`` (clamped to <= 0).
    """
    orig = meta['orig_size']
    if bool(meta['is_anomaly']):
        comps = list(meta.get('component_bboxes') or [])
        if not comps and meta.get('gt_box_px') is not None:
            comps = [meta['gt_box_px']]
        if not new_boxes or not comps:
            return float(miss)
        px = [to_pixels(b, orig) for b in new_boxes]
        return float(set_localization_reward(px, comps, orig, iou_threshold, geometry_weight)['reward'])
    # Normal image: quality is dominated by false-positive cost.
    if not new_boxes:
        return float(correct_reject)
    return float(min(0.0, fp_cost_per_box * len(new_boxes)))


class BoxOutcomeModel(nn.Module):
    """Small MLP scoring ``A`` actions per (global, crop, candidate) context.

    Inputs (per sample, batch ``B``, ``A = len(ALL_ACTIONS)`` actions):

    * ``global_features``  [B, Dg] — pooled ref+test representation;
    * ``crop_features``    [B, Dc] — pooled crop representation (or None -> learned
      "no crop" embedding, so degenerate crops are still scored);
    * ``candidate_box``    [B, 4]  — the selected candidate (0-1000);
    * ``action_ids``       [B, A]  — long indices into ``ALL_ACTIONS``.

    Output: [B, A] predicted qualities. The controller takes ``argmax`` (``keep``
    is included so "do nothing" is always a candidate).
    """

    def __init__(self, global_dim: int, crop_dim: int, num_actions: Optional[int] = None,
                 box_dim: int = 64, action_dim: int = 32, hidden: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        num_actions = num_actions or len(ALL_ACTIONS)
        self.action_emb = nn.Embedding(num_actions, action_dim)
        self.box_proj = nn.Sequential(nn.Linear(4, box_dim), nn.ReLU())
        self.global_proj = nn.Linear(global_dim, hidden)
        self.crop_proj = nn.Linear(crop_dim, hidden)
        self.no_crop_emb = nn.Parameter(torch.zeros(crop_dim))
        self.ctx_proj = nn.Sequential(nn.Linear(hidden + hidden + box_dim, hidden), nn.ReLU())
        self.act_proj = nn.Linear(action_dim, hidden)
        self.fuse = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, global_features: torch.Tensor, crop_features: Optional[torch.Tensor],
                candidate_box: torch.Tensor, action_ids: torch.Tensor) -> torch.Tensor:
        B = int(global_features.shape[0])
        g = self.global_proj(global_features)                                   # [B, hidden]
        if crop_features is None:
            c = self.no_crop_emb.expand(B, -1)
        else:
            c = crop_features
        c = self.crop_proj(c)                                                   # [B, hidden]
        b = self.box_proj(candidate_box)                                        # [B, box_dim]
        ctx = self.ctx_proj(torch.cat([g, c, b], dim=-1))                       # [B, hidden]
        act = self.act_proj(self.action_emb(action_ids))                        # [B, A, hidden]
        h = ctx.unsqueeze(1) + act                                              # [B, A, hidden]
        h = self.fuse(h)
        return self.head(h).squeeze(-1)                                         # [B, A]

    def select_action(self, global_features: torch.Tensor, crop_features: Optional[torch.Tensor],
                      candidate_box: torch.Tensor, action_ids: torch.Tensor) -> torch.Tensor:
        """Argmax action index per sample (controller decision)."""
        return self.forward(global_features, crop_features, candidate_box, action_ids).argmax(dim=-1)


def default_action_ids(batch_size: int, device) -> torch.Tensor:
    """``[B, A]`` long tensor enumerating ``ALL_ACTIONS`` for every sample."""
    ids = torch.arange(len(ALL_ACTIONS), device=device)
    return ids.unsqueeze(0).expand(batch_size, -1)


def apply_selected_action(boxes: Sequence[Sequence[float]], selected_index: int,
                          action_index: int) -> List[List[float]]:
    """Apply the action at ``action_index`` in ``ALL_ACTIONS`` (controller glue)."""
    action = ALL_ACTIONS[int(action_index)]
    new_boxes, _removed = apply_action(boxes, selected_index, action)
    return new_boxes


def save_box_outcome(model: BoxOutcomeModel, path) -> None:
    """Persist the predictor (weights + geometry needed to rebuild it)."""
    import os
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(
        state_dict=model.state_dict(),
        global_dim=int(model.global_proj.in_features),
        crop_dim=int(model.crop_proj.in_features),
        num_actions=int(model.action_emb.num_embeddings),
    ), path)


def load_box_outcome(path, global_dim: Optional[int] = None, crop_dim: Optional[int] = None,
                     num_actions: Optional[int] = None, device='cpu') -> BoxOutcomeModel:
    """Rebuild the predictor from a checkpoint (dims taken from the file if omitted)."""
    ckpt = torch.load(path, map_location=device)
    model = BoxOutcomeModel(
        global_dim=global_dim or int(ckpt['global_dim']),
        crop_dim=crop_dim or int(ckpt['crop_dim']),
        num_actions=num_actions or int(ckpt['num_actions']),
    )
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    return model
