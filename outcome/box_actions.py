"""Discrete geometric actions for the "observe -> propose -> apply -> refine" loop.

Phase 2 of the world-model plan. After reading the crop, the model (or, later,
the frozen action-outcome predictor) picks *one* action per candidate. Every
action maps deterministically to a concrete new box — there is no free-form
``refine`` that hides the geometry. Coordinates are in the 0-1000 normalized
space the Qwen-VL policy already emits (the same convention as ``B0``/``B1``).

Actions:

* ``keep``      — leave the selected candidate unchanged.
* ``reject``    — delete the selected candidate (a false alarm / true normal).
* ``expand`` / ``shrink`` — grow/shrink all four sides by one step.
* ``expand_{left,right,up,down}`` / ``shrink_{left,right,up,down}`` — one side.
* ``shift_{left,right,up,down}`` — translate the box by one step.

Version 1 deliberately excludes *adding* new candidates: that keeps the action
space finite and the "did observation help" attribution clean.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

# Coordinate space: boxes are [x1, y1, x2, y2] in 0-1000.
IMAGE_SPAN = 1000.0

KEEP = 'keep'
REJECT = 'reject'
EXPAND = 'expand'
SHRINK = 'shrink'

_EXPAND_DIR = ('expand_left', 'expand_right', 'expand_up', 'expand_down')
_SHRINK_DIR = ('shrink_left', 'shrink_right', 'shrink_up', 'shrink_down')
_SHIFT_DIR = ('shift_left', 'shift_right', 'shift_up', 'shift_down')

# Actions that modify the box geometry (exclude keep/reject).
GEOMETRIC_ACTIONS: Tuple[str, ...] = (EXPAND, SHRINK) + _EXPAND_DIR + _SHRINK_DIR + _SHIFT_DIR
# Terminal actions leave a boolean state (keep the box / drop the box).
TERMINAL_ACTIONS: Tuple[str, ...] = (KEEP, REJECT)
ALL_ACTIONS: Tuple[str, ...] = TERMINAL_ACTIONS + GEOMETRIC_ACTIONS

_ACTION_DESC = {
    KEEP: 'keep the candidate box as-is',
    REJECT: 'reject the candidate (false alarm, remove it)',
    EXPAND: 'expand the box on all four sides',
    SHRINK: 'shrink the box on all four sides',
    'expand_left': 'expand the box left edge',
    'expand_right': 'expand the box right edge',
    'expand_up': 'expand the box top edge',
    'expand_down': 'expand the box bottom edge',
    'shrink_left': 'shrink the box left edge',
    'shrink_right': 'shrink the box right edge',
    'shrink_up': 'shrink the box top edge',
    'shrink_down': 'shrink the box bottom edge',
    'shift_left': 'shift the box left',
    'shift_right': 'shift the box right',
    'shift_up': 'shift the box up',
    'shift_down': 'shift the box down',
}


def action_description(action: str) -> str:
    """Human-readable description of an action (for prompts / logging)."""
    return _ACTION_DESC.get(action, action)


def _step(box: Sequence[float], step_frac: float) -> Tuple[float, float]:
    """Per-action step size in x/y, relative to the box's own extent."""
    w = float(box[2]) - float(box[0])
    h = float(box[3]) - float(box[1])
    return step_frac * max(w, 1e-6), step_frac * max(h, 1e-6)


def _clamp_box(box: Sequence[float], span: float = IMAGE_SPAN,
               min_span: float = 1.0) -> List[float]:
    """Clamp ``[x1,y1,x2,y2]`` to ``[0, span]``, guaranteeing a positive extent."""
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = min(max(x1, 0.0), span)
    x2 = min(max(x2, 0.0), span)
    y1 = min(max(y1, 0.0), span)
    y2 = min(max(y2, 0.0), span)
    if x2 - x1 < min_span:
        x2 = min(x1 + min_span, span)
        x1 = max(x2 - min_span, 0.0)
    if y2 - y1 < min_span:
        y2 = min(y1 + min_span, span)
        y1 = max(y2 - min_span, 0.0)
    return [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)]


def apply_geometric(box: Sequence[float], action: str,
                    step_frac: float = 0.1, span: float = IMAGE_SPAN) -> List[float]:
    """Return the new box after a *geometric* action (keep/reject not allowed).

    Shrink steps never invert the box: a degenerate shrink collapses to the
    minimum 1-unit span instead of producing ``x1 >= x2``.
    """
    dx, dy = _step(box, step_frac)
    x1, y1, x2, y2 = [float(v) for v in box]
    if action == EXPAND:
        n = [x1 - dx, y1 - dy, x2 + dx, y2 + dy]
    elif action == SHRINK:
        n = [x1 + dx, y1 + dy, x2 - dx, y2 - dy]
    elif action == 'expand_left':
        n = [x1 - dx, y1, x2, y2]
    elif action == 'expand_right':
        n = [x1, y1, x2 + dx, y2]
    elif action == 'expand_up':
        n = [x1, y1 - dy, x2, y2]
    elif action == 'expand_down':
        n = [x1, y1, x2, y2 + dy]
    elif action == 'shrink_left':
        n = [x1 + dx, y1, x2, y2]
    elif action == 'shrink_right':
        n = [x1, y1, x2 - dx, y2]
    elif action == 'shrink_up':
        n = [x1, y1 + dy, x2, y2]
    elif action == 'shrink_down':
        n = [x1, y1, x2, y2 - dy]
    elif action == 'shift_left':
        n = [x1 - dx, y1, x2 - dx, y2]
    elif action == 'shift_right':
        n = [x1 + dx, y1, x2 + dx, y2]
    elif action == 'shift_up':
        n = [x1, y1 - dy, x2, y2 - dy]
    elif action == 'shift_down':
        n = [x1, y1 + dy, x2, y2 + dy]
    else:
        raise ValueError(f'unknown geometric action: {action!r}')
    return _clamp_box(n, span)


def propose_actions(box: Optional[Sequence[float]], image_size: float = IMAGE_SPAN,
                    step_frac: float = 0.1) -> List[str]:
    """The full finite action candidate set for one candidate box.

    Returns ``ALL_ACTIONS`` in canonical order so the action-outcome predictor's
    score vector has a stable indexing. ``box`` is only used to validate that a
    candidate exists; when ``None`` no geometric action is meaningful.
    """
    if box is None:
        return [REJECT]
    return list(ALL_ACTIONS)


def validate_action(action: str, boxes: Sequence[Sequence[float]],
                    selected_index: Optional[int] = None) -> bool:
    """Whether ``action`` is a legal choice given the current box set."""
    if action not in ALL_ACTIONS:
        return False
    n = len(boxes)
    if action == REJECT:
        return n > 0
    # keep / geometric actions require a selected candidate.
    idx = selected_index if selected_index is not None else (0 if n > 0 else -1)
    return 0 <= idx < n


def apply_action(boxes: Sequence[Sequence[float]], selected_index: int, action: str,
                 image_size: float = IMAGE_SPAN, step_frac: float = 0.1) -> Tuple[List[List[float]], bool]:
    """Apply ``action`` to ``boxes[selected_index]``.

    Returns ``(new_boxes, removed)`` where ``removed`` is True iff the action was
    ``reject`` (the selected candidate was deleted; the remaining boxes keep their
    order). ``keep`` / geometric actions replace the selected box and return
    ``removed=False``. When ``reject`` empties the last candidate, the caller must
    make the final normal/anomalous decision consistent with an empty box set.
    """
    boxes = [list(b) for b in boxes]
    n = len(boxes)
    if not 0 <= selected_index < n:
        raise IndexError(f'selected_index {selected_index} out of range for {n} boxes')
    if action == REJECT:
        boxes.pop(selected_index)
        return boxes, True
    if action == KEEP:
        return boxes, False
    boxes[selected_index] = apply_geometric(boxes[selected_index], action, step_frac, image_size)
    return boxes, False
