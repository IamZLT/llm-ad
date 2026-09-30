"""Action-conditioned world model: parse ``[imagine]`` predictions.

The imagine stage must score every legal observation action. A plan that names
only the preferred action is invalid: that is a policy output, not
:math:`W(s, a) -> (evidence, gain)`.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


EVIDENCE_TYPES = {
    "supported",
    "undershoot",
    "overshoot",
    "shifted",
    "false_alarm",
    "missing_region",
    "complete_coverage",
    "no_new_evidence",
    "uncertain",
}


@dataclass(frozen=True)
class ObservationAction:
    name: str
    candidate_index: Optional[int]
    cost: float


@dataclass
class ActionPrediction:
    action: str
    evidence: str
    expected_gain: float


@dataclass
class WorldModelPlan:
    predictions: Dict[str, ActionPrediction] = field(default_factory=dict)
    valid: bool = False
    error: Optional[str] = None


PLAN_LINE = re.compile(
    r"action\s*=\s*([a-zA-Z0-9_]+)\s*;"
    r"\s*evidence\s*=\s*([a-zA-Z0-9_]+)\s*;"
    r"\s*gain\s*=\s*(-?\d+(?:\.\d+)?)",
    re.I,
)


def parse_imagine_plan(text: str, legal_actions: List[ObservationAction]) -> WorldModelPlan:
    """Parse one prediction line per legal action. Duplicates or gaps are invalid."""
    legal = {a.name for a in legal_actions}
    predictions = {}

    for match in PLAN_LINE.finditer(text or ""):
        action = match.group(1).lower()
        evidence = match.group(2).lower()
        try:
            gain = float(match.group(3))
        except ValueError:
            continue
        if action not in legal or not math.isfinite(gain):
            continue
        if evidence not in EVIDENCE_TYPES:
            evidence = "uncertain"
        gain = max(-1.0, min(1.0, gain))
        if action in predictions:
            return WorldModelPlan(
                predictions=predictions,
                valid=False,
                error=f"duplicate_action:{action}",
            )
        predictions[action] = ActionPrediction(
            action=action,
            evidence=evidence,
            expected_gain=gain,
        )

    missing = legal - set(predictions)
    if missing:
        return WorldModelPlan(
            predictions=predictions,
            valid=False,
            error=f"missing_actions:{sorted(missing)}",
        )
    return WorldModelPlan(predictions=predictions, valid=True)
