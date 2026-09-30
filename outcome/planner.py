"""Observation-action policy. This module never reads ground truth.

``world_model`` selects the legal action with the highest predicted gain minus
cost. ``rule`` reproduces the old box-count heuristic. ``random`` samples a
legal action. An invalid world-model plan stops; it does not fall back to the
rule planner.
"""
from __future__ import annotations

import random

from outcome.world_model import ObservationAction


def build_observation_actions(boxes, cfg):
    pcfg = (cfg.get("outcome") or {}).get("planner") or {}
    max_zoom = int(pcfg.get("max_zoom_candidates", 3))
    actions = []
    for i in range(min(len(boxes), max_zoom)):
        actions.append(ObservationAction(
            name=f"zoom_box_{i}",
            candidate_index=i,
            cost=float(pcfg.get("zoom_cost", 1.0)),
        ))
    actions.append(ObservationAction(
        name="global_scan",
        candidate_index=None,
        cost=float(pcfg.get("global_scan_cost", 2.0)),
    ))
    actions.append(ObservationAction(
        name="stop",
        candidate_index=None,
        cost=0.0,
    ))
    return actions


def select_world_model_action(plan, actions, cfg):
    pcfg = (cfg.get("outcome") or {}).get("planner") or {}
    if not plan.valid:
        return "stop", {"score": 0.0, "fallback": "invalid_world_model_plan"}
    lambda_cost = float(pcfg.get("cost_weight", 0.05))
    best_action = None
    best_score = -float("inf")
    for action in actions:
        pred = plan.predictions[action.name]
        score = pred.expected_gain - lambda_cost * action.cost
        if score > best_score:
            best_score = score
            best_action = action.name
    return best_action, {"score": best_score, "fallback": None}


def select_random_action(actions, rng=None):
    rng = rng or random
    return rng.choice(actions).name


def select_rule_action(boxes, cfg):
    """Old ``plan_observations`` decision, kept only as the rule baseline."""
    zcfg = (cfg.get("outcome") or {}).get("zoom") or {}
    if not boxes or len(boxes) > 1 or bool(zcfg.get("coverage_scan", False)):
        return "global_scan"
    return "zoom_box_0"


def select_action(plan, boxes, actions, cfg, rng=None):
    mode = str(((cfg.get("outcome") or {}).get("planner") or {}).get("mode", "world_model"))
    if mode == "world_model":
        return select_world_model_action(plan, actions, cfg)
    if mode == "rule":
        return select_rule_action(boxes, cfg), {"score": None, "fallback": None}
    if mode == "random":
        return select_random_action(actions, rng), {"score": None, "fallback": None}
    raise ValueError(f"unknown planner mode: {mode}")
