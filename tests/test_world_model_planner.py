"""World-model planner: predicted gain selects the executed observation."""
from PIL import Image

from outcome.observation import execute_observation_action
from outcome.planner import build_observation_actions, select_world_model_action
from outcome.world_model import parse_imagine_plan


CFG = {
    "outcome": {
        "planner": {
            "cost_weight": 0.05,
            "zoom_cost": 1.0,
            "global_scan_cost": 2.0,
        },
        "zoom": {"expand": 0.3, "min_pad_frac": 0.04, "max_area_frac": 0.6},
    }
}


def _plan(gains, actions):
    text = "\n".join(
        f"action={name}; evidence=uncertain; gain={gain}"
        for name, gain in gains.items()
    )
    return parse_imagine_plan(text, actions)


def test_swapping_imagine_gains_changes_the_selected_action():
    boxes = [[100, 100, 200, 200]]
    actions = build_observation_actions(boxes, CFG)
    first, _ = select_world_model_action(
        _plan({"zoom_box_0": 0.30, "global_scan": 0.10, "stop": 0.00}, actions),
        actions, CFG)
    swapped, _ = select_world_model_action(
        _plan({"zoom_box_0": 0.10, "global_scan": 0.30, "stop": 0.00}, actions),
        actions, CFG)
    assert first == "zoom_box_0"
    assert swapped == "global_scan"


def test_invalid_plan_stops_instead_of_using_the_rule():
    boxes = [[100, 100, 200, 200]]
    actions = build_observation_actions(boxes, CFG)
    plan = _plan({"zoom_box_0": 0.90}, actions)
    action, meta = select_world_model_action(plan, actions, CFG)
    assert action == "stop"
    assert meta["fallback"] == "invalid_world_model_plan"
    assert plan.valid is False


def test_degenerate_zoom_does_not_become_a_global_scan():
    image = Image.new("RGB", (200, 200), "white")
    boxes = [[0, 0, 1000, 1000]]
    cfg = {
        "outcome": {
            "zoom": {"expand": 0.0, "min_pad_frac": 0.0, "max_area_frac": 0.1},
        }
    }
    execution = execute_observation_action(
        image, boxes, image.size, "zoom_box_0", cfg)
    assert execution.executed is False
    assert execution.skip_reason == "degenerate_zoom"
    assert execution.observations == []
