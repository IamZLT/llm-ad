"""World-model planner: predicted gain selects the executed observation."""
from PIL import Image

from outcome.observation import execute_observation_action
from outcome.planner import build_observation_actions, select_world_model_action
from outcome.planner_supervision import (
    build_action_supervision, sample_belief_boxes, summarize_supervision,
)
from outcome.state_quality import state_quality
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


def test_normal_false_alarm_has_positive_zoom_gain():
    boxes = [[100, 100, 200, 200]]
    rows = build_action_supervision(boxes, [], (1000, 1000), CFG)
    zoom = next(row for row in rows if row.action == "zoom_box_0")
    assert zoom.evidence == "false_alarm"
    assert abs(zoom.target_gain - 0.25) < 1e-6
    assert zoom.target_boxes == []
    assert state_quality(boxes, [], (1000, 1000), CFG) == 0.75
    assert state_quality([], [], (1000, 1000), CFG) == 1.0


def test_each_action_owns_its_next_boxes():
    gt = [[100, 100, 400, 400]]
    b0 = [[100, 100, 280, 280]]
    rows = build_action_supervision(b0, gt, (1000, 1000), CFG)
    by_name = {row.action: row for row in rows}
    assert by_name["stop"].target_boxes == b0
    assert abs(by_name["stop"].target_gain) < 1e-8
    assert by_name["zoom_box_0"].evidence == "boundary_undershoot"
    assert by_name["zoom_box_0"].target_gain > 0
    assert by_name["zoom_box_0"].target_boxes == [gt[0]]
    missed = [[700, 700, 900, 900]]
    partial = [gt[0]]
    scan_rows = build_action_supervision(partial, gt + missed, (1000, 1000), CFG)
    scan = next(row for row in scan_rows if row.action == "global_scan")
    assert scan.evidence == "missing_region"
    assert scan.target_gain > 0
    assert len(scan.target_boxes) == 2


def test_planner_targets_are_not_mostly_zero():
    records = []
    gt = [[120, 80, 360, 300], [700, 600, 900, 880]]
    for i in range(400):
        _mode, boxes = sample_belief_boxes(gt, is_anomaly=True)
        records.append(build_action_supervision(boxes, gt, (1000, 1000), CFG))
        if i % 5 == 0:
            _mode, normal_boxes = sample_belief_boxes([], is_anomaly=False)
            records.append(build_action_supervision(normal_boxes, [], (1000, 1000), CFG))
    summary = summarize_supervision(records, CFG)
    assert summary["all_zero_rate"] < 0.45
    assert summary["zoom_positive_rate"] > 0.25
    assert summary["scan_positive_rate"] > 0.15
    assert summary["best_action_zoom_rate"] > 0.05
    assert summary["best_action_scan_rate"] > 0.05
    assert summary["best_action_stop_rate"] > 0.05
    assert summary["gain_std"] > 0.02
