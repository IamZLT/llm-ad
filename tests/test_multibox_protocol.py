"""Tests for outcome-multibox-v1 protocol + set-level reward."""
import pytest

from outcome.protocol_multibox import (
    count_reward,
    focus_reward,
    parse_boxes_list,
    parse_output,
    parse_verify,
    score_output,
    set_localization_reward,
)

ORIG = (1000, 1000)


def _anomaly_meta(comps=[[100, 100, 200, 200], [500, 500, 600, 600]]):
    return dict(orig_size=ORIG, is_anomaly=True,
                gt_box_px=[100, 100, 600, 600], component_bboxes=comps,
                num_components=len(comps), image_path='x.png')


def _normal_meta():
    return dict(orig_size=ORIG, is_anomaly=False, gt_box_px=None,
                component_bboxes=[], num_components=0, image_path='x.png')


def _output(ground, verify, answer):
    return (f"<understand>\nstructure\n</understand>\n"
            f"<compare>\ndifference\n</compare>\n"
            f"<ground>\n{ground}\n</ground>\n"
            f"<verify>\n{verify}\n</verify>\n"
            f"<answer>\n{answer}\n</answer>")


def test_parse_boxes_list_states():
    assert parse_boxes_list('null') == ('null', [])
    assert parse_boxes_list('[]') == ('empty', [])
    state, boxes = parse_boxes_list('[[100,100,200,200],[300,300,400,400]]')
    assert state == 'list' and boxes == [[100, 100, 200, 200], [300, 300, 400, 400]]
    assert parse_boxes_list('[[100,100,200]]')[0] == 'invalid'


def test_parse_verify_action():
    assert parse_verify('keep; both look anomalous')[0] == 'keep'
    assert parse_verify('reject; looks like noise')[0] == 'reject'
    assert parse_verify('garbage text')[0] is None


def test_parse_output_multibox_anomaly():
    text = _output(
        'candidate_bboxes_2d=[[100,100,200,200],[300,300,400,400]]',
        'keep; both look anomalous',
        '{"is_anomaly":true,"bboxes_2d":[[100,100,200,200],[300,300,400,400]],"description":"two defects"}',
    )
    p = parse_output(text)
    assert p['task_valid'] and p['is_anomaly'] is True
    assert p['bboxes_2d'] == [[100, 100, 200, 200], [300, 300, 400, 400]]
    assert p['candidate_state'] == 'list'
    assert p['verify_action'] == 'keep'
    assert p['protocol_core'] and p['protocol_strict']


def test_parse_output_normal_can_have_nonempty_hypothesis():
    text = _output(
        'candidate_bboxes_2d=[[100,100,200,200]]',
        'reject; looks like noise',
        '{"is_anomaly":false,"bboxes_2d":[],"description":"normal"}',
    )
    p = parse_output(text)
    assert p['task_valid'] and p['is_anomaly'] is False
    assert p['bboxes_2d'] == []
    assert p['candidate_bboxes_2d'] == [[100, 100, 200, 200]]
    assert p['protocol_core']


def test_parse_output_answer_only_is_strict_json():
    # verify with an action but no semicolon passes core, fails strict
    text = _output(
        'candidate_bboxes_2d=[[100,100,200,200]]',
        'keep evidence without semicolon',
        '{"is_anomaly":true,"bboxes_2d":[[100,100,200,200]],"description":"d"}',
    )
    p = parse_output(text)
    assert p['protocol_core'] is True
    assert p['protocol_strict'] is False


def test_set_reward_missed_defect_penalized_by_denominator():
    r = set_localization_reward([[100, 100, 200, 200]], [[100, 100, 200, 200], [500, 500, 600, 600]], ORIG)
    assert r['reward'] == pytest.approx(0.5)  # 1 found of 2 -> 1/2


def test_set_reward_duplicate_box_penalized_by_denominator():
    pred = [[100, 100, 200, 200], [100, 100, 200, 200], [100, 100, 200, 200]]
    r = set_localization_reward(pred, [[100, 100, 200, 200]], ORIG)
    assert r['reward'] == pytest.approx(1 / 3)


def test_score_output_correct_anomaly_gets_mask_reward():
    text = _output('candidate_bboxes_2d=[]', 'keep; evidence',
                   '{"is_anomaly":true,"bboxes_2d":[[100,100,200,200]],"description":"d"}')
    p = parse_output(text)
    s = score_output(p, _anomaly_meta(), protocol_weight=0.01)
    assert s['correct'] is True
    # task = 0.5*mask_iou(0.5) + 0.2*count(0.5) + 0.3*loc(0.5) + 0.2*cand_cov(0)
    #        + 0.1*clip(delta=0.5, +-0.2) = 0.25 + 0.1 + 0.15 + 0 + 0.02 = 0.52
    assert s['task'] == pytest.approx(0.52)
    assert s['loc_reward'] == pytest.approx(0.5)
    assert s['mask_iou'] == pytest.approx(0.5)
    assert s['cand_coverage'] == pytest.approx(0.0)  # empty candidate -> no coverage


def test_score_output_correct_normal_gets_positive():
    text = _output('candidate_bboxes_2d=[]', 'reject; noise',
                   '{"is_anomaly":false,"bboxes_2d":[],"description":"normal"}')
    p = parse_output(text)
    s = score_output(p, _normal_meta(), protocol_weight=0.01)
    assert s['correct'] is True and s['task'] == 1.0
    assert s['focus_reward'] == pytest.approx(0.0)  # no candidate -> no focus bonus


def test_score_output_anomaly_task_is_mask_iou_when_others_zero():
    # With dense/count/refine disabled the anomaly task equals the BBox-Mask IoU.
    text = _output('candidate_bboxes_2d=[]', 'keep; evidence',
                   '{"is_anomaly":true,"bboxes_2d":[[100,100,200,200]],"description":"d"}')
    p = parse_output(text)
    s = score_output(p, _anomaly_meta(), protocol_weight=0.01,
                     localization={'iou_weight': 1.0, 'dense_weight': 0.0,
                                   'count_weight': 0.0, 'refine_weight': 0.0})
    # one of two components covered -> mask_iou = 1/2
    assert s['task'] == pytest.approx(0.5)
    assert s['mask_iou'] == pytest.approx(0.5)


def test_score_output_anomaly_task_blends_mask_iou_and_dense():
    text = _output('candidate_bboxes_2d=[]', 'keep; evidence',
                   '{"is_anomaly":true,"bboxes_2d":[[100,100,200,200]],"description":"d"}')
    p = parse_output(text)
    s = score_output(p, _anomaly_meta(), protocol_weight=0.01,
                     localization={'iou_weight': 0.5, 'dense_weight': 0.5,
                                   'count_weight': 0.0, 'refine_weight': 0.0})
    # mask_iou=0.5 and loc_reward=0.5 -> 0.5*0.5 + 0.5*0.5 = 0.5
    assert s['task'] == pytest.approx(0.5)


def test_count_reward_stages():
    assert count_reward(2, 2) == 1.0
    assert count_reward(1, 2) == 0.5
    assert count_reward(0, 2) == -0.1
    assert count_reward(3, 1) == -0.1


def test_focus_reward_stages():
    # band shape: 1..max_candidates all earn the bonus (SFT teaches normals to
    # read out ALL H candidates, up to 3, before rejecting them)
    assert focus_reward(0) == 0.0
    assert focus_reward(1) == 0.5
    assert focus_reward(2) == 0.5
    assert focus_reward(3) == 0.5
    assert focus_reward(4) == -0.1
    assert focus_reward(2, max_candidates=1) == -0.1


def test_score_output_correct_normal_focus_reward_applied():
    # normal with exactly one candidate box -> focus reward 0.5 boosts the task
    text = _output('candidate_bboxes_2d=[[100,100,200,200]]', 'reject; noise',
                   '{"is_anomaly":false,"bboxes_2d":[],"description":"normal"}')
    p = parse_output(text)
    s = score_output(p, _normal_meta(), protocol_weight=0.01)
    assert s['correct'] is True
    assert s['focus_reward'] == pytest.approx(0.5)
    assert s['task'] == pytest.approx(1.0 + 0.2 * 0.5)


def test_score_output_anomaly_count_reward_in_task():
    # 1 pred vs 2 GT -> count_reward=0.5 enters the task with count_weight
    text = _output('candidate_bboxes_2d=[]', 'keep; evidence',
                   '{"is_anomaly":true,"bboxes_2d":[[100,100,200,200]],"description":"d"}')
    p = parse_output(text)
    s = score_output(p, _anomaly_meta(), protocol_weight=0.01)
    assert s['count_reward'] == pytest.approx(0.5)
    # 0.5*0.5 (mask_iou) + 0.2*0.5 (count) + 0.3*0.5 (dense) + 0.2*0 (coverage)
    # + 0.1*clip(0.5, +-0.2) (empty candidate -> discover bonus) = 0.52
    assert s['task'] == pytest.approx(0.52)


def test_score_output_wrong_decision_gets_negative_one():
    text = _output('candidate_bboxes_2d=[]', 'reject; noise',
                   '{"is_anomaly":false,"bboxes_2d":[],"description":"normal"}')
    p = parse_output(text)
    s = score_output(p, _anomaly_meta(), protocol_weight=0.01)
    assert s['correct'] is False and s['task'] == -1.0


def test_score_output_reports_union_iou_benchmark():
    text = _output('candidate_bboxes_2d=[]', 'keep; evidence',
                   '{"is_anomaly":true,"bboxes_2d":[[100,100,600,600]],"description":"d"}')
    p = parse_output(text)
    s = score_output(p, _anomaly_meta(), protocol_weight=0.01)
    assert s['union_iou'] == pytest.approx(1.0)
    assert s['set_f_reward'] < 1.0  # one union box does not recover both components


def test_delta_refine_recorded():
    text = _output('candidate_bboxes_2d=[[100,100,200,200],[500,500,600,600]]', 'keep; evidence',
                   '{"is_anomaly":true,"bboxes_2d":[[100,100,200,200],[500,500,600,600]],"description":"d"}')
    p = parse_output(text)
    s = score_output(p, _anomaly_meta(), protocol_weight=0.01)
    assert s['set_f_reward'] == pytest.approx(1.0)
    assert s['delta_refine'] == pytest.approx(0.0)  # candidate already perfect
    assert s['cand_coverage'] == pytest.approx(1.0)  # both components hit


def test_cand_coverage_rewards_gt_recall_without_count_penalty():
    # candidate covers both GT components but with loose boxes and one extra box:
    # coverage is full (no count penalty at the candidate stage)
    text = _output('candidate_bboxes_2d=[[90,90,210,210],[490,490,610,610],[800,800,900,900]]',
                   'refine; evidence',
                   '{"is_anomaly":true,"bboxes_2d":[[100,100,200,200],[500,500,600,600]],"description":"d"}')
    p = parse_output(text)
    s = score_output(p, _anomaly_meta(), protocol_weight=0.01)
    assert s['cand_coverage'] == pytest.approx(1.0)
    # task = 0.5*1 + 0.2*1 + 0.3*1 + 0.2*1 + 0.1*clip(1 - candd, +-0.2)
    assert s['task'] == pytest.approx(1.2 + 0.1 * min(max(1.0 - s['set_c_reward'], -0.2), 0.2))


def test_cand_coverage_partial_when_component_missed():
    text = _output('candidate_bboxes_2d=[[100,100,200,200]]', 'discover; evidence',
                   '{"is_anomaly":true,"bboxes_2d":[[100,100,200,200],[500,500,600,600]],"description":"d"}')
    p = parse_output(text)
    s = score_output(p, _anomaly_meta(), protocol_weight=0.01)
    assert s['cand_coverage'] == pytest.approx(0.5)  # 1 of 2 components hit


def test_refine_term_is_two_sided_and_clipped():
    # perfect final, empty candidate -> delta = +1.0 -> clipped to +refine_clip
    text = _output('candidate_bboxes_2d=[]', 'discover; evidence',
                   '{"is_anomaly":true,"bboxes_2d":[[100,100,200,200],[500,500,600,600]],"description":"d"}')
    p = parse_output(text)
    s = score_output(p, _anomaly_meta(), protocol_weight=0.01)
    assert s['delta_refine'] == pytest.approx(1.0)
    # 0.5 + 0.2 + 0.3 + 0.2*0 (coverage) + 0.1*0.2 (clipped bonus)
    assert s['task'] == pytest.approx(1.02)
    # degraded final, perfect candidate -> delta = -0.5 -> clipped to -0.2
    text = _output('candidate_bboxes_2d=[[100,100,200,200],[500,500,600,600]]', 'refine; evidence',
                   '{"is_anomaly":true,"bboxes_2d":[[100,100,200,200]],"description":"d"}')
    p = parse_output(text)
    s = score_output(p, _anomaly_meta(), protocol_weight=0.01)
    assert s['delta_refine'] == pytest.approx(-0.5)
    # 0.5*0.5 + 0.2*0.5 + 0.3*0.5 + 0.2*1.0 (candidate covers both) - 0.1*0.2
    assert s['task'] == pytest.approx(0.25 + 0.1 + 0.15 + 0.2 - 0.02)


def test_giant_candidate_box_is_excluded_anti_hack():
    # a whole-image candidate must not earn coverage or focus rewards
    text = _output('candidate_bboxes_2d=[[0,0,1000,1000]]', 'keep; evidence',
                   '{"is_anomaly":true,"bboxes_2d":[[100,100,200,200],[500,500,600,600]],"description":"d"}')
    p = parse_output(text)
    s = score_output(p, _anomaly_meta(), protocol_weight=0.01)
    assert s['cand_coverage'] == pytest.approx(0.0)
    text = _output('candidate_bboxes_2d=[[0,0,1000,1000]]', 'reject; noise',
                   '{"is_anomaly":false,"bboxes_2d":[],"description":"normal"}')
    p = parse_output(text)
    s = score_output(p, _normal_meta(), protocol_weight=0.01)
    assert s['focus_reward'] == pytest.approx(0.0)
    assert s['task'] == pytest.approx(1.0)


def test_focus_band_applies_to_normal_with_three_candidates():
    text = _output('candidate_bboxes_2d=[[100,100,200,200],[300,300,400,400],[500,500,600,600]]',
                   'reject; noise',
                   '{"is_anomaly":false,"bboxes_2d":[],"description":"normal"}')
    p = parse_output(text)
    s = score_output(p, _normal_meta(), protocol_weight=0.01)
    assert s['focus_reward'] == pytest.approx(0.5)
    assert s['task'] == pytest.approx(1.1)


def test_candidate_hits_comp_center_rule():
    from outcome.protocol_multibox import candidate_hits_comp
    # small candidate strictly inside a larger GT comp: IoU 0.09 < 0.1, center inside
    assert candidate_hits_comp([120, 120, 180, 180], [100, 100, 300, 300]) is True
    # far-away candidate: neither overlap nor center
    assert candidate_hits_comp([700, 700, 760, 760], [100, 100, 300, 300]) is False
    # edge overlap with center outside but IoU >= threshold
    assert candidate_hits_comp([90, 100, 130, 300], [100, 100, 300, 300]) is True
    # edge overlap, center outside, IoU below threshold
    assert candidate_hits_comp([90, 100, 105, 300], [100, 100, 300, 300]) is False


def test_cand_coverage_counts_center_hit():
    # point-like candidate inside the GT comp must count as coverage even at IoU<0.1
    text = _output('candidate_bboxes_2d=[[120,120,180,180]]', 'refine; evidence',
                   '{"is_anomaly":true,"bboxes_2d":[[100,100,300,300]],"description":"d"}')
    p = parse_output(text)
    s = score_output(p, _anomaly_meta(comps=[[100, 100, 300, 300]]), protocol_weight=0.01)
    assert s['cand_coverage'] == pytest.approx(1.0)


def test_more_than_max_boxes_is_invalid():
    boxes = '[[10,10,20,20],[30,30,40,40],[50,50,60,60]]'
    text = _output('candidate_bboxes_2d=[]', 'keep; evidence',
                   f'{{"is_anomaly":true,"bboxes_2d":{boxes},"description":"d"}}')
    p = parse_output(text, max_boxes=2)
    assert p['task_valid'] is False
    assert p['bboxes_2d'] == []
    assert p['num_boxes'] == 0


def test_candidate_more_than_max_boxes_is_invalid():
    text = _output(
        'candidate_bboxes_2d=[[10,10,20,20],[30,30,40,40],[50,50,60,60]]',
        'keep; evidence',
        '{"is_anomaly":true,"bboxes_2d":[[10,10,20,20]],"description":"d"}',
    )
    p = parse_output(text, max_boxes=2)
    assert p['candidate_state'] == 'invalid'
    assert p['candidate_bboxes_2d'] == []
