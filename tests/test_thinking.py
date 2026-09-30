"""Qwen native internal CoT: staged think headers, SFT labels, protocol gating."""
from outcome.thinking import (
    THINK_STAGES,
    parse_think_stages,
    split_native_think,
    staged_sft_target,
    think_block_for_sft,
    think_body_char_spans,
    unsupervised_think_labels,
    visible_sft_target,
)
from outcome.protocol_multibox import parse_output
from train_region_sft import build_labels, build_sft_target


def test_parse_think_requires_five_stages_in_order():
    ok = parse_think_stages(
        '[understand]\nref is glossy\n[compare]\nscratch vs print\n'
        '[localize]\nupper left\n[imagine]\nkeep; box matches the defect\n'
        '[confirm]\nkeep; real defect'
    )
    assert ok['ok'] and ok['filled'] and ok['no_early_boxes']
    assert ok['headers'] == list(THINK_STAGES)
    assert 'glossy' in ok['bodies']['understand']


def test_parse_think_rejects_chinese_headers():
    ok = parse_think_stages('[理解]\na\n[对比]\nb\n[定位]\nc\n[确认]\nd')
    assert not ok['ok']


def test_parse_think_rejects_missing_or_shuffled_stages():
    assert not parse_think_stages('[understand]\nx\n[confirm]\ny')['ok']
    assert not parse_think_stages(
        '[compare]\na\n[understand]\nb\n[localize]\nc\n[imagine]\nd\n[confirm]\ne'
    )['ok']


def test_parse_think_placeholder_is_not_filled():
    info = parse_think_stages(think_block_for_sft().replace('<think>', '').replace('</think>', ''))
    assert info['ok']
    assert not info['filled']


def test_early_box_leak_in_understand():
    info = parse_think_stages(
        '[understand]\ncandidate_bboxes_2d=[[1,2,3,4]]\n[compare]\nok\n'
        '[localize]\nleft\n[imagine]\nkeep; ok\n[confirm]\nkeep; x'
    )
    assert info['ok'] and not info['no_early_boxes']


def test_sft_masks_tokens_before_answer():
    class _Tok:
        def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
            ids = [ord(c) % 97 + 1 for c in text]
            out = {'input_ids': ids}
            if return_offsets_mapping:
                out['offset_mapping'] = [(i, i + 1) for i in range(len(text))]
            return out

    target = visible_sft_target('{"is_anomaly": false, "bboxes_2d": [], "description": "ok"}')
    tok = _Tok()
    ids = tok(target)['input_ids']
    labels = unsupervised_think_labels(tok, target)
    assert len(labels) == len(ids)
    spans = think_body_char_spans(target)
    assert spans == [(0, target.index('<answer>'))]
    for i, (s, e) in enumerate(tok(target, return_offsets_mapping=True)['offset_mapping']):
        in_prefix = any(s < ce and e > cs for cs, ce in spans)
        if in_prefix:
            assert labels[i] == -100
        else:
            assert labels[i] == ids[i]


def test_build_labels_masks_prompt_supervises_whole_target():
    # Single-pass SFT: the entire target (stage headers, bodies, </think>, JSON,
    # closing tags) is supervised; only the prompt is masked.
    ids = list(range(1, 50))
    labels = build_labels([0, 0], ids)
    assert labels[:2] == [-100, -100]
    assert labels[2:] == ids


def test_legacy_five_block_still_strict_without_think():
    text = (
        '<understand>\nstructure\n</understand>\n'
        '<compare>\ndifference\n</compare>\n'
        '<ground>\ncandidate_bboxes_2d=[[100,100,200,200]]\n</ground>\n'
        '<verify>\nkeep; both look anomalous\n</verify>\n'
        '<answer>\n{"is_anomaly":true,"bboxes_2d":[[100,100,200,200]],"description":"d"}\n</answer>'
    )
    p = parse_output(text)
    assert p['protocol_core'] and p['protocol_strict']
    assert not p['think_ok']
    p_req = parse_output(text, thinking_required=True)
    assert p_req['protocol_core'] is False


def test_thinking_required_accepts_freeform_cot():
    text = (
        'Reference looks uniform; inspection has a local scratch not explained by print.\n'
        '</think>\n\n'
        '<answer>\n{"is_anomaly":true,"bboxes_2d":[[100,100,200,200]],"description":"d"}\n</answer>'
    )
    p = parse_output(text, thinking_required=True)
    assert p['think_ok'] and p['think_filled']
    assert p['protocol_core']
    assert not p['protocol_strict']
    assert p['candidate_state'] == 'missing'
    body, after = split_native_think(text)
    assert 'scratch' in body and '<answer>' in after


def test_thinking_required_still_parses_optional_stages():
    text = (
        '<think>\n[understand]\nref looks smooth\n[compare]\nlocal scratch not print\n'
        '[localize]\ncandidate_bboxes_2d=[[100,100,200,200]]; in the upper left\n'
        '[imagine]\nkeep; this box matches the defect\n'
        '[confirm]\nunchanged; matches the difference\n</think>\n'
        '<answer>\n{"is_anomaly":true,"bboxes_2d":[[100,100,200,200]],"description":"d"}\n</answer>'
    )
    p = parse_output(text, thinking_required=True)
    assert p['think_ok'] and p['think_filled']
    assert p['candidate_bboxes_2d'] == [[100, 100, 200, 200]]
    assert p['verify_action'] == 'unchanged'
    assert p['imagine_action'] == 'keep'
    assert p['protocol_core'] and p['protocol_strict']


def test_sft_target_with_thinking_uses_stage_headers_not_xml():
    meta = dict(is_anomaly=True, orig_size=[1000, 1000], class_name='bottle',
                gt_box_px=[100, 100, 200, 200], component_bboxes=[[100, 100, 200, 200]],
                prior_candidates=[dict(bbox_2d=[100, 100, 200, 200])])
    target = build_sft_target(meta, multibox=True, thinking=True, sft_cfg={'p_keep': 1.0})
    assert target.startswith('[understand]')
    assert '[compare]' in target and '[localize]' in target and '[imagine]' in target and '[confirm]' in target
    assert '</think>' in target and '<answer>' in target
    assert '<understand>' not in target and '<ground>' not in target and '<think>' not in target
    p = parse_output(target, thinking_required=True)
    assert p['think_ok'] and p['think_filled']
    assert p['protocol_core'] and p['protocol_strict']
    assert p['candidate_bboxes_2d'] == [[100.0, 100.0, 200.0, 200.0]]
    assert 'action=zoom_box_0' in target and 'action=global_scan' in target and 'action=stop' in target
    assert p['confirm_valid'] and p['update_action'] in ('keep', 'refine', 'reject', 'discover')
    assert p['bboxes_2d'] == [[100.0, 100.0, 200.0, 200.0]]
    assert 'scratch' not in target


def test_thinking_trace_uses_prose_before_answer_when_unclosed():
    from outcome.thinking import thinking_trace
    from outcome.visualize_multibox import format_outcome_case_text

    text = (
        'Image 2 matches the reference bottle.\n'
        '<answer>\n{"is_anomaly":false,"bboxes_2d":[],"description":"ok"}\n</answer>'
    )
    cot, ans = thinking_trace(text)
    assert 'matches the reference' in cot
    assert ans.startswith('<answer>')
    parsed = parse_output(text, thinking_required=True)
    meta = dict(image_path='x', class_name='bottle', is_anomaly=False, num_components=0)
    blob = format_outcome_case_text(0, meta, text, parsed, 0.0, 0.0, True)
    assert '=== think ===' in blob and 'matches the reference' in blob
    assert '=== answer ===' in blob

