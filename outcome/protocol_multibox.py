"""outcome-multibox-v1: multi-box detection protocol and set-level reward.

Keeps the five-stage semantics but only <answer> is strict JSON; <ground> and
<verify> are plain text lines to lower protocol syntax entropy for direct GRPO.
Reuses the DCLR localization term and geometry helpers from ``outcome.protocol``
and the Hungarian/metrics helpers from ``outcome.metrics``.
"""
from __future__ import annotations

import json
import re

from outcome.metrics import component_metrics, hungarian_matching, mask_iou, set_giou, union_box
from outcome.protocol import iou, load_object, localization_reward, to_pixels, valid_box
from outcome.thinking import parse_think_stages, split_native_think, thinking_enabled as _thinking_enabled

VERSION = 'outcome-multibox-v1'
TAGS = ('understand', 'compare', 'ground', 'verify', 'answer')
ALL_TAGS = ('think',) + TAGS
BLOCK = re.compile(r'<(think|understand|compare|ground|verify|answer)>(.*?)</\1>', re.S)
VERIFY = re.compile(r'^\s*(keep|refine|reject|discover|none)\b\s*[;:,\-]?\s*(.*)$', re.I)
VERIFY_ACTIONS = ('keep', 'refine', 'reject', 'discover', 'none')

DEFAULT_MAX_BOXES = 16  # VisA-train findContours GT: no-merge p95=9/p99=20/max=63;
# merge_kernel_ratio=0.01 → p95=5/p99=10/max=28, so 16 still covers p99 either way


def parse_boxes_list(text: str):
    """Parse ``candidate_bboxes_2d=[[x1,y1,x2,y2],...]`` (or ``[]`` / ``null``).

    An optional ``; upper left, center`` location suffix (or a location prefix)
    is ignored; only the numeric list is returned.

    Returns ``(state, boxes)`` where state is one of 'list','empty','null',
    'invalid' and boxes is a list of validated [x1,y1,x2,y2] floats.
    """
    text = text.strip()
    assigned = re.search(r'candidate_bboxes_2d\s*=\s*(null|\[.*\])', text, flags=re.I | re.S)
    if assigned:
        text = assigned.group(1).strip()
    else:
        text = re.sub(r'^\s*candidate_bboxes_2d\s*=\s*', '', text, flags=re.I)
    if text == '':
        return 'invalid', []
    if text.lower() == 'null':
        return 'null', []
    if text == '[]':
        return 'empty', []
    groups = re.findall(r'\[([^\[\]]*)\]', text)
    if not groups:
        return 'invalid', []
    boxes = []
    for g in groups:
        parts = [p.strip() for p in g.split(',')]
        if len(parts) != 4:
            return 'invalid', []
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            return 'invalid', []
        if not valid_box(vals):
            return 'invalid', []
        boxes.append(vals)
    return 'list', boxes


def parse_verify(text: str):
    """Return ``(action, evidence)``; action is None when no keyword is found."""
    stripped = text.strip()
    m = VERIFY.match(stripped)
    if m:
        return m.group(1).lower(), m.group(2).strip()
    for action in VERIFY_ACTIONS:
        if re.search(rf'\b{action}\b', text, re.I):
            return action, stripped
    return None, stripped


def parse_output_cfg(text: str, cfg: dict, max_boxes=None) -> dict:
    """parse_output using outcome.max_boxes / outcome.thinking.enabled from cfg."""
    oc = cfg.get('outcome') or {}
    n = int(max_boxes if max_boxes is not None else oc.get('max_boxes', DEFAULT_MAX_BOXES))
    mode = str(oc.get('reasoning_mode', 'fsm'))
    return parse_output(
        text,
        max_boxes=n,
        thinking_required=(mode == 'fsm' and _thinking_enabled(cfg)),
        answer_only=(mode in ('loop', 'direct')),
    )


def parse_output(text: str, max_boxes: int = DEFAULT_MAX_BOXES, thinking_required: bool = False,
                 answer_only: bool = False) -> dict:
    """Parse multi-box output. Only <answer> is strict JSON.

    Legacy: five XML blocks understand/compare/ground/verify/answer.
    Thinking: Qwen native CoT, then a single visible ``<answer>`` JSON.
    Optional ``[localize]``/``[confirm]`` inside think are parsed if present.
    """
    result = dict(task_valid=False, decision_valid=False, final_geometry_valid=False,
                  is_anomaly=None, bboxes_2d=[], candidate_bboxes_2d=[], candidate_state='missing',
                  verify_action=None, verify_evidence='', action=None, description='', tags={},
                  answer_keys=[], protocol_core=False, protocol_strict=False, num_boxes=0,
                  think_ok=False, think_filled=False, think_no_early_boxes=True,
                  think_headers=[], think_bodies={})
    blocks = list(BLOCK.finditer(text))
    result['tags'] = {m[1]: m[2].strip() for m in blocks}
    answers = list(re.finditer(r'<answer>(.*?)</answer>', text, re.S))
    if (len(answers) == 1 and text.count('<answer>') == 1 and text.count('</answer>') == 1
            and not text[answers[0].end():].strip()):
        try:
            obj = load_object(answers[0][1])
            result['answer_keys'] = sorted(obj)
            pred = obj.get('is_anomaly')
            result['decision_valid'] = type(pred) is bool
            result['is_anomaly'] = pred if type(pred) is bool else None
            boxes_raw = obj.get('bboxes_2d')
            if type(pred) is bool:
                if pred is True:
                    geom = (isinstance(boxes_raw, list) and 1 <= len(boxes_raw) <= max_boxes
                            and all(valid_box(b) for b in boxes_raw))
                    result['bboxes_2d'] = [list(b) for b in boxes_raw] if geom else []
                else:
                    geom = isinstance(boxes_raw, list) and len(boxes_raw) == 0
                    result['bboxes_2d'] = []
                result['final_geometry_valid'] = bool(geom)
                result['task_valid'] = result['decision_valid'] and bool(geom)
            desc = obj.get('description', '')
            result['description'] = desc if isinstance(desc, str) else ''
        except (ValueError, TypeError):
            pass
    result['num_boxes'] = len(result['bboxes_2d'])

    think_body, after_think = split_native_think(text)
    if not think_body and result['tags'].get('think'):
        think_body = result['tags']['think']
    think_info = parse_think_stages(think_body)
    n_open = len(re.findall(r'<think>', text, flags=re.I))
    n_close = len(re.findall(r'</think>', text, flags=re.I))
    answer_after_think = bool(re.fullmatch(r'\s*<answer>.*?</answer>\s*', after_think or '', re.S | re.I))
    native_think = n_close == 1 and n_open in (0, 1) and answer_after_think
    result['think_ok'] = bool(native_think)
    result['think_filled'] = bool(think_body) and not re.match(r'^[\s.。…\-–—]*$', think_body)
    result['think_no_early_boxes'] = bool(think_info['no_early_boxes'])
    result['think_headers'] = list(think_info['headers'])
    result['think_bodies'] = dict(think_info['bodies'])

    names = [m[1] for m in blocks]
    has_think = bool(names) and names[0] == 'think'
    think_closed = n_open == n_close
    think_once = (n_open == 1 and think_closed) if has_think else (n_open == 0)

    if thinking_required:
        ground_text = think_info['bodies'].get('localize', '')
        verify_text = think_info['bodies'].get('confirm', '')
        ordered = native_think
        structure = native_think and text.count('<answer>') == text.count('</answer>') == 1
    else:
        ground_text = result['tags'].get('ground', '')
        verify_text = result['tags'].get('verify', '')
        ordered = names == (['think'] + list(TAGS) if has_think else list(TAGS))
        structure = (ordered
                     and not BLOCK.sub('', text).strip()
                     and all(text.count(f'<{t}>') == text.count(f'</{t}>') == 1 for t in TAGS)
                     and think_once)

    if thinking_required and not ground_text.strip():
        cstate, cboxes = 'missing', []
    else:
        cstate, cboxes = parse_boxes_list(ground_text)
        if cstate == 'list' and len(cboxes) > max_boxes:
            cstate = 'invalid'
            cboxes = []
    result['candidate_state'] = cstate
    result['candidate_bboxes_2d'] = cboxes

    vaction, vevidence = parse_verify(verify_text)
    result['verify_action'] = vaction
    result['action'] = vaction
    result['verify_evidence'] = vevidence

    candidate_ok = cstate in ('null', 'empty', 'list')
    verify_ok = vaction is not None
    if thinking_required:
        understand_ok = bool(think_info['bodies'].get('understand', '').strip())
        compare_ok = bool(think_info['bodies'].get('compare', '').strip())
    else:
        understand_ok = bool((result['tags'].get('understand') or '').strip())
        compare_ok = bool((result['tags'].get('compare') or '').strip())
    desc_ok = bool(result['description'].strip())

    if thinking_required:
        core = bool(native_think and result['task_valid'] and desc_ok)
    else:
        core = bool(ordered and result['task_valid'] and candidate_ok and verify_ok and desc_ok)
    result['protocol_core'] = core

    ground_strict = bool(re.fullmatch(
        r'\s*candidate_bboxes_2d\s*=\s*(null|\[.*\])\s*(;\s*\S.*)?\s*', ground_text, re.I | re.S))
    verify_strict = bool(re.fullmatch(r'\s*(keep|refine|reject|discover|none)\s*;\s*\S.*', verify_text, re.I))
    answer_strict = (result['task_valid']
                     and set(result['answer_keys']) == {'is_anomaly', 'bboxes_2d', 'description'}
                     and desc_ok)
    if answer_only:
        structure = (
            text.count('<answer>') == 1
            and text.count('</answer>') == 1
        )
        core = bool(structure and result['task_valid'] and desc_ok)
        result['protocol_core'] = core
        result['protocol_strict'] = bool(
            core
            and answer_strict
            and not text[:answers[0].start()].strip()
            and not text[answers[0].end():].strip()
        )
        return result
    if thinking_required:
        result['protocol_strict'] = bool(
            structure and answer_strict and think_info['ok'] and think_info['filled'])
    else:
        result['protocol_strict'] = bool(
            structure and result['task_valid'] and candidate_ok and verify_ok
            and ground_strict and verify_strict and answer_strict and understand_ok and compare_ok)
    return result


def count_reward(m: int, n: int) -> float:
    """Explicit box-count reward (AD-FM eq. 5): penalizes wrong enumeration.

    |m-n|=0 -> 1.0, |m-n|=1 -> 0.5, |m-n|>=2 -> -0.1. This gives GRPO a direct
    signal for detecting exactly the right number of defects, which the implicit
    max(N,M) denominator of set_iou cannot isolate.
    """
    d = abs(int(m) - int(n))
    if d == 0:
        return 1.0
    if d == 1:
        return 0.5
    return -0.1


def focus_reward(m: int, max_candidates: int = 3) -> float:
    """Focus reward for correctly-rejected normal samples (AD-FM eq. 6).

    Encourages the model to actually localize suspicious regions before rejecting
    them: 0 boxes -> 0.0, 1..max_candidates boxes -> 0.5, more -> -0.1. Uses the
    candidate box count because a normal sample's final answer must be empty.
    The band (rather than exactly-one) matches the SFT cold-start, which teaches
    normals to read out ALL H region hints (up to ``max_candidates``) and then
    reject them in <verify>.
    """
    if m == 0:
        return 0.0
    if m <= int(max_candidates):
        return 0.5
    return -0.1


def _area_frac_1000(b) -> float:
    """Box area as a fraction of the image, in the 0-1000 coordinate system."""
    return max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1])) / 1e6


def candidate_hits_comp(cand, comp, iou_threshold: float = 0.10) -> bool:
    """Whether a candidate box 'marks' a GT component.

    H candidates are patch-aligned boxes that usually sit *inside* the defect
    region, where plain IoU systematically undercounts (a 60px candidate inside
    a 200px GT box scores 0.09). A candidate therefore counts as marking the
    component when it overlaps at ``iou_threshold`` OR its center lands inside
    the component box. Works in any consistent coordinate system (both boxes in
    px, or both in 0-1000).
    """
    if iou(cand, comp) >= float(iou_threshold):
        return True
    cx = 0.5 * (float(cand[0]) + float(cand[2]))
    cy = 0.5 * (float(cand[1]) + float(cand[3]))
    return float(comp[0]) <= cx <= float(comp[2]) and float(comp[1]) <= cy <= float(comp[3])


def set_localization_reward(pred_boxes, gt_components, orig_size, iou_threshold=0.30, geometry_weight=0.30):
    """DCLR-style set reward via Hungarian matching.

    S_ij = R_loc(B_i, G_j) for every pred/GT pair; matching maximizes sum S_ij.
    R_set = sum(matched S_ij) / max(N, M) penalizes missed defects and duplicate
    boxes through the denominator without a separate count penalty.
    """
    if not pred_boxes or not gt_components:
        return dict(reward=0.0, matched_pairs=[], s_sum=0.0)
    N, M = len(pred_boxes), len(gt_components)
    S = [[localization_reward(p, g, orig_size, iou_threshold, geometry_weight)['loc_reward']
          for g in gt_components] for p in pred_boxes]
    pairs = hungarian_matching(S)
    s_sum = float(sum(S[i][j] for i, j in pairs))
    reward = s_sum / max(N, M)
    return dict(reward=reward, matched_pairs=pairs, s_sum=s_sum)


def validate_gt(meta):
    w, h = meta['orig_size']
    if w <= 0 or h <= 0:
        raise ValueError('invalid original image size')
    if meta['is_anomaly']:
        comps = meta.get('component_bboxes') or []
        gt = meta.get('gt_box_px')
        has_comps = comps and all(valid_box(b, max(w, h)) and b[2] <= w and b[3] <= h for b in comps)
        if not has_comps and not (gt is not None and valid_box(gt, max(w, h)) and gt[2] <= w and gt[3] <= h):
            raise ValueError(f"anomalous sample missing/invalid component GT: {meta.get('image_path')}")
    elif (meta.get('gt_box_px') is not None) or (meta.get('component_bboxes')):
        raise ValueError('normal sample must have null GT')


def score_output(parsed, meta, protocol_weight=0.01, localization=None, max_boxes=DEFAULT_MAX_BOXES):
    """Fine-grained reward: classification + GIoU/count/focus/refine (AD-FM style).

    ``R_task`` decomposes as:

    * wrong decision (or invalid output)  -> ``wrong_decision`` (default -1)
    * correct normal  -> ``normal_correct + focus_weight * focus_reward(n_cand)``
    * correct anomaly -> ``cls_weight + iou_weight * mask_iou
                          + count_weight * count_reward(N, M)
                          + dense_weight * loc_reward
                          + cand_weight * cand_coverage
                          + refine_weight * clip(delta_refine, +-refine_clip)``

    ``mask_iou`` is the AD-Copilot BBox-Mask IoU: the predicted and GT boxes are
    each rasterized to a binary mask and IoU is computed in mask space, so it does
    not pair boxes nor care how many boxes the model emits (robust to irregular/
    disconnected defects and to output granularity). ``count_reward`` still gives a
    direct signal for box enumeration, ``loc_reward`` is the DCLR dense term, and
    ``cand_coverage`` is the candidate-stage GT recall (fraction of GT components
    marked by any candidate — IoU at ``cand_iou_threshold`` or candidate center
    inside the component, see ``candidate_hits_comp``; no count penalty) so the
    <ground> stage carries proposal responsibility and the empty-candidate hack
    forfeits it. The refine term is two-sided but clipped (improving the candidate
    earns a small capped bonus, degrading it is penalized symmetrically), which
    keeps "candidate = loose superset, final = precise" strictly better than
    "candidate == final" without leaving room for refine farming.
    ``focus_reward`` rewards localizing-then-rejecting on normal samples.
    Candidate boxes larger than ``cand_max_area_frac`` of the image are excluded
    from coverage/focus accounting (anti giant-box hack). Hungarian
    ``set_giou`` / ``set_iou`` / ``union_iou`` are reported as diagnostics only,
    never in reward.
    """
    validate_gt(meta)
    if not 0 <= protocol_weight <= 0.1:
        raise ValueError('protocol_weight must be in [0, 0.1]')
    loc = localization or {}
    iou_threshold = float(loc.get('iou_threshold', 0.30))
    geometry_weight = float(loc.get('geometry_weight', 0.30))
    normal_correct = float(loc.get('normal_correct', 1.0))
    wrong_decision = float(loc.get('wrong_decision', -1.0))
    false_positive_penalty = float(loc.get('false_positive_penalty', wrong_decision))
    cls_weight = float(loc.get('cls_weight', 0.0))
    iou_weight = float(loc.get('iou_weight', 0.5))
    count_weight = float(loc.get('count_weight', 0.2))
    dense_weight = float(loc.get('dense_weight', 0.3))
    focus_weight = float(loc.get('focus_weight', 0.2))
    refine_weight = float(loc.get('refine_weight', 0.1))
    cand_weight = float(loc.get('cand_weight', 0.2))
    cand_iou_threshold = float(loc.get('cand_iou_threshold', 0.10))
    cand_max_area_frac = float(loc.get('cand_max_area_frac', 0.10))
    refine_clip = abs(float(loc.get('refine_clip', 0.2)))
    focus_max_candidates = int(loc.get('focus_max_candidates', 3))
    reject_weight = float(loc.get('reject_weight', 0.0))
    correct = parsed['task_valid'] and parsed['is_anomaly'] == bool(meta['is_anomaly'])
    pred_px = [to_pixels(b, meta['orig_size']) for b in parsed['bboxes_2d']]
    union_iou_val = (iou(union_box(pred_px), meta.get('gt_box_px'))
                     if (meta['is_anomaly'] and pred_px) else 0.0)
    comps = meta.get('component_bboxes') or ([meta['gt_box_px']] if meta.get('gt_box_px') else [])
    mask_iou_val = mask_iou(pred_px, comps, meta['orig_size']) if meta['is_anomaly'] else 0.0
    cand_boxes = [b for b in parsed['candidate_bboxes_2d'] if _area_frac_1000(b) <= cand_max_area_frac]
    cand_px = [to_pixels(b, meta['orig_size']) for b in cand_boxes]
    setd = dict(reward=0.0, matched_pairs=[], s_sum=0.0)
    candd = dict(reward=0.0, matched_pairs=[], s_sum=0.0)
    raw_set_iou = 0.0
    set_giou_val = 0.0
    count_val = 0.0
    focus_val = 0.0
    cand_cov_val = 0.0
    if meta['is_anomaly'] and correct:
        setd = set_localization_reward(parsed['bboxes_2d'], comps, meta['orig_size'], iou_threshold, geometry_weight)
        raw_set_iou = float(component_metrics(pred_px, comps)['set_iou'])
        set_giou_val = float(set_giou(pred_px, comps))
        count_val = float(count_reward(len(pred_px), len(comps)))
        if comps:
            hits = sum(1 for g in comps
                       if any(candidate_hits_comp(c, g, cand_iou_threshold) for c in cand_px))
            cand_cov_val = float(hits / len(comps))
    if meta['is_anomaly']:
        candd = set_localization_reward(cand_boxes, comps, meta['orig_size'], iou_threshold, geometry_weight)
    loc_reward = setd['reward']
    delta_refine = loc_reward - candd['reward']
    # Reject shaping: a [confirm]=reject triggers the confirm-reloop (a zero-H /
    # pure-text re-localization). Reward it only when the final localization is
    # actually good (mask_iou), so a reject that rescues a bad localization is
    # rewarded while a pointless reject is penalized. (2*mask_iou - 1) makes the
    # term positive above IoU 0.5 and negative below, tying reject directly to
    # localization quality. Anomaly-only: normal-sample rejects are covered by
    # focus_reward / the empty final answer.
    reject_bonus = 0.0
    if not correct:
        # False positive (normal image declared anomalous) gets a heavier
        # penalty than a miss (anomaly declared normal): in industrial QC a
        # false alarm is costlier than a missed defect.
        is_false_positive = (not meta['is_anomaly']) and parsed['is_anomaly'] is True
        task = false_positive_penalty if is_false_positive else wrong_decision
    elif not meta['is_anomaly']:
        focus_val = float(focus_reward(len(cand_boxes), focus_max_candidates))
        task = normal_correct + focus_weight * focus_val
    else:
        task = (cls_weight
                + iou_weight * mask_iou_val
                + count_weight * count_val
                + dense_weight * loc_reward
                + cand_weight * cand_cov_val
                + refine_weight * float(min(max(delta_refine, -refine_clip), refine_clip)))
        if parsed['verify_action'] == 'reject':
            reject_bonus = reject_weight * (2.0 * mask_iou_val - 1.0)
            task = task + reject_bonus
    protocol_core = float(parsed['protocol_core'])
    return dict(task=task, protocol=protocol_core, protocol_core=protocol_core,
                protocol_strict=float(parsed['protocol_strict']),
                total=task + protocol_weight * protocol_core,
                loc_reward=loc_reward, set_c_reward=candd['reward'], set_f_reward=loc_reward,
                delta_refine=delta_refine,
                mask_iou=mask_iou_val, union_iou=union_iou_val,
                raw_iou=raw_set_iou, set_iou=raw_set_iou, set_giou=set_giou_val,
                count_reward=count_val, focus_reward=focus_val, cand_coverage=cand_cov_val,
                reject_bonus=reject_bonus, verify_action=parsed['verify_action'],
                correct=bool(correct), matched_pairs=setd['matched_pairs'], s_sum=setd['s_sum'])
