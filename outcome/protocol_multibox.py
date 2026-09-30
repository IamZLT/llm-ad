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
# ``[confirm]`` evaluates the *refine loop itself* — whether this trajectory's
# refinement improved the final box relative to the first candidate (the sign of
# ``delta_refine``) — rather than echoing ``[imagine]``'s box-quality verdict.
# This keeps the two stages orthogonally useful: imagine = "is the box right?",
# confirm = "did my fix make it better / worse / stay the same?".
REFINE_VERIFY = re.compile(r'^\s*(improved|unchanged|degraded)\b\s*[;:,\-]?\s*(.*)$', re.I)
REFINE_VERIFY_ACTIONS = ('improved', 'unchanged', 'degraded')

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
    assigned = re.search(r'candidate_bboxes_2d\s*=', text, flags=re.I)
    if assigned:
        rest = text[assigned.end():].lstrip()
        if rest.lower().startswith('null'):
            text = 'null'
        elif rest.startswith('['):
            # Bracket-match the box list so a trailing ``; location`` suffix or a
            # following ``[imagine]``/``[confirm]`` header cannot be swallowed by a
            # greedy regex (a greedy ``\[.*\]`` over-matches past ``]]`` into the
            # next bracketed stage header, corrupting the parse).
            depth = 0
            end = -1
            for j, c in enumerate(rest):
                if c == '[':
                    depth += 1
                elif c == ']':
                    depth -= 1
                    if depth == 0:
                        end = j + 1
                        break
            if end > 0:
                text = rest[:end].strip()
            else:
                text = rest
        else:
            text = rest
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


def parse_confirm(text: str) -> dict:
    """Parse ``observed=...; consistency=...; update=...`` from ``[confirm]``."""
    match = re.search(
        r"observed\s*=\s*([a-zA-Z0-9_]+)\s*;\s*"
        r"consistency\s*=\s*(supported|contradicted|uncertain)\s*;\s*"
        r"update\s*=\s*([a-zA-Z0-9_]+)",
        text or "",
        re.I,
    )
    if not match:
        return dict(valid=False, observed_evidence=None,
                    prediction_consistency=None, update_action=None)
    return dict(
        valid=True,
        observed_evidence=match.group(1).lower(),
        prediction_consistency=match.group(2).lower(),
        update_action=match.group(3).lower(),
    )


def parse_refine_verify(text: str):
    """Parse a ``[confirm]`` body into ``(improved|unchanged|degraded, evidence)``."""
    stripped = text.strip()
    m = REFINE_VERIFY.match(stripped)
    if m:
        return m.group(1).lower(), m.group(2).strip()
    for action in REFINE_VERIFY_ACTIONS:
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
                  verify_action=None, imagine_action=None, verify_evidence='', action=None,
                  description='', tags={},
                  answer_keys=[], protocol_core=False, protocol_strict=False, num_boxes=0,
                  think_ok=False, think_filled=False, think_no_early_boxes=True,
                  think_headers=[], think_bodies={},
                  # Two-stage observation bookkeeping (phase 1): B0/B1 are the initial
                  # [localize] boxes and the final <answer> boxes respectively.
                  pre_observation_prediction='', selected_action=None, predicted_effect=None,
                  observed_evidence=None, prediction_consistency=None, update_action=None,
                  confirm_valid=False)
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
    result['first_candidate_bboxes_2d'] = []

    names = [m[1] for m in blocks]
    has_think = bool(names) and names[0] == 'think'
    think_closed = n_open == n_close
    think_once = (n_open == 1 and think_closed) if has_think else (n_open == 0)

    if thinking_required:
        ground_text = think_info['bodies'].get('localize', '')
        first_ground_text = think_info['first_bodies'].get('localize', ground_text)
        imagine_text = think_info['bodies'].get('imagine', '')
        verify_text = think_info['bodies'].get('confirm', '')
        ordered = native_think
        structure = native_think and text.count('<answer>') == text.count('</answer>') == 1
    else:
        ground_text = result['tags'].get('ground', '')
        first_ground_text = ground_text
        imagine_text = ''
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

    # First-round candidate (the *starting* box before any reject/refine): used by
    # score_output to measure delta_refine = final - first (the true "did refine help"
    # improvement), since the last-round candidate equals the final answer in a
    # well-formed trajectory and would otherwise zero out the refine term.
    if first_ground_text.strip():
        _fcstate, _fcboxes = parse_boxes_list(first_ground_text)
        result['first_candidate_bboxes_2d'] = _fcboxes if _fcstate == 'list' else []
    else:
        result['first_candidate_bboxes_2d'] = []

    if thinking_required:
        confirm = parse_confirm(verify_text)
        result['confirm_valid'] = confirm['valid']
        result['observed_evidence'] = confirm['observed_evidence']
        result['prediction_consistency'] = confirm['prediction_consistency']
        result['update_action'] = confirm['update_action']
        if confirm['valid']:
            vaction, vevidence = confirm['update_action'], confirm['observed_evidence']
        else:
            vaction, vevidence = parse_refine_verify(verify_text)
    else:
        # Legacy <verify> tag keeps the old box-quality action vocabulary.
        vaction, vevidence = parse_verify(verify_text)
    result['verify_action'] = vaction
    result['action'] = vaction
    result['verify_evidence'] = vevidence
    if 'action=' in imagine_text.lower():
        iaction = None
    else:
        iaction, _ = parse_verify(imagine_text) if imagine_text.strip() else (None, '')
    result['imagine_action'] = iaction
    # Phase-1 observation fields. ``pre_observation_prediction`` is the raw
    # [imagine] body (written before the crop); ``predicted_effect`` is the
    # [confirm] verdict (improved/unchanged/degraded — a *prediction*, not a
    # verified fact, since inference has no GT). ``selected_action`` stays None
    # until the phase-2 geometric action module introduces keep/expand/.../reject.
    result['pre_observation_prediction'] = imagine_text.strip()
    result['predicted_effect'] = vaction
    result['selected_action'] = None

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


def match_boxes_at_iou(pred_boxes, gt_boxes, threshold=0.5):
    """One-to-one matching: maximize threshold-satisfying matches, then IoU.

    ``pred_boxes`` and ``gt_boxes`` must use the same coordinate system. A valid
    match (IoU >= ``threshold``) is worth more than all secondary IoU terms summed,
    so the Hungarian assignment first maximizes the number of matches and only then
    breaks ties by raw IoU.
    """
    if not pred_boxes or not gt_boxes:
        return []

    ious = [
        [iou(p, g) for g in gt_boxes]
        for p in pred_boxes
    ]

    bonus = min(len(pred_boxes), len(gt_boxes)) + 1.0
    scores = [
        [
            bonus * float(v >= threshold) + v
            for v in row
        ]
        for row in ious
    ]

    pairs = hungarian_matching(scores)
    return [
        (i, j, float(ious[i][j]))
        for i, j in pairs
        if ious[i][j] >= threshold
    ]


def diagnose_box_set(pred_boxes, gt_boxes, threshold=0.5):
    """Set-level tp/fp/fn from one-to-one IoU matching.

    ``fn`` counts GT components unmatched at the given IoU threshold, so it bundles
    both "region entirely missed" and "region found but poorly localized" — it must
    not be read as "no box was emitted anywhere near this region".
    """
    matches = match_boxes_at_iou(
        pred_boxes, gt_boxes, threshold=threshold
    )
    tp = len(matches)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp

    return {
        "matches": matches,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "complete": fn == 0 and fp == 0,
    }


def objective_verdict(cand_px, comps, iou_threshold: float = 0.10, keep_iou: float = 0.50) -> str:
    """Objective verdict for a candidate box *set* against GT components.

    This is the ground-truth label the world-model's ``[imagine]`` / ``[confirm]``
    verdict is *calibrated* against — the model is rewarded for predicting this
    outcome, not for echoing its own answer:

    * ``'keep'``    — the candidate set exactly covers every GT component (complete).
    * ``'refine'``  — some GT component is only loosely supported (correctable), or
      the set is a superset/duplicate (fp>0).
    * ``'reject'``  — no candidate has support from any GT component, yet candidates
      were emitted (false alarm / unrelated region).
    * ``'discover'`` — no candidate at all, but GT components exist (must search).
    * ``'none'``    — no GT components and no candidate (true normal).

    A candidate that marks a GT component with low IoU is still ``'refine'`` (it has
    support and is fixable); only a completely unsupported candidate set becomes
    ``'reject'``. The ``keep`` decision is set-level, so a set that misses any GT
    component can never be ``'keep'`` no matter how precisely the found boxes fit.
    """
    if not comps:
        return 'none' if not cand_px else 'reject'

    if not cand_px:
        return 'discover'

    diag = diagnose_box_set(cand_px, comps, keep_iou)
    if diag['complete']:
        return 'keep'

    has_support = any(
        candidate_hits_comp(c, g, iou_threshold)
        for c in cand_px
        for g in comps
    )

    return 'refine' if has_support else 'reject'


def refine_verdict(delta_refine: float, eps: float = 0.05) -> str:
    """Sign of the refine loop's net effect, as a three-way verdict.

    ``delta_refine = final_box_reward - first_candidate_box_reward`` measures how
    much the internal reject/refine loop moved the box. ``[confirm]`` is scored
    against this objective sign — ``'improved'`` (net gain), ``'unchanged'``
    (no meaningful change, e.g. a single-round keep or an empty normal), or
    ``'degraded'`` (refining made it worse). ``eps`` absorbs numeric noise and
    treats tiny refinements as "unchanged" so a first-round box that is already
    right does not get rewarded for a no-op re-localization.
    """
    if delta_refine > eps:
        return 'improved'
    if delta_refine < -eps:
        return 'degraded'
    return 'unchanged'


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


def score_output(parsed, meta, protocol_weight=0.01, localization=None, max_boxes=DEFAULT_MAX_BOXES,
                 trace=None):
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
    Hungarian ``set_giou`` / ``set_iou`` / ``union_iou`` are reported as
    diagnostics only, never in reward.
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
    refine_clip = abs(float(loc.get('refine_clip', 0.2)))
    focus_max_candidates = int(loc.get('focus_max_candidates', 3))
    discrim_weight = float(loc.get('discrim_weight', 0.0))
    imagine_weight = float(loc.get('imagine_weight', discrim_weight))
    keep_iou_threshold = float(loc.get('keep_iou_threshold', 0.50))
    refine_eps = float(loc.get('refine_eps', 0.05))
    correct = parsed['task_valid'] and parsed['is_anomaly'] == bool(meta['is_anomaly'])
    pred_px = [to_pixels(b, meta['orig_size']) for b in parsed['bboxes_2d']]
    union_iou_val = (iou(union_box(pred_px), meta.get('gt_box_px'))
                     if (meta['is_anomaly'] and pred_px) else 0.0)
    comps = meta.get('component_bboxes') or ([meta['gt_box_px']] if meta.get('gt_box_px') else [])
    mask_iou_val = mask_iou(pred_px, comps, meta['orig_size']) if meta['is_anomaly'] else 0.0
    cand_boxes = [b for b in parsed['candidate_bboxes_2d']]
    cand_px = [to_pixels(b, meta['orig_size']) for b in cand_boxes]
    first_cand_boxes = [b for b in parsed.get('first_candidate_bboxes_2d', [])]
    setd = dict(reward=0.0, matched_pairs=[], s_sum=0.0)
    candd = dict(reward=0.0, matched_pairs=[], s_sum=0.0)
    first_candd = dict(reward=0.0, matched_pairs=[], s_sum=0.0)
    raw_set_iou = 0.0
    set_giou_val = 0.0
    count_val = 0.0
    focus_val = 0.0
    cand_cov_val = 0.0
    if meta['is_anomaly']:
        # Localization geometry is scored for EVERY anomalous sample, not gated on
        # ``correct``. Gating it meant a wrong decision froze loc_reward at 0, so the
        # per-token localization advantage had no gradient during the long cold-start
        # phase (lz=0) and mIoU never lifted.
        setd = set_localization_reward(parsed['bboxes_2d'], comps, meta['orig_size'], iou_threshold, geometry_weight)
        if pred_px:
            raw_set_iou = float(component_metrics(pred_px, comps)['set_iou'])
            set_giou_val = float(set_giou(pred_px, comps))
        count_val = float(count_reward(len(pred_px), len(comps)))
        candd = set_localization_reward(cand_boxes, comps, meta['orig_size'], iou_threshold, geometry_weight)
        first_candd = set_localization_reward(first_cand_boxes, comps, meta['orig_size'], iou_threshold, geometry_weight)
        if comps:
            hits = sum(1 for g in comps
                       if any(candidate_hits_comp(c, g, cand_iou_threshold) for c in cand_px))
            cand_cov_val = float(hits / len(comps))
    # Q0 = candidate-box quality (B0), Q1 = final-box quality (B1). The final
    # answer's coordinates are supervised by Q1 only; the candidate boxes carry
    # their own Q0 signal through cand_weight. We no longer take max(Q0, Q1) for
    # the final-coordinate advantage — that rewarded a correct candidate even when
    # the final box degraded, giving the final coords a positive signal they did
    # not earn (and hiding refinement regressions).
    q0 = candd['reward']
    q1 = setd['reward']
    loc_reward = q1
    cand_reward = q0
    # delta_refine = final - first candidate: the true "did observation/refinement
    # improve the box" increment (Q1 - Q(B0)). Using the last-round candidate here
    # would compare the final box against itself (== 0) in a well-formed trajectory.
    delta_refine = q1 - first_candd['reward']
    # World-model rehearsal calibration. The two stages are scored against two
    # *different* objective labels so they cannot collapse into echoes:
    #   * [imagine] predicts the candidate box's objective quality (keep/refine/
    #     reject/none) — "is this box right?"
    #   * [confirm] predicts the sign of the refine loop's net effect
    #     (improved/unchanged/degraded) — "did refining actually fix it?"
    # ``imagine_correct`` rewards the quality rehearsal; ``discrim_correct`` rewards
    # the refine-loop verdict (sign of ``delta_refine``), NOT the box-quality label.
    verify_action = parsed.get('verify_action')
    imagine_action = parsed.get('imagine_action')
    obj_verdict = objective_verdict(cand_px, comps, cand_iou_threshold, keep_iou_threshold)
    candidate_diag = diagnose_box_set(cand_px, comps, keep_iou_threshold)
    final_diag = diagnose_box_set(pred_px, comps, keep_iou_threshold)
    imagine_correct = 1.0 if imagine_action == obj_verdict else 0.0
    refine_obj = refine_verdict(delta_refine, refine_eps)
    discrim_correct = 1.0 if verify_action == refine_obj else 0.0
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
    planner_signal = 0.0
    confirm_signal = 0.0
    gain_error = None
    predicted_gain = None
    observation_cost = 0.0
    from outcome.planner_supervision import build_action_supervision
    from outcome.state_quality import state_quality
    gt_for_quality = comps if meta['is_anomaly'] else []
    state_q0 = state_quality(cand_boxes, gt_for_quality, meta['orig_size'], loc)
    state_q1 = state_quality(parsed['bboxes_2d'], gt_for_quality, meta['orig_size'], loc)
    supervision_rows = build_action_supervision(
        cand_boxes, gt_for_quality, meta['orig_size'], loc)
    oracle = max(supervision_rows, key=lambda row: row.target_gain) if supervision_rows else None
    oracle_action = oracle.action if oracle else None
    oracle_gain = oracle.target_gain if oracle else None
    if trace is not None and getattr(trace, 'rounds', None):
        rnd = trace.rounds[-1]
        actual_gain = float(state_q1 - state_q0)
        rnd.actual_gain = actual_gain
        predicted_gain = rnd.selected_predicted_gain
        observation_cost = float(rnd.observation_cost or 0.0)
        if predicted_gain is not None:
            gain_error = abs(float(predicted_gain) - actual_gain)
            rnd.gain_error = gain_error
        planner_signal = (actual_gain
                          - float(loc.get('cost_weight', 0.05)) * observation_cost
                          - float(loc.get('gain_calib_weight', 0.05)) * (gain_error or 0.0))
        confirm_signal = 0.5 * float(state_q1) + 0.5 * actual_gain
        if parsed.get('confirm_valid'):
            rnd.observed_evidence = parsed.get('observed_evidence')
            rnd.prediction_consistency = parsed.get('prediction_consistency')
            rnd.update_action = parsed.get('update_action')
    protocol_core = float(parsed['protocol_core'])
    return dict(task=task, protocol=protocol_core, protocol_core=protocol_core,
                protocol_strict=float(parsed['protocol_strict']),
                total=task + protocol_weight * protocol_core,
                loc_reward=loc_reward, cand_reward=cand_reward, q0=q0, q1=q1,
                set_c_reward=candd['reward'], set_f_reward=setd['reward'],
                delta_refine=delta_refine,
                mask_iou=mask_iou_val, union_iou=union_iou_val,
                raw_iou=raw_set_iou, set_iou=raw_set_iou, set_giou=set_giou_val,
                count_reward=count_val, focus_reward=focus_val, cand_coverage=cand_cov_val,
                discrim_correct=discrim_correct, imagine_correct=imagine_correct,
                verify_signal=discrim_weight * discrim_correct + imagine_weight * imagine_correct,
                verify_action=verify_action, imagine_action=imagine_action,
                selected_action=parsed.get('selected_action'),
                pre_observation_prediction=parsed.get('pre_observation_prediction', ''),
                predicted_effect=parsed.get('predicted_effect'),
                objective_verdict=obj_verdict, refine_verdict=refine_obj,
                correct=bool(correct), matched_pairs=setd['matched_pairs'], s_sum=setd['s_sum'],
                candidate_tp=candidate_diag['tp'], candidate_fp=candidate_diag['fp'],
                candidate_fn=candidate_diag['fn'], final_tp=final_diag['tp'],
                final_fp=final_diag['fp'], final_fn=final_diag['fn'],
                final_set_complete=bool(parsed['task_valid'] and final_diag['complete']),
                false_keep=bool(imagine_action == 'keep' and not candidate_diag['complete']),
                planner_signal=planner_signal, confirm_signal=confirm_signal,
                gain_error=gain_error, predicted_gain=predicted_gain,
                observation_cost=observation_cost,
                state_q0=state_q0, state_q1=state_q1,
                oracle_action=oracle_action, oracle_gain=oracle_gain,
                planner_regret=(None if oracle_gain is None
                                else float(oracle_gain) - float(state_q1 - state_q0)))
