"""Qwen native internal CoT for outcome-multibox.

When thinking is on, ``prompt.enable_thinking`` opens ``<think>`` in the chat
template. Visible output is still only ``<answer>`` JSON. Stages live *inside*
the think body as ``[understand] [compare] [localize] [imagine] [confirm]`` — not
as extra XML blocks. ``[imagine]`` is the world-model rehearsal step: the model
predicts whether its candidate box matches the defect (keep/refine/reject/none).
``[confirm]`` evaluates the refine loop itself — whether refining improved the
final box relative to the first candidate (improved/unchanged/degraded) — so the
two stages stay orthogonally useful instead of echoing each other. SFT supervises
those headers plus the JSON; RL scores ``[localize]`` boxes with the candidate
rewards, calibrates ``[imagine]`` against the objective box verdict, and
calibrates ``[confirm]`` against the sign of ``delta_refine``.
Completions look like ``[understand]...[confirm]</think>\\n\\n<answer>JSON</answer>``.
"""
from __future__ import annotations

import re

THINK_STAGES = ('understand', 'compare', 'localize', 'imagine', 'confirm')
# A (localize, imagine, confirm) triple is one world-model "predict → rehearse →
# decide" round; multi-round trajectories append more triples.
ROUND_STAGES = ('localize', 'imagine', 'confirm')
UNSUPERVISED_STAGES = ('understand', 'compare')
STAGE_CANON = {
    'understand': 'understand',
    'compare': 'compare',
    'localize': 'localize',
    'imagine': 'imagine',
    'confirm': 'confirm',
}
STAGE_HEADER_RE = re.compile(
    r'\[(understand|compare|localize|imagine|confirm)\]',
    re.I,
)
THINK_BLOCK_RE = re.compile(r'<think>(.*?)</think>', re.S | re.I)
BOX_LEAK_RE = re.compile(
    r'candidate_bboxes_2d|candidate_bbox_2d|\[\s*\d+(?:\.\d+)?\s*,\s*\d+',
    re.I,
)
# Localization signal spans: the coordinates the model actually writes. These are
# the tokens whose quality is measured directly by ``loc_reward`` (DCLR), so RL
# gives them their own per-token advantage instead of averaging them in with the
# long prose/confirm text (which would dilute their gradient ~10x).
LOCALIZE_SPAN_RE = re.compile(
    r'\[localize\](.*?)(?=\[imagine\]|\[confirm\]|\</think\>|<answer>)',
    re.S | re.I,
)
BBOX_VAL_RE = re.compile(r'"bboxes_2d"\s*:\s*([^"]*?)(?=,\s*"|\s*})', re.S)
_PLACEHOLDER_BODY = re.compile(r'^[\s.。…\-–—]*$')
SFT_THINK_PLACEHOLDER = '...'
THINK_CLOSE_PREFIX = '</think>\n\n'
ANSWER_OPEN_RE = re.compile(r'<answer>', re.I)


def thinking_enabled(cfg) -> bool:
    think = (cfg.get('outcome') or {}).get('thinking') or {}
    return bool(think.get('enabled', False))


def visible_sft_target(answer_json: str) -> str:
    """Close native think, then JSON. Used when think content is unlabeled."""
    return f'{THINK_CLOSE_PREFIX}<answer>\n{answer_json}\n</answer>'


def staged_sft_target(understand, compare, localize, imagine, confirm, answer_json: str,
                      extra_rounds=()) -> str:
    """Think continuation: stage headers, close think, then JSON.

    The chat template already opened ``<think>``, so this must not emit a second
    opening tag. ``[localize]`` / ``[imagine]`` / ``[confirm]`` are the world-model
    "predict → rehearse → decide" triple, not XML.

    ``[imagine]`` is the internal rehearsal step: the model predicts whether the
    candidate box it just wrote actually matches the defect (keep / refine / reject /
    none). ``[confirm]`` then evaluates the refine loop's net effect — whether the
    final box improved over the first candidate (improved / unchanged / degraded) —
    so the two stages answer different questions. This is the LLM's own lookahead:
    it reasons about the box's correctness and its own refinement from the image
    features already in its hidden state, with no external zoom.

    ``extra_rounds`` is a list of ``(localize, imagine, confirm)`` triples appended
    after the first round, encoding reject→relocalize / refine→refine multi-round
    trajectories (P0 intermediate states) so the model learns to re-localize inside
    one pass.

    The whole chain (stage headers included) is supervised token-by-token so the
    model learns to write the full thought chain in one autoregressive pass — the
    world-model "internal thinking" formulation, replacing the external FSM that
    used to inject the ``[stage]`` markers itself.
    """
    body = (
        f'[understand]\n{understand}\n'
        f'[compare]\n{compare}\n'
    )
    for loc, img, conf in [(localize, imagine, confirm)] + list(extra_rounds):
        body += f'[localize]\n{loc}\n[imagine]\n{img}\n[confirm]\n{conf}\n'
    return f'{body}{THINK_CLOSE_PREFIX}<answer>\n{answer_json}\n</answer>'


def whole_target_labels(tokenizer, target_text: str) -> list:
    """Token labels for a full (single-pass) SFT target: supervise everything.

    Unlike the old staged scheme, nothing is masked inside the target — the model
    is trained to emit the complete ``[understand]...[confirm]</think><answer>``
    chain itself, so every token (headers, bodies, JSON, closing tags) is a label.
    """
    enc = tokenizer(target_text, add_special_tokens=False, return_offsets_mapping=True)
    ids, _ = _encoding_fields(enc)
    return list(ids)


def thinking_trace(text: str) -> tuple:
    """Split a completion into (cot, visible_answer) for logs / TensorBoard.

    Prefers a closed ``</think>``. If the model skipped the close tag (common
    after answer-only SFT), the prose before ``<answer>`` is still the CoT.
    """
    text = text or ''
    think, after = split_native_think(text)
    if think or re.search(r'</think>', text, re.I):
        return think, (after or '').strip()
    opened = re.search(r'<answer>', text, re.I)
    if opened:
        return text[:opened.start()].strip(), text[opened.start():].strip()
    return '', text.strip()


def visible_answer_text(target: str) -> str:
    """TB / logs: show the JSON answer, not the think-close prefix."""
    m = ANSWER_OPEN_RE.search(target or '')
    if not m:
        return target or ''
    return (target or '')[m.start():]


def think_block_for_sft(localize=None, imagine=None, confirm=None) -> str:
    """Wrapped staged think (parser tests). Training uses ``staged_sft_target``."""
    parts = [
        '<think>',
        '[understand]', SFT_THINK_PLACEHOLDER,
        '[compare]', SFT_THINK_PLACEHOLDER,
        '[localize]', localize if localize is not None else SFT_THINK_PLACEHOLDER,
        '[imagine]', imagine if imagine is not None else SFT_THINK_PLACEHOLDER,
        '[confirm]', confirm if confirm is not None else SFT_THINK_PLACEHOLDER,
        '</think>',
    ]
    return '\n'.join(parts) + '\n'


def split_native_think(text: str) -> tuple:
    """Split a completion into (think_body, after_think).

    Qwen leaves the opening ``<think>`` in the prompt, so rollouts often start
    inside the think body and only emit ``</think>``.
    """
    text = text or ''
    wrapped = THINK_BLOCK_RE.search(text)
    if wrapped:
        return wrapped.group(1).strip(), text[wrapped.end():]
    close = re.search(r'</think>', text, re.I)
    if close:
        return text[:close.start()].strip(), text[close.end():]
    return '', text


def parse_think_stages(think_text: str) -> dict:
    """Parse ``[understand] ... [compare] ...`` inside a think body.

    Returns headers, per-stage bodies (last occurrence), first-occurrence bodies,
    and whether the five stages appear exactly once in the required order (with
    optional extra localize/imagine/confirm triples for multi-round trajectories).
    """
    empty = dict(ok=False, filled=False, no_early_boxes=True, headers=[], bodies={},
                 first_bodies={})
    text = think_text or ''
    hits = list(STAGE_HEADER_RE.finditer(text))
    if not hits:
        return empty
    headers = []
    bodies = {}
    first_bodies = {}
    for i, m in enumerate(hits):
        canon = STAGE_CANON[m.group(1).lower()]
        headers.append(canon)
        start = m.end()
        end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        body = text[start:end].strip()
        if canon not in first_bodies:
            first_bodies[canon] = body
        bodies[canon] = body
    ok = headers == list(THINK_STAGES)
    if not ok:
        # Multi-round: a reject/refine adds extra (localize, imagine, confirm) triples,
        # so headers may be U,C,(L,I,V)*. The `bodies` dict keeps the LAST occurrence
        # of each stage (the final re-localization result), which is what RL scores.
        base = headers[:len(THINK_STAGES)] == list(THINK_STAGES)
        extras = headers[len(THINK_STAGES):]
        n_rounds = len(extras) // len(ROUND_STAGES)
        ok = (base and len(extras) % len(ROUND_STAGES) == 0
              and all(extras[i::len(ROUND_STAGES)] == [s] * n_rounds
                      for i, s in enumerate(ROUND_STAGES)))
    filled = ok and all(not _PLACEHOLDER_BODY.match(bodies.get(s, '')) for s in THINK_STAGES)
    no_early = True
    if ok:
        for stage in ('understand', 'compare'):
            if BOX_LEAK_RE.search(bodies.get(stage, '')):
                no_early = False
                break
    return dict(ok=ok, filled=filled, no_early_boxes=no_early, headers=headers,
                bodies=bodies, first_bodies=first_bodies)


def think_body_char_spans(target_text: str) -> list:
    """Character spans to leave unsupervised: everything before ``<answer>``."""
    text = target_text or ''
    m = ANSWER_OPEN_RE.search(text)
    if not m or m.start() == 0:
        return []
    return [(0, m.start())]


def _encoding_fields(enc):
    ids = enc['input_ids'] if isinstance(enc, dict) else enc.input_ids
    if hasattr(ids, 'tolist'):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    offs = enc.get('offset_mapping') if isinstance(enc, dict) else getattr(enc, 'offset_mapping', None)
    if offs is not None and hasattr(offs, 'tolist'):
        offs = offs.tolist()
    if offs and isinstance(offs[0], (list, tuple)) and offs[0] and isinstance(offs[0][0], (list, tuple)):
        offs = offs[0]
    return list(ids), offs


def unsupervised_think_labels(tokenizer, target_text: str) -> list:
    """Token labels for a full SFT target: -100 before ``<answer>``, ids after."""
    try:
        enc = tokenizer(target_text, add_special_tokens=False, return_offsets_mapping=True)
    except TypeError:
        enc = tokenizer(target_text, add_special_tokens=False)
    ids, offs = _encoding_fields(enc)
    labels = list(ids)
    spans = think_body_char_spans(target_text)
    if not spans:
        return labels
    if offs:
        for i, pair in enumerate(offs):
            s, e = int(pair[0]), int(pair[1])
            if e <= s:
                continue
            for cs, ce in spans:
                if s < ce and e > cs:
                    labels[i] = -100
                    break
        return labels
    for cs, ce in spans:
        n0 = len(tokenizer(target_text[:cs], add_special_tokens=False).input_ids)
        n1 = len(tokenizer(target_text[:ce], add_special_tokens=False).input_ids)
        for i in range(n0, min(n1, len(labels))):
            labels[i] = -100
    return labels


def loc_char_spans(text: str) -> list:
    """Character spans carrying localization signal (the actual coordinates).

    Two sources: (1) every ``[localize]`` candidate-box line inside the think
    body, and (2) the final ``bboxes_2d`` value in the ``<answer>`` JSON. Returns
    ``(start, end)`` char offsets into ``text`` (decoded completion).
    """
    spans = []
    for m in LOCALIZE_SPAN_RE.finditer(text or ''):
        spans.append((m.start(), m.end()))
    for m in BBOX_VAL_RE.finditer(text or ''):
        spans.append((m.start(), m.end()))
    return spans


def loc_token_mask(tokenizer, text: str) -> list:
    """Per-token 0/1 mask over ``text`` (decoded completion): 1 = localization token.

    Re-tokenizes the decoded completion with char offsets and marks any token whose
    char span overlaps a localization span. Mirrors ``unsupervised_think_labels``.
    Falls back to all-zero (no per-token split) when offsets are unavailable.
    """
    text = text or ''
    try:
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    except TypeError:
        return []
    ids, offs = _encoding_fields(enc)
    mask = [0] * len(ids)
    spans = loc_char_spans(text)
    if not spans or not offs:
        return mask
    for i, pair in enumerate(offs):
        s, e = int(pair[0]), int(pair[1])
        if e <= s:
            continue
        for cs, ce in spans:
            if s < ce and e > cs:
                mask[i] = 1
                break
    return mask


IMAGINE_SPAN_RE = re.compile(
    r'\[imagine\](.*?)(?=\[confirm\]|\</think\>|<answer>)',
    re.S | re.I,
)
CONFIRM_SPAN_RE = re.compile(
    r'\[confirm\](.*?)(?=\</think\>|<answer>)',
    re.S | re.I,
)


def verify_char_spans(text: str) -> list:
    """Character spans carrying the rehearsal/verification signal.

    The ``[imagine]`` (box-quality rehearsal) and ``[confirm]`` (refine-loop
    verdict) bodies. These tokens get a dedicated calibration advantage so the
    "is my box right?" / "did my fix help?" reasoning is not averaged away by the
    long prose. Returns ``(start, end)`` char offsets.
    """
    spans = []
    for m in IMAGINE_SPAN_RE.finditer(text or ''):
        spans.append((m.start(), m.end()))
    for m in CONFIRM_SPAN_RE.finditer(text or ''):
        spans.append((m.start(), m.end()))
    return spans


def verify_token_mask(tokenizer, text: str) -> list:
    """Per-token 0/1 mask: 1 = imagine/confirm reasoning token."""
    text = text or ''
    try:
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    except TypeError:
        return []
    ids, offs = _encoding_fields(enc)
    mask = [0] * len(ids)
    spans = verify_char_spans(text)
    if not spans or not offs:
        return mask
    for i, pair in enumerate(offs):
        s, e = int(pair[0]), int(pair[1])
        if e <= s:
            continue
        for cs, ce in spans:
            if s < ce and e > cs:
                mask[i] = 1
                break
    return mask
