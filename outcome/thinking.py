"""Qwen native internal CoT for outcome-multibox.

When thinking is on, ``prompt.enable_thinking`` opens ``<think>`` in the chat
template. Visible output is still only ``<answer>`` JSON. Stages live *inside*
the think body as ``[understand] [compare] [localize] [confirm]`` — not as
extra XML blocks. SFT supervises those headers plus the JSON; RL can then
score ``[localize]`` boxes with the existing candidate rewards.
Completions look like ``[understand]...[confirm]</think>\\n\\n<answer>JSON</answer>``.
"""
from __future__ import annotations

import re

THINK_STAGES = ('understand', 'compare', 'localize', 'confirm')
UNSUPERVISED_STAGES = ('understand', 'compare')
STAGE_CANON = {
    'understand': 'understand',
    'compare': 'compare',
    'localize': 'localize',
    'confirm': 'confirm',
    '理解': 'understand',
    '对比': 'compare',
    '定位': 'localize',
    '确认': 'confirm',
}
STAGE_HEADER_RE = re.compile(
    r'\[(understand|compare|localize|confirm|理解|对比|定位|确认)\]',
    re.I,
)
THINK_BLOCK_RE = re.compile(r'<think>(.*?)</think>', re.S | re.I)
BOX_LEAK_RE = re.compile(
    r'candidate_bboxes_2d|candidate_bbox_2d|\[\s*\d+(?:\.\d+)?\s*,\s*\d+',
    re.I,
)
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


def staged_sft_target(understand, compare, localize, confirm, answer_json: str) -> str:
    """Think continuation: four stage headers, close think, then JSON.

    The chat template already opened ``<think>``, so this must not emit a second
    opening tag. ``[localize]`` / ``[confirm]`` are ground / verify, not XML.
    """
    body = (
        f'[understand]\n{understand}\n'
        f'[compare]\n{compare}\n'
        f'[localize]\n{localize}\n'
        f'[confirm]\n{confirm}\n'
    )
    return f'{body}{THINK_CLOSE_PREFIX}<answer>\n{answer_json}\n</answer>'


STAGED_MARKERS = ('[understand]', '[compare]', '[localize]', '[confirm]',
                  '</think>', '<answer>')


def staged_marker_char_spans(target_text: str) -> list:
    """Character spans of the FSM-controlled markers in a staged SFT target.

    ``outcome.staged_decode`` injects these markers itself; the model never
    samples them. SFT must therefore mask them (label=-100) so teacher-forcing
    matches the staged decoder's per-stage continuation instead of teaching the
    model to write the whole chain in one go (which is what makes it rush to
    ``</think><answer>`` in the first stage).
    """
    text = target_text or ''
    spans = []
    for marker in STAGED_MARKERS:
        start = 0
        while True:
            idx = text.find(marker, start)
            if idx < 0:
                break
            spans.append((idx, idx + len(marker)))
            start = idx + len(marker)
    return spans


def staged_sft_labels(tokenizer, target_text: str) -> list:
    """Token labels for a staged SFT target with FSM markers masked (-100).

    Only the four stage bodies and the JSON answer are supervised; the
    ``[stage]`` headers, ``</think>`` and ``<answer>`` wrappers are masked.
    ``</answer>`` is NOT masked: the model emits it itself to end the answer
    stage (it is the staged decoder's stop string).
    """
    enc = tokenizer(target_text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = _encoding_fields(enc)
    labels = list(ids)
    spans = staged_marker_char_spans(target_text)
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


def think_block_for_sft(localize=None, confirm=None) -> str:
    """Wrapped staged think (parser tests). Training uses ``staged_sft_target``."""
    parts = [
        '<think>',
        '[understand]', SFT_THINK_PLACEHOLDER,
        '[compare]', SFT_THINK_PLACEHOLDER,
        '[localize]', localize if localize is not None else SFT_THINK_PLACEHOLDER,
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

    Returns headers, per-stage bodies, and whether the four stages appear
    exactly once in the required order.
    """
    empty = dict(ok=False, filled=False, no_early_boxes=True, headers=[], bodies={})
    text = think_text or ''
    hits = list(STAGE_HEADER_RE.finditer(text))
    if not hits:
        return empty
    headers = []
    bodies = {}
    for i, m in enumerate(hits):
        canon = STAGE_CANON[m.group(1).lower()]
        headers.append(canon)
        start = m.end()
        end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        bodies[canon] = text[start:end].strip()
    ok = headers == list(THINK_STAGES)
    filled = ok and all(not _PLACEHOLDER_BODY.match(bodies.get(s, '')) for s in THINK_STAGES)
    no_early = True
    if ok:
        for stage in ('understand', 'compare'):
            if BOX_LEAK_RE.search(bodies.get(stage, '')):
                no_early = False
                break
    return dict(ok=ok, filled=filled, no_early_boxes=no_early, headers=headers, bodies=bodies)


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
