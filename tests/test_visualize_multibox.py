"""Tests for class-stratified eval-case selection in outcome.visualize_multibox."""
from __future__ import annotations

from outcome.visualize_multibox import stratify_cases_by_class


def _case(cls):
    return {'meta': {'class_name': cls}, 'parsed': {}}


def test_round_robin_covers_every_class():
    # cases arrive grouped by class, like stratified_eval_indices yields
    cases = [_case('bottle')] * 13 + [_case('cable')] * 13 + [_case('capsule')] * 13
    picked = stratify_cases_by_class(cases, 16)
    assert len(picked) == 16
    classes = [c['meta']['class_name'] for c in picked]
    # every class appears, and the head is no longer a bottle run
    assert set(classes) == {'bottle', 'cable', 'capsule'}
    assert classes[:3] == ['bottle', 'cable', 'capsule']


def test_each_class_represented_when_slots_exceed_classes():
    classes = [f'cls{i:02d}' for i in range(15)]  # MVTec-like
    cases = []
    for cls in classes:
        cases.extend(_case(cls) for _ in range(13))
    picked = stratify_cases_by_class(cases, 16)
    assert len(picked) == 16
    assert len({c['meta']['class_name'] for c in picked}) == 15


def test_fewer_slots_than_classes_takes_first_sorted():
    cases = [_case('b')] * 5 + [_case('a')] * 5 + [_case('c')] * 5
    picked = stratify_cases_by_class(cases, 2)
    assert [c['meta']['class_name'] for c in picked] == ['a', 'b']


def test_empty_and_zero():
    assert stratify_cases_by_class([], 16) == []
    assert stratify_cases_by_class([_case('a')], 0) == []


def test_leftover_slots_fill_after_full_round():
    cases = [_case('a')] * 3 + [_case('b')] * 1
    picked = stratify_cases_by_class(cases, 4)
    assert len(picked) == 4
    assert [c['meta']['class_name'] for c in picked].count('a') == 3
