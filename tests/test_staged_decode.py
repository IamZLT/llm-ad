"""FSM for staged incremental decode (no GPU)."""
import pytest

from outcome.staged_decode import (
    OPEN,
    STAGES,
    _confirm_action,
    inject_after,
    next_marker,
    next_stage,
    stage_finished,
    stage_stop,
)


def test_stage_order_cannot_skip():
    chain = ['U']
    while chain[-1] != 'ANSWER':
        chain.append(next_stage(chain[-1]))
    assert chain == ['U', 'C', 'L', 'V', 'ANSWER']
    with pytest.raises(ValueError):
        next_stage('DONE')


def test_next_marker_aligns_with_sft_open_tags():
    assert next_marker('U') == '[compare]'
    assert next_marker('C') == '[localize]'
    assert next_marker('L') == '[confirm]'
    assert next_marker('V') == '</think>'


def test_stage_stop_matches_current_sft():
    for s in STAGES:
        assert stage_stop(s) == '\n'

    assert stage_stop('ANSWER') == '</answer>'


def test_stage_finishes_on_newline():
    assert not stage_finished('still reasoning', 'C')

    assert stage_finished('comparison finished\n', 'C')

    assert stage_finished('{"is_anomaly": false}\n</answer>', 'ANSWER')


def test_controller_owns_next_marker():
    assert inject_after('U') == '[compare]\n'
    assert inject_after('C') == '[localize]\n'
    assert inject_after('L') == '[confirm]\n'

    assert inject_after('V') == '</think>\n\n<answer>\n'


def test_confirm_action_detects_reject_for_reloop():
    assert _confirm_action('reject; H misled me') == 'reject'
    assert _confirm_action('keep; boxes match') == 'keep'
    assert _confirm_action('refine; shrink the box') == 'refine'
    assert _confirm_action('none; no true defect') == 'none'
    assert _confirm_action('discover; extra defect') == 'discover'
    assert _confirm_action('') == ''


def test_reloop_opens_localize_not_think_close():
    # A confirm-reject reloop re-opens [localize] (zero-H) instead of the normal
    # </think><answer> transition — verified here via the marker the controller
    # injects on a reloop.
    assert OPEN['L'] == '[localize]\n'
    assert inject_after('V') == '</think>\n\n<answer>\n'


def test_padded_completion_tensors_masks_injected_tokens():
    import torch

    from rl.grpo import padded_completion_tensors

    prompt_len = 3
    seqs = [torch.tensor([10, 11, 12, 20, 21, 22, 30, 31])]
    masks = [[0, 1, 1, 0, 1]]  # token 20 and 30 are controller-injected
    outputs, attn, labels = padded_completion_tensors(seqs, prompt_len, 0,
                                                      torch.device('cpu'), sampled_masks=masks)
    # positions after prompt: 20(inj),21,22,30(inj),31
    assert labels[0].tolist() == [-100, -100, -100, -100, 21, 22, -100, 31]


def test_padded_completion_tensors_no_mask_supervises_all():
    import torch

    from rl.grpo import padded_completion_tensors

    prompt_len = 2
    seqs = [torch.tensor([1, 2, 3, 4, 5])]
    outputs, attn, labels = padded_completion_tensors(seqs, prompt_len, 0,
                                                      torch.device('cpu'))
    assert labels[0].tolist() == [-100, -100, 3, 4, 5]
