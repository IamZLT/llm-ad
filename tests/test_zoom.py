"""Unit tests for two-scale H + original-image crop helpers."""
from PIL import Image

from types import SimpleNamespace

from outcome.inputs import (
    OutcomeCollator,
    crop_original_by_box1000,
    format_region_hints,
    pixel_budget,
    zoom_prompt_suffix,
)


def test_crop_center_box_stays_inside_and_expands():
    img = Image.new('RGB', (1000, 1000), color=(10, 20, 30))
    crop = crop_original_by_box1000(img, [400, 400, 600, 600], expand=1.5)
    assert crop is not None
    # 200px box * 1.5 = 300px, centered at 500 → [350, 650]
    assert crop.size == (300, 300)


def test_crop_clamps_to_image_border():
    img = Image.new('RGB', (200, 100), color=0)
    crop = crop_original_by_box1000(img, [0, 0, 100, 200], expand=2.0)
    assert crop is not None
    w, h = crop.size
    assert w <= 200 and h <= 100
    assert w >= 2 and h >= 2


def test_crop_rejects_degenerate_box():
    img = Image.new('RGB', (64, 64), color=0)
    assert crop_original_by_box1000(img, None) is None
    assert crop_original_by_box1000(img, [10, 10]) is None


def test_zoom_suffix_empty_when_no_crops():
    assert zoom_prompt_suffix(0) == ''
    assert zoom_prompt_suffix(-1) == ''


def test_zoom_suffix_mentions_full_image_coords():
    one = zoom_prompt_suffix(1)
    assert 'Image 3' in one
    assert '[0,1000]' in one
    three = zoom_prompt_suffix(3)
    assert 'Images 3-5' in three
    assert 'FULL IMAGE' in three


def test_format_region_hints_includes_peak_coords():
    text = format_region_hints([dict(peak_2d=[512.4, 380.6])], 1)
    assert '<|region|>' in text
    assert '@[512,381]' in text
    assert format_region_hints([], 1) == '<|region|>'


def test_format_region_hints_groups_extent_tokens_after_peak():
    props = [dict(id='h1', peak_2d=[100, 200]), dict(id='h2', peak_2d=[300, 400])]
    text = format_region_hints(props, 5, owners=[0, 0, 0, 1, 1])
    assert text == 'h1:<|region|>@[100,200] <|region|> <|region|> h2:<|region|>@[300,400] <|region|>'


def test_2b_prompt_does_not_embed_extra_region_special_tokens():
    """Instruction text must not contain <|region|> besides {region_tokens}.

    An extra literal copy makes collate raise region tokenization mismatch.
    """
    from pathlib import Path
    import yaml
    from outcome.protocol import render_prompt

    cfg = yaml.safe_load(Path('configs/qwen35_2b_outcome_multibox.yaml').read_text())
    tmpl = cfg['prompt']['template']
    assert tmpl.count('<|region|>') == 0
    rendered = render_prompt(cfg, 'bottle', region_tokens='h1:<|region|>@[10,20] <|region|>')
    assert rendered.count('<|region|>') == 2


def test_zoom_defaults_off_when_key_missing():
    dummy = SimpleNamespace(cfg={'outcome': {}})
    assert OutcomeCollator._zoom_cfg(dummy) == {}
    dummy.cfg = {'outcome': {'zoom': {'enabled': False, 'h_image_size': 768}}}
    assert OutcomeCollator._zoom_cfg(dummy)['enabled'] is False


def test_pixel_budget_restores_processor_caps():
    img = SimpleNamespace(min_pixels=100, max_pixels=200, size={'shortest_edge': 100, 'longest_edge': 200})
    proc = SimpleNamespace(image_processor=img)
    with pixel_budget(proc, 768):
        assert img.max_pixels == 768 * 768
        assert img.size['longest_edge'] == 768 * 768
    assert img.min_pixels == 100
    assert img.max_pixels == 200
    assert img.size == {'shortest_edge': 100, 'longest_edge': 200}
