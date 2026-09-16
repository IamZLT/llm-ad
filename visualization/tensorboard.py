"""PIL + TensorBoard helpers shared by GRPO and outcome visualizers."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from utils.common import is_main_process


def _font(size=14):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def pil_to_tb(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB")).transpose(2, 0, 1)


def orig_box_to_resized(box, orig_wh, dest_wh):
    ow, oh = float(orig_wh[0]), float(orig_wh[1])
    dw, dh = float(dest_wh[0]), float(dest_wh[1])
    return [box[0] * dw / ow, box[1] * dh / oh, box[2] * dw / ow, box[3] * dh / oh]


def _resize_to_height(im: Image.Image, height: int) -> Image.Image:
    im = im.convert("RGB")
    if im.size[1] == height:
        return im
    w = max(1, int(round(im.size[0] * height / max(im.size[1], 1))))
    return im.resize((w, height), Image.Resampling.BILINEAR)


def _caption_bar(text: str, width: int, height: int = 22) -> Image.Image:
    bar = Image.new("RGB", (width, height), (28, 28, 28))
    draw = ImageDraw.Draw(bar)
    draw.text((4, 3), text, fill=(230, 230, 230), font=_font(12))
    return bar


def hstack_labeled(parts: Sequence[Tuple[str, Image.Image]], height: int = 224) -> Image.Image:
    cols = []
    for label, im in parts:
        body = _resize_to_height(im, height)
        cols.append(Image.fromarray(np.vstack([np.asarray(_caption_bar(label, body.size[0])), np.asarray(body)])))
    if not cols:
        return Image.new("RGB", (height, height), (0, 0, 0))
    h = max(c.size[1] for c in cols)
    padded = []
    for c in cols:
        if c.size[1] < h:
            canvas = Image.new("RGB", (c.size[0], h), (0, 0, 0))
            canvas.paste(c, (0, 0))
            c = canvas
        padded.append(c)
    return Image.fromarray(np.hstack([np.asarray(c) for c in padded]))


def vstack_labeled(rows: Sequence[Tuple[str, Image.Image]], width: Optional[int] = None) -> Image.Image:
    if not rows:
        return Image.new("RGB", (224, 224), (0, 0, 0))
    width = width or max(im.size[0] for _, im in rows)
    stacked = []
    for title, im in rows:
        im = im.convert("RGB")
        if im.size[0] != width:
            h = max(1, int(round(im.size[1] * width / max(im.size[0], 1))))
            im = im.resize((width, h), Image.Resampling.BILINEAR)
        stacked.append(_caption_bar(title, width, height=24))
        stacked.append(im)
    return Image.fromarray(np.vstack([np.asarray(x) for x in stacked]))


def _as_pil(heat, size) -> Image.Image:
    if isinstance(heat, Image.Image):
        im = heat.convert("RGB")
        return im if im.size == size else im.resize(size, Image.Resampling.BILINEAR)
    if torch.is_tensor(heat):
        from models.anomaly_prior import heatmap_to_pil
        return heatmap_to_pil(heat, size)
    arr = np.asarray(heat)
    if arr.ndim == 2:
        lo, hi = float(arr.min()), float(arr.max())
        norm = np.zeros_like(arr, dtype=np.float32) if hi - lo < 1e-8 else (arr - lo) / (hi - lo)
        rgb = np.stack([norm, np.zeros_like(norm), 1.0 - norm], axis=-1)
        im = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), mode="RGB")
        return im if im.size == size else im.resize(size, Image.Resampling.BILINEAR)
    raise TypeError(f"unsupported heatmap type: {type(heat)}")


def make_heatmap_panel(ref, test, heat, alpha: float = 0.45, prior_points=None) -> Image.Image:
    test_rgb = test.convert("RGB")
    heat_rgb = _as_pil(heat, test_rgb.size)
    overlay = Image.blend(test_rgb, heat_rgb, float(np.clip(alpha, 0.0, 1.0)))
    if prior_points:
        draw = ImageDraw.Draw(overlay)
        w, h = overlay.size
        for pt in prior_points:
            if pt is None or len(pt) < 2:
                continue
            x, y = float(pt[0]) * w / 1000.0, float(pt[1]) * h / 1000.0
            r = 5
            draw.ellipse([x - r, y - r, x + r, y + r], outline=(255, 255, 0), width=2)
    return hstack_labeled([("REF", ref), ("TEST", test), ("H", overlay)])


def _free_tensorboard_port(port: int) -> None:
    """Kill leftover tensorboard servers bound to ``port`` (typically a previous SFT run)."""
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
    except Exception:
        return
    for line in out.splitlines():
        low = line.lower()
        if "tensorboard" not in low:
            continue
        if f"--port {port}" not in line and f"--port={port}" not in line:
            continue
        try:
            pid = int(line.split(None, 1)[0])
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, 15)
            print(f"[tensorboard] 释放端口 {port}：已停止旧实例 pid={pid}", flush=True)
        except ProcessLookupError:
            pass
        except PermissionError:
            print(f"[tensorboard] 无法停止 pid={pid}（无权限），端口 {port} 可能仍被占用", flush=True)


def start_tensorboard(logdir, cfg: dict, default_port: int = 6006) -> None:
    """Start TensorBoard on cfg port, replacing any previous instance on that port."""
    if not is_main_process():
        return
    tb_cfg = cfg.get("tensorboard") or {}
    port = int(tb_cfg.get("port", default_port))
    host = str(tb_cfg.get("host", "0.0.0.0"))
    logdir = Path(logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    _free_tensorboard_port(port)
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "tensorboard.main",
             "--logdir", str(logdir), "--port", str(port), "--host", host],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
        )
        print(f"[tensorboard] 已启动 pid={proc.pid} http://{host}:{port} (logdir={logdir})", flush=True)
    except Exception as exc:
        print(f"[tensorboard] 启动失败: {exc}", flush=True)
