"""PIL + TensorBoard helpers shared by GRPO and outcome visualizers."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

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


def draw_case_boxes(image, *, gt_orig=None, pred_orig=None, cand_orig=None, orig_wh=None) -> Image.Image:
    im = image.copy().convert("RGB")
    draw = ImageDraw.Draw(im)
    font = _font(14)
    orig = orig_wh or im.size

    def _one(box, color, label):
        if box is None:
            return
        xy = orig_box_to_resized(box, orig, im.size)
        draw.rectangle(xy, outline=color, width=3)
        draw.text((xy[0] + 3, max(0, xy[1] - 18)), label, fill=color, font=font)

    _one(gt_orig, (0, 220, 0), "GT")
    _one(cand_orig, (255, 165, 0), "Bc")
    _one(pred_orig, (255, 0, 0), "Bf")
    return im


def _fmt_box(box) -> str:
    if box is None:
        return "null"
    return "[" + ",".join(str(int(round(v))) for v in box) + "]"


def format_case_text(*, step, meta, response, parsed, iou, rec_ok, iou_c=0.0) -> str:
    return "\n".join(
        [
            f"step={step}",
            f"image={meta.get('image_path')}",
            f"class={meta.get('class_name')} anomaly_gt={meta.get('is_anomaly')} rec_ok={rec_ok} "
            f"iou_f={iou:.3f} iou_c={iou_c:.3f}",
            f"pred={parsed.get('is_anomaly')} bbox_2d={_fmt_box(parsed.get('bbox_2d'))} "
            f"candidate_bbox_2d={_fmt_box(parsed.get('candidate_bbox_2d'))}",
            f"valid={parsed.get('trajectory_valid')} action={parsed.get('action')}",
            f"description={parsed.get('description') or ''}",
            "",
            response or "",
        ]
    )


def log_heatmap_and_case(
    writer,
    *,
    step: int,
    tag_prefix: str,
    meta: dict,
    response: str,
    parsed: dict,
    iou: float,
    rec_ok: bool,
    overlay_alpha: float = 0.45,
    iou_c: float = 0.0,
    log_heatmap: bool = True,
    log_case: bool = True,
) -> None:
    if writer is None:
        return
    ref, test, heat = meta.get("ref"), meta.get("test"), meta.get("heatmap")
    orig = tuple(meta.get("orig_size") or (test.size if test is not None else (1, 1)))
    if log_heatmap and ref is not None and test is not None and heat is not None:
        panel = make_heatmap_panel(ref, test, heat, alpha=overlay_alpha, prior_points=meta.get("prior_points"))
        writer.add_image(f"{tag_prefix}/1_heatmap", pil_to_tb(panel), step)
    if log_case and test is not None:
        pred = parsed.get("bbox_2d")
        cand = parsed.get("candidate_bbox_2d")
        from reasoning.rewards import qwen1000_to_pixels_strict, valid_bbox_1000
        pred_px = qwen1000_to_pixels_strict(pred, orig) if valid_bbox_1000(pred) else None
        cand_px = qwen1000_to_pixels_strict(cand, orig) if valid_bbox_1000(cand) else None
        vis = draw_case_boxes(
            test, gt_orig=meta.get("gt_box_px"), pred_orig=pred_px, cand_orig=cand_px, orig_wh=orig
        )
        writer.add_image(f"{tag_prefix}/2_bbox", pil_to_tb(vis), step)
    writer.add_text(
        f"{tag_prefix}/3_cot",
        format_case_text(step=step, meta=meta, response=response, parsed=parsed, iou=iou, rec_ok=rec_ok, iou_c=iou_c),
        step,
    )
    writer.flush()


def log_eval_cases_grid(writer, *, step: int, cases: List[dict], overlay_alpha: float = 0.45, save_dir=None, max_cases: int = 16) -> None:
    if writer is None:
        return
    rows, texts = [], []
    for ci, c in enumerate(cases[:max_cases]):
        meta, parsed = c["meta"], c["parsed"]
        ref, test, heat = meta.get("ref"), meta.get("test"), meta.get("heatmap")
        orig = tuple(meta.get("orig_size") or (test.size if test is not None else (1, 1)))
        parts = []
        if ref is not None and test is not None and heat is not None:
            parts.append(("H", make_heatmap_panel(ref, test, heat, alpha=overlay_alpha, prior_points=meta.get("prior_points"))))
        if test is not None:
            from reasoning.rewards import qwen1000_to_pixels_strict, valid_bbox_1000
            pred = parsed.get("bbox_2d")
            cand = parsed.get("candidate_bbox_2d")
            vis = draw_case_boxes(
                test,
                gt_orig=meta.get("gt_box_px"),
                pred_orig=qwen1000_to_pixels_strict(pred, orig) if valid_bbox_1000(pred) else None,
                cand_orig=qwen1000_to_pixels_strict(cand, orig) if valid_bbox_1000(cand) else None,
                orig_wh=orig,
            )
            parts.append(("bbox", vis))
        if parts:
            title = (
                f"#{ci} {meta.get('class_name')} gt_anom={meta.get('is_anomaly')} "
                f"pred={parsed.get('is_anomaly')} iou={float(c.get('iou', 0.0)):.2f}"
            )
            rows.append((title, hstack_labeled(parts)))
        texts.append(
            format_case_text(
                step=step, meta=meta, response=c.get("response", ""), parsed=parsed,
                iou=float(c.get("iou", 0.0)), rec_ok=bool(c.get("rec_ok", False)),
                iou_c=float(c.get("iou_c", 0.0)),
            )
        )
    if rows:
        grid = vstack_labeled(rows)
        writer.add_image("eval/cases_grid", pil_to_tb(grid), step)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            grid.save(os.path.join(save_dir, f"cases_step{int(step):08d}.png"))
    writer.add_text("eval/cases_cot", "\n\n".join(texts), step)
    writer.flush()


def format_grpo_group_text(*, step, meta, texts, details, advantages, logprobs) -> str:
    lines = [
        f"step={step} image={meta.get('image_path')} class={meta.get('class_name')} "
        f"is_anomaly={meta.get('is_anomaly')}",
    ]
    for i, (text, det, adv, lp) in enumerate(zip(texts, details, advantages, logprobs)):
        lines.append(
            f"--- tau[{i}] Rf={float(det.get('R_final', 0)):.3f} adv={float(adv):.3f} lp={float(lp):.3f} "
            f"iou_f={float(det.get('R_iou', 0)):.3f} iou_c={float(det.get('R_iou_c', 0)):.3f}"
        )
        lines.append(text or "")
        lines.append("")
    return "\n".join(lines)


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return float(sum(xs) / len(xs)) if xs else 0.0


def _as_list(x) -> List[float]:
    if x is None:
        return []
    if torch.is_tensor(x):
        return [float(v) for v in x.detach().flatten().cpu().tolist()]
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x]
    return [float(x)]


_DETAIL_TAGS = (
    ("R_ground", "grpo/R_ground"),
    ("R_reason", "grpo/R_reason"),
    ("R_final", "grpo/R_final"),
    ("R_iou_c", "grpo/R_iou_c"),
    ("R_iou", "grpo/R_iou"),
    ("delta_iou", "grpo/delta_iou"),
    ("R_dir", "grpo/R_dir"),
    ("raw_iou_f", "grpo/raw_iou_f"),
    ("raw_iou_c", "grpo/raw_iou_c"),
    ("R_fmt", "grpo/R_fmt"),
    ("R_dense_c", "grpo/R_dense_c"),
    ("R_dense_f", "grpo/R_dense_f"),
    ("delta_dense", "grpo/delta_dense"),
    ("candidate_area_ratio", "grpo/candidate_area_ratio"),
    ("final_area_ratio", "grpo/final_area_ratio"),
    ("pred_gt_area_ratio", "grpo/pred_gt_area_ratio"),
    ("full_image_box", "grpo/full_image_box_rate"),
    ("h_anchor", "grpo/h_anchor"),
    ("h_follow", "grpo/h_follow_rate"),
    ("h_override", "grpo/h_override_rate"),
)

_EXTRA_TAGS = {
    "loss_pg": "grpo/pg_loss",
    "loss_kl": "grpo/kl",
    "rho_mean": "grpo/rho",
    "clip_frac": "grpo/clip_frac",
    "resample": "grpo/resample_n",
    "skipped": "grpo/skip_rate",
    "ref_gap": "grpo/ref_gap",
    "kl_contrib": "grpo/kl_contrib",
    "strict_protocol_rate": "grpo/protocol_rate",
    "strict_trajectory_rate": "protocol/strict_trajectory_rate",
    "strict_answer_rate": "protocol/strict_answer_rate",
    "strict_final_valid_rate": "protocol/strict_final_valid_rate",
    "task_protocol_rate": "grpo/protocol_rate",
    "task_trajectory_rate": "protocol/task_trajectory_rate",
    "task_answer_rate": "protocol/task_answer_rate",
    "task_final_valid_rate": "protocol/task_final_valid_rate",
    "task_candidate_valid_rate": "grpo/candidate_valid_rate",
}

_ALWAYS_ZERO = (
    "grpo/pg_loss",
    "grpo/kl",
    "grpo/rho",
    "grpo/clip_frac",
    "grpo/kl_contrib",
    "grpo/ref_gap",
    "grpo/protocol_rate",
    "grpo/trajectory_valid_rate",
    "grpo/candidate_valid_rate",
    "grpo/final_valid_rate",
    "grpo/box_pair_valid_rate",
    "grpo/unique_response_rate",
    "grpo/resample_n",
    "grpo/skip_rate",
    "protocol/strict_trajectory_rate",
    "protocol/task_trajectory_rate",
    "protocol/strict_answer_rate",
    "protocol/task_answer_rate",
    "protocol/strict_final_valid_rate",
    "protocol/task_final_valid_rate",
)


def log_grpo_scalars(
    writer,
    *,
    step: int,
    loss: float,
    rewards,
    details,
    advantages=None,
    seq_lp=None,
    texts=None,
    lr: float = 0.0,
    params=None,
    grad_norm=None,
    extra=None,
    opt_step=None,
    is_anomaly=None,
    strict_stats=None,
    task_stats=None,
) -> None:
    """Write the allowlisted GRPO scalars. ``params`` and per-tau curves are never logged."""
    if writer is None:
        return
    extra = extra or {}
    details = list(details or [])
    values = {}
    values["grpo/loss"] = float(loss)
    values["grpo/lr"] = float(lr)
    values["grpo/grad_norm"] = float(grad_norm) if grad_norm is not None else 0.0
    r = _as_list(rewards)
    values["grpo/reward_std"] = float(np.std(r)) if len(r) > 1 else 0.0
    for key, tag in _DETAIL_TAGS:
        values[tag] = _mean(float(d.get(key, 0.0) or 0.0) for d in details)
    for tag in _ALWAYS_ZERO:
        values.setdefault(tag, 0.0)
    if texts:
        values["grpo/unique_response_rate"] = len(set(texts)) / max(len(texts), 1)
    if extra.get("kl_contrib") is None and "loss_kl" in extra and "kl_beta" in extra:
        extra = dict(extra)
        extra["kl_contrib"] = float(extra["loss_kl"]) * float(extra["kl_beta"])
    for src, tag in _EXTRA_TAGS.items():
        if src in extra:
            values[tag] = float(extra[src])
    for stats, prefix in ((strict_stats, "strict"), (task_stats, "task")):
        if not stats:
            continue
        mapping = {
            "protocol_rate": "grpo/protocol_rate" if prefix == "strict" else "grpo/protocol_rate",
            "trajectory_valid_rate": f"protocol/{prefix}_trajectory_rate",
            "answer_rate": f"protocol/{prefix}_answer_rate",
            "final_valid_rate": f"protocol/{prefix}_final_valid_rate" if prefix == "strict" else "protocol/task_final_valid_rate",
            "candidate_valid_rate": "grpo/candidate_valid_rate",
            "box_pair_valid_rate": "grpo/box_pair_valid_rate",
            "unique_response_rate": "grpo/unique_response_rate",
        }
        for k, tag in mapping.items():
            if k in stats:
                values[tag] = float(stats[k])
    for tag, value in values.items():
        writer.add_scalar(tag, float(value), step)
    writer.flush()


def tensorboard_event_dir(output_dir) -> str:
    return str(Path(output_dir) / "tb")


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


def auto_start_tensorboard(cfg: dict, output_dir) -> None:
    start_tensorboard(tensorboard_event_dir(output_dir), cfg)


def log_grpo_run_config(writer, cfg: dict) -> None:
    if writer is None:
        return
    import json
    writer.add_text("grpo/0_config", json.dumps(cfg, ensure_ascii=False, indent=2, default=str), 0)
    writer.flush()
