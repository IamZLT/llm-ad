#!/usr/bin/env python3
"""Browse ANNOS paper boxes and run the local outcome-multibox model."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from flask import Flask, jsonify, render_template, request, send_file

from viewer.annos_index import (
    DATASETS,
    SOURCE_MAP,
    list_classes,
    list_local_classes,
    list_local_samples,
    list_samples,
    list_sources,
    sample_detail,
)

app = Flask(__name__, template_folder="templates")
app.config["JSON_SORT_KEYS"] = False
app.config["TEMPLATES_AUTO_RELOAD"] = True

_LOCK = threading.Lock()
_MODEL = None
_LOADED = {}

DEFAULT_CFG = ROOT / "configs" / "qwen35_2b_outcome_multibox.yaml"
DEFAULT_ADAPTER = ROOT / "outputs/train/qwen35_2b_outcome_multibox/train_hybrid_rl3/checkpoint-1000"


def _safe_file(path: str) -> Path:
    p = Path(path).resolve()
    allowed = [DATASETS.resolve(), (ROOT / "outputs").resolve()]
    if not any(p == a or a in p.parents for a in allowed):
        raise ValueError("path outside datasets/outputs")
    if not p.is_file():
        raise FileNotFoundError(str(p))
    return p


def gpu_status() -> list[dict]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=8,
        )
    except Exception:
        return []
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        used, total, free = float(parts[2]), float(parts[3]), float(parts[4])
        rows.append(
            dict(
                index=int(parts[0]),
                name=parts[1],
                used_mb=used,
                total_mb=total,
                free_mb=free,
            )
        )
    return rows


def pick_gpu(min_free_mb: float = 18000) -> dict:
    gpus = gpu_status()
    if not gpus:
        return dict(index=None, warning="未检测到 NVIDIA GPU，将尝试 CPU（会非常慢）")
    best = max(gpus, key=lambda g: g["free_mb"])
    if best["free_mb"] < min_free_mb:
        best = dict(
            best,
            warning=(
                f"GPU {best['index']} 仅剩 {best['free_mb']:.0f} MB；"
                "训练占用中时推理可能 OOM。浏览标注不需要 GPU。"
            ),
        )
    return best


def list_adapters() -> list[dict]:
    rows = []
    hybrid = ROOT / "outputs/train/qwen35_2b_outcome_multibox/train_hybrid_rl3"
    if hybrid.is_dir():
        ckpts = []
        for p in hybrid.glob("checkpoint-*"):
            if (p / "adapter_config.json").exists():
                try:
                    step = int(p.name.split("-")[-1])
                except ValueError:
                    continue
                ckpts.append((step, p))
        for step, p in sorted(ckpts):
            tag = " 推荐" if step == 1000 else ""
            rows.append(dict(id=str(p), label=f"hybrid {p.name}{tag}"))
    old = ROOT / "outputs/train/qwen35_2b_outcome_multibox/train_20260913_053232_599794"
    for n in (1100, 1200, 400, 200):
        p = old / f"checkpoint-{n}"
        if (p / "adapter_config.json").exists():
            rows.append(dict(id=str(p), label=f"old 2B {p.name}"))
    return rows


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/meta")
def api_meta():
    adapters = list_adapters()
    default = str(DEFAULT_ADAPTER) if DEFAULT_ADAPTER.exists() else (adapters[0]["id"] if adapters else "")
    return jsonify(
        dict(
            sources=list_sources(),
            adapters=adapters,
            default_adapter=default,
            gpus=gpu_status(),
            model_loaded=bool(_LOADED.get("adapter")),
            loaded_adapter=_LOADED.get("adapter") or "",
        )
    )


@app.get("/api/gpu")
def api_gpu():
    return jsonify(dict(gpus=gpu_status(), pick=pick_gpu(), loaded=_LOADED))


@app.get("/api/classes")
def api_classes():
    source = request.args.get("source", "mvtec")
    mode = request.args.get("mode", "annos")
    if mode == "local":
        if source not in SOURCE_MAP:
            return jsonify(dict(error="该数据源没有本地图像，无法用模型测试")), 400
        names = list_local_classes(source)
    else:
        names = list_classes(source)
    return jsonify(dict(classes=names))


@app.get("/api/samples")
def api_samples():
    source = request.args.get("source", "mvtec")
    cls = request.args.get("cls", "")
    mode = request.args.get("mode", "annos")
    if not cls:
        return jsonify(dict(error="need cls")), 400
    if mode == "local":
        if source not in SOURCE_MAP:
            return jsonify(dict(error="该数据源没有本地图像")), 400
        rows = list_local_samples(source, cls)
    else:
        rows = list_samples(source, cls)
    defects = sorted({r["defect"] for r in rows})
    return jsonify(dict(samples=rows, defects=defects))


@app.get("/api/sample")
def api_sample():
    source = request.args.get("source", "mvtec")
    cls = request.args.get("cls", "")
    name = request.args.get("name", "")
    try:
        return jsonify(sample_detail(source, cls, name))
    except FileNotFoundError as e:
        return jsonify(dict(error=str(e))), 404


@app.get("/api/file")
def api_file():
    try:
        path = _safe_file(request.args.get("path", ""))
    except (ValueError, FileNotFoundError) as e:
        return jsonify(dict(error=str(e))), 400
    mime = "image/png"
    suf = path.suffix.lower()
    if suf in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    return send_file(path, mimetype=mime)


def _load_model(adapter: str):
    global _MODEL, _LOADED
    adapter = str(Path(adapter).resolve())
    if _MODEL is not None and _LOADED.get("adapter") == adapter:
        return _MODEL

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        chosen = pick_gpu()
        if chosen.get("index") is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(chosen["index"])
            os.environ["LOCAL_RANK"] = "0"
            os.environ["RANK"] = "0"

    from outcome.engine_multibox import load_model
    from utils.config import load_yaml_config

    if _MODEL is not None:
        import torch

        try:
            del _MODEL["model"]
            del _MODEL["processor"]
            del _MODEL["prior"]
        except Exception:
            pass
        _MODEL = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cfg = load_yaml_config(str(DEFAULT_CFG))
    cfg.setdefault("distributed", {})["num_gpu"] = 1
    model, processor, prior = load_model(cfg, adapter=adapter, fresh_lora=False)
    model.eval()
    _MODEL = dict(model=model, processor=processor, prior=prior, cfg=cfg)
    _LOADED = dict(adapter=adapter, gpu=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    return _MODEL


@app.post("/api/predict")
def api_predict():
    body = request.get_json(force=True, silent=True) or {}
    source = body.get("source", "mvtec")
    cls = body.get("cls") or body.get("class_name")
    name = body.get("name")
    adapter = body.get("adapter") or str(DEFAULT_ADAPTER)
    ref_path = body.get("ref")
    if not (source and cls and name and ref_path):
        return jsonify(dict(error="need source, cls, name, ref")), 400
    if source not in SOURCE_MAP:
        return jsonify(dict(error="该数据源没有本地图像，无法跑模型")), 400
    try:
        detail = sample_detail(source, cls, name)
        img_path = _safe_file(detail["image"])
        ref_path = _safe_file(ref_path)
    except (ValueError, FileNotFoundError, TypeError) as e:
        return jsonify(dict(error=str(e))), 400
    if img_path.resolve() == ref_path.resolve():
        return jsonify(dict(error="参考图必须与检测图不同")), 400

    from PIL import Image

    from outcome.inputs_multibox import OutcomeMultiboxCollator
    from outcome.policy import generate_group
    from outcome.protocol_multibox import parse_output_cfg, to_pixels
    from rl.grpo import move_batch

    with _LOCK:
        try:
            pack = _load_model(adapter)
        except Exception as e:
            return jsonify(dict(error=f"加载模型失败: {e}")), 500
        test = Image.open(img_path).convert("RGB")
        ref = Image.open(ref_path).convert("RGB")
        item = dict(
            test=test,
            ref=ref,
            image_path=str(img_path),
            ref_path=str(ref_path),
            orig_size=test.size,
            class_name=cls,
            gt_box_px=None,
            component_bboxes=[],
            num_components=0,
            mask_area_fraction=0.0,
            union_area_fraction=0.0,
            is_anomaly=bool(detail.get("is_anomaly")),
            defect_type=detail.get("defect"),
        )
        cfg = pack["cfg"]
        max_boxes = int(cfg["outcome"].get("max_boxes", 16))
        try:
            batch = move_batch(
                OutcomeMultiboxCollator(pack["processor"], pack["prior"], cfg)([item]),
                next(pack["model"].parameters()).device,
            )
            completion = generate_group(pack["model"], pack["processor"], batch, cfg, group=1, sample=False)[0]
        except Exception as e:
            return jsonify(dict(error=f"推理失败（GPU 占用或 OOM？）: {e}")), 500
        parsed = parse_output_cfg(completion.text, cfg, max_boxes=max_boxes)
        pred_px = [to_pixels(b, test.size) for b in parsed.get("bboxes_2d") or []]
        cand_px = [to_pixels(b, test.size) for b in parsed.get("candidate_bboxes_2d") or []]
        h_px = []
        meta = (batch.get("_meta") or [{}])[0]
        for c in meta.get("prior_candidates") or []:
            box = c.get("bbox_2d")
            if box:
                h_px.append(
                    dict(
                        bbox_px=to_pixels(box, test.size),
                        id=c.get("id"),
                        peak=c.get("peak_2d"),
                    )
                )
        return jsonify(
            dict(
                is_anomaly=parsed.get("is_anomaly"),
                bboxes_2d=parsed.get("bboxes_2d") or [],
                bboxes_px=pred_px,
                candidate_bboxes_2d=parsed.get("candidate_bboxes_2d") or [],
                candidate_px=cand_px,
                h_boxes=h_px,
                verify_action=parsed.get("verify_action"),
                imagine_action=parsed.get("imagine_action"),
                description=(parsed.get("description") or ""),
                text=completion.text,
                stop_reason=completion.stop_reason,
                adapter=adapter,
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5006)
    parser.add_argument("--gpu", default=None, help="Force CUDA_VISIBLE_DEVICES, e.g. 0")
    args = parser.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(f"ANNOS bbox viewer  http://127.0.0.1:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
