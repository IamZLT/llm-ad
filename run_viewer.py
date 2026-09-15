#!/usr/bin/env python3
"""Start the ANNOS bbox viewer + local model test bench."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def main():
    parser = argparse.ArgumentParser(description="ANNOS bbox 可视化 + 本地模型测试")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5006)
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES，例如 0")
    args = parser.parse_args()
    from viewer.server import main as serve

    sys.argv = ["viewer.server", "--host", args.host, "--port", str(args.port)]
    if args.gpu is not None:
        sys.argv += ["--gpu", str(args.gpu)]
    serve()


if __name__ == "__main__":
    main()
