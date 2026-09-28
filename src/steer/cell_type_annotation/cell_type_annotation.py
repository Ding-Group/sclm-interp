#!/usr/bin/env python3
"""Evaluate generated cell-type annotations with an SAE-steered model.

This task entry point uses the shared cell-type dataset and prompt, then
delegates SAE intervention, generation, and scoring to
``sae_steer_inference.py``.

Usage:
    python src/steer/cell_type_annotation/cell_type_annotation.py
    python src/steer/cell_type_annotation/cell_type_annotation.py --config configs/steer.yaml
    python src/steer/cell_type_annotation/cell_type_annotation.py --features 12 34 --alpha 1.0
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.steer import sae_steer_inference


def main() -> None:
    args = sae_steer_inference.parse_args()
    cfg = sae_steer_inference.load_eval_config(args.config)
    sae_steer_inference.apply_cli_overrides(cfg, args)
    sae_steer_inference.run_eval(cfg)


if __name__ == "__main__":
    main()
