#!/usr/bin/env python3
"""
Run the SAE analysis suite from one unified config.

Each evaluation lives in this package and is configured under
``evaluations.<name>`` in configs/eval_sae.yaml, sharing the checkpoint,
activation data, model settings, device, and seed with the others.

Results are written to
``results/<evaluation>/<checkpoint_name>/<dataset_name>/``, matching the
layout produced by src/evaluate/gene_features.py. Setting ``output.dir`` for
an evaluation overrides that layout.

Usage:
    python src/evaluate/sae_analysis/eval_sae.py
    python src/evaluate/sae_analysis/eval_sae.py --config configs/eval_sae.yaml
    python src/evaluate/sae_analysis/eval_sae.py --only feature_analysis,pca
    python src/evaluate/sae_analysis/eval_sae.py --skip reconstruction
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluate.model_loading import (
    PROJECT_ROOT,
    evaluation_enabled,
    load_yaml_mapping,
    project_path,
    reject_train_config_references,
)
from evaluate.sae_analysis import (
    feature_analysis,
    pca_analysis,
    pca_vs_sae,
    reconstruction_eval,
)

_DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "eval_sae.yaml"


class Evaluation(NamedTuple):
    """One runnable evaluation: its config key, title, and entry points."""

    name: str
    title: str
    load_config: Callable[[str | Path | None], SimpleNamespace]
    run: Callable[[SimpleNamespace], None]


EVALUATIONS: tuple[Evaluation, ...] = (
    Evaluation(
        "feature_analysis",
        "Feature activation",
        feature_analysis.load_config,
        feature_analysis.run,
    ),
    Evaluation(
        "reconstruction",
        "Reconstruction and variance explained",
        reconstruction_eval.load_config,
        reconstruction_eval.run,
    ),
    Evaluation(
        "pca",
        "PCA alignment",
        pca_analysis.load_config,
        pca_analysis.run,
    ),
    Evaluation(
        "pca_vs_sae",
        "SAE vs PCA variance explained",
        pca_vs_sae.load_config,
        pca_vs_sae.run,
    ),
)

EVALUATION_NAMES = tuple(evaluation.name for evaluation in EVALUATIONS)


def _name_set(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def parse_args() -> argparse.Namespace:
    names = ", ".join(EVALUATION_NAMES)
    parser = argparse.ArgumentParser(
        description="Run configured SAE evaluations from configs/eval_sae.yaml."
    )
    parser.add_argument("--config", default=None, help="Path to unified YAML config.")
    parser.add_argument("--only", default=None, help=f"Comma-separated subset: {names}.")
    parser.add_argument(
        "--skip", default=None, help=f"Comma-separated modules to skip: {names}."
    )
    return parser.parse_args()


def selected_evaluations(
    config_path: str | Path,
    *,
    only: set[str] | None = None,
    skip: set[str] | None = None,
) -> list[Evaluation]:
    """Resolve which evaluations to run from the config and CLI filters."""
    raw = load_yaml_mapping(config_path)
    reject_train_config_references(raw)
    only = only or set()
    skip = skip or set()

    unknown = (only | skip) - set(EVALUATION_NAMES)
    if unknown:
        raise ValueError(f"Unknown evaluation(s): {', '.join(sorted(unknown))}")

    return [
        evaluation
        for evaluation in EVALUATIONS
        if (not only or evaluation.name in only)
        and evaluation.name not in skip
        and evaluation_enabled(raw, evaluation.name)
    ]


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config) if args.config is not None else _DEFAULT_CONFIG
    evaluations = selected_evaluations(
        config_path,
        only=_name_set(args.only),
        skip=_name_set(args.skip),
    )

    if not evaluations:
        print("No evaluations selected.")
        return

    print(f"Using config: {config_path}")
    print("Selected evaluations: " + ", ".join(e.name for e in evaluations))

    for evaluation in evaluations:
        print(f"\n{'#' * 72}")
        print(f"# {evaluation.title} ({evaluation.name})")
        print(f"{'#' * 72}")
        evaluation.run(evaluation.load_config(config_path))


if __name__ == "__main__":
    main()
