#!/usr/bin/env python3
"""
Reconstruction evaluation for trained SAE checkpoints.

Streams activation splits through one or more SAE checkpoints and reports
reconstruction error, sparsity, and centered variance-explained metrics. This
replaces the former ``variance_explained`` and ``recon_error`` modules, which
computed identical metrics from the same accumulator and differed only in the
axis they swept (splits versus checkpoints); this module sweeps both.

Evaluated corpora:
    1) Each configured activation split (``data.splits``).
    2) The pooled corpus of those splits, when ``data.combined`` is set.
    3) (Optional) Additional held-out directories in ``data.unseen_dirs``,
       reported separately and excluded from the pooling.

Metrics per split and checkpoint:
    - Fraction of variance explained (FVE), aggregated across all dims:
      FVE = 1 - sum_d Var(x_d - xhat_d) / sum_d Var(x_d)
    - Mean per-dim FVE (uniform average over coordinates, skipping dims with
      no variance to explain).
    - Reconstruction MSE (mean E[r²] over dims) and normalized MSE
      (total residual energy / total signal energy, i.e. E[‖r‖²] / E[‖x‖²],
      both uncentered — NOT MSE / Var(x)).
    - Mean L0 (mean number of active features per sample).
    - Scale-free companions, which weight every row equally instead of letting
      high-norm rows dominate:
        * mean/std/min cosine similarity between x and xhat;
        * FVE and per-dim FVE recomputed on unit-normalized rows
          (fve_normalized);
        * nmse_normalized = E[‖r‖²/‖x‖²], the unweighted mean per-row relative
          error, versus the norm-weighted nmse above.
      A gap between fve and fve_normalized means the fit is carried by a
      high-norm minority of rows, which the raw metrics cannot reveal.

All paths and hyperparameters are read from eval_sae.yaml.

Usage:
    python src/evaluate/sae_analysis/reconstruction_eval.py
    python src/evaluate/sae_analysis/reconstruction_eval.py --config configs/eval_sae.yaml
    python src/evaluate/sae_analysis/reconstruction_eval.py --splits data_test --max-samples 50000
    python src/evaluate/sae_analysis/reconstruction_eval.py --checkpoint a.ckpt --checkpoint b.ckpt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

# Make src/ importable when run as a script from anywhere in the project.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluate.model_loading import (
    PROJECT_ROOT,
    evaluation_config,
    load_sae,
    load_yaml_mapping,
    project_path,
    checkpoint_file_stem,
    resolve_checkpoint_path,
    resolve_output_dir,
)
from evaluate.sae_analysis.activation_data import (
    SplitSource,
    describe_source,
    normalize_dir_entries,
    prepare_split_source,
    resolve_device,
    resolve_split_specs,
)
from evaluate.sae_analysis.reconstruction import (
    METRIC_HEADERS,
    accumulate_reconstruction_sums,
    combine_sums,
    json_scalars,
    metric_cells,
    per_dim_arrays,
    reconstruction_metrics,
)
from evaluate.sae_analysis.reporting import format_table, write_table

_DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "eval_sae.yaml"
_COMBINED_LABEL = "combined"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def resolve_checkpoint_paths(checkpoint_cfg: dict[str, Any]) -> list[Path]:
    """Resolve a single checkpoint.path/dir or a checkpoint.paths sweep."""
    paths = checkpoint_cfg.get("paths")
    if paths is None:
        return [resolve_checkpoint_path(checkpoint_cfg)]
    if isinstance(paths, (str, Path)):
        raise ValueError("checkpoint.paths must be a list of .ckpt paths.")
    if not isinstance(paths, list) or not paths:
        raise ValueError("checkpoint.paths must be a non-empty list of .ckpt paths.")

    resolved_paths: list[Path] = []
    for path in paths:
        resolved = project_path(path)
        if resolved is None:
            raise ValueError("checkpoint.paths entries must not be null.")
        resolved_paths.append(resolved)
    return resolved_paths


def load_config(path: str | Path | None = None) -> SimpleNamespace:
    """Load reconstruction config from YAML and flatten into a SimpleNamespace."""
    raw = evaluation_config(
        load_yaml_mapping(path or _DEFAULT_CONFIG), "reconstruction"
    )

    ckpt = raw.get("checkpoint", {})
    data = raw.get("data", {})
    out = raw.get("output", {})
    model = raw.get("model", {})

    checkpoint_paths = resolve_checkpoint_paths(ckpt)
    # A sweep pools several checkpoints into one result set, so no single
    # checkpoint names the directory unless they all share a run directory.
    experiment_names = {path.parent.name for path in checkpoint_paths}
    if len(experiment_names) > 1 and not out.get("dir"):
        raise ValueError(
            "checkpoint.paths spans several checkpoint directories "
            f"({', '.join(sorted(experiment_names))}), so the reconstruction "
            "result directory cannot be derived from one checkpoint. Set output.dir."
        )
    # Likewise for the checkpoint level: an explicit checkpoint.name labels a
    # sweep, otherwise the swept checkpoints must share one file stem.
    checkpoint_name = ckpt.get("name")
    if checkpoint_name is None:
        stems = {path.stem for path in checkpoint_paths}
        if len(stems) > 1 and not out.get("dir"):
            raise ValueError(
                "checkpoint.paths sweeps several checkpoints "
                f"({', '.join(sorted(stems))}), so no single checkpoint names the "
                "result directory. Set checkpoint.name to label the sweep, or "
                "output.dir for an exact path."
            )
        checkpoint_name = checkpoint_paths[0].stem
    else:
        checkpoint_name = checkpoint_file_stem(checkpoint_name)

    flat: dict[str, Any] = {
        # checkpoint
        "checkpoint_paths": checkpoint_paths,
        "layer": ckpt.get("layer"),
        "pooling": ckpt.get("pooling"),
        "base_model": ckpt.get("base_model"),
        "prompt_prefix": ckpt.get("prompt_prefix"),
        # data
        "data_cfg": dict(data),
        "split_specs": resolve_split_specs(
            data,
            legacy_split_key="split",
            default_splits=("data_test",),
        ),
        "combine_splits": bool(data.get("combined", False)),
        "unseen_specs": normalize_dir_entries(data.get("unseen_dirs")),
        "max_samples": data.get("max_samples"),
        "max_unseen_samples": data.get("max_unseen_samples"),
        "max_shards": data.get("max_shards"),
        # output
        "output_dir": resolve_output_dir(
            out,
            "reconstruction",
            checkpoint_path=checkpoint_paths[0],
            data_dir=data.get("dir"),
            checkpoint_name=checkpoint_name,
        ),
        "seed": out.get("seed", 42),
        # model defaults can be inferred from checkpoint metadata
        "sae_type": model.get("sae_type"),
    }
    for section in ("model", "eval", "infrastructure"):
        flat.update(raw.get(section, {}))
    flat.setdefault("batch_size", 8192)
    flat.setdefault("device", "auto")

    return SimpleNamespace(**flat)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SAE reconstruction error and variance explained."
    )
    parser.add_argument("--config", default=None, help="Path to YAML config file.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--splits",
        default=None,
        help="Comma-separated split names to evaluate, e.g. data_train,data_test.",
    )
    combined = parser.add_mutually_exclusive_group()
    combined.add_argument("--combined", dest="combined", action="store_true", default=None)
    combined.add_argument("--no-combined", dest="combined", action="store_false")
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help="Override checkpoint path. Repeat to evaluate multiple checkpoints.",
    )
    return parser.parse_args()


def apply_cli_overrides(cfg: SimpleNamespace, args: argparse.Namespace) -> None:
    if args.max_samples is not None:
        cfg.max_samples = args.max_samples
    if args.max_shards is not None:
        cfg.max_shards = args.max_shards
    if args.output_dir is not None:
        cfg.output_dir = project_path(args.output_dir)
    if args.combined is not None:
        cfg.combine_splits = args.combined
    if args.splits is not None:
        splits = [s.strip() for s in args.splits.split(",") if s.strip()]
        if not splits:
            raise ValueError("--splits must name at least one split.")
        cfg.split_specs = resolve_split_specs({**cfg.data_cfg, "splits": splits})
    if args.checkpoint is not None:
        cfg.checkpoint_paths = []
        for path in args.checkpoint:
            resolved = project_path(path)
            if resolved is None:
                raise ValueError("--checkpoint values must not be empty.")
            cfg.checkpoint_paths.append(resolved)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _safe_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "checkpoint"


def checkpoint_labels(checkpoint_paths: list[Path]) -> list[str]:
    """Short unique names for the evaluated checkpoints."""
    stems = [path.stem for path in checkpoint_paths]
    duplicate_stems = {stem for stem in stems if stems.count(stem) > 1}
    labels: list[str] = []
    seen: set[str] = set()
    for path in checkpoint_paths:
        label = path.stem
        if label in duplicate_stems:
            label = f"{path.parent.name}__{path.stem}"
        label = _safe_key(label)
        base = label
        suffix = 2
        while label in seen:
            label = f"{base}_{suffix}"
            suffix += 1
        seen.add(label)
        labels.append(label)
    return labels


def source_meta(source: SplitSource) -> dict[str, Any]:
    return {
        "split": source.spec.name,
        "data_dir": str(source.spec.path),
        "num_shards": len(source.shard_paths),
        "rows_evaluated": source.num_rows,
        "rows_available": source.total_rows,
    }


def combined_meta(sources: list[SplitSource]) -> dict[str, Any]:
    return {
        "split": "+".join(s.spec.name for s in sources),
        "data_dir": [str(s.spec.path) for s in sources],
        "num_shards": sum(len(s.shard_paths) for s in sources),
        "rows_evaluated": sum(s.num_rows for s in sources),
        "rows_available": sum(s.total_rows for s in sources),
    }


def split_table(metrics_by_checkpoint: dict[str, dict[str, Any]]) -> str:
    headers = ("Checkpoint", *METRIC_HEADERS)
    rows = [
        [label, *metric_cells(metrics)]
        for label, metrics in metrics_by_checkpoint.items()
    ]
    return format_table(headers, rows)


def overview_table(results: dict[str, dict[str, dict[str, Any]]]) -> str:
    headers = ("Split", "Checkpoint", *METRIC_HEADERS)
    rows = [
        [split, label, *metric_cells(metrics)]
        for split, per_checkpoint in results.items()
        for label, metrics in per_checkpoint.items()
    ]
    return format_table(headers, rows, n_key_cols=2)


def save_split_results(
    output_dir: Path,
    split_meta: dict[str, Any],
    run_meta: dict[str, Any],
    metrics_by_checkpoint: dict[str, dict[str, Any]],
    checkpoint_meta: dict[str, dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "meta": {**run_meta, **split_meta},
        "checkpoints": {
            label: {**json_scalars(metrics), "checkpoint": checkpoint_meta[label]}
            for label, metrics in metrics_by_checkpoint.items()
        },
    }
    with open(output_dir / "reconstruction.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    npz_data: dict[str, np.ndarray] = {}
    for label, metrics in metrics_by_checkpoint.items():
        npz_data.update(per_dim_arrays(metrics, label))
    np.savez_compressed(output_dir / "reconstruction_per_dim.npz", **npz_data)

    write_table(output_dir, split_table(metrics_by_checkpoint))


def save_overview(
    output_dir: Path,
    run_meta: dict[str, Any],
    split_meta_by_label: dict[str, dict[str, Any]],
    results: dict[str, dict[str, dict[str, Any]]],
    checkpoint_meta: dict[str, dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "meta": run_meta,
        "checkpoints": checkpoint_meta,
        "splits": {
            label: {
                **split_meta_by_label[label],
                "checkpoints": {
                    name: json_scalars(metrics)
                    for name, metrics in per_checkpoint.items()
                },
            }
            for label, per_checkpoint in results.items()
        },
    }
    with open(output_dir / "reconstruction.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    write_table(output_dir, overview_table(results))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def prepare_sources(cfg: SimpleNamespace) -> tuple[list[SplitSource], list[SplitSource]]:
    """Resolve split and unseen sources, keeping their labels distinct."""
    sources = [
        prepare_split_source(
            spec,
            max_shards=cfg.max_shards,
            max_samples=cfg.max_samples,
            seed=cfg.seed,
        )
        for spec in cfg.split_specs
    ]

    # Unseen directories are held-out corpora, not splits of the training data,
    # so they are reported separately and excluded from the combined pooling.
    taken = {source.label for source in sources} | {_COMBINED_LABEL}
    unseen_sources: list[SplitSource] = []
    for spec in cfg.unseen_specs:
        if spec.label in taken:
            spec = replace(spec, label=f"unseen__{spec.label}")
        taken.add(spec.label)
        unseen_sources.append(
            prepare_split_source(
                spec,
                max_shards=cfg.max_shards,
                max_samples=cfg.max_unseen_samples,
                seed=cfg.seed,
            )
        )
    return sources, unseen_sources


def run(cfg: SimpleNamespace) -> None:
    checkpoint_paths = [Path(path) for path in cfg.checkpoint_paths]
    output_dir = Path(cfg.output_dir)
    device = resolve_device(cfg.device)

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Resolve shards and sampled rows once, so every checkpoint sees the same rows.
    sources, unseen_sources = prepare_sources(cfg)
    combine = bool(cfg.combine_splits) and len(sources) >= 2
    if cfg.combine_splits and not combine:
        print("Skipping combined analysis: fewer than two splits.")

    print(f"\n{'=' * 60}")
    print(f"Checkpoints : {len(checkpoint_paths)}")
    print(f"Device      : {device}")
    print(f"Output      : {output_dir}")
    print("Splits      :")
    for source in sources + unseen_sources:
        print(f"  {describe_source(source)}")

    labels = checkpoint_labels(checkpoint_paths)
    results: dict[str, dict[str, dict[str, Any]]] = {
        source.label: {} for source in sources
    }
    if combine:
        results[_COMBINED_LABEL] = {}
    for source in unseen_sources:
        results[source.label] = {}
    checkpoint_meta: dict[str, dict[str, Any]] = {}

    for label, ckpt_path in zip(labels, checkpoint_paths):
        print(f"\n{'-' * 60}")
        print(f"Checkpoint : {label} ({ckpt_path})")
        sae, sae_meta = load_sae(ckpt_path, cfg.sae_type, device=device)
        sae_meta.update(
            {
                "label": label,
                "layer": cfg.layer,
                "pooling": cfg.pooling,
                "checkpoint": str(ckpt_path),
            }
        )
        print(
            f"SAE        : {sae_meta['sae_type']}, d_model={sae_meta['d_model']}, "
            f"d_hidden={sae_meta['d_hidden']}, expansion={sae_meta['expansion']}"
        )
        checkpoint_meta[label] = sae_meta

        per_split_sums = []
        for source in sources:
            print(f"  Evaluating {source.spec.name} ({source.num_rows:,} rows) ...")
            sums = accumulate_reconstruction_sums(
                sae, source, batch_size=cfg.batch_size, device=device
            )
            per_split_sums.append(sums)
            results[source.label][label] = reconstruction_metrics(sums)

        if combine:
            results[_COMBINED_LABEL][label] = reconstruction_metrics(
                combine_sums(per_split_sums)
            )

        for source in unseen_sources:
            print(f"  Evaluating unseen '{source.label}' ({source.num_rows:,} rows) ...")
            sums = accumulate_reconstruction_sums(
                sae, source, batch_size=cfg.batch_size, device=device
            )
            results[source.label][label] = reconstruction_metrics(sums)

        del sae
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    print("\n" + overview_table(results))

    run_meta = {
        "max_samples": cfg.max_samples,
        "max_unseen_samples": cfg.max_unseen_samples,
        "max_shards": cfg.max_shards,
        "seed": cfg.seed,
        "device": device,
        "batch_size": cfg.batch_size,
        "num_checkpoints": len(checkpoint_paths),
    }
    split_meta_by_label = {source.label: source_meta(source) for source in sources}
    if combine:
        split_meta_by_label[_COMBINED_LABEL] = combined_meta(sources)
    for source in unseen_sources:
        split_meta_by_label[source.label] = {**source_meta(source), "unseen": True}

    print(f"Saving results to {output_dir} ...")
    for split_label, metrics_by_checkpoint in results.items():
        save_split_results(
            output_dir / split_label,
            split_meta_by_label[split_label],
            run_meta,
            metrics_by_checkpoint,
            checkpoint_meta,
        )
    save_overview(output_dir, run_meta, split_meta_by_label, results, checkpoint_meta)

    print("\nDone.")
    print("  reconstruction.json                 - metrics for every split and checkpoint")
    print("  summary_table.txt                   - printable cross-split summary")
    print("  <split>/reconstruction.json         - per-split metrics")
    print("  <split>/reconstruction_per_dim.npz  - per-dim reconstruction/FVE arrays")
    print("  <split>/summary_table.txt           - per-split printable summary")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    apply_cli_overrides(cfg, args)
    run(cfg)


if __name__ == "__main__":
    main()
