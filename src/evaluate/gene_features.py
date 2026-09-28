#!/usr/bin/env python3
"""Map a gene vocabulary to activated features from a trained SAE.

It loads pre-extracted vocabulary embeddings, encodes every gene with the
matching SAE, and saves each gene's activated features in descending activation
order. ``src/evaluate/enrich/enrich.py`` consumes these outputs to run GO
enrichment on the resulting per-feature gene sets.

Outputs are written to::

    <output.dir>/<sae_name>/<dataset_name>/<base_model_name>/

Files:
    gene_to_features.jsonl
        One record per gene: ``{"gene": str, "features": [int, ...]}``.
    feature_to_genes.jsonl
        Optional inverse mapping, sorted by activation magnitude.
    gene_feature_activations.npz
        Full post-threshold CSR-like arrays: indices, values, and offsets.
    genes.json
        Ordered gene list corresponding to NPZ row offsets.
    summary.json
        Configuration, resolved identities, SAE metadata, counts, and paths.
    gene_activation_summary.csv
        Per-gene activation counts and strength statistics for plotting/filtering.
    feature_activation_summary.csv
        Per-feature gene prevalence and activation-strength statistics.
    top_gene_feature_pairs.csv
        Strongest individual gene-feature pairs across the vocabulary.
    figures/
        Compact PNG diagnostics for sparsity, strength, and prevalent features.

Usage:
    .venv/bin/python src/evaluate/gene_features.py
    .venv/bin/python src/evaluate/gene_features.py --threshold 2 --top-k 100
    .venv/bin/python src/evaluate/gene_features.py --limit 20 --device cpu
    .venv/bin/python src/evaluate/gene_features.py --figures-only \
        --results-dir results/gene_features/<sae>/<dataset>/<model>
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from safetensors import safe_open


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluate.model_loading import (  # noqa: E402
    dataset_identity_from_path,
    load_checkpoint_metadata,
    load_sae,
    project_path,
    reject_train_config_references,
    resolve_checkpoint_path,
    safe_path_name,
)
from sae import SAE  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "gene_features.yaml"


@dataclass(frozen=True)
class GeneFeaturesConfig:
    config_path: Path
    checkpoint_path: Path
    sae_type: str | None
    vocab_dir: Path
    pooling: str
    limit: int | None
    threshold: float
    top_k: int | None
    device: str
    batch_size: int
    output_root: Path
    sae_name: str | None
    dataset_name: str | None
    base_model_name: str | None
    write_inverse: bool
    generate_figures: bool
    figure_top_n: int
    figure_bins: int
    figure_dpi: int
    figure_top_pairs: int


@dataclass(frozen=True)
class OutputIdentity:
    sae_name: str
    dataset_name: str
    base_model_name: str


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    section = raw.get(name, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"Expected config section {name!r} to be a mapping.")
    return section


def _required_path(value: Any, setting: str) -> Path:
    if not value:
        raise ValueError(f"Set {setting} in the gene-features config.")
    resolved = project_path(value)
    if resolved is None:
        raise ValueError(f"Set {setting} in the gene-features config.")
    return resolved


def load_config(path: str | Path | None = None) -> GeneFeaturesConfig:
    config_path = project_path(path or DEFAULT_CONFIG)
    if config_path is None or not config_path.is_file():
        raise FileNotFoundError(f"Gene-features config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    reject_train_config_references(raw)

    checkpoint = _section(raw, "checkpoint")
    vocab = _section(raw, "vocab")
    filtering = _section(raw, "filter")
    infrastructure = _section(raw, "infrastructure")
    output = _section(raw, "output")
    figures = _section(raw, "figures")

    checkpoint_path = resolve_checkpoint_path(checkpoint)
    vocab_dir = _required_path(vocab.get("dir"), "vocab.dir")
    output_root = _required_path(output.get("dir"), "output.dir")
    pooling = str(vocab.get("pooling", "last"))
    if pooling not in {"last", "mean"}:
        raise ValueError("vocab.pooling must be 'last' or 'mean'.")

    limit_value = vocab.get("limit")
    limit = None if limit_value is None else int(limit_value)
    if limit is not None and limit < 1:
        raise ValueError("vocab.limit must be null or at least 1.")

    threshold = float(filtering.get("threshold", 0.0))
    if threshold < 0:
        raise ValueError("filter.threshold must be non-negative.")
    top_k_value = filtering.get("top_k")
    top_k = None if top_k_value is None else int(top_k_value)
    if top_k is not None and top_k < 1:
        raise ValueError("filter.top_k must be null or at least 1.")

    device = str(infrastructure.get("device", "auto"))
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("infrastructure.device must be auto, cpu, or cuda.")
    batch_size = int(infrastructure.get("batch_size", 512))
    if batch_size < 1:
        raise ValueError("infrastructure.batch_size must be at least 1.")

    write_inverse = output.get("write_inverse", True)
    if not isinstance(write_inverse, bool):
        raise ValueError("output.write_inverse must be true or false.")

    generate_figures = figures.get("generate", True)
    if not isinstance(generate_figures, bool):
        raise ValueError("figures.generate must be true or false.")
    figure_top_n = int(figures.get("top_n", 30))
    figure_bins = int(figures.get("bins", 60))
    figure_dpi = int(figures.get("dpi", 180))
    figure_top_pairs = int(figures.get("top_pairs", 1000))
    if figure_top_n < 1:
        raise ValueError("figures.top_n must be at least 1.")
    if figure_bins < 2:
        raise ValueError("figures.bins must be at least 2.")
    if figure_dpi < 1:
        raise ValueError("figures.dpi must be at least 1.")
    if figure_top_pairs < 1:
        raise ValueError("figures.top_pairs must be at least 1.")

    return GeneFeaturesConfig(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        sae_type=checkpoint.get("sae_type"),
        vocab_dir=vocab_dir,
        pooling=pooling,
        limit=limit,
        threshold=threshold,
        top_k=top_k,
        device=device,
        batch_size=batch_size,
        output_root=output_root,
        sae_name=output.get("sae_name"),
        dataset_name=output.get("dataset_name"),
        base_model_name=output.get("base_model_name"),
        write_inverse=write_inverse,
        generate_figures=generate_figures,
        figure_top_n=figure_top_n,
        figure_bins=figure_bins,
        figure_dpi=figure_dpi,
        figure_top_pairs=figure_top_pairs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map pre-extracted gene vocabulary embeddings to SAE features."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="YAML configuration path.",
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output.dir; identity subdirectories are still appended.",
    )
    parser.add_argument("--sae-name", default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--base-model-name", default=None)
    parser.add_argument(
        "--write-inverse",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write feature_to_genes.jsonl.",
    )
    parser.add_argument(
        "--figures",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Generate CSV visualization tables and PNG figures.",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Regenerate visualization artifacts from an existing result directory.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help=(
            "Existing gene-feature result directory for --figures-only. "
            "If omitted, it is resolved from the config identities."
        ),
    )
    parser.add_argument("--figure-top-n", type=int, default=None)
    parser.add_argument("--figure-bins", type=int, default=None)
    parser.add_argument("--figure-dpi", type=int, default=None)
    parser.add_argument(
        "--figure-top-pairs",
        type=int,
        default=None,
        help="Rows to retain in top_gene_feature_pairs.csv.",
    )
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def load_vocab(
    vocab_dir: str | Path,
    pooling: str,
    *,
    limit: int | None = None,
) -> tuple[torch.Tensor, list[str]]:
    """Load aligned vocabulary embeddings and gene names."""

    vocab_dir = Path(vocab_dir)
    embeddings_path = vocab_dir / f"vocab_{pooling}.safetensors"
    genes_path = vocab_dir / "genes.json"
    if not embeddings_path.is_file():
        raise FileNotFoundError(f"Vocabulary embeddings not found: {embeddings_path}")
    if not genes_path.is_file():
        raise FileNotFoundError(f"Vocabulary gene list not found: {genes_path}")

    with safe_open(embeddings_path, framework="pt", device="cpu") as f:
        if "embeddings" not in f.keys():
            raise KeyError(f"Missing 'embeddings' tensor in {embeddings_path}")
        embeddings = f.get_tensor("embeddings").float()
    with genes_path.open("r", encoding="utf-8") as f:
        raw_genes = json.load(f)
    if not isinstance(raw_genes, list):
        raise ValueError(f"Expected a JSON list in {genes_path}")
    genes = [str(gene) for gene in raw_genes]

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected vocabulary embeddings with shape (genes, d_model), "
            f"got {tuple(embeddings.shape)}"
        )
    if len(genes) != embeddings.shape[0]:
        raise ValueError(
            f"Gene count {len(genes)} does not match embedding rows "
            f"{embeddings.shape[0]}."
        )
    if len(set(genes)) != len(genes):
        raise ValueError(f"Gene names must be unique in {genes_path}")

    if limit is not None:
        embeddings = embeddings[:limit]
        genes = genes[:limit]
    return embeddings, genes


_safe_name = safe_path_name


def _vocab_path_identity(vocab_dir: Path) -> tuple[str | None, str | None]:
    """Infer ``(base_model_name, dataset_name)`` from the extraction layout."""

    return dataset_identity_from_path(vocab_dir)


def resolve_output_identity(
    cfg: GeneFeaturesConfig,
    checkpoint_metadata: dict[str, Any],
    *,
    sae_name: str | None = None,
    dataset_name: str | None = None,
    base_model_name: str | None = None,
) -> OutputIdentity:
    inferred_model, inferred_dataset = _vocab_path_identity(cfg.vocab_dir)
    dataset_meta = checkpoint_metadata.get("dataset") or {}
    if not isinstance(dataset_meta, dict):
        dataset_meta = {}

    return OutputIdentity(
        sae_name=_safe_name(
            sae_name or cfg.sae_name or cfg.checkpoint_path.parent.name,
            "SAE name",
        ),
        # The vocabulary being encoded names the output, so encoding one SAE
        # against several vocabularies keeps the results apart. The checkpoint's
        # training dataset stays recorded in its model_metadata.yaml.
        dataset_name=_safe_name(
            dataset_name
            or cfg.dataset_name
            or inferred_dataset
            or dataset_meta.get("name")
            or cfg.vocab_dir.name,
            "dataset name",
        ),
        base_model_name=_safe_name(
            base_model_name or cfg.base_model_name or inferred_model,
            "base-model name",
        ),
    )


def validate_alignment(
    cfg: GeneFeaturesConfig,
    checkpoint_metadata: dict[str, Any],
) -> None:
    """Validate layer, pooling, and base-model path metadata.

    A vocabulary from a dataset other than the SAE's training set is a
    legitimate cross-dataset run, not an error: the output path is named after
    the vocabulary, so those results stay separate. Layer and pooling
    mismatches remain fatal because they make the encoding meaningless.
    """

    model_meta = checkpoint_metadata.get("model") or {}
    dataset_meta = checkpoint_metadata.get("dataset") or {}
    if not isinstance(model_meta, dict):
        model_meta = {}
    if not isinstance(dataset_meta, dict):
        dataset_meta = {}

    path_parts = set(cfg.vocab_dir.parts)
    metadata_dataset = dataset_meta.get("name")
    if metadata_dataset and metadata_dataset not in path_parts:
        print(
            f"Note: SAE was trained on {metadata_dataset!r}, but the vocabulary "
            f"comes from {cfg.vocab_dir}. Running cross-dataset."
        )
    layer = model_meta.get("layer")
    if layer is not None and f"layer{layer}" not in path_parts:
        raise ValueError(
            f"Checkpoint layer{layer} does not match vocabulary directory "
            f"{cfg.vocab_dir}."
        )
    pooling = model_meta.get("pooling_method")
    if pooling is not None and str(pooling) != cfg.pooling:
        raise ValueError(
            f"Checkpoint pooling {pooling!r} does not match vocab.pooling "
            f"{cfg.pooling!r}."
        )
@torch.inference_mode()
def encode_gene_features(
    sae: SAE,
    embeddings: torch.Tensor,
    *,
    batch_size: int,
    device: str,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode embeddings into CSR-like active feature arrays."""

    all_indices: list[np.ndarray] = []
    all_values: list[np.ndarray] = []
    offsets = [0]
    n_genes = int(embeddings.shape[0])

    for start in range(0, n_genes, batch_size):
        stop = min(start + batch_size, n_genes)
        encoded = sae.encode(embeddings[start:stop].to(device)).cpu()
        for row in encoded:
            active = row > threshold
            feature_ids = (
                active.nonzero(as_tuple=False)
                .squeeze(-1)
                .to(torch.int32)
                .numpy()
            )
            magnitudes = row[active].float().numpy()
            all_indices.append(feature_ids)
            all_values.append(magnitudes)
            offsets.append(offsets[-1] + len(feature_ids))
        print(f"  [{stop:,}/{n_genes:,}] genes encoded")

    indices = (
        np.concatenate(all_indices).astype(np.int32, copy=False)
        if all_indices
        else np.empty(0, dtype=np.int32)
    )
    values = (
        np.concatenate(all_values).astype(np.float32, copy=False)
        if all_values
        else np.empty(0, dtype=np.float32)
    )
    return indices, values, np.asarray(offsets, dtype=np.int64)


def build_gene_to_features(
    genes: list[str],
    indices: np.ndarray,
    values: np.ndarray,
    offsets: np.ndarray,
    *,
    top_k: int | None,
) -> list[dict[str, Any]]:
    """Build gene -> feature IDs sorted by descending activation magnitude."""

    records: list[dict[str, Any]] = []
    for gene_index, gene in enumerate(genes):
        start = int(offsets[gene_index])
        stop = int(offsets[gene_index + 1])
        order = np.argsort(values[start:stop])[::-1]
        if top_k is not None:
            order = order[:top_k]
        feature_ids = indices[start:stop][order]
        records.append(
            {
                "gene": gene,
                "features": [int(feature_id) for feature_id in feature_ids],
            }
        )
    return records


def build_feature_to_genes(
    genes: list[str],
    indices: np.ndarray,
    values: np.ndarray,
    offsets: np.ndarray,
    *,
    top_k: int | None,
) -> list[dict[str, Any]]:
    """Build feature -> genes sorted by descending activation magnitude."""

    accumulator: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for gene_index, gene in enumerate(genes):
        start = int(offsets[gene_index])
        stop = int(offsets[gene_index + 1])
        for feature_id, magnitude in zip(
            indices[start:stop], values[start:stop], strict=True
        ):
            accumulator[int(feature_id)].append((gene, float(magnitude)))

    records: list[dict[str, Any]] = []
    for feature_id, entries in sorted(accumulator.items()):
        ordered = sorted(entries, key=lambda item: item[1], reverse=True)
        if top_k is not None:
            ordered = ordered[:top_k]
        records.append(
            {
                "feature": feature_id,
                "genes": [gene for gene, _ in ordered],
            }
        )
    return records


def _validate_sparse_results(
    genes: list[str],
    indices: np.ndarray,
    values: np.ndarray,
    offsets: np.ndarray,
    d_hidden: int,
) -> None:
    """Check the invariants needed by the visualization summaries."""

    if indices.ndim != 1 or values.ndim != 1 or offsets.ndim != 1:
        raise ValueError("Sparse activation arrays must all be one-dimensional.")
    if len(indices) != len(values):
        raise ValueError("Sparse activation indices and values have different lengths.")
    if len(offsets) != len(genes) + 1:
        raise ValueError("Sparse activation offsets do not align with genes.json.")
    if len(offsets) and (int(offsets[0]) != 0 or int(offsets[-1]) != len(indices)):
        raise ValueError("Sparse activation offsets do not span the stored pairs.")
    if np.any(np.diff(offsets) < 0):
        raise ValueError("Sparse activation offsets must be non-decreasing.")
    if d_hidden < 1:
        raise ValueError("SAE d_hidden must be at least 1.")
    if len(indices) and (int(indices.min()) < 0 or int(indices.max()) >= d_hidden):
        raise ValueError("A stored feature ID falls outside the SAE feature range.")


def build_visualization_summaries(
    genes: list[str],
    indices: np.ndarray,
    values: np.ndarray,
    offsets: np.ndarray,
    *,
    d_hidden: int,
    top_pairs: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build tidy gene, feature, and strongest-pair tables from sparse arrays.

    Every ``mean_activation_when_active`` divides by the number of surviving
    activations, not by the vocabulary size: the sparse arrays hold only pairs
    above ``filter.threshold`` (and, for a TopK SAE, only the k features the
    architecture leaves non-zero per gene). The column is therefore a mean over
    the retained tail and is floored near the threshold, so it says how hard a
    feature fires where it fires, never how often. Prevalence lives in
    ``n_genes``/``gene_fraction``; ranking on strength alone mixes near-dense
    features in with focused ones.
    """

    _validate_sparse_results(genes, indices, values, offsets, d_hidden)
    feature_counts = np.bincount(indices, minlength=d_hidden).astype(np.int64)
    feature_sums = np.bincount(
        indices, weights=values.astype(np.float64, copy=False), minlength=d_hidden
    )
    feature_max = np.full(d_hidden, -np.inf, dtype=np.float64)
    feature_top_gene = np.full(d_hidden, -1, dtype=np.int64)

    gene_rows: list[dict[str, Any]] = []
    for gene_index, gene in enumerate(genes):
        start = int(offsets[gene_index])
        stop = int(offsets[gene_index + 1])
        row_indices = indices[start:stop]
        row_values = values[start:stop]
        count = stop - start
        if count:
            local_top = int(np.argmax(row_values))
            top_feature = int(row_indices[local_top])
            max_activation = float(row_values[local_top])
            total_activation = float(
                row_values.astype(np.float64, copy=False).sum()
            )
            better = row_values > feature_max[row_indices]
            better_features = row_indices[better]
            feature_max[better_features] = row_values[better]
            feature_top_gene[better_features] = gene_index
        else:
            top_feature = None
            max_activation = 0.0
            total_activation = 0.0
        gene_rows.append(
            {
                "gene": gene,
                "n_active_features": count,
                "total_activation": total_activation,
                "mean_activation_when_active": (
                    total_activation / count if count else 0.0
                ),
                "max_activation": max_activation,
                "top_feature": top_feature,
            }
        )

    n_genes = len(genes)
    feature_rows: list[dict[str, Any]] = []
    for feature_id in np.flatnonzero(feature_counts):
        feature_id = int(feature_id)
        count = int(feature_counts[feature_id])
        top_gene_index = int(feature_top_gene[feature_id])
        total_activation = float(feature_sums[feature_id])
        feature_rows.append(
            {
                "feature": feature_id,
                "n_genes": count,
                "gene_fraction": count / n_genes if n_genes else 0.0,
                "total_activation": total_activation,
                "mean_activation_when_active": total_activation / count,
                "max_activation": float(feature_max[feature_id]),
                "top_gene": genes[top_gene_index],
            }
        )

    n_top = min(top_pairs, len(values))
    if n_top == len(values):
        selected = np.arange(len(values), dtype=np.int64)
    elif n_top:
        selected = np.argpartition(values, -n_top)[-n_top:]
    else:
        selected = np.empty(0, dtype=np.int64)
    selected = selected[np.argsort(values[selected], kind="stable")[::-1]]
    pair_gene_indices = np.searchsorted(offsets, selected, side="right") - 1
    pair_rows = [
        {
            "rank": rank,
            "gene": genes[int(gene_index)],
            "feature": int(indices[pair_index]),
            "activation": float(values[pair_index]),
        }
        for rank, (pair_index, gene_index) in enumerate(
            zip(selected, pair_gene_indices, strict=True), start=1
        )
    ]
    return gene_rows, feature_rows, pair_rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_pyplot() -> Any:
    """Import a headless pyplot only when figures are requested."""

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_figure(fig: Any, path: Path, *, dpi: int, plt: Any) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_visualization_artifacts(
    output_dir: Path,
    genes: list[str],
    indices: np.ndarray,
    values: np.ndarray,
    offsets: np.ndarray,
    *,
    d_hidden: int,
    top_n: int,
    bins: int,
    dpi: int,
    top_pairs: int,
) -> dict[str, Any]:
    """Write plotting tables and compact PNG diagnostics for one result set."""

    gene_rows, feature_rows, pair_rows = build_visualization_summaries(
        genes,
        indices,
        values,
        offsets,
        d_hidden=d_hidden,
        top_pairs=top_pairs,
    )
    gene_summary_path = output_dir / "gene_activation_summary.csv"
    feature_summary_path = output_dir / "feature_activation_summary.csv"
    top_pairs_path = output_dir / "top_gene_feature_pairs.csv"
    _write_csv(
        gene_summary_path,
        gene_rows,
        [
            "gene",
            "n_active_features",
            "total_activation",
            "mean_activation_when_active",
            "max_activation",
            "top_feature",
        ],
    )
    _write_csv(
        feature_summary_path,
        feature_rows,
        [
            "feature",
            "n_genes",
            "gene_fraction",
            "total_activation",
            "mean_activation_when_active",
            "max_activation",
            "top_gene",
        ],
    )
    _write_csv(
        top_pairs_path,
        pair_rows,
        ["rank", "gene", "feature", "activation"],
    )

    plt = _load_pyplot()
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    style = {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.22,
        "grid.linestyle": "--",
    }
    figure_paths: dict[str, str] = {}

    with plt.rc_context(style):
        gene_counts = np.diff(offsets)
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.hist(gene_counts, bins=bins, color="#0072B2")
        ax.axvline(
            float(gene_counts.mean()) if len(gene_counts) else 0.0,
            color="#D55E00",
            linestyle="--",
            label="mean",
        )
        ax.set(
            title="Active SAE features per gene",
            xlabel="Active features",
            ylabel="Genes",
        )
        ax.legend(frameon=False)
        path = figure_dir / "active_features_per_gene.png"
        _save_figure(fig, path, dpi=dpi, plt=plt)
        figure_paths["active_features_per_gene"] = str(path)

        fig, ax = plt.subplots(figsize=(8, 4.8))
        positive_values = values[values > 0]
        if len(positive_values) and float(positive_values.max()) > float(
            positive_values.min()
        ):
            activation_bins = np.geomspace(
                float(positive_values.min()),
                float(positive_values.max()),
                bins + 1,
            )
            ax.hist(positive_values, bins=activation_bins, color="#009E73")
            ax.set_xscale("log")
            activation_label = "Activation (log scale)"
        else:
            ax.hist(values, bins=bins, color="#009E73")
            activation_label = "Activation"
        ax.set(
            title="Stored gene-feature activation strengths",
            xlabel=activation_label,
            ylabel="Gene-feature pairs",
        )
        path = figure_dir / "activation_strength_distribution.png"
        _save_figure(fig, path, dpi=dpi, plt=plt)
        figure_paths["activation_strength_distribution"] = str(path)

        feature_counts = np.asarray(
            [row["n_genes"] for row in feature_rows], dtype=np.int64
        )
        feature_means = np.asarray(
            [row["mean_activation_when_active"] for row in feature_rows],
            dtype=np.float64,
        )
        fig, ax = plt.subplots(figsize=(8, 5.2))
        ax.scatter(
            feature_counts,
            feature_means,
            s=12,
            alpha=0.45,
            color="#CC79A7",
            linewidths=0,
        )
        if len(feature_counts) and int(feature_counts.max()) > 1:
            ax.set_xscale("log")
        ax.set(
            title="Feature prevalence versus activation strength",
            xlabel="Genes activating feature",
            ylabel="Mean activation when active",
        )
        path = figure_dir / "feature_prevalence_vs_strength.png"
        _save_figure(fig, path, dpi=dpi, plt=plt)
        figure_paths["feature_prevalence_vs_strength"] = str(path)

        ordered_features = sorted(
            feature_rows,
            key=lambda row: (row["n_genes"], row["mean_activation_when_active"]),
            reverse=True,
        )[:top_n]
        ordered_features.reverse()
        fig, ax = plt.subplots(
            figsize=(9, max(4.8, 0.28 * len(ordered_features) + 1.8))
        )
        labels = [
            f"F{row['feature']} · {row['top_gene']}" for row in ordered_features
        ]
        ax.barh(
            labels,
            [row["n_genes"] for row in ordered_features],
            color="#E69F00",
        )
        ax.set(
            title=f"Top {len(ordered_features)} features by gene prevalence",
            xlabel="Genes activating feature",
            ylabel="Feature · strongest gene",
        )
        path = figure_dir / "top_features_by_gene_prevalence.png"
        _save_figure(fig, path, dpi=dpi, plt=plt)
        figure_paths["top_features_by_gene_prevalence"] = str(path)

    return {
        "gene_summary": str(gene_summary_path),
        "feature_summary": str(feature_summary_path),
        "top_pairs": str(top_pairs_path),
        "figures_dir": str(figure_dir),
        "figures": figure_paths,
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")


def load_gene_to_features(path: str | Path) -> dict[str, list[int]]:
    """Load ``gene_to_features.jsonl`` into a gene -> feature IDs dictionary."""

    mapping: dict[str, list[int]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "gene" not in record or "features" not in record:
                raise ValueError(f"Malformed mapping at {path}:{line_number}")
            mapping[str(record["gene"])] = [
                int(feature_id) for feature_id in record["features"]
            ]
    return mapping


def save_results(
    output_dir: Path,
    genes: list[str],
    indices: np.ndarray,
    values: np.ndarray,
    offsets: np.ndarray,
    *,
    cfg: GeneFeaturesConfig,
    identity: OutputIdentity,
    sae_metadata: dict[str, Any],
    threshold: float,
    top_k: int | None,
    limit: int | None,
    batch_size: int,
    device: str,
    write_inverse: bool,
    generate_figures: bool,
    figure_top_n: int,
    figure_bins: int,
    figure_dpi: int,
    figure_top_pairs: int,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "gene_feature_activations.npz"
    genes_path = output_dir / "genes.json"
    gene_mapping_path = output_dir / "gene_to_features.jsonl"
    inverse_path = output_dir / "feature_to_genes.jsonl"
    summary_path = output_dir / "summary.json"

    np.savez_compressed(
        npz_path,
        indices=indices,
        values=values,
        offsets=offsets,
    )
    with genes_path.open("w", encoding="utf-8") as f:
        json.dump(genes, f, indent=2)
        f.write("\n")
    write_jsonl(
        gene_mapping_path,
        build_gene_to_features(
            genes, indices, values, offsets, top_k=top_k
        ),
    )
    if write_inverse:
        write_jsonl(
            inverse_path,
            build_feature_to_genes(
                genes, indices, values, offsets, top_k=top_k
            ),
        )

    total_active = int(len(indices))
    active_counts = np.diff(offsets)
    outputs = {
        "gene_to_features": gene_mapping_path,
        "gene_feature_activations": npz_path,
        "genes": genes_path,
    }
    if write_inverse:
        outputs["feature_to_genes"] = inverse_path

    visualization: dict[str, Any] = {}
    if generate_figures:
        visualization = write_visualization_artifacts(
            output_dir,
            genes,
            indices,
            values,
            offsets,
            d_hidden=int(sae_metadata["d_hidden"]),
            top_n=figure_top_n,
            bins=figure_bins,
            dpi=figure_dpi,
            top_pairs=figure_top_pairs,
        )

    summary = {
        "inputs": {
            "config": str(cfg.config_path),
            "checkpoint": str(cfg.checkpoint_path),
            "vocab_dir": str(cfg.vocab_dir),
            "pooling": cfg.pooling,
        },
        "identity": {
            "sae_name": identity.sae_name,
            "dataset_name": identity.dataset_name,
            "base_model_name": identity.base_model_name,
        },
        "settings": {
            "threshold": threshold,
            "top_k": top_k,
            "limit": limit,
            "batch_size": batch_size,
            "device": device,
            "write_inverse": write_inverse,
            "figures": {
                "generate": generate_figures,
                "top_n": figure_top_n,
                "bins": figure_bins,
                "dpi": figure_dpi,
                "top_pairs": figure_top_pairs,
            },
        },
        "sae": sae_metadata,
        "vocabulary": {
            "n_genes": len(genes),
            "total_active_features": total_active,
            "mean_active_features_per_gene": (
                float(active_counts.mean()) if len(active_counts) else 0.0
            ),
            "min_active_features_per_gene": (
                int(active_counts.min()) if len(active_counts) else 0
            ),
            "max_active_features_per_gene": (
                int(active_counts.max()) if len(active_counts) else 0
            ),
        },
        "results": {
            "output_dir": str(output_dir),
            **{name: str(path) for name, path in outputs.items()},
            "visualization": visualization,
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    outputs["summary"] = summary_path
    if visualization:
        outputs["figures"] = Path(visualization["figures_dir"])
    return outputs


def _load_saved_sparse_results(
    output_dir: Path,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    npz_path = output_dir / "gene_feature_activations.npz"
    genes_path = output_dir / "genes.json"
    summary_path = output_dir / "summary.json"
    for path in (npz_path, genes_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(f"Existing gene-feature result not found: {path}")

    with np.load(npz_path, allow_pickle=False) as store:
        required = {"indices", "values", "offsets"}
        missing = required.difference(store.files)
        if missing:
            raise KeyError(f"Missing arrays in {npz_path}: {sorted(missing)}")
        indices = store["indices"]
        values = store["values"]
        offsets = store["offsets"]
    with genes_path.open("r", encoding="utf-8") as f:
        raw_genes = json.load(f)
    if not isinstance(raw_genes, list):
        raise ValueError(f"Expected a JSON list in {genes_path}")
    genes = [str(gene) for gene in raw_genes]
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    if not isinstance(summary, dict):
        raise ValueError(f"Expected a JSON object in {summary_path}")
    return genes, indices, values, offsets, summary


def regenerate_visualization_artifacts(
    output_dir: Path,
    *,
    top_n: int,
    bins: int,
    dpi: int,
    top_pairs: int,
) -> dict[str, Any]:
    """Generate artifacts from a completed evaluation and refresh its summary."""

    genes, indices, values, offsets, summary = _load_saved_sparse_results(output_dir)
    sae_summary = summary.get("sae") or {}
    if not isinstance(sae_summary, dict) or "d_hidden" not in sae_summary:
        raise ValueError(f"Missing sae.d_hidden in {output_dir / 'summary.json'}")
    visualization = write_visualization_artifacts(
        output_dir,
        genes,
        indices,
        values,
        offsets,
        d_hidden=int(sae_summary["d_hidden"]),
        top_n=top_n,
        bins=bins,
        dpi=dpi,
        top_pairs=top_pairs,
    )
    settings = summary.setdefault("settings", {})
    settings["figures"] = {
        "generate": True,
        "top_n": top_n,
        "bins": bins,
        "dpi": dpi,
        "top_pairs": top_pairs,
    }
    results = summary.setdefault("results", {})
    results["visualization"] = visualization
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    return visualization


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    threshold = cfg.threshold if args.threshold is None else args.threshold
    top_k = cfg.top_k if args.top_k is None else args.top_k
    limit = cfg.limit if args.limit is None else args.limit
    batch_size = cfg.batch_size if args.batch_size is None else args.batch_size
    requested_device = cfg.device if args.device is None else args.device
    output_root = (
        cfg.output_root
        if args.output_dir is None
        else _required_path(args.output_dir, "--output-dir")
    )
    write_inverse = (
        cfg.write_inverse if args.write_inverse is None else args.write_inverse
    )
    generate_figures = (
        cfg.generate_figures if args.figures is None else args.figures
    )
    figure_top_n = (
        cfg.figure_top_n if args.figure_top_n is None else args.figure_top_n
    )
    figure_bins = cfg.figure_bins if args.figure_bins is None else args.figure_bins
    figure_dpi = cfg.figure_dpi if args.figure_dpi is None else args.figure_dpi
    figure_top_pairs = (
        cfg.figure_top_pairs
        if args.figure_top_pairs is None
        else args.figure_top_pairs
    )

    if threshold < 0:
        raise ValueError("--threshold must be non-negative.")
    if top_k is not None and top_k < 1:
        raise ValueError("--top-k must be at least 1.")
    if limit is not None and limit < 1:
        raise ValueError("--limit must be at least 1.")
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if figure_top_n < 1:
        raise ValueError("--figure-top-n must be at least 1.")
    if figure_bins < 2:
        raise ValueError("--figure-bins must be at least 2.")
    if figure_dpi < 1:
        raise ValueError("--figure-dpi must be at least 1.")
    if figure_top_pairs < 1:
        raise ValueError("--figure-top-pairs must be at least 1.")
    if args.results_dir is not None and not args.figures_only:
        raise ValueError("--results-dir can only be used with --figures-only.")
    if args.figures_only and not generate_figures:
        raise ValueError("--figures-only cannot be combined with --no-figures.")

    if args.figures_only and args.results_dir is not None:
        output_dir = project_path(args.results_dir)
        if output_dir is None:
            raise ValueError("--results-dir must name a result directory.")
        print(f"Regenerating visualization artifacts from {output_dir}")
        visualization = regenerate_visualization_artifacts(
            output_dir,
            top_n=figure_top_n,
            bins=figure_bins,
            dpi=figure_dpi,
            top_pairs=figure_top_pairs,
        )
        print(f"Gene summary   : {visualization['gene_summary']}")
        print(f"Feature summary: {visualization['feature_summary']}")
        print(f"Top pairs      : {visualization['top_pairs']}")
        print(f"Figures        : {visualization['figures_dir']}")
        return

    checkpoint_metadata = load_checkpoint_metadata(cfg.checkpoint_path)
    identity = resolve_output_identity(
        cfg,
        checkpoint_metadata,
        sae_name=args.sae_name,
        dataset_name=args.dataset_name,
        base_model_name=args.base_model_name,
    )
    validate_alignment(cfg, checkpoint_metadata)
    output_dir = (
        output_root
        / identity.sae_name
        / identity.dataset_name
        / identity.base_model_name
    )

    if args.figures_only:
        print(f"Regenerating visualization artifacts from {output_dir}")
        visualization = regenerate_visualization_artifacts(
            output_dir,
            top_n=figure_top_n,
            bins=figure_bins,
            dpi=figure_dpi,
            top_pairs=figure_top_pairs,
        )
        print(f"Gene summary   : {visualization['gene_summary']}")
        print(f"Feature summary: {visualization['feature_summary']}")
        print(f"Top pairs      : {visualization['top_pairs']}")
        print(f"Figures        : {visualization['figures_dir']}")
        return

    device = resolve_device(requested_device)
    print(f"Checkpoint : {cfg.checkpoint_path}")
    print(f"Vocabulary : {cfg.vocab_dir} (pooling={cfg.pooling})")
    print(f"Device     : {device}")
    print(f"Threshold  : {threshold}")
    print(f"Output     : {output_dir}")

    sae, sae_metadata = load_sae(
        cfg.checkpoint_path,
        sae_type=cfg.sae_type,
        device=device,
    )
    embeddings, genes = load_vocab(cfg.vocab_dir, cfg.pooling, limit=limit)
    if int(embeddings.shape[1]) != int(sae_metadata["d_model"]):
        raise ValueError(
            f"Vocabulary width {embeddings.shape[1]} does not match SAE d_model "
            f"{sae_metadata['d_model']}."
        )
    print(
        f"Loaded {len(genes):,} genes with d_model={embeddings.shape[1]:,}; "
        f"SAE has {sae_metadata['d_hidden']:,} features."
    )

    indices, values, offsets = encode_gene_features(
        sae,
        embeddings,
        batch_size=batch_size,
        device=device,
        threshold=threshold,
    )
    outputs = save_results(
        output_dir,
        genes,
        indices,
        values,
        offsets,
        cfg=cfg,
        identity=identity,
        sae_metadata=sae_metadata,
        threshold=threshold,
        top_k=top_k,
        limit=limit,
        batch_size=batch_size,
        device=device,
        write_inverse=write_inverse,
        generate_figures=generate_figures,
        figure_top_n=figure_top_n,
        figure_bins=figure_bins,
        figure_dpi=figure_dpi,
        figure_top_pairs=figure_top_pairs,
    )

    print(f"Activated pairs: {len(indices):,}")
    print(f"Gene mapping   : {outputs['gene_to_features']}")
    if "feature_to_genes" in outputs:
        print(f"Inverse mapping: {outputs['feature_to_genes']}")
    print(f"Sparse values  : {outputs['gene_feature_activations']}")
    if "figures" in outputs:
        print(f"Figures        : {outputs['figures']}")
    print(f"Summary        : {outputs['summary']}")


if __name__ == "__main__":
    main()
