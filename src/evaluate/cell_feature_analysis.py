#!/usr/bin/env python3
"""Rank SAE features that distinguish each cell type.

The cell activation extractor writes one base-model final-state activation per
cell under ``datasets/cells``. Each safetensors shard contains aligned
``activations`` and ``cell_type_ids`` tensors, while ``cell_types.json`` maps
the integer IDs to labels. This script streams those rows through the matching
SAE and aggregates feature statistics without retaining a cell-by-feature
matrix in memory.

When enabled, the gene analysis reuses the matching hierarchical
``results/gene_features`` output and joins each selected SAE feature to its
highest-activation vocabulary genes. Features absent from that lookup are
kept with an empty gene list so the join never silently drops candidates.

The main ranking statistic is ``specificity_margin``:

    mean activation in the target cell type
    - maximum mean activation in any other cell type

A positive margin therefore identifies a feature whose average activation is
stronger for the target than for every competing type. The output also keeps
activation prevalence, mean magnitude when active, specificity share, and a
one-vs-rest standardized effect size. These are observational candidates for
later steering or ablation experiments; they are not by themselves causal
evidence that a feature governs the model's cell-type decision.

Two one-vs-rest discrimination metrics accompany the margin:

    cohens_d_vs_rest  standardized mean difference between the target type and
                      every other cell, using the degrees-of-freedom weighted
                      pooled standard deviation
    auroc_vs_rest     probability that the feature ranks a random target cell
                      above a random cell of any other type, ties counted half

``auroc_vs_rest`` is evaluated exactly rather than by sampling. The encoder
leaves most features at exactly zero, so for any feature the cells on which it
does not fire form one large tie block whose contribution to the Mann-Whitney
statistic is available in closed form; only the non-zero activations are ranked.
This keeps the streaming design intact, since the retained sparse entries are a
small fraction of a full cell-by-feature matrix.

The three statistics rank features differently: the margin rewards a large
absolute activation gap, while Cohen's d and the AUROC reward clean separation
and are insensitive to a feature's overall scale. Top-N selection, including the
features carried into the gene analysis, continues to use the margin, so the two
new columns are read as diagnostics on that ranking rather than as a substitute
for it.

Example:
    .venv/bin/python src/evaluate/cell_feature_analysis.py

    .venv/bin/python src/evaluate/cell_feature_analysis.py \
        --config configs/cell_analysis.yaml \
        --cells-dir <cell-activation-representation-dir> \
        --checkpoint <sae-checkpoint.ckpt> \
        --top-n 100
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from safetensors import safe_open
from scipy import stats

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluate.model_loading import (  # noqa: E402
    dataset_identity_from_path,
    load_checkpoint_metadata,
    load_sae,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "cell_analysis.yaml"

# Non-zero activations retained for the rank statistics, at roughly 12 bytes
# each. A top-k encoder keeps well under this; the cap exists so a dense
# encoder fails with an explanation instead of exhausting memory.
MAX_NONZERO_ENTRIES = 64_000_000


@dataclass
class FeatureAggregates:
    """Sufficient statistics for cell-type-by-feature comparisons.

    The ``nonzero_*`` triplet holds every strictly positive activation as
    (feature, cell type, value). It is retained only when rank statistics are
    requested, and is ``None`` otherwise.
    """

    cell_types: list[str]
    cell_counts: np.ndarray
    activation_sum: np.ndarray
    activation_sum_sq: np.ndarray
    active_count: np.ndarray
    n_shards: int
    nonzero_feature_ids: np.ndarray | None = None
    nonzero_type_ids: np.ndarray | None = None
    nonzero_values: np.ndarray | None = None


def project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _config_section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    section = raw.get(name, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"Expected config section {name!r} to be a mapping.")
    return section


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = project_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Cell analysis config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")

    data = _config_section(raw, "data")
    checkpoint = _config_section(raw, "checkpoint")
    analysis = _config_section(raw, "analysis")
    infrastructure = _config_section(raw, "infrastructure")
    figures = _config_section(raw, "figures")
    gene_analysis = _config_section(raw, "gene_analysis")
    output = _config_section(raw, "output")

    cells_dir = data.get("cells_dir")
    checkpoint_path = checkpoint.get("path")
    output_dir = output.get("dir")
    if not cells_dir:
        raise ValueError("Set data.cells_dir in the cell analysis config.")
    if not checkpoint_path:
        raise ValueError("Set checkpoint.path in the cell analysis config.")
    if not output_dir:
        raise ValueError("Set output.dir in the cell analysis config.")

    splits = data.get("splits")
    if splits is not None and (
        not isinstance(splits, list)
        or any(not isinstance(split, str) or not split for split in splits)
    ):
        raise ValueError("data.splits must be null or a list of split names.")

    device = str(infrastructure.get("device", "auto"))
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError(
            "infrastructure.device must be one of: auto, cpu, cuda."
        )

    generate_figures = figures.get("generate", True)
    if not isinstance(generate_figures, bool):
        raise ValueError("figures.generate must be true or false.")
    heatmap_cmap = str(figures.get("heatmap_cmap", "viridis"))
    try:
        plt.get_cmap(heatmap_cmap)
    except ValueError as error:
        raise ValueError(
            f"figures.heatmap_cmap is not a Matplotlib colormap: {heatmap_cmap}"
        ) from error
    heatmap_gamma = float(figures.get("heatmap_gamma", 0.65))
    if heatmap_gamma <= 0:
        raise ValueError("figures.heatmap_gamma must be greater than zero.")

    compute_auroc = analysis.get("auroc", True)
    if not isinstance(compute_auroc, bool):
        raise ValueError("analysis.auroc must be true or false.")

    gene_analysis_enabled = gene_analysis.get("enabled", True)
    if not isinstance(gene_analysis_enabled, bool):
        raise ValueError("gene_analysis.enabled must be true or false.")
    gene_analysis_top_features = int(
        gene_analysis.get("top_features_per_cell_type", 15)
    )
    top_genes_per_feature = int(gene_analysis.get("top_genes_per_feature", 10))
    if gene_analysis_top_features < 1:
        raise ValueError(
            "gene_analysis.top_features_per_cell_type must be at least 1."
        )
    if top_genes_per_feature < 1:
        raise ValueError("gene_analysis.top_genes_per_feature must be at least 1.")

    return {
        "config": str(config_path),
        "cells_dir": str(cells_dir),
        "checkpoint": str(checkpoint_path),
        "splits": splits,
        "output_dir": str(output_dir),
        "sae_name": output.get("sae_name"),
        "dataset_name": output.get("dataset_name"),
        "top_n": int(analysis.get("top_n", 50)),
        "activation_threshold": float(
            analysis.get("activation_threshold", 0.0)
        ),
        "compute_auroc": compute_auroc,
        "batch_size": int(infrastructure.get("batch_size", 256)),
        "device": device,
        "generate_figures": generate_figures,
        "figure_top_n": int(figures.get("top_n", 15)),
        "heatmap_top_n": int(figures.get("heatmap_top_n", 10)),
        "heatmap_cmap": heatmap_cmap,
        "heatmap_gamma": heatmap_gamma,
        "figure_dpi": int(figures.get("dpi", 180)),
        "gene_analysis_enabled": gene_analysis_enabled,
        "gene_features_root": str(
            gene_analysis.get("results_dir", "results/gene_features")
        ),
        "feature_to_genes": gene_analysis.get("feature_to_genes"),
        "gene_analysis_base_model_name": gene_analysis.get("base_model_name"),
        "gene_analysis_top_features": gene_analysis_top_features,
        "top_genes_per_feature": top_genes_per_feature,
    }


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    config_args, _ = config_parser.parse_known_args()
    defaults = load_config(config_args.config)

    parser = argparse.ArgumentParser(
        description=(
            "Encode extracted cell-level expressions with an SAE and rank "
            "cell-type-specific decision-feature candidates."
        )
    )
    parser.add_argument(
        "--config",
        default=defaults["config"],
        help="YAML configuration path (default: configs/cell_analysis.yaml).",
    )
    parser.add_argument(
        "--cells-dir",
        default=defaults["cells_dir"],
        help=(
            "Cell activation representation directory containing data_* split "
            "directories, or one split directory itself."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=defaults["checkpoint"],
        help="Exact SAE .ckpt file corresponding to the extracted layer.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=defaults["splits"],
        help="Override the data.splits list from the YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        default=defaults["output_dir"],
        help="Output root; SAE and dataset subdirectories are appended.",
    )
    parser.add_argument(
        "--sae-name",
        default=defaults["sae_name"],
        help="Override the SAE output directory name.",
    )
    parser.add_argument(
        "--dataset-name",
        default=defaults["dataset_name"],
        help="Override the dataset output directory name.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=defaults["top_n"],
        help="Number of positive-margin features to save per cell type.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=defaults["batch_size"],
        help="Number of cell activations encoded at once.",
    )
    parser.add_argument(
        "--activation-threshold",
        type=float,
        default=defaults["activation_threshold"],
        help="Feature value above which a feature counts as active.",
    )
    parser.add_argument(
        "--auroc",
        action=argparse.BooleanOptionalAction,
        default=defaults["compute_auroc"],
        help=(
            "Compute one-vs-rest AUROC per feature. Disabling it also skips "
            "retaining non-zero activations, which is the memory-heavy part."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=defaults["device"],
        help="Device used for SAE encoding.",
    )
    parser.add_argument(
        "--figures",
        action=argparse.BooleanOptionalAction,
        default=defaults["generate_figures"],
        help="Generate or skip visual artifacts.",
    )
    parser.add_argument(
        "--figure-top-n",
        type=int,
        default=defaults["figure_top_n"],
        help="Features per cell type in bars and scatter highlights.",
    )
    parser.add_argument(
        "--heatmap-top-n",
        type=int,
        default=defaults["heatmap_top_n"],
        help="Features per cell type in the activation heatmap.",
    )
    parser.add_argument(
        "--heatmap-cmap",
        default=defaults["heatmap_cmap"],
        help="Matplotlib colormap used by the activation heatmap.",
    )
    parser.add_argument(
        "--heatmap-gamma",
        type=float,
        default=defaults["heatmap_gamma"],
        help="Power normalization for smoother heatmap color transitions.",
    )
    parser.add_argument(
        "--figure-dpi",
        type=int,
        default=defaults["figure_dpi"],
        help="Resolution of generated PNG figures.",
    )
    parser.add_argument(
        "--gene-analysis",
        action=argparse.BooleanOptionalAction,
        default=defaults["gene_analysis_enabled"],
        help="Join top cell-type features to the matching gene-feature results.",
    )
    parser.add_argument(
        "--gene-features-root",
        default=defaults["gene_features_root"],
        help="Root directory containing hierarchical gene-feature results.",
    )
    parser.add_argument(
        "--feature-to-genes",
        default=defaults["feature_to_genes"],
        help="Optional exact feature_to_genes.jsonl path override.",
    )
    parser.add_argument(
        "--base-model-name",
        default=defaults["gene_analysis_base_model_name"],
        help="Override the base-model directory used for the gene lookup.",
    )
    parser.add_argument(
        "--gene-analysis-top-features",
        type=int,
        default=defaults["gene_analysis_top_features"],
        help="Top ranked features per cell type included in the gene analysis.",
    )
    parser.add_argument(
        "--top-genes-per-feature",
        type=int,
        default=defaults["top_genes_per_feature"],
        help="Maximum genes listed for each selected feature.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    return requested


def _is_split_dir(path: Path) -> bool:
    return (path / "cell_types.json").is_file() and any(
        path.glob("*.safetensors")
    )


def discover_split_dirs(
    cells_dir: str | Path,
    requested_splits: list[str] | None = None,
) -> list[Path]:
    """Resolve a representation directory or a single split directory."""

    root = project_path(cells_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Cell activation directory does not exist: {root}")

    if _is_split_dir(root):
        if requested_splits and root.name not in requested_splits:
            raise ValueError(
                f"{root} is split {root.name!r}, which is not in --splits "
                f"{requested_splits}."
            )
        return [root]

    available = {
        path.name: path
        for path in sorted(root.iterdir())
        if path.is_dir() and _is_split_dir(path)
    }
    if not available:
        raise FileNotFoundError(
            f"No split directories with cell_types.json and safetensors shards in {root}"
        )

    if requested_splits is None:
        return [available[name] for name in sorted(available)]

    missing = [name for name in requested_splits if name not in available]
    if missing:
        raise FileNotFoundError(
            f"Requested split(s) not found under {root}: {', '.join(missing)}. "
            f"Available: {', '.join(sorted(available))}"
        )
    return [available[name] for name in requested_splits]


def load_cell_types(split_dir: Path) -> list[str]:
    path = split_dir / "cell_types.json"
    with path.open("r", encoding="utf-8") as f:
        labels = json.load(f)
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"Expected a non-empty JSON list in {path}")

    cleaned = [str(label).strip() for label in labels]
    if any(not label for label in cleaned) or len(set(cleaned)) != len(cleaned):
        raise ValueError(f"Cell-type labels must be non-empty and unique in {path}")
    return cleaned


def _first_shard(split_dirs: list[Path]) -> Path:
    for split_dir in split_dirs:
        shards = sorted(split_dir.glob("*.safetensors"))
        if shards:
            return shards[0]
    raise FileNotFoundError("No safetensors shards found in the selected splits.")


def activation_width(split_dirs: list[Path]) -> int:
    shard = _first_shard(split_dirs)
    with safe_open(shard, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        required = {"activations", "cell_type_ids"}
        if not required.issubset(keys):
            raise KeyError(f"{shard} is missing tensor(s): {sorted(required - keys)}")
        shape = tuple(f.get_slice("activations").get_shape())
    if len(shape) != 2:
        raise ValueError(f"Expected activations with shape (cells, d_model), got {shape}")
    return int(shape[1])


def validate_checkpoint_alignment(
    split_dirs: list[Path],
    checkpoint_metadata: dict[str, Any],
    sae_metadata: dict[str, Any],
) -> None:
    """Catch common layer, dataset, pooling, prefix, and width mismatches."""

    d_model = activation_width(split_dirs)
    if d_model != int(sae_metadata["d_model"]):
        raise ValueError(
            "Cell activations and SAE have different widths: "
            f"{d_model} != {sae_metadata['d_model']}"
        )

    path_parts = set(split_dirs[0].parts)
    model_meta = checkpoint_metadata.get("model") or {}
    dataset_meta = checkpoint_metadata.get("dataset") or {}

    layer = model_meta.get("layer")
    path_layers = {
        part for part in path_parts if re.fullmatch(r"layer\d+", part)
    }
    if layer is not None and path_layers and f"layer{layer}" not in path_layers:
        raise ValueError(
            f"Checkpoint layer{layer} does not match cell path layer(s) "
            f"{sorted(path_layers)}."
        )

    # Cells from a dataset other than the SAE's training set are a legitimate
    # cross-dataset run, not an error: the output path is named after the cell
    # activations, so those results stay separate. Layer, pooling, and prefix
    # mismatches below remain fatal because they make the encoding meaningless.
    dataset_name = dataset_meta.get("name")
    if dataset_name and dataset_name not in path_parts:
        print(
            f"Note: SAE was trained on {dataset_name!r}, but the cell "
            f"activations come from {split_dirs[0]}. Running cross-dataset."
        )

    pooling = model_meta.get("pooling_method")
    prompt_prefix = model_meta.get("prompt_prefix")
    representation_parts = [
        part for part in path_parts if part.startswith("cell_activations_")
    ]
    if pooling and prompt_prefix is not None and representation_parts:
        prefix_tag = "prefix" if bool(prompt_prefix) else "no_prefix"
        expected = f"cell_activations_{pooling}_{prefix_tag}"
        if expected not in representation_parts:
            raise ValueError(
                f"Checkpoint expects {expected!r}, but cell path contains "
                f"{sorted(representation_parts)}."
            )


def _safe_output_name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    if not safe:
        raise ValueError(f"{label} does not contain a safe path component.")
    return safe


def resolve_output_identity(
    checkpoint_path: Path,
    checkpoint_metadata: dict[str, Any],
    *,
    cells_dir: str | Path | None = None,
    sae_name_override: str | None,
    dataset_name_override: str | None,
) -> tuple[str, str]:
    """Resolve safe output components for the SAE run and source dataset.

    The dataset names the cell activations being analyzed, read off their
    ``<base_model>/layer<N>/<dataset>/`` layout, so analyzing one SAE against
    several datasets keeps the results apart. It falls back to the checkpoint's
    recorded training dataset when the path does not follow that layout.
    """

    sae_name = sae_name_override or checkpoint_path.parent.name
    dataset_meta = checkpoint_metadata.get("dataset") or {}
    dataset_name = dataset_name_override
    if not dataset_name and cells_dir is not None:
        _, dataset_name = dataset_identity_from_path(cells_dir)
    if not dataset_name:
        dataset_name = dataset_meta.get("name")
    if not dataset_name:
        raise ValueError(
            f"Cannot resolve a dataset name: {cells_dir} does not follow "
            "<base_model>/layer<N>/<dataset>/ and the checkpoint metadata "
            "records no dataset.name. Set output.dataset_name or pass "
            "--dataset-name."
        )
    return (
        _safe_output_name(sae_name, "SAE name"),
        _safe_output_name(dataset_name, "dataset name"),
    )


def resolve_base_model_name(
    split_dirs: list[Path],
    override: str | None,
) -> str:
    if override:
        return _safe_output_name(override, "base-model name")
    parts = split_dirs[0].parts
    for index, part in enumerate(parts):
        if re.fullmatch(r"layer\d+", part) and index > 0:
            return _safe_output_name(parts[index - 1], "base-model name")
    raise ValueError(
        "Could not infer the base-model name from the cell activation path. "
        "Set gene_analysis.base_model_name or pass --base-model-name."
    )


def resolve_feature_to_genes_path(
    *,
    explicit_path: str | Path | None,
    results_root: str | Path,
    sae_name: str,
    dataset_name: str,
    base_model_name: str,
) -> Path:
    path = (
        project_path(explicit_path)
        if explicit_path
        else project_path(results_root)
        / sae_name
        / dataset_name
        / base_model_name
        / "feature_to_genes.jsonl"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Matching gene-feature lookup not found: {path}. "
            "Run src/evaluate/gene_features.py for this SAE and dataset, or "
            "set gene_analysis.feature_to_genes explicitly."
        )
    return path


def validate_gene_feature_lookup(
    feature_to_genes_path: Path,
    *,
    checkpoint_path: Path,
    dataset_name: str,
    base_model_name: str,
) -> dict[str, Any]:
    summary_path = feature_to_genes_path.parent / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Gene-feature summary is required to validate the lookup: {summary_path}"
        )
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    if not isinstance(summary, dict):
        raise ValueError(f"Expected a JSON object in {summary_path}")

    identity = summary.get("identity") or {}
    inputs = summary.get("inputs") or {}
    mismatches: list[str] = []
    if identity.get("dataset_name") != dataset_name:
        mismatches.append(
            f"dataset {identity.get('dataset_name')!r} != {dataset_name!r}"
        )
    if identity.get("base_model_name") != base_model_name:
        mismatches.append(
            f"base model {identity.get('base_model_name')!r} != "
            f"{base_model_name!r}"
        )
    lookup_checkpoint = inputs.get("checkpoint")
    if (
        not lookup_checkpoint
        or Path(lookup_checkpoint).resolve() != checkpoint_path.resolve()
    ):
        mismatches.append("checkpoint path does not match the cell-type SAE")
    if mismatches:
        raise ValueError(
            f"Gene-feature lookup {feature_to_genes_path} is incompatible: "
            + "; ".join(mismatches)
        )
    return summary


@torch.inference_mode()
def aggregate_features(
    split_dirs: list[Path],
    sae: torch.nn.Module,
    d_hidden: int,
    *,
    device: str,
    batch_size: int,
    activation_threshold: float,
    collect_nonzero: bool = True,
    max_nonzero_entries: int = MAX_NONZERO_ENTRIES,
) -> FeatureAggregates:
    """Stream cell rows through the SAE and aggregate sufficient statistics.

    With ``collect_nonzero`` the strictly positive activations are kept as well,
    which is what the rank statistics in :func:`one_vs_rest_auroc` need. A
    top-k encoder makes this cheap; ``max_nonzero_entries`` bounds the cost for
    encoders that turn out to be denser than expected.
    """

    split_labels = {split_dir: load_cell_types(split_dir) for split_dir in split_dirs}
    cell_types = sorted(
        {label for labels in split_labels.values() for label in labels}
    )
    global_id = {label: idx for idx, label in enumerate(cell_types)}
    n_types = len(cell_types)

    cell_counts = np.zeros(n_types, dtype=np.int64)
    activation_sum = np.zeros((n_types, d_hidden), dtype=np.float64)
    activation_sum_sq = np.zeros((n_types, d_hidden), dtype=np.float64)
    active_count = np.zeros((n_types, d_hidden), dtype=np.int64)
    n_shards = 0

    nonzero_features: list[np.ndarray] = []
    nonzero_types: list[np.ndarray] = []
    nonzero_values: list[np.ndarray] = []
    n_nonzero = 0

    for split_dir in split_dirs:
        local_labels = split_labels[split_dir]
        local_to_global = torch.tensor(
            [global_id[label] for label in local_labels], dtype=torch.long
        )
        shard_paths = sorted(split_dir.glob("*.safetensors"))
        print(f"Encoding {split_dir.name}: {len(shard_paths)} shard(s)")

        for shard_path in shard_paths:
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                keys = set(f.keys())
                required = {"activations", "cell_type_ids"}
                if not required.issubset(keys):
                    raise KeyError(
                        f"{shard_path} is missing tensor(s): "
                        f"{sorted(required - keys)}"
                    )
                activations = f.get_tensor("activations")
                cell_type_ids = f.get_tensor("cell_type_ids").long()

            if activations.ndim != 2:
                raise ValueError(
                    f"Expected 2D activations in {shard_path}, got {activations.shape}"
                )
            if cell_type_ids.ndim != 1 or len(cell_type_ids) != len(activations):
                raise ValueError(
                    f"Unaligned cell_type_ids in {shard_path}: "
                    f"{tuple(cell_type_ids.shape)} for {len(activations)} rows"
                )
            if len(cell_type_ids) == 0:
                n_shards += 1
                continue
            if (
                int(cell_type_ids.min()) < 0
                or int(cell_type_ids.max()) >= len(local_labels)
            ):
                raise ValueError(f"Out-of-range cell_type_ids in {shard_path}")

            mapped_ids = local_to_global[cell_type_ids]
            for start in range(0, len(activations), batch_size):
                stop = min(start + batch_size, len(activations))
                inputs = activations[start:stop].to(device=device, dtype=torch.float32)
                feature_values = sae.encode(inputs).float()
                if feature_values.shape != (stop - start, d_hidden):
                    raise ValueError(
                        "Unexpected SAE encoding shape: "
                        f"{tuple(feature_values.shape)}, expected "
                        f"{(stop - start, d_hidden)}"
                    )

                batch_ids = mapped_ids[start:stop]
                batch_ids_device = batch_ids.to(device)

                if collect_nonzero:
                    rows_nz, cols_nz = torch.nonzero(
                        feature_values > 0, as_tuple=True
                    )
                    n_nonzero += int(rows_nz.numel())
                    if n_nonzero > max_nonzero_entries:
                        raise MemoryError(
                            f"SAE encodings hold more than {max_nonzero_entries:,} "
                            "non-zero activations, too many to rank in memory. "
                            "Rerun with --no-auroc to skip the rank statistics, "
                            "or raise max_nonzero_entries if the machine can "
                            "afford roughly 12 bytes per entry."
                        )
                    nonzero_features.append(
                        cols_nz.to(torch.int32).cpu().numpy()
                    )
                    nonzero_types.append(
                        batch_ids_device[rows_nz].to(torch.int32).cpu().numpy()
                    )
                    nonzero_values.append(
                        feature_values[rows_nz, cols_nz].cpu().numpy()
                    )

                sum_batch = torch.zeros(
                    (n_types, d_hidden), device=device, dtype=torch.float32
                )
                sum_sq_batch = torch.zeros_like(sum_batch)
                active_batch = torch.zeros_like(sum_batch)
                sum_batch.index_add_(0, batch_ids_device, feature_values)
                sum_sq_batch.index_add_(0, batch_ids_device, feature_values.square())
                active_batch.index_add_(
                    0,
                    batch_ids_device,
                    (feature_values > activation_threshold).float(),
                )

                cell_counts += torch.bincount(
                    batch_ids, minlength=n_types
                ).numpy()
                activation_sum += sum_batch.cpu().numpy()
                activation_sum_sq += sum_sq_batch.cpu().numpy()
                active_count += active_batch.cpu().numpy().astype(np.int64)

            n_shards += 1

    missing = [cell_types[i] for i, count in enumerate(cell_counts) if count == 0]
    if missing:
        raise ValueError(f"No cells were found for cell type(s): {', '.join(missing)}")
    if len(cell_types) < 2:
        raise ValueError("At least two cell types are required for specificity ranking.")

    if collect_nonzero:
        empty_int = np.zeros(0, dtype=np.int32)
        empty_float = np.zeros(0, dtype=np.float32)
        packed_features = (
            np.concatenate(nonzero_features) if nonzero_features else empty_int
        )
        packed_types = (
            np.concatenate(nonzero_types) if nonzero_types else empty_int
        )
        packed_values = (
            np.concatenate(nonzero_values) if nonzero_values else empty_float
        )
        density = n_nonzero / max(int(cell_counts.sum()) * d_hidden, 1)
        print(
            f"Retained {n_nonzero:,} non-zero activations "
            f"({density:.2%} density) for rank statistics"
        )
    else:
        packed_features = packed_types = packed_values = None

    return FeatureAggregates(
        cell_types=cell_types,
        cell_counts=cell_counts,
        activation_sum=activation_sum,
        activation_sum_sq=activation_sum_sq,
        active_count=active_count,
        n_shards=n_shards,
        nonzero_feature_ids=packed_features,
        nonzero_type_ids=packed_types,
        nonzero_values=packed_values,
    )


def one_vs_rest_auroc(
    aggregates: FeatureAggregates,
    d_hidden: int,
) -> np.ndarray:
    """Exact one-vs-rest AUROC for every (cell type, feature) pair.

    For feature ``j`` and type ``a`` this is the probability that ``j`` scores a
    randomly drawn type-``a`` cell above a randomly drawn cell of any other
    type, counting ties as half -- the Mann-Whitney U statistic normalized by
    ``n_a * (n - n_a)``.

    Most cells score exactly zero on most features, so the four blocks of the
    pairwise comparison are evaluated separately and only the non-zero block
    needs ranking. Writing ``p`` for the type-``a`` cells on which the feature
    fires and ``z`` for the other-type cells on which it does not:

    * non-zero vs non-zero -- midranks over the non-zero activations alone;
    * non-zero vs zero     -- all ``p * z`` pairs favour the positive cell;
    * zero vs non-zero     -- no pair favours the positive cell;
    * zero vs zero         -- every pair ties, contributing one half each.

    A feature that never fires ties every pair and scores 0.5.
    """

    features = aggregates.nonzero_feature_ids
    types = aggregates.nonzero_type_ids
    values = aggregates.nonzero_values
    if features is None or types is None or values is None:
        raise ValueError(
            "Rank statistics need the non-zero activations; call "
            "aggregate_features with collect_nonzero=True."
        )

    n_types = len(aggregates.cell_types)
    counts_positive = aggregates.cell_counts.astype(np.float64)
    counts_negative = float(aggregates.cell_counts.sum()) - counts_positive
    if np.any(counts_negative <= 0):
        raise ValueError(
            "One-vs-rest AUROC needs at least one cell outside every type."
        )
    pair_counts = counts_positive * counts_negative

    # A feature that never fires separates nothing, which is the 0.5 default.
    auroc = np.full((n_types, d_hidden), 0.5, dtype=np.float64)
    if features.size == 0:
        return auroc

    order = np.argsort(features, kind="stable")
    features = features[order]
    types = types[order].astype(np.int64, copy=False)
    values = values[order]
    feature_ids = np.arange(d_hidden, dtype=features.dtype)
    starts = np.searchsorted(features, feature_ids, side="left")
    stops = np.searchsorted(features, feature_ids, side="right")

    for feature_id in range(d_hidden):
        start, stop = int(starts[feature_id]), int(stops[feature_id])
        if stop <= start:
            continue
        block_types = types[start:stop]
        midranks = stats.rankdata(values[start:stop])

        # Non-zero cells of each type, and their rank sums within the block.
        n_firing = np.bincount(block_types, minlength=n_types).astype(np.float64)
        rank_sum = np.bincount(block_types, weights=midranks, minlength=n_types)

        # Block 1: both cells fire. Blocks 2 and 4: the negative cell does not.
        firing_elsewhere = float(stop - start) - n_firing
        silent_negatives = counts_negative - firing_elsewhere
        u = rank_sum - n_firing * (n_firing + 1.0) / 2.0
        u += n_firing * silent_negatives
        u += 0.5 * (counts_positive - n_firing) * silent_negatives

        auroc[:, feature_id] = u / pair_counts

    return auroc


def calculate_metrics(
    aggregates: FeatureAggregates,
    *,
    compute_auroc: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Calculate target-vs-other metrics and rank every feature per cell type."""

    counts = aggregates.cell_counts.astype(np.float64)
    sums = aggregates.activation_sum
    sum_sq = aggregates.activation_sum_sq
    active_count = aggregates.active_count
    means = sums / counts[:, None]
    variances = np.maximum(sum_sq / counts[:, None] - np.square(means), 0.0)
    activation_rates = active_count / counts[:, None]
    mean_when_active = np.divide(
        sums,
        active_count,
        out=np.zeros_like(sums),
        where=active_count > 0,
    )

    total_count = counts.sum()
    total_sum = sums.sum(axis=0)
    total_sum_sq = sum_sq.sum(axis=0)
    mean_mass = means.sum(axis=0)
    d_hidden = means.shape[1]

    if compute_auroc:
        print("Computing one-vs-rest AUROC")
        auroc = one_vs_rest_auroc(aggregates, d_hidden)
    else:
        auroc = np.full((len(aggregates.cell_types), d_hidden), np.nan)
    cohens_d = np.zeros((len(aggregates.cell_types), d_hidden), dtype=np.float64)

    rows: list[dict[str, Any]] = []

    for type_idx, cell_type in enumerate(aggregates.cell_types):
        other_indices = np.asarray(
            [idx for idx in range(len(aggregates.cell_types)) if idx != type_idx]
        )
        other_means = means[other_indices]
        strongest_other_offset = np.argmax(other_means, axis=0)
        strongest_other_idx = other_indices[strongest_other_offset]
        feature_ids = np.arange(d_hidden)
        strongest_other_mean = other_means[
            strongest_other_offset, feature_ids
        ]
        max_other_rate = activation_rates[other_indices].max(axis=0)

        rest_count = total_count - counts[type_idx]
        rest_mean = (total_sum - sums[type_idx]) / rest_count
        rest_variance = np.maximum(
            (total_sum_sq - sum_sq[type_idx]) / rest_count - np.square(rest_mean),
            0.0,
        )
        mean_difference_vs_rest = means[type_idx] - rest_mean
        pooled_std = np.sqrt((variances[type_idx] + rest_variance) / 2.0)
        standardized_effect = np.divide(
            mean_difference_vs_rest,
            pooled_std,
            out=np.zeros_like(mean_difference_vs_rest),
            where=pooled_std > 0,
        )

        # Cohen's d proper: the pooled standard deviation weights each group by
        # its degrees of freedom, unlike the equal-weight standardized effect
        # above, and the two diverge whenever the groups are unbalanced. The
        # streamed variances are population variances, so n * var recovers the
        # (n - 1) * s^2 term without a second pass.
        pooled_dof = max(counts[type_idx] + rest_count - 2.0, 1.0)
        cohens_pooled_std = np.sqrt(
            (counts[type_idx] * variances[type_idx] + rest_count * rest_variance)
            / pooled_dof
        )
        cohens_d[type_idx] = np.divide(
            mean_difference_vs_rest,
            cohens_pooled_std,
            out=np.zeros_like(mean_difference_vs_rest),
            where=cohens_pooled_std > 0,
        )
        specificity_margin = means[type_idx] - strongest_other_mean
        specificity_share = np.divide(
            means[type_idx],
            mean_mass,
            out=np.zeros_like(means[type_idx]),
            where=mean_mass > 0,
        )
        other_active_count = active_count.sum(axis=0) - active_count[type_idx]

        order = np.lexsort((-means[type_idx], -specificity_margin))
        rank_by_feature = np.empty(d_hidden, dtype=np.int64)
        rank_by_feature[order] = np.arange(1, d_hidden + 1)

        for feature_id in range(d_hidden):
            rows.append(
                {
                    "cell_type": cell_type,
                    "rank": int(rank_by_feature[feature_id]),
                    "feature_id": feature_id,
                    "specificity_margin": float(specificity_margin[feature_id]),
                    "mean_activation": float(means[type_idx, feature_id]),
                    "mean_activation_when_active": float(
                        mean_when_active[type_idx, feature_id]
                    ),
                    "activation_rate": float(
                        activation_rates[type_idx, feature_id]
                    ),
                    "strongest_other_cell_type": aggregates.cell_types[
                        int(strongest_other_idx[feature_id])
                    ],
                    "strongest_other_mean_activation": float(
                        strongest_other_mean[feature_id]
                    ),
                    "max_other_activation_rate": float(max_other_rate[feature_id]),
                    "rest_mean_activation": float(rest_mean[feature_id]),
                    "mean_difference_vs_rest": float(
                        mean_difference_vs_rest[feature_id]
                    ),
                    "standardized_effect_vs_rest": float(
                        standardized_effect[feature_id]
                    ),
                    "cohens_d_vs_rest": float(cohens_d[type_idx, feature_id]),
                    "auroc_vs_rest": float(auroc[type_idx, feature_id]),
                    "specificity_share": float(specificity_share[feature_id]),
                    "active_cells": int(active_count[type_idx, feature_id]),
                    "cell_count": int(aggregates.cell_counts[type_idx]),
                    "exclusive_to_cell_type": bool(
                        active_count[type_idx, feature_id] > 0
                        and other_active_count[feature_id] == 0
                    ),
                }
            )

    arrays = {
        "feature_ids": np.arange(d_hidden, dtype=np.int64),
        "cell_types": np.asarray(aggregates.cell_types),
        "cell_counts": aggregates.cell_counts,
        "mean_activation": means,
        "mean_activation_when_active": mean_when_active,
        "activation_rate": activation_rates,
        "activation_sum": sums,
        "activation_sum_sq": sum_sq,
        "active_count": active_count,
        "cohens_d_vs_rest": cohens_d,
        "auroc_vs_rest": auroc,
    }
    return rows, arrays


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: list[str] | None = None,
) -> None:
    if fieldnames is None:
        if not rows:
            raise ValueError(f"fieldnames are required for an empty table: {path}")
        fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_top_genes_for_features(
    path: Path,
    feature_ids: set[int],
    *,
    top_genes_per_feature: int,
) -> dict[int, tuple[list[str], int]]:
    """Load only selected feature records from a feature-to-genes JSONL file."""

    if not feature_ids:
        return {}

    lookup: dict[int, tuple[list[str], int]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed JSON at {path}:{line_number}") from error
            if not isinstance(record, dict) or not {
                "feature",
                "genes",
            }.issubset(record):
                raise ValueError(f"Malformed feature lookup at {path}:{line_number}")
            feature_id = int(record["feature"])
            if feature_id not in feature_ids:
                continue
            if feature_id in lookup:
                raise ValueError(f"Duplicate feature {feature_id} in {path}")
            genes_value = record["genes"]
            if not isinstance(genes_value, list):
                raise ValueError(
                    f"Expected a genes list for feature {feature_id} in {path}"
                )
            genes = [str(gene) for gene in genes_value]
            lookup[feature_id] = (
                genes[:top_genes_per_feature],
                len(genes),
            )
            if len(lookup) == len(feature_ids):
                break
    return lookup


def write_top_feature_gene_analysis(
    output_dir: Path,
    top_rows: list[dict[str, Any]],
    *,
    feature_to_genes_path: Path,
    gene_feature_summary: dict[str, Any],
    top_features_per_cell_type: int,
    top_genes_per_feature: int,
) -> dict[str, Any]:
    selected = [
        row
        for row in top_rows
        if int(row["rank"]) <= top_features_per_cell_type
    ]
    feature_ids = {int(row["feature_id"]) for row in selected}
    gene_lookup = load_top_genes_for_features(
        feature_to_genes_path,
        feature_ids,
        top_genes_per_feature=top_genes_per_feature,
    )

    records: list[dict[str, Any]] = []
    for row in selected:
        feature_id = int(row["feature_id"])
        lookup_found = feature_id in gene_lookup
        genes, available_count = gene_lookup.get(feature_id, ([], 0))
        records.append(
            {
                "cell_type": row["cell_type"],
                "rank": int(row["rank"]),
                "feature_id": feature_id,
                "specificity_margin": float(row["specificity_margin"]),
                "mean_activation": float(row["mean_activation"]),
                "activation_rate": float(row["activation_rate"]),
                "standardized_effect_vs_rest": float(
                    row["standardized_effect_vs_rest"]
                ),
                "cohens_d_vs_rest": float(row["cohens_d_vs_rest"]),
                "auroc_vs_rest": float(row["auroc_vs_rest"]),
                "specificity_share": float(row["specificity_share"]),
                "lookup_found": lookup_found,
                "n_genes_available_in_lookup": available_count,
                "n_genes_listed": len(genes),
                "top_genes": genes,
            }
        )

    jsonl_path = output_dir / "top_feature_genes.jsonl"
    csv_path = output_dir / "top_feature_genes.csv"
    analysis_path = output_dir / "top_feature_gene_analysis.md"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")

    csv_rows = [
        {
            **{key: value for key, value in record.items() if key != "top_genes"},
            "top_genes": ";".join(record["top_genes"]),
        }
        for record in records
    ]
    if csv_rows:
        write_csv(csv_path, csv_rows)
    else:
        csv_path.write_text("cell_type,rank,feature_id,top_genes\n", encoding="utf-8")

    lookup_threshold = (gene_feature_summary.get("settings") or {}).get(
        "threshold"
    )
    matched = sum(bool(record["lookup_found"]) for record in records)
    missing_feature_ids = sorted(
        {
            int(record["feature_id"])
            for record in records
            if not record["lookup_found"]
        }
    )
    with analysis_path.open("w", encoding="utf-8") as f:
        f.write("# Top cell-type feature gene analysis\n\n")
        f.write(f"Gene-feature source: `{feature_to_genes_path}`\n\n")
        if lookup_threshold is not None:
            f.write(
                "Genes are ordered by their vocabulary activation for each "
                f"feature after the gene-feature threshold `{lookup_threshold}`.\n\n"
            )
        f.write(
            f"Matched `{matched}` of `{len(records)}` selected cell-type feature "
            f"rows; at most `{top_genes_per_feature}` genes are shown per feature.\n\n"
        )
        f.write(
            "These genes are vocabulary associations for interpreting each SAE "
            "feature; they are not a causal marker-gene claim.\n\n"
        )
        f.write(
            "Features are ranked by specificity margin. Cohen's d and AUROC "
            "are one-vs-rest discrimination diagnostics on that ranking: the "
            "margin measures the size of the activation gap to the nearest "
            "competing type, while the two rank-and-scale statistics measure "
            "how cleanly the feature separates this type from all others.\n\n"
        )
        cell_types = list(dict.fromkeys(str(row["cell_type"]) for row in records))
        for cell_type in cell_types:
            f.write(f"## {cell_type.replace('_', ' ')}\n\n")
            f.write(
                "| Rank | Feature | Specificity margin | Mean activation | "
                "Activation rate | Cohen's d | AUROC | Top genes |\n"
            )
            f.write("|---:|---:|---:|---:|---:|---:|---:|---|\n")
            for record in records:
                if record["cell_type"] != cell_type:
                    continue
                genes_text = ", ".join(record["top_genes"]) or "—"
                auroc_value = record["auroc_vs_rest"]
                auroc_text = (
                    "—" if np.isnan(auroc_value) else f"{auroc_value:.3f}"
                )
                f.write(
                    f"| {record['rank']} | F{record['feature_id']} | "
                    f"{record['specificity_margin']:.3f} | "
                    f"{record['mean_activation']:.3f} | "
                    f"{record['activation_rate']:.3f} | "
                    f"{record['cohens_d_vs_rest']:.3f} | "
                    f"{auroc_text} | {genes_text} |\n"
                )
            f.write("\n")

    return {
        "jsonl": jsonl_path,
        "csv": csv_path,
        "analysis": analysis_path,
        "selected_feature_rows": len(records),
        "matched_feature_rows": matched,
        "missing_feature_ids": missing_feature_ids,
    }


def _top_positive_rows(
    rows: list[dict[str, Any]],
    cell_type: str,
    top_n: int,
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row["cell_type"] == cell_type and row["specificity_margin"] > 0
    ]
    return sorted(selected, key=lambda row: row["rank"])[:top_n]


def _plot_specificity_bars(
    figure_dir: Path,
    rows: list[dict[str, Any]],
    cell_types: list[str],
    *,
    top_n: int,
    dpi: int,
) -> Path:
    colors = plt.get_cmap("tab10")
    fig_height = max(5.5, 0.34 * top_n + 2.2)
    fig, axes = plt.subplots(
        1,
        len(cell_types),
        figsize=(max(11.0, 4.8 * len(cell_types)), fig_height),
        squeeze=False,
        constrained_layout=True,
    )

    for type_idx, cell_type in enumerate(cell_types):
        ax = axes[0, type_idx]
        selected = list(
            reversed(_top_positive_rows(rows, cell_type, top_n))
        )
        values = [float(row["specificity_margin"]) for row in selected]
        labels = [f"F{row['feature_id']}" for row in selected]
        bars = ax.barh(labels, values, color=colors(type_idx), alpha=0.85)
        if bars:
            ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
        ax.set_title(cell_type.replace("_", " "))
        ax.set_xlabel("Specificity margin")
        ax.grid(axis="x", alpha=0.25, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Top cell-type-specific SAE features",
        fontsize=15,
        fontweight="bold",
    )
    path = figure_dir / "top_specificity_margins.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _plot_activation_heatmap(
    figure_dir: Path,
    rows: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    cell_types: list[str],
    *,
    top_n: int,
    cmap: str,
    gamma: float,
    dpi: int,
) -> Path:
    selected = [
        row
        for cell_type in cell_types
        for row in _top_positive_rows(rows, cell_type, top_n)
    ]
    if not selected:
        raise ValueError("No positive-margin features are available for the heatmap.")

    feature_ids = np.asarray(
        [int(row["feature_id"]) for row in selected], dtype=np.int64
    )
    means = np.asarray(arrays["mean_activation"], dtype=np.float64)
    raw_matrix = means[:, feature_ids].T
    row_max = raw_matrix.max(axis=1, keepdims=True)
    normalized = np.divide(
        raw_matrix,
        row_max,
        out=np.zeros_like(raw_matrix),
        where=row_max > 0,
    )

    fig_height = max(8.0, 0.29 * len(selected) + 2.5)
    fig, ax = plt.subplots(
        figsize=(max(8.0, 2.2 * len(cell_types) + 3.0), fig_height),
        constrained_layout=True,
    )
    color_norm = matplotlib.colors.PowerNorm(
        gamma=gamma,
        vmin=0.0,
        vmax=1.0,
    )
    image = ax.imshow(
        normalized,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=color_norm,
    )
    ax.set_xticks(np.arange(len(cell_types)))
    ax.set_xticklabels([label.replace("_", " ") for label in cell_types])
    ax.set_yticks(np.arange(len(selected)))
    ax.set_yticklabels(
        [
            f"F{row['feature_id']}  ({row['cell_type'].replace('_', ' ')})"
            for row in selected
        ],
        fontsize=8,
    )
    ax.set_xlabel("Cell type")
    ax.set_ylabel("Feature (ranked target cell type)")
    ax.set_title(
        "Top-feature mean activations\n"
        "Color is normalized within each feature; labels show raw magnitude",
        fontweight="bold",
    )

    for row_idx in range(raw_matrix.shape[0]):
        for column_idx in range(raw_matrix.shape[1]):
            scaled_color = float(color_norm(normalized[row_idx, column_idx]))
            color = "white" if scaled_color < 0.52 else "black"
            ax.text(
                column_idx,
                row_idx,
                f"{raw_matrix[row_idx, column_idx]:.1f}",
                ha="center",
                va="center",
                fontsize=7,
                color=color,
            )

    previous_target = selected[0]["cell_type"]
    for row_idx, row in enumerate(selected[1:], start=1):
        if row["cell_type"] != previous_target:
            ax.axhline(row_idx - 0.5, color="white", linewidth=2.0)
            previous_target = row["cell_type"]

    colorbar = fig.colorbar(image, ax=ax, shrink=0.65, pad=0.03)
    colorbar.set_label("Mean activation / feature maximum")
    path = figure_dir / "top_feature_activation_heatmap.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _plot_specificity_prevalence(
    figure_dir: Path,
    rows: list[dict[str, Any]],
    cell_types: list[str],
    *,
    top_n: int,
    dpi: int,
) -> Path:
    colors = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(10.5, 7.0), constrained_layout=True)

    for type_idx, cell_type in enumerate(cell_types):
        candidates = [
            row
            for row in rows
            if row["cell_type"] == cell_type
            and row["specificity_margin"] > 0
            and row["mean_activation"] > 0
        ]
        ax.scatter(
            [row["activation_rate"] for row in candidates],
            [row["specificity_margin"] for row in candidates],
            s=11,
            alpha=0.18,
            color=colors(type_idx),
            edgecolors="none",
            label=cell_type.replace("_", " "),
        )
        top_rows = _top_positive_rows(rows, cell_type, top_n)
        ax.scatter(
            [row["activation_rate"] for row in top_rows],
            [row["specificity_margin"] for row in top_rows],
            s=48,
            alpha=0.9,
            color=colors(type_idx),
            edgecolors="white",
            linewidths=0.6,
        )
        annotation_offsets = ((-46, 8), (7, 7), (-46, -15))
        for row in top_rows[:1]:
            ax.annotate(
                f"F{row['feature_id']}",
                (row["activation_rate"], row["specificity_margin"]),
                xytext=annotation_offsets[type_idx % len(annotation_offsets)],
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Activation rate in target cell type")
    ax.set_ylabel("Specificity margin")
    ax.set_title(
        "Feature specificity versus prevalence",
        fontsize=15,
        fontweight="bold",
    )
    ax.grid(alpha=0.22, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title="Target cell type", frameon=False)
    path = figure_dir / "specificity_vs_activation_rate.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def write_figures(
    output_dir: Path,
    rows: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    *,
    top_n: int,
    heatmap_top_n: int,
    heatmap_cmap: str,
    heatmap_gamma: float,
    dpi: int,
) -> dict[str, Path]:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    cell_types = [str(label) for label in arrays["cell_types"].tolist()]
    return {
        "top_specificity_margins": _plot_specificity_bars(
            figure_dir, rows, cell_types, top_n=top_n, dpi=dpi
        ),
        "top_feature_activation_heatmap": _plot_activation_heatmap(
            figure_dir,
            rows,
            arrays,
            cell_types,
            top_n=heatmap_top_n,
            cmap=heatmap_cmap,
            gamma=heatmap_gamma,
            dpi=dpi,
        ),
        "specificity_vs_activation_rate": _plot_specificity_prevalence(
            figure_dir, rows, cell_types, top_n=top_n, dpi=dpi
        ),
    }


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    aggregates: FeatureAggregates,
    *,
    config_path: Path,
    output_root: Path,
    sae_name: str,
    dataset_name: str,
    base_model_name: str | None,
    split_dirs: list[Path],
    checkpoint_path: Path,
    sae_metadata: dict[str, Any],
    top_n: int,
    activation_threshold: float,
    batch_size: int,
    device: str,
    generate_figures: bool,
    figure_top_n: int,
    heatmap_top_n: int,
    heatmap_cmap: str,
    heatmap_gamma: float,
    figure_dpi: int,
    gene_analysis_enabled: bool,
    feature_to_genes_path: Path | None,
    gene_feature_summary: dict[str, Any],
    gene_analysis_top_features: int,
    top_genes_per_feature: int,
    compute_auroc: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = sorted(rows, key=lambda row: (row["cell_type"], row["rank"]))
    top_rows = [
        row
        for row in all_rows
        if row["rank"] <= top_n and row["specificity_margin"] > 0
    ]

    all_path = output_dir / "all_cell_type_features.csv"
    top_path = output_dir / "top_cell_type_features.csv"
    arrays_path = output_dir / "cell_type_feature_stats.npz"
    summary_path = output_dir / "summary.json"
    write_csv(all_path, all_rows)
    write_csv(top_path, top_rows, fieldnames=list(all_rows[0]))
    np.savez_compressed(arrays_path, **arrays)
    gene_analysis_outputs: dict[str, Any] = {}
    if gene_analysis_enabled:
        if feature_to_genes_path is None:
            raise ValueError(
                "feature_to_genes_path is required when gene analysis is enabled."
            )
        gene_analysis_outputs = write_top_feature_gene_analysis(
            output_dir,
            top_rows,
            feature_to_genes_path=feature_to_genes_path,
            gene_feature_summary=gene_feature_summary,
            top_features_per_cell_type=gene_analysis_top_features,
            top_genes_per_feature=top_genes_per_feature,
        )
    figure_paths = (
        write_figures(
            output_dir,
            all_rows,
            arrays,
            top_n=figure_top_n,
            heatmap_top_n=heatmap_top_n,
            heatmap_cmap=heatmap_cmap,
            heatmap_gamma=heatmap_gamma,
            dpi=figure_dpi,
        )
        if generate_figures
        else {}
    )

    top_counts = {
        cell_type: sum(row["cell_type"] == cell_type for row in top_rows)
        for cell_type in aggregates.cell_types
    }
    summary = {
        "inputs": {
            "config": str(config_path),
            "split_dirs": [str(path) for path in split_dirs],
            "checkpoint": str(checkpoint_path),
            "feature_to_genes": (
                str(feature_to_genes_path) if feature_to_genes_path else None
            ),
        },
        "settings": {
            "top_n": top_n,
            "activation_threshold": activation_threshold,
            "batch_size": batch_size,
            "device": device,
            "ranking_metric": "specificity_margin",
            "discrimination_metrics": (
                ["cohens_d_vs_rest", "auroc_vs_rest"]
                if compute_auroc
                else ["cohens_d_vs_rest"]
            ),
            "auroc": compute_auroc,
            "figures": {
                "generate": generate_figures,
                "top_n": figure_top_n,
                "heatmap_top_n": heatmap_top_n,
                "heatmap_cmap": heatmap_cmap,
                "heatmap_gamma": heatmap_gamma,
                "dpi": figure_dpi,
            },
            "gene_analysis": {
                "enabled": gene_analysis_enabled,
                "base_model_name": base_model_name,
                "top_features_per_cell_type": gene_analysis_top_features,
                "top_genes_per_feature": top_genes_per_feature,
            },
        },
        "sae": sae_metadata,
        "data": {
            "cell_types": aggregates.cell_types,
            "cell_counts": {
                label: int(count)
                for label, count in zip(
                    aggregates.cell_types, aggregates.cell_counts, strict=True
                )
            },
            "total_cells": int(aggregates.cell_counts.sum()),
            "shards": aggregates.n_shards,
        },
        "results": {
            "output_root": str(output_root),
            "output_dir": str(output_dir),
            "sae_name": sae_name,
            "dataset_name": dataset_name,
            "base_model_name": base_model_name,
            "positive_margin_features_written": top_counts,
            "all_features_csv": str(all_path),
            "top_features_csv": str(top_path),
            "aggregate_arrays": str(arrays_path),
            "figure_feature_selection": {
                "source_csv": str(top_path),
                "bar_and_scatter_filter": f"rank <= {figure_top_n}",
                "heatmap_filter": f"rank <= {heatmap_top_n}",
            },
            "figures": {
                name: str(path) for name, path in figure_paths.items()
            },
            "gene_analysis": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in gene_analysis_outputs.items()
            },
        },
        "interpretation": (
            "Positive specificity_margin features are observational candidates. "
            "Use feature ablation or activation steering to test causal control."
        ),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    return {
        "all": all_path,
        "top": top_path,
        "arrays": arrays_path,
        "summary": summary_path,
        "figures": figure_paths,
        "gene_analysis": gene_analysis_outputs,
    }


def main() -> None:
    args = parse_args()
    if args.top_n < 1:
        raise ValueError("--top-n must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.figure_top_n < 1:
        raise ValueError("--figure-top-n must be at least 1.")
    if args.heatmap_top_n < 1:
        raise ValueError("--heatmap-top-n must be at least 1.")
    if args.heatmap_gamma <= 0:
        raise ValueError("--heatmap-gamma must be greater than zero.")
    try:
        plt.get_cmap(args.heatmap_cmap)
    except ValueError as error:
        raise ValueError(
            f"--heatmap-cmap is not a Matplotlib colormap: {args.heatmap_cmap}"
        ) from error
    if args.figure_dpi < 1:
        raise ValueError("--figure-dpi must be at least 1.")
    if args.gene_analysis_top_features < 1:
        raise ValueError("--gene-analysis-top-features must be at least 1.")
    if args.top_genes_per_feature < 1:
        raise ValueError("--top-genes-per-feature must be at least 1.")

    split_dirs = discover_split_dirs(args.cells_dir, args.splits)
    checkpoint_path = project_path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SAE checkpoint does not exist: {checkpoint_path}")
    output_root = project_path(args.output_dir)
    config_path = project_path(args.config)
    device = resolve_device(args.device)

    print(f"Loading SAE: {checkpoint_path}")
    sae, sae_metadata = load_sae(checkpoint_path, device=device)
    checkpoint_metadata = load_checkpoint_metadata(checkpoint_path)
    validate_checkpoint_alignment(
        split_dirs, checkpoint_metadata, sae_metadata
    )
    sae_name, dataset_name = resolve_output_identity(
        checkpoint_path,
        checkpoint_metadata,
        cells_dir=args.cells_dir,
        sae_name_override=args.sae_name,
        dataset_name_override=args.dataset_name,
    )
    base_model_name: str | None = None
    feature_to_genes_path: Path | None = None
    gene_feature_summary: dict[str, Any] = {}
    if args.gene_analysis:
        base_model_name = resolve_base_model_name(
            split_dirs, args.base_model_name
        )
        feature_to_genes_path = resolve_feature_to_genes_path(
            explicit_path=args.feature_to_genes,
            results_root=args.gene_features_root,
            sae_name=sae_name,
            dataset_name=dataset_name,
            base_model_name=base_model_name,
        )
        gene_feature_summary = validate_gene_feature_lookup(
            feature_to_genes_path,
            checkpoint_path=checkpoint_path,
            dataset_name=dataset_name,
            base_model_name=base_model_name,
        )
        print(f"Using gene-feature lookup: {feature_to_genes_path}")
    output_dir = output_root / sae_name / dataset_name
    print(
        f"SAE: {sae_metadata['sae_type']} with "
        f"{sae_metadata['d_hidden']:,} features on {device}"
    )

    aggregates = aggregate_features(
        split_dirs,
        sae,
        int(sae_metadata["d_hidden"]),
        device=device,
        batch_size=args.batch_size,
        activation_threshold=args.activation_threshold,
        collect_nonzero=args.auroc,
    )
    rows, arrays = calculate_metrics(aggregates, compute_auroc=args.auroc)
    outputs = write_outputs(
        output_dir,
        rows,
        arrays,
        aggregates,
        config_path=config_path,
        output_root=output_root,
        sae_name=sae_name,
        dataset_name=dataset_name,
        base_model_name=base_model_name,
        split_dirs=split_dirs,
        checkpoint_path=checkpoint_path,
        sae_metadata=sae_metadata,
        top_n=args.top_n,
        activation_threshold=args.activation_threshold,
        batch_size=args.batch_size,
        device=device,
        generate_figures=args.figures,
        figure_top_n=args.figure_top_n,
        heatmap_top_n=args.heatmap_top_n,
        heatmap_cmap=args.heatmap_cmap,
        heatmap_gamma=args.heatmap_gamma,
        figure_dpi=args.figure_dpi,
        gene_analysis_enabled=args.gene_analysis,
        feature_to_genes_path=feature_to_genes_path,
        gene_feature_summary=gene_feature_summary,
        gene_analysis_top_features=args.gene_analysis_top_features,
        top_genes_per_feature=args.top_genes_per_feature,
        compute_auroc=args.auroc,
    )

    print(
        "Cells by type: "
        + ", ".join(
            f"{label}={int(count):,}"
            for label, count in zip(
                aggregates.cell_types, aggregates.cell_counts, strict=True
            )
        )
    )
    print(f"Top decision-feature candidates: {outputs['top']}")
    print(f"All feature statistics: {outputs['all']}")
    if outputs["figures"]:
        print(f"Figures: {output_dir / 'figures'}")
    if outputs["gene_analysis"]:
        print(
            "Top-feature gene analysis: "
            f"{outputs['gene_analysis']['analysis']}"
        )
    print(f"Summary: {outputs['summary']}")


if __name__ == "__main__":
    main()
