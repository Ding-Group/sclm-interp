#!/usr/bin/env python3
"""Compare SAE features between sibling-confusion failures and correct calls.

The cell-type annotation runs under ``results/cell_type_annotation`` record one
prediction per cell, in the same order the cell activation extractor wrote its
``datasets/cells`` shards. That alignment lets a failure bucket computed from
the generated text be carried straight onto the matching SAE feature vectors.

The script does three things for one target cell type (default
``cd14_monocytes``):

1. Bucket every incorrect prediction for the target by how the answer relates
   to the gold label in the blood lineage (``wrong_sibling``,
   ``under_specified``, ``wrong_lineage``, ``off_target``, ``vacuous``), and
   keep the ``wrong_sibling`` indices — cells the model placed in the right
   lineage but resolved to the wrong terminal type.
2. Collect the indices of the target cells the model got right.
3. Encode both groups through the layer's SAE and test, feature by feature,
   whether the sibling failures carry a different feature profile than the
   correct calls.

The comparison reports per-feature Welch t-tests with Benjamini-Hochberg FDR
control, rank AUROC, Cohen's d and activation-rate differences, plus a
permutation test on the distance between the two group centroids so a null
result is reportable as a null result. It also scores each cell against the
cell-type marker features from a matching ``results/cell_type_features`` run,
which answers the direct question: is the monocyte signature weaker in the
cells the model called a dendritic cell?

These are observational contrasts on a small, self-selected group of cells.
They identify features worth intervening on; they are not causal evidence that
a feature drives the model's answer.

Example:
    .venv/bin/python src/analysis/annotation_analysis.py

    .venv/bin/python src/analysis/annotation_analysis.py \
        --predictions-dir results/cell_type_annotation/base/purified-pbmc-subset-2/C2S-Scale-Gemma-2-2B \
        --cell-type cd14_monocytes \
        --layer 20 \
        --checkpoint checkpoints/gemma-2b/layer20_topk_exp8_last_no_prefix_23/last.ckpt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
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

from evaluate.model_loading import load_checkpoint_metadata, load_sae  # noqa: E402


DEFAULT_PREDICTIONS_DIR = (
    "results/cell_type_annotation/base/purified-pbmc-subset-2/C2S-Scale-Gemma-2-2B"
)
DEFAULT_CHECKPOINT = (
    "checkpoints/gemma-2b/layer20_topk_exp8_last_no_prefix_23/last.ckpt"
)
DEFAULT_LAYER = 20
DEFAULT_POOLING_DIRNAME = "cell_activations_last_no_prefix"
DEFAULT_SPLITS = ("data_train", "data_val", "data_test")
DEFAULT_OUTPUT_ROOT = "results/annotation_failure_analysis"
CELL_TYPE_FEATURES_ROOT = "results/cell_type_features"
GENE_FEATURES_ROOT = "results/gene_features"

_TAG_RE = re.compile(r"<[^>]*>")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


# ---------------------------------------------------------------------------
# Cell ontology
# ---------------------------------------------------------------------------
#
# Hand-encoded from the standard Cell Ontology blood hierarchy, covering the
# terms these runs actually emit plus the near neighbours of those terms. It is
# used only to explain *why* a prediction failed; scoring stays with the run's
# own label matcher, so nothing here can turn a wrong answer into a right one.
#
# ``lineage``     - myeloid / lymphoid / other branch of haematopoiesis.
# ``granularity`` - terminal: a committed mature identity, the granularity the
#                   task asks for; intermediate: a real cell class that is
#                   coarser than that; root: contentless; progenitor: right
#                   tissue, wrong maturation stage; non_blood: outside blood.

@dataclass(frozen=True)
class TermInfo:
    lineage: str
    granularity: str


CELL_TERMS: dict[str, TermInfo] = {
    # Myeloid
    "monocyte": TermInfo("myeloid", "terminal"),
    "classical monocyte": TermInfo("myeloid", "terminal"),
    "non classical monocyte": TermInfo("myeloid", "terminal"),
    "intermediate monocyte": TermInfo("myeloid", "terminal"),
    "cd14 positive monocyte": TermInfo("myeloid", "terminal"),
    "cd16 positive monocyte": TermInfo("myeloid", "terminal"),
    "macrophage": TermInfo("myeloid", "terminal"),
    "inflammatory macrophage": TermInfo("myeloid", "terminal"),
    "alveolar macrophage": TermInfo("myeloid", "terminal"),
    "microglial cell": TermInfo("myeloid", "terminal"),
    "dendritic cell": TermInfo("myeloid", "terminal"),
    "conventional dendritic cell": TermInfo("myeloid", "terminal"),
    "myeloid dendritic cell": TermInfo("myeloid", "terminal"),
    "plasmacytoid dendritic cell": TermInfo("myeloid", "terminal"),
    "plasmacytoid dendritic cell human": TermInfo("myeloid", "terminal"),
    "neutrophil": TermInfo("myeloid", "terminal"),
    "eosinophil": TermInfo("myeloid", "terminal"),
    "basophil": TermInfo("myeloid", "terminal"),
    "mast cell": TermInfo("myeloid", "terminal"),
    "granulocyte": TermInfo("myeloid", "intermediate"),
    "myeloid leukocyte": TermInfo("myeloid", "intermediate"),
    "myeloid cell": TermInfo("myeloid", "intermediate"),
    "mononuclear phagocyte": TermInfo("myeloid", "intermediate"),
    "phagocyte": TermInfo("myeloid", "intermediate"),
    "platelet": TermInfo("myeloid", "terminal"),
    "megakaryocyte": TermInfo("myeloid", "terminal"),
    "erythrocyte": TermInfo("myeloid", "terminal"),
    # Lymphoid
    "b cell": TermInfo("lymphoid", "terminal"),
    "b lymphocyte": TermInfo("lymphoid", "terminal"),
    "naive b cell": TermInfo("lymphoid", "terminal"),
    "memory b cell": TermInfo("lymphoid", "terminal"),
    "plasma cell": TermInfo("lymphoid", "terminal"),
    "plasmablast": TermInfo("lymphoid", "terminal"),
    "t cell": TermInfo("lymphoid", "terminal"),
    "mature t cell": TermInfo("lymphoid", "terminal"),
    "cd4 positive alpha beta t cell": TermInfo("lymphoid", "terminal"),
    "cd8 positive alpha beta t cell": TermInfo("lymphoid", "terminal"),
    "naive thymus derived cd4 positive alpha beta t cell": TermInfo(
        "lymphoid", "terminal"
    ),
    "naive t cell": TermInfo("lymphoid", "terminal"),
    "memory t cell": TermInfo("lymphoid", "terminal"),
    "regulatory t cell": TermInfo("lymphoid", "terminal"),
    "natural killer cell": TermInfo("lymphoid", "terminal"),
    "nk cell": TermInfo("lymphoid", "terminal"),
    "innate lymphoid cell": TermInfo("lymphoid", "terminal"),
    "lymphocyte": TermInfo("lymphoid", "intermediate"),
    # Lineage-agnostic blood terms
    "leukocyte": TermInfo("blood", "intermediate"),
    "hematopoietic cell": TermInfo("blood", "intermediate"),
    "blood cell": TermInfo("blood", "intermediate"),
    "peripheral blood mononuclear cell": TermInfo("blood", "intermediate"),
    # Progenitor / stem
    "hematopoietic stem cell": TermInfo("blood", "progenitor"),
    "progenitor cell": TermInfo("blood", "progenitor"),
    "erythroid progenitor cell": TermInfo("blood", "progenitor"),
    "proerythroblast": TermInfo("blood", "progenitor"),
    "common myeloid progenitor": TermInfo("myeloid", "progenitor"),
    "stem cell": TermInfo("other", "progenitor"),
    # Contentless
    "native cell": TermInfo("other", "root"),
    "cell": TermInfo("other", "root"),
    "animal cell": TermInfo("other", "root"),
    "eukaryotic cell": TermInfo("other", "root"),
    # Outside blood
    "neuron": TermInfo("other", "non_blood"),
    "astrocyte": TermInfo("other", "non_blood"),
    "ciliated cell": TermInfo("other", "non_blood"),
    "ciliated epithelial cell": TermInfo("other", "non_blood"),
    "epithelial cell": TermInfo("other", "non_blood"),
    "endothelial cell": TermInfo("other", "non_blood"),
    "endothelial cell of hepatic sinusoid": TermInfo("other", "non_blood"),
    "fibroblast": TermInfo("other", "non_blood"),
    "malignant cell": TermInfo("other", "non_blood"),
    "neoplastic cell": TermInfo("other", "non_blood"),
}

# IS-A ancestors of each gold label. A prediction naming one of these is the
# right lineage at too coarse a granularity, which the run still scores wrong.
GOLD_ANCESTORS: dict[str, set[str]] = {
    "cd14_monocytes": {
        "monocyte",
        "mononuclear phagocyte",
        "phagocyte",
        "myeloid leukocyte",
        "myeloid cell",
        "leukocyte",
        "hematopoietic cell",
        "blood cell",
        "peripheral blood mononuclear cell",
    },
    "b_cells": {
        "b lymphocyte",
        "lymphocyte",
        "leukocyte",
        "hematopoietic cell",
        "blood cell",
        "peripheral blood mononuclear cell",
    },
    "cd56_nk": {
        "natural killer cell",
        "nk cell",
        "innate lymphoid cell",
        "lymphocyte",
        "leukocyte",
        "hematopoietic cell",
        "blood cell",
        "peripheral blood mononuclear cell",
    },
}

GOLD_LINEAGE: dict[str, str] = {
    "cd14_monocytes": "myeloid",
    "b_cells": "lymphoid",
    "cd56_nk": "lymphoid",
}

BUCKETS = (
    "wrong_sibling",
    "under_specified",
    "wrong_lineage",
    "off_target",
    "vacuous",
    "unclassified",
)


def normalize_answer(text: Any) -> str:
    """Normalize a generated answer to the run's label-matching token form."""
    stripped = _TAG_RE.sub(" ", str(text))
    return " ".join(_TOKEN_RE.findall(stripped.lower()))


def bucket_answer(gold_cell_type: str, answer: str) -> str:
    """Classify why a wrong answer failed, relative to the gold label."""
    if gold_cell_type not in GOLD_ANCESTORS:
        raise KeyError(
            f"No ontology entry for gold label {gold_cell_type!r}. Add it to "
            "GOLD_ANCESTORS and GOLD_LINEAGE before analyzing this cell type."
        )
    if answer in GOLD_ANCESTORS[gold_cell_type]:
        return "under_specified"

    term = CELL_TERMS.get(answer)
    if term is None:
        return "unclassified"
    if term.granularity == "root":
        return "vacuous"
    if term.granularity in {"progenitor", "non_blood"}:
        return "off_target"

    gold_lineage = GOLD_LINEAGE[gold_cell_type]
    if term.lineage == gold_lineage:
        return "wrong_sibling"
    if term.lineage in {"myeloid", "lymphoid"}:
        return "wrong_lineage"
    # Blood-generic intermediates that are not ancestors of this gold label.
    return "under_specified" if term.granularity == "intermediate" else "off_target"


# ---------------------------------------------------------------------------
# Prediction loading
# ---------------------------------------------------------------------------

@dataclass
class CellRecord:
    split: str
    index: int
    cell_name: str
    cell_type: str
    correct: bool
    answer: str
    bucket: str | None


@dataclass
class SplitGroups:
    split: str
    n_predictions: int
    failure_indices: list[int] = field(default_factory=list)
    correct_indices: list[int] = field(default_factory=list)


def project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_predictions(predictions_dir: Path, split: str) -> list[dict[str, Any]]:
    path = predictions_dir / split / "predictions.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Predictions file does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_no} is not valid JSON") from error
    for position, row in enumerate(rows):
        if int(row.get("index", position)) != position:
            raise ValueError(
                f"{path} is not in index order at position {position}: "
                f"found index {row.get('index')!r}. The SAE activations are "
                "matched positionally, so a reordered file cannot be used."
            )
    return rows


def collect_groups(
    predictions_dir: Path,
    splits: Iterable[str],
    cell_type: str,
    failure_buckets: set[str],
) -> tuple[list[SplitGroups], list[CellRecord]]:
    """Split the target cell type into failure-bucket and correct groups."""
    groups: list[SplitGroups] = []
    records: list[CellRecord] = []

    for split in splits:
        rows = load_predictions(predictions_dir, split)
        split_groups = SplitGroups(split=split, n_predictions=len(rows))
        for position, row in enumerate(rows):
            if row.get("cell_type") != cell_type:
                continue
            answer = normalize_answer(row.get("generated_text", ""))
            correct = bool(row.get("correct"))
            bucket = None if correct else bucket_answer(cell_type, answer)
            records.append(
                CellRecord(
                    split=split,
                    index=position,
                    cell_name=str(row.get("cell_name", "")),
                    cell_type=cell_type,
                    correct=correct,
                    answer=answer,
                    bucket=bucket,
                )
            )
            if correct:
                split_groups.correct_indices.append(position)
            elif bucket in failure_buckets:
                split_groups.failure_indices.append(position)
        groups.append(split_groups)

    return groups, records


# ---------------------------------------------------------------------------
# Activations and SAE encoding
# ---------------------------------------------------------------------------

def load_split_activations(
    split_dir: Path,
) -> tuple[torch.Tensor, list[str], np.ndarray]:
    """Load one split's activations, labels, and per-row cell-type labels."""
    labels_path = split_dir / "cell_types.json"
    if not labels_path.is_file():
        raise FileNotFoundError(f"Missing cell type labels: {labels_path}")
    with labels_path.open("r", encoding="utf-8") as f:
        labels = json.load(f)
    if not isinstance(labels, list) or any(not isinstance(x, str) for x in labels):
        raise ValueError(f"{labels_path} must contain a list of label strings.")

    shard_paths = sorted(split_dir.glob("*.safetensors"))
    if not shard_paths:
        raise FileNotFoundError(f"No safetensors shards in {split_dir}")

    activation_chunks: list[torch.Tensor] = []
    id_chunks: list[torch.Tensor] = []
    for shard_path in shard_paths:
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            keys = set(f.keys())
            required = {"activations", "cell_type_ids"}
            if not required.issubset(keys):
                raise KeyError(
                    f"{shard_path} is missing tensor(s): {sorted(required - keys)}"
                )
            activations = f.get_tensor("activations")
            cell_type_ids = f.get_tensor("cell_type_ids").long()
        if activations.ndim != 2:
            raise ValueError(
                f"Expected 2D activations in {shard_path}, got {tuple(activations.shape)}"
            )
        if cell_type_ids.ndim != 1 or len(cell_type_ids) != len(activations):
            raise ValueError(
                f"Unaligned cell_type_ids in {shard_path}: "
                f"{tuple(cell_type_ids.shape)} for {len(activations)} rows"
            )
        activation_chunks.append(activations)
        id_chunks.append(cell_type_ids)

    activations = torch.cat(activation_chunks, dim=0)
    ids = torch.cat(id_chunks, dim=0).numpy()
    if ids.size and (ids.min() < 0 or ids.max() >= len(labels)):
        raise ValueError(f"Out-of-range cell_type_ids in {split_dir}")
    row_labels = np.asarray(labels, dtype=object)[ids]
    return activations, labels, row_labels


def verify_alignment(
    split: str,
    row_labels: np.ndarray,
    predictions: list[dict[str, Any]],
) -> None:
    """Fail loudly if activations and predictions are not row-aligned."""
    if len(row_labels) != len(predictions):
        raise ValueError(
            f"{split}: {len(row_labels)} activation rows but "
            f"{len(predictions)} predictions. The two were produced from "
            "different cell sets and cannot be matched by position."
        )
    mismatches = [
        position
        for position, row in enumerate(predictions)
        if row_labels[position] != row.get("cell_type")
    ]
    if mismatches:
        head = mismatches[:5]
        raise ValueError(
            f"{split}: cell type disagrees between activations and predictions "
            f"at {len(mismatches)} row(s), first at {head}. Positional matching "
            "is invalid for this pair of directories."
        )


@torch.no_grad()
def encode_rows(
    sae: torch.nn.Module,
    activations: torch.Tensor,
    row_indices: np.ndarray,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Encode selected activation rows into SAE feature space."""
    if len(row_indices) == 0:
        return np.zeros((0, int(sae.d_hidden)), dtype=np.float32)
    selected = activations[torch.from_numpy(np.asarray(row_indices, dtype=np.int64))]
    out = np.empty((len(selected), int(sae.d_hidden)), dtype=np.float32)
    for start in range(0, len(selected), batch_size):
        batch = selected[start : start + batch_size].to(device=device, dtype=torch.float32)
        out[start : start + len(batch)] = sae.encode(batch).float().cpu().numpy()
    return out


# ---------------------------------------------------------------------------
# Feature statistics
# ---------------------------------------------------------------------------

def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Return BH-adjusted q-values for a vector of p-values."""
    p_values = np.asarray(p_values, dtype=np.float64)
    n = p_values.size
    if n == 0:
        return p_values
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def rank_auroc(fail: np.ndarray, correct: np.ndarray) -> np.ndarray:
    """Per-feature AUROC for separating failures from correct calls.

    0.5 is no separation; >0.5 means the feature runs higher in the failures.
    Computed from midrank sums so ties (a sparse SAE produces many exact
    zeros) are handled correctly.
    """
    n_fail, n_correct = len(fail), len(correct)
    if n_fail == 0 or n_correct == 0:
        return np.full(fail.shape[1] if fail.ndim == 2 else 0, np.nan)
    combined = np.vstack([fail, correct])
    ranks = np.apply_along_axis(stats.rankdata, 0, combined)
    rank_sum_fail = ranks[:n_fail].sum(axis=0)
    u = rank_sum_fail - n_fail * (n_fail + 1) / 2.0
    return u / (n_fail * n_correct)


@dataclass
class FeatureComparison:
    feature_ids: np.ndarray
    mean_fail: np.ndarray
    mean_correct: np.ndarray
    rate_fail: np.ndarray
    rate_correct: np.ndarray
    mean_difference: np.ndarray
    rate_difference: np.ndarray
    cohens_d: np.ndarray
    auroc: np.ndarray
    t_statistic: np.ndarray
    p_value: np.ndarray
    q_value: np.ndarray


def compare_features(
    fail: np.ndarray,
    correct: np.ndarray,
    min_active_cells: int,
    activation_threshold: float,
) -> FeatureComparison:
    """Test every sufficiently active feature for a between-group difference."""
    active_fail = fail > activation_threshold
    active_correct = correct > activation_threshold
    total_active = active_fail.sum(axis=0) + active_correct.sum(axis=0)
    keep = np.flatnonzero(total_active >= min_active_cells)
    if keep.size == 0:
        raise ValueError(
            "No feature is active in at least "
            f"{min_active_cells} of the selected cells; nothing to compare."
        )

    fail_kept = fail[:, keep]
    correct_kept = correct[:, keep]

    mean_fail = fail_kept.mean(axis=0)
    mean_correct = correct_kept.mean(axis=0)
    rate_fail = active_fail[:, keep].mean(axis=0)
    rate_correct = active_correct[:, keep].mean(axis=0)

    var_fail = fail_kept.var(axis=0, ddof=1) if len(fail_kept) > 1 else np.zeros_like(mean_fail)
    var_correct = (
        correct_kept.var(axis=0, ddof=1) if len(correct_kept) > 1 else np.zeros_like(mean_correct)
    )
    n_fail, n_correct = len(fail_kept), len(correct_kept)
    pooled = np.sqrt(
        ((n_fail - 1) * var_fail + (n_correct - 1) * var_correct)
        / max(n_fail + n_correct - 2, 1)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        cohens_d = np.where(pooled > 0, (mean_fail - mean_correct) / pooled, 0.0)

    t_stat, p_value = stats.ttest_ind(
        fail_kept, correct_kept, axis=0, equal_var=False, nan_policy="omit"
    )
    t_stat = np.nan_to_num(np.asarray(t_stat, dtype=np.float64), nan=0.0)
    p_value = np.nan_to_num(np.asarray(p_value, dtype=np.float64), nan=1.0)

    return FeatureComparison(
        feature_ids=keep.astype(np.int64),
        mean_fail=mean_fail,
        mean_correct=mean_correct,
        rate_fail=rate_fail,
        rate_correct=rate_correct,
        mean_difference=mean_fail - mean_correct,
        rate_difference=rate_fail - rate_correct,
        cohens_d=cohens_d,
        auroc=rank_auroc(fail_kept, correct_kept),
        t_statistic=t_stat,
        p_value=p_value,
        q_value=benjamini_hochberg(p_value),
    )


def centroid_permutation_test(
    fail: np.ndarray,
    correct: np.ndarray,
    n_permutations: int,
    seed: int,
) -> dict[str, float]:
    """Permutation test on the L2 distance between the two group centroids.

    The observed statistic is compared against the null built by reshuffling
    the failure/correct labels, so a group that is no more separated than a
    random split of the same cells reports a large p-value.
    """
    n_fail, n_correct = len(fail), len(correct)
    combined = np.vstack([fail, correct])
    total = combined.sum(axis=0)

    observed = float(np.linalg.norm(fail.mean(axis=0) - correct.mean(axis=0)))
    rng = np.random.default_rng(seed)
    n_total = n_fail + n_correct
    null = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        idx = rng.choice(n_total, size=n_fail, replace=False)
        sum_fail = combined[idx].sum(axis=0)
        centroid_fail = sum_fail / n_fail
        centroid_correct = (total - sum_fail) / n_correct
        null[i] = np.linalg.norm(centroid_fail - centroid_correct)

    # Add-one estimator: never reports p = 0 from a finite null.
    p_value = float((np.sum(null >= observed) + 1) / (n_permutations + 1))
    return {
        "observed_centroid_distance": observed,
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=1)) if n_permutations > 1 else 0.0,
        "p_value": p_value,
        "n_permutations": int(n_permutations),
    }


# ---------------------------------------------------------------------------
# Marker feature scores
# ---------------------------------------------------------------------------

def load_marker_features(
    path: Path,
    top_n: int,
) -> dict[str, list[int]]:
    """Read per-cell-type marker features from a cell_feature_analysis run."""
    markers: dict[str, list[tuple[int, float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            margin = float(row["specificity_margin"])
            if margin <= 0:
                continue
            markers.setdefault(row["cell_type"], []).append(
                (int(row["feature_id"]), margin)
            )
    return {
        cell_type: [
            feature_id
            for feature_id, _ in sorted(entries, key=lambda item: -item[1])[:top_n]
        ]
        for cell_type, entries in markers.items()
    }


def marker_scores(
    features: np.ndarray,
    marker_ids: list[int],
) -> np.ndarray:
    """Mean activation of one cell type's marker features, per cell."""
    if not marker_ids:
        return np.zeros(len(features), dtype=np.float32)
    return features[:, np.asarray(marker_ids, dtype=np.int64)].mean(axis=1)


# ---------------------------------------------------------------------------
# Gene annotation
# ---------------------------------------------------------------------------

def load_feature_genes(path: Path, top_genes: int) -> dict[int, list[str]]:
    lookup: dict[int, list[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            genes = record.get("genes") or []
            lookup[int(record["feature"])] = [str(g) for g in genes[:top_genes]]
    return lookup


def load_feature_specificity(path: Path) -> dict[int, tuple[str, float]]:
    """Map each ranked feature to the cell type it is most specific for."""
    best: dict[int, tuple[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            feature_id = int(row["feature_id"])
            margin = float(row["specificity_margin"])
            current = best.get(feature_id)
            if current is None or margin > current[1]:
                best[feature_id] = (row["cell_type"], margin)
    return best


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_volcano(
    comparison: FeatureComparison,
    fdr: float,
    out_path: Path,
    dpi: int,
) -> None:
    # Very large t-statistics underflow the q-value to exactly zero, which would
    # otherwise stretch the axis to -log10(tiny) and flatten everything else.
    # Those points are floored one decade below the smallest resolved q-value
    # and drawn as triangles so the cap is visible rather than silent.
    q = comparison.q_value
    resolved = q[q > 0]
    floor = resolved.min() / 10.0 if resolved.size else 1e-300
    capped = q <= 0
    y = -np.log10(np.where(capped, floor, q))
    significant = q < fdr

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(
        comparison.cohens_d[~significant],
        y[~significant],
        s=8,
        color="#c8c8c8",
        alpha=0.6,
        label=f"q >= {fdr:g}",
        linewidths=0,
    )
    shown = significant & ~capped
    ax.scatter(
        comparison.cohens_d[shown],
        y[shown],
        s=14,
        color="#c2452d",
        alpha=0.85,
        label=f"q < {fdr:g}",
        linewidths=0,
    )
    if capped.any():
        ax.scatter(
            comparison.cohens_d[capped],
            y[capped],
            s=26,
            marker="^",
            color="#c2452d",
            label=f"q below float resolution (capped, n={int(capped.sum())})",
            linewidths=0,
        )
    ax.axhline(-np.log10(fdr), color="#666666", linestyle="--", linewidth=0.8)
    ax.axvline(0.0, color="#666666", linewidth=0.8)
    ax.set_xlabel("Cohen's d  (positive = higher in sibling failures)")
    ax.set_ylabel("-log10 BH q-value")
    ax.set_title("SAE feature differences: sibling failures vs correct calls")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_top_features(
    comparison: FeatureComparison,
    selected: np.ndarray,
    out_path: Path,
    dpi: int,
) -> None:
    if selected.size == 0:
        return
    labels = [f"{comparison.feature_ids[i]}" for i in selected]
    y = np.arange(len(selected))
    height = 0.38

    fig, ax = plt.subplots(figsize=(8.0, max(3.0, 0.42 * len(selected) + 1.4)))
    ax.barh(
        y + height / 2,
        comparison.mean_fail[selected],
        height=height,
        color="#c2452d",
        label="sibling failures",
    )
    ax.barh(
        y - height / 2,
        comparison.mean_correct[selected],
        height=height,
        color="#3f6f9c",
        label="correct calls",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean SAE activation")
    ax.set_ylabel("Feature ID")
    ax.set_title("Largest group differences by |Cohen's d|")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_marker_scores(
    scores_fail: dict[str, np.ndarray],
    scores_correct: dict[str, np.ndarray],
    out_path: Path,
    dpi: int,
) -> None:
    cell_types = sorted(scores_fail)
    if not cell_types:
        return
    fig, ax = plt.subplots(figsize=(1.9 * len(cell_types) + 3.0, 5.0))
    positions = np.arange(len(cell_types))
    width = 0.34

    for offset, scores, color, label in (
        (-width / 2, scores_correct, "#3f6f9c", "correct calls"),
        (width / 2, scores_fail, "#c2452d", "sibling failures"),
    ):
        data = [scores[cell_type] for cell_type in cell_types]
        parts = ax.boxplot(
            data,
            positions=positions + offset,
            widths=width * 0.85,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.0},
        )
        for patch in parts["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
            patch.set_linewidth(0.6)
        parts["boxes"][0].set_label(label)

    ax.set_xticks(positions)
    ax.set_xticklabels(cell_types)
    ax.set_ylabel("Mean activation over cell-type marker features")
    ax.set_title("Marker-feature score by group")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(
    path: Path,
    summary: dict[str, Any],
    comparison: FeatureComparison,
    top_rows: list[dict[str, Any]],
) -> None:
    lines: list[str] = []
    lines.append(f"# Annotation failure analysis: {summary['cell_type']}\n")
    lines.append(
        f"Predictions: `{summary['predictions_dir']}`  \n"
        f"SAE: `{summary['checkpoint']}` (layer {summary['layer']}, "
        f"{summary['sae']['d_hidden']:,} features)  \n"
        f"Cells: `{summary['cells_dir']}`  \n"
        f"Splits: {', '.join(summary['splits'])}\n"
    )

    lines.append("## 1. Failure buckets\n")
    lines.append("| Bucket | Cells | Share of failures |")
    lines.append("|---|---:|---:|")
    total_failures = summary["n_failures_total"]
    for bucket, count in summary["failure_buckets"].items():
        share = count / total_failures if total_failures else 0.0
        lines.append(f"| `{bucket}` | {count} | {share:.1%} |")
    lines.append(
        f"\n{total_failures} incorrect predictions for `{summary['cell_type']}` "
        f"out of {summary['n_target_cells']} cells "
        f"({summary['n_correct']} correct, "
        f"{summary['n_correct'] / summary['n_target_cells']:.1%}).\n"
    )
    if summary["unclassified_answers"]:
        lines.append(
            "Answers with no ontology entry (counted as `unclassified`, never "
            "silently dropped): "
            + ", ".join(f"`{a}` ({n})" for a, n in summary["unclassified_answers"])
            + "\n"
        )

    lines.append("### Sibling answers in the analyzed group\n")
    lines.append("| Answer | Cells |")
    lines.append("|---|---:|")
    for answer, count in summary["failure_group_answers"]:
        lines.append(f"| `{answer}` | {count} |")
    lines.append("")

    lines.append("## 2. Groups compared\n")
    lines.append("| Split | Target cells | Group A (failures) | Group B (correct) |")
    lines.append("|---|---:|---:|---:|")
    for split in summary["per_split"]:
        lines.append(
            f"| `{split['split']}` | {split['n_target']} | "
            f"{split['n_failures']} | {split['n_correct']} |"
        )
    lines.append(
        f"| **pooled** | {summary['n_target_cells']} | "
        f"{summary['n_failure_group']} | {summary['n_correct_group']} |\n"
    )

    lines.append("## 3. SAE feature comparison\n")
    perm = summary["centroid_permutation_test"]
    lines.append(
        f"Features tested (active in >= {summary['min_active_cells']} of the pooled "
        f"cells): {len(comparison.feature_ids):,} of {summary['sae']['d_hidden']:,}.  \n"
        f"Features passing BH FDR < {summary['fdr']:g}: "
        f"{summary['n_significant']:,} "
        f"({summary['n_significant_higher_in_failures']:,} higher in failures, "
        f"{summary['n_significant_lower_in_failures']:,} lower).\n"
    )
    lines.append(
        f"Group centroid separation: {perm['observed_centroid_distance']:.3f} "
        f"vs a permutation null of {perm['null_mean']:.3f} +/- {perm['null_std']:.3f} "
        f"over {perm['n_permutations']:,} shuffles (p = {perm['p_value']:.4f}).\n"
    )

    if top_rows:
        lines.append(f"### Top {len(top_rows)} features by |Cohen's d|\n")
        lines.append(
            "| Feature | d | AUROC | Mean (fail) | Mean (correct) | "
            "Rate (fail) | Rate (correct) | q | Specific to | Top genes |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
        for row in top_rows:
            genes = ", ".join(str(row["top_genes"]).split(";")[:6])
            lines.append(
                f"| {row['feature_id']} | {row['cohens_d']:.2f} | "
                f"{row['auroc']:.3f} | {row['mean_fail']:.3f} | "
                f"{row['mean_correct']:.3f} | {row['rate_fail']:.2f} | "
                f"{row['rate_correct']:.2f} | {row['q_value']:.2e} | "
                f"{row['specific_to_cell_type'] or '-'} | {genes or '-'} |"
            )
        lines.append("")

    if summary["marker_scores"]:
        lines.append("## 4. Marker-feature scores\n")
        lines.append(
            "Mean activation over the top "
            f"{summary['marker_top_n']} specificity-ranked features of each cell "
            "type, per cell.\n"
        )
        lines.append(
            "| Marker set | Failures (mean) | Correct (mean) | Difference | "
            "AUROC | p (Welch) |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|")
        for entry in summary["marker_scores"]:
            lines.append(
                f"| `{entry['cell_type']}` | {entry['mean_fail']:.3f} | "
                f"{entry['mean_correct']:.3f} | {entry['difference']:+.3f} | "
                f"{entry['auroc']:.3f} | {entry['p_value']:.3e} |"
            )
        lines.append("")

    lines.append("## Caveats\n")
    lines.append(
        "- The ontology buckets are hand-encoded from the standard CL blood "
        "hierarchy and cover the terms these runs emit. They explain failures; "
        "the run's own label matcher scores them.\n"
        "- Failure cells are a self-selected group, so every contrast here is "
        "observational. A feature that separates the groups is a candidate for "
        "a steering or ablation test, not evidence that it caused the answer.\n"
        "- Group sizes are unbalanced and the failure group is small on single "
        "splits; the permutation test and FDR control are reported so a null "
        "result reads as a null result.\n"
        "- Activations are matched to predictions by row position, verified "
        "against the shard cell-type labels before any statistic is computed.\n"
    )

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare SAE features between sibling-confusion failures and "
            "correct predictions for one cell type."
        )
    )
    parser.add_argument(
        "--predictions-dir",
        default=DEFAULT_PREDICTIONS_DIR,
        help="Annotation run directory containing <split>/predictions.jsonl.",
    )
    parser.add_argument(
        "--cell-type",
        default="cd14_monocytes",
        help="Gold cell type to analyze.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Splits to pool. Fewer splits means a smaller failure group.",
    )
    parser.add_argument(
        "--failure-buckets",
        nargs="+",
        default=["wrong_sibling"],
        choices=list(BUCKETS),
        help="Failure buckets forming the comparison group.",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="SAE checkpoint whose features are compared.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=DEFAULT_LAYER,
        help="Model layer; must match the checkpoint and the cell activations.",
    )
    parser.add_argument(
        "--cells-dir",
        default=None,
        help=(
            "Cell activation directory holding <split>/ shards. Defaults to "
            "datasets/cells/<base model>/layer<N>/<dataset>/"
            f"{DEFAULT_POOLING_DIRNAME}."
        ),
    )
    parser.add_argument(
        "--cell-type-features",
        default=None,
        help=(
            "all_cell_type_features.csv for marker scoring and the specificity "
            "join. Defaults to the matching results/cell_type_features run."
        ),
    )
    parser.add_argument(
        "--feature-to-genes",
        default=None,
        help=(
            "feature_to_genes.jsonl for annotating top features. Defaults to "
            "the matching results/gene_features run."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Output directory. Defaults under {DEFAULT_OUTPUT_ROOT}/.",
    )
    parser.add_argument("--top-n", type=int, default=40, help="Features reported in detail.")
    parser.add_argument(
        "--marker-top-n",
        type=int,
        default=20,
        help="Marker features per cell type used for the marker score.",
    )
    parser.add_argument(
        "--top-genes",
        type=int,
        default=10,
        help="Genes retained per reported feature.",
    )
    parser.add_argument(
        "--fdr",
        type=float,
        default=0.05,
        help="Benjamini-Hochberg false discovery rate.",
    )
    parser.add_argument(
        "--min-active-cells",
        type=int,
        default=5,
        help="Skip features active in fewer than this many pooled cells.",
    )
    parser.add_argument(
        "--activation-threshold",
        type=float,
        default=0.0,
        help="A feature counts as active above this value.",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=1000,
        help="Shuffles for the centroid separation test. 0 disables it.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Permutation seed.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--no-figures", action="store_true", help="Skip figure generation."
    )
    return parser.parse_args(argv)


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but CUDA is not available.")
    return requested


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    predictions_dir = project_path(args.predictions_dir)
    if not predictions_dir.is_dir():
        raise FileNotFoundError(f"Predictions directory does not exist: {predictions_dir}")

    # results/cell_type_annotation/<run>/<dataset>/<base model>
    base_model_name = predictions_dir.name
    dataset_name = predictions_dir.parent.name
    run_name = predictions_dir.parent.parent.name

    checkpoint_path = project_path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SAE checkpoint does not exist: {checkpoint_path}")
    sae_name = checkpoint_path.parent.name

    checkpoint_metadata = load_checkpoint_metadata(checkpoint_path)
    metadata_layer = (checkpoint_metadata.get("model") or {}).get("layer")
    if metadata_layer is not None and int(metadata_layer) != args.layer:
        raise ValueError(
            f"Checkpoint {checkpoint_path} was trained on layer {metadata_layer}, "
            f"but --layer is {args.layer}. Feature indices from one layer do not "
            "transfer to another."
        )

    if args.cells_dir:
        cells_dir = project_path(args.cells_dir)
    else:
        cells_dir = (
            PROJECT_ROOT
            / "datasets"
            / "cells"
            / base_model_name
            / f"layer{args.layer}"
            / dataset_name
            / DEFAULT_POOLING_DIRNAME
        )
    if not cells_dir.is_dir():
        raise FileNotFoundError(
            f"Cell activation directory does not exist: {cells_dir}. Pass "
            "--cells-dir if the extraction lives elsewhere."
        )

    output_dir = (
        project_path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT
        / DEFAULT_OUTPUT_ROOT
        / run_name
        / dataset_name
        / base_model_name
        / sae_name
        / args.cell_type
    )
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_figures:
        figures_dir.mkdir(parents=True, exist_ok=True)

    failure_buckets = set(args.failure_buckets)

    # --- Steps 1 and 2: buckets, failure indices, correct indices ------------
    groups, records = collect_groups(
        predictions_dir, args.splits, args.cell_type, failure_buckets
    )
    if not records:
        raise ValueError(
            f"No cells with gold label {args.cell_type!r} in "
            f"{predictions_dir} across splits {args.splits}."
        )

    bucket_counts = Counter(r.bucket for r in records if not r.correct)
    unclassified = Counter(
        r.answer for r in records if r.bucket == "unclassified"
    ).most_common()
    failure_group_answers = Counter(
        r.answer for r in records if r.bucket in failure_buckets
    ).most_common()

    n_failure_group = sum(len(g.failure_indices) for g in groups)
    n_correct_group = sum(len(g.correct_indices) for g in groups)
    print(
        f"{args.cell_type}: {len(records)} cells, "
        f"{sum(1 for r in records if r.correct)} correct, "
        f"{sum(1 for r in records if not r.correct)} incorrect."
    )
    print(
        "Failure buckets: "
        + ", ".join(f"{bucket}={bucket_counts.get(bucket, 0)}" for bucket in BUCKETS)
    )
    print(
        f"Comparison groups: {n_failure_group} "
        f"({'/'.join(sorted(failure_buckets))}) vs {n_correct_group} correct."
    )

    if n_failure_group < 3 or n_correct_group < 3:
        raise ValueError(
            f"Need at least 3 cells per group; got {n_failure_group} failures "
            f"and {n_correct_group} correct. Pool more splits or widen "
            "--failure-buckets."
        )

    # --- Step 3: SAE features ------------------------------------------------
    device = resolve_device(args.device)
    sae, sae_metadata = load_sae(checkpoint_path, device=device)
    print(
        f"SAE: {sae_metadata['sae_type']} with "
        f"{sae_metadata['d_hidden']:,} features on {device}"
    )

    fail_blocks: list[np.ndarray] = []
    correct_blocks: list[np.ndarray] = []
    per_split: list[dict[str, Any]] = []

    for split_group in groups:
        split_dir = cells_dir / split_group.split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing cell activation split: {split_dir}")
        activations, _, row_labels = load_split_activations(split_dir)
        verify_alignment(
            split_group.split,
            row_labels,
            load_predictions(predictions_dir, split_group.split),
        )
        if activations.shape[1] != int(sae_metadata["d_model"]):
            raise ValueError(
                f"{split_dir} activations are {activations.shape[1]}-dimensional "
                f"but the SAE expects {sae_metadata['d_model']}."
            )

        fail_blocks.append(
            encode_rows(
                sae,
                activations,
                np.asarray(split_group.failure_indices, dtype=np.int64),
                device,
                args.batch_size,
            )
        )
        correct_blocks.append(
            encode_rows(
                sae,
                activations,
                np.asarray(split_group.correct_indices, dtype=np.int64),
                device,
                args.batch_size,
            )
        )
        per_split.append(
            {
                "split": split_group.split,
                "n_predictions": split_group.n_predictions,
                "n_target": sum(1 for r in records if r.split == split_group.split),
                "n_failures": len(split_group.failure_indices),
                "n_correct": len(split_group.correct_indices),
                "failure_indices": split_group.failure_indices,
                "correct_indices": split_group.correct_indices,
            }
        )

    fail_features = np.vstack(fail_blocks)
    correct_features = np.vstack(correct_blocks)

    comparison = compare_features(
        fail_features,
        correct_features,
        min_active_cells=args.min_active_cells,
        activation_threshold=args.activation_threshold,
    )
    significant = comparison.q_value < args.fdr
    n_significant = int(significant.sum())
    n_higher = int((significant & (comparison.mean_difference > 0)).sum())

    permutation = (
        centroid_permutation_test(
            fail_features, correct_features, args.permutations, args.seed
        )
        if args.permutations > 0
        else {
            "observed_centroid_distance": float(
                np.linalg.norm(
                    fail_features.mean(axis=0) - correct_features.mean(axis=0)
                )
            ),
            "null_mean": float("nan"),
            "null_std": float("nan"),
            "p_value": float("nan"),
            "n_permutations": 0,
        }
    )

    # --- Annotation joins ----------------------------------------------------
    cell_type_features_path = (
        project_path(args.cell_type_features)
        if args.cell_type_features
        else PROJECT_ROOT
        / CELL_TYPE_FEATURES_ROOT
        / sae_name
        / dataset_name
        / "all_cell_type_features.csv"
    )
    feature_genes_path = (
        project_path(args.feature_to_genes)
        if args.feature_to_genes
        else PROJECT_ROOT
        / GENE_FEATURES_ROOT
        / sae_name
        / dataset_name
        / base_model_name
        / "feature_to_genes.jsonl"
    )

    specificity: dict[int, tuple[str, float]] = {}
    markers: dict[str, list[int]] = {}
    if cell_type_features_path.is_file():
        specificity = load_feature_specificity(cell_type_features_path)
        markers = load_marker_features(cell_type_features_path, args.marker_top_n)
    else:
        print(f"No cell-type feature table at {cell_type_features_path}; skipping join.")

    feature_genes: dict[int, list[str]] = {}
    if feature_genes_path.is_file():
        feature_genes = load_feature_genes(feature_genes_path, args.top_genes)
    else:
        print(f"No feature-to-gene lookup at {feature_genes_path}; skipping join.")

    marker_summary: list[dict[str, Any]] = []
    scores_fail: dict[str, np.ndarray] = {}
    scores_correct: dict[str, np.ndarray] = {}
    for cell_type in sorted(markers):
        ids = markers[cell_type]
        fail_score = marker_scores(fail_features, ids)
        correct_score = marker_scores(correct_features, ids)
        scores_fail[cell_type] = fail_score
        scores_correct[cell_type] = correct_score
        t_stat, p_value = stats.ttest_ind(fail_score, correct_score, equal_var=False)
        auroc = float(
            rank_auroc(fail_score[:, None], correct_score[:, None])[0]
        )
        marker_summary.append(
            {
                "cell_type": cell_type,
                "n_marker_features": len(ids),
                "mean_fail": float(fail_score.mean()),
                "mean_correct": float(correct_score.mean()),
                "difference": float(fail_score.mean() - correct_score.mean()),
                "auroc": auroc,
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
            }
        )

    # --- Write outputs -------------------------------------------------------
    write_csv(
        output_dir / "failure_buckets.csv",
        ["split", "index", "cell_name", "cell_type", "answer", "bucket", "in_group"],
        (
            {
                "split": r.split,
                "index": r.index,
                "cell_name": r.cell_name,
                "cell_type": r.cell_type,
                "answer": r.answer,
                "bucket": r.bucket,
                "in_group": r.bucket in failure_buckets,
            }
            for r in records
            if not r.correct
        ),
    )

    write_csv(
        output_dir / "group_cells.csv",
        ["split", "index", "cell_name", "group", "answer", "bucket"],
        (
            {
                "split": r.split,
                "index": r.index,
                "cell_name": r.cell_name,
                "group": "correct" if r.correct else "failure",
                "answer": r.answer,
                "bucket": r.bucket or "",
            }
            for r in records
            if r.correct or r.bucket in failure_buckets
        ),
    )

    with (output_dir / "group_indices.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "cell_type": args.cell_type,
                "failure_buckets": sorted(failure_buckets),
                "splits": {
                    split["split"]: {
                        "failure_indices": split["failure_indices"],
                        "correct_indices": split["correct_indices"],
                    }
                    for split in per_split
                },
            },
            f,
            indent=2,
        )

    comparison_fields = [
        "feature_id",
        "cohens_d",
        "auroc",
        "mean_fail",
        "mean_correct",
        "mean_difference",
        "rate_fail",
        "rate_correct",
        "rate_difference",
        "t_statistic",
        "p_value",
        "q_value",
        "significant",
        "specific_to_cell_type",
        "specificity_margin",
        "top_genes",
    ]

    def comparison_row(i: int) -> dict[str, Any]:
        feature_id = int(comparison.feature_ids[i])
        spec = specificity.get(feature_id)
        return {
            "feature_id": feature_id,
            "cohens_d": float(comparison.cohens_d[i]),
            "auroc": float(comparison.auroc[i]),
            "mean_fail": float(comparison.mean_fail[i]),
            "mean_correct": float(comparison.mean_correct[i]),
            "mean_difference": float(comparison.mean_difference[i]),
            "rate_fail": float(comparison.rate_fail[i]),
            "rate_correct": float(comparison.rate_correct[i]),
            "rate_difference": float(comparison.rate_difference[i]),
            "t_statistic": float(comparison.t_statistic[i]),
            "p_value": float(comparison.p_value[i]),
            "q_value": float(comparison.q_value[i]),
            "significant": bool(comparison.q_value[i] < args.fdr),
            "specific_to_cell_type": spec[0] if spec else "",
            "specificity_margin": f"{spec[1]:.6f}" if spec else "",
            "top_genes": ";".join(feature_genes.get(feature_id, [])),
        }

    order = np.argsort(-np.abs(comparison.cohens_d))
    write_csv(
        output_dir / "feature_comparison.csv",
        comparison_fields,
        (comparison_row(int(i)) for i in order),
    )

    top_selected = order[: args.top_n]
    top_rows = [comparison_row(int(i)) for i in top_selected]
    write_csv(output_dir / "top_differential_features.csv", comparison_fields, top_rows)

    if marker_summary:
        write_csv(
            output_dir / "marker_scores.csv",
            [
                "cell_type",
                "n_marker_features",
                "mean_fail",
                "mean_correct",
                "difference",
                "auroc",
                "t_statistic",
                "p_value",
            ],
            marker_summary,
        )

    if not args.no_figures:
        plot_volcano(comparison, args.fdr, figures_dir / "feature_volcano.png", args.dpi)
        plot_top_features(
            comparison,
            top_selected[: min(args.top_n, 25)],
            figures_dir / "top_differential_features.png",
            args.dpi,
        )
        plot_marker_scores(
            scores_fail, scores_correct, figures_dir / "marker_scores.png", args.dpi
        )

    summary: dict[str, Any] = {
        "cell_type": args.cell_type,
        "predictions_dir": str(predictions_dir),
        "run_name": run_name,
        "dataset_name": dataset_name,
        "base_model_name": base_model_name,
        "cells_dir": str(cells_dir),
        "checkpoint": str(checkpoint_path),
        "sae_name": sae_name,
        "layer": args.layer,
        "splits": list(args.splits),
        "failure_bucket_selection": sorted(failure_buckets),
        "sae": sae_metadata,
        "n_target_cells": len(records),
        "n_correct": sum(1 for r in records if r.correct),
        "n_failures_total": sum(1 for r in records if not r.correct),
        "failure_buckets": {bucket: int(bucket_counts.get(bucket, 0)) for bucket in BUCKETS},
        "unclassified_answers": unclassified,
        "failure_group_answers": failure_group_answers,
        "n_failure_group": n_failure_group,
        "n_correct_group": n_correct_group,
        "per_split": per_split,
        "fdr": args.fdr,
        "min_active_cells": args.min_active_cells,
        "activation_threshold": args.activation_threshold,
        "n_features_tested": int(len(comparison.feature_ids)),
        "n_significant": n_significant,
        "n_significant_higher_in_failures": n_higher,
        "n_significant_lower_in_failures": n_significant - n_higher,
        "max_abs_cohens_d": float(np.abs(comparison.cohens_d).max()),
        "max_abs_auroc_deviation": float(np.abs(comparison.auroc - 0.5).max()),
        "centroid_permutation_test": permutation,
        "marker_top_n": args.marker_top_n,
        "marker_scores": marker_summary,
        "cell_type_features_path": (
            str(cell_type_features_path) if specificity else None
        ),
        "feature_to_genes_path": str(feature_genes_path) if feature_genes else None,
        "outputs": {
            "failure_buckets": str(output_dir / "failure_buckets.csv"),
            "group_cells": str(output_dir / "group_cells.csv"),
            "group_indices": str(output_dir / "group_indices.json"),
            "feature_comparison": str(output_dir / "feature_comparison.csv"),
            "top_differential_features": str(
                output_dir / "top_differential_features.csv"
            ),
            "report": str(output_dir / "report.md"),
        },
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_report(output_dir / "report.md", summary, comparison, top_rows)

    print(
        f"Tested {len(comparison.feature_ids):,} features; "
        f"{n_significant:,} pass BH FDR < {args.fdr:g} "
        f"({n_higher:,} higher in failures)."
    )
    print(
        "Centroid separation p = "
        f"{permutation['p_value']:.4f} over {permutation['n_permutations']:,} shuffles."
    )
    print(f"Wrote results to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
