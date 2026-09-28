"""Shared helpers for SAE feature gene-set enrichment scripts.

The enrichment backends consume the outputs of ``src/evaluate/gene_features.py``:

    feature_to_genes.jsonl
        ``{"feature": int, "genes": [gene_symbol, ...]}`` per activated feature.
    feature_activation_summary.csv
        Per-feature prevalence and activation statistics used to rank features
        when only the top-k should be enriched.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

RANK_COLUMNS = (
    "n_genes",
    "gene_fraction",
    "total_activation",
    "mean_activation_when_active",
    "max_activation",
)

# Column names written by older gene_features.py runs. The strength column was
# renamed to ``mean_activation_when_active`` because it divides by the number of
# above-threshold activations, not by the vocabulary -- the same meaning the name
# already carries in cell_feature_analysis.py, where a plain ``mean_activation``
# instead averages over every cell including the zeros. Existing result
# directories keep working without being regenerated.
_LEGACY_COLUMNS = {"mean_activation": "mean_activation_when_active"}


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not isinstance(rec, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            yield rec


def write_jsonl_record(handle, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    handle.flush()


def list_field(record: dict[str, Any], key: str) -> list[Any]:
    value = record.get(key)
    return value if isinstance(value, list) else []


def unique_preserving_order(values: Iterable[Any]) -> list[Any]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def chunks(values: list[Any], size: int) -> Iterator[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def maybe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _gene_name(value: Any) -> str | None:
    if isinstance(value, dict):
        value = value.get("gene", value.get("symbol", value.get("name")))
    if value is None:
        return None
    gene = str(value).strip()
    return gene or None


def normalize_feature_gene_record(
    record: dict[str, Any],
    *,
    path: Path | None = None,
    line_no: int | None = None,
) -> dict[str, Any]:
    """
    Normalize current and slightly richer feature-to-genes records.

    The retrieval pipeline writes ``{"feature": int, "genes": [str, ...]}``.
    Some downstream experiments also carry gene metadata records; those are
    accepted when each item has a ``gene``, ``symbol``, or ``name`` field.
    """
    feature = maybe_int(record.get("feature"))
    if feature is None:
        loc = f" at {path}:{line_no}" if path is not None and line_no is not None else ""
        raise ValueError(f"Malformed feature-to-genes record{loc}: missing feature")

    genes_raw = record.get("genes")
    if not isinstance(genes_raw, list):
        loc = f" at {path}:{line_no}" if path is not None and line_no is not None else ""
        raise ValueError(f"Malformed feature-to-genes record{loc}: missing genes list")

    genes = unique_preserving_order(
        gene for value in genes_raw if (gene := _gene_name(value)) is not None
    )
    return {"feature": feature, "genes": genes}


def iter_feature_to_genes(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not isinstance(rec, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            yield normalize_feature_gene_record(rec, path=path, line_no=line_no)


def load_feature_to_genes(path: Path) -> list[dict[str, Any]]:
    return list(iter_feature_to_genes(path))


def meets_min_genes(n_genes: int, min_genes: int) -> bool:
    return n_genes >= min_genes


def truncate_genes(genes: list[str], max_genes: int | None) -> list[str]:
    """
    Cap a feature gene set at ``max_genes`` entries, keeping the first ones.

    src/evaluate/gene_features.py writes every gene list sorted by descending
    activation magnitude, so the retained head is the slice of genes the feature
    responds to most strongly. Truncating rather than discarding keeps large
    gene sets in the enrichment instead of dropping exactly the broad,
    high-activation features that a hard cap would silently remove.
    """
    if max_genes is None or len(genes) <= max_genes:
        return genes
    return genes[:max_genes]


def gene_bounds_label(min_genes: int, max_genes: int | None) -> str:
    if max_genes is None:
        return f">= {min_genes} genes"
    return f">= {min_genes} genes, truncated to {max_genes}"


def count_qualifying(
    records: Iterable[dict[str, Any]],
    min_genes: int,
) -> int:
    """Count records with enough genes to enrich. Truncation never drops a
    record below ``min_genes``, so only the lower bound can disqualify one."""
    return sum(1 for rec in records if meets_min_genes(len(rec["genes"]), min_genes))


def load_completed_features(path: Path) -> set[int]:
    completed: set[int] = set()
    if not path.exists():
        return completed

    for rec in iter_jsonl(path):
        feature = maybe_int(rec.get("feature"))
        if feature is not None:
            completed.add(feature)
    return completed


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------


def parse_feature_selector(values: Iterable[Any] | None) -> list[int]:
    """
    Parse an explicit feature selection into ordered, de-duplicated IDs.

    Accepts plain integers and strings holding comma-separated IDs or inclusive
    ranges, so ``[0, "5,9", "10-12"]`` yields ``[0, 5, 9, 10, 11, 12]``.
    """
    if not values:
        return []

    selected: list[int] = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError(f"Invalid feature selector: {value!r}")
        if isinstance(value, int):
            selected.append(value)
            continue

        for part in str(value).split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part.lstrip("-"):
                lo_s, hi_s = part.split("-", 1)
                lo, hi = int(lo_s), int(hi_s)
                if hi < lo:
                    raise ValueError(f"Invalid feature range: {part}")
                selected.extend(range(lo, hi + 1))
            else:
                selected.append(int(part))

    if any(feature < 0 for feature in selected):
        raise ValueError("Feature IDs must be non-negative.")
    return list(dict.fromkeys(selected))


def load_feature_summary(path: Path) -> list[dict[str, Any]]:
    """Load ``feature_activation_summary.csv`` written by gene_features.py.

    Legacy headers are mapped onto their current names, so a directory produced
    before the rename still ranks.
    """

    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "feature" not in reader.fieldnames:
            raise ValueError(f"Missing 'feature' column in {path}")

        rows: list[dict[str, Any]] = []
        for line_no, row in enumerate(reader, start=2):
            feature = maybe_int(row.get("feature"))
            if feature is None:
                raise ValueError(f"Malformed feature row at {path}:{line_no}")
            parsed: dict[str, Any] = {"feature": feature}
            for column in RANK_COLUMNS:
                if column in row:
                    parsed[column] = maybe_float(row[column])
            for legacy, column in _LEGACY_COLUMNS.items():
                if column not in parsed and legacy in row:
                    parsed[column] = maybe_float(row[legacy])
            rows.append(parsed)
    return rows


def rank_features(summary_rows: list[dict[str, Any]], rank_by: str) -> list[int]:
    """Order feature IDs by ``rank_by`` descending, breaking ties on feature ID."""

    if rank_by not in RANK_COLUMNS:
        valid = ", ".join(RANK_COLUMNS)
        raise ValueError(f"Unknown rank_by '{rank_by}'. Valid columns: {valid}.")

    missing = [row["feature"] for row in summary_rows if row.get(rank_by) is None]
    if missing:
        raise ValueError(
            f"Column '{rank_by}' is missing or unparsable for "
            f"{len(missing):,} feature rows."
        )
    return [
        row["feature"]
        for row in sorted(
            summary_rows,
            key=lambda row: (-float(row[rank_by]), row["feature"]),
        )
    ]


def load_selected_feature_genes(
    path: Path,
    features: Iterable[int] | None = None,
) -> dict[int, list[str]]:
    """
    Load feature gene sets from ``feature_to_genes.jsonl``.

    Passing ``features`` keeps only those IDs, so a top-k run never holds the
    full mapping in memory.
    """
    wanted = None if features is None else set(features)
    selected: dict[int, list[str]] = {}
    for rec in iter_feature_to_genes(path):
        feature = rec["feature"]
        if wanted is not None and feature not in wanted:
            continue
        selected[feature] = rec["genes"]
        if wanted is not None and len(selected) == len(wanted):
            break
    return selected


def select_feature_records(
    input_path: Path,
    *,
    mode: str,
    top_k: int | None = None,
    rank_by: str = "n_genes",
    features: Iterable[Any] | None = None,
    summary_path: Path | None = None,
    min_genes: int = 0,
    max_genes: int | None = None,
) -> list[dict[str, Any]]:
    """
    Build the ordered ``{"feature", "genes", "n_genes_total"}`` records to enrich.

    Gene sets longer than ``max_genes`` are truncated to their highest-activating
    ``max_genes`` genes rather than dropped, so a cap narrows what is enriched
    without removing features from the selection. ``n_genes_total`` records the
    size before truncation; backends need it to report what was cut, since the
    truncated list no longer carries that count.

    Modes:
        ``all``     every feature in ``input_path``, in file order.
        ``top_k``   the ``top_k`` highest-ranked features from
                    ``feature_activation_summary.csv``, best first. Only
                    ``min_genes`` filters the candidate pool; ``max_genes`` does
                    not, since an oversized gene set is truncated rather than
                    skipped.
        ``features`` the explicitly configured feature IDs, in the given order
                    and without gene-count filtering.
    """
    if mode == "all":
        return [
            {
                **rec,
                "genes": truncate_genes(rec["genes"], max_genes),
                "n_genes_total": len(rec["genes"]),
            }
            for rec in load_feature_to_genes(input_path)
        ]

    if mode == "top_k":
        if top_k is None or top_k < 1:
            raise ValueError("select.top_k must be at least 1 when mode='top_k'.")
        if summary_path is None or not summary_path.exists():
            raise FileNotFoundError(
                f"Feature summary required for mode='top_k': {summary_path}"
            )
        summary_rows = [
            row
            for row in load_feature_summary(summary_path)
            if row.get("n_genes") is None
            or meets_min_genes(int(row["n_genes"]), min_genes)
        ]
        ordered = rank_features(summary_rows, rank_by)[:top_k]
        if len(ordered) < top_k:
            print(
                f"[warning] only {len(ordered):,} of the requested {top_k:,} features "
                f"have {gene_bounds_label(min_genes, max_genes)}"
            )
    elif mode == "features":
        ordered = parse_feature_selector(features)
        if not ordered:
            raise ValueError("Set select.features to at least one ID when mode='features'.")
    else:
        raise ValueError(f"Unknown select.mode '{mode}'. Valid: all, top_k, features.")

    gene_sets = load_selected_feature_genes(input_path, ordered)
    missing = [feature for feature in ordered if feature not in gene_sets]
    if missing:
        preview = ", ".join(str(feature) for feature in missing[:10])
        suffix = ", ..." if len(missing) > 10 else ""
        print(
            f"[warning] {len(missing):,} selected features are absent from "
            f"{input_path.name} (never activated): {preview}{suffix}"
        )
    return [
        {
            "feature": feature,
            "genes": truncate_genes(gene_sets[feature], max_genes),
            "n_genes_total": len(gene_sets[feature]),
        }
        for feature in ordered
        if feature in gene_sets
    ]
