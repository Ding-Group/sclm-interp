#!/usr/bin/env python3
"""
g:Profiler Gene Ontology enrichment for SAE features.

Reads a feature_to_genes.jsonl file produced by src/evaluate/gene_features.py,
then queries the g:Profiler API (https://biit.cs.ut.ee/gprofiler) for each
feature's gene list to identify over-represented GO terms (Biological Process,
Molecular Function, Cellular Component).

Prefer src/evaluate/enrich/enrich.py for config-driven runs where the provider,
the GO sources, and the feature selection should be set in YAML.  Use this
script directly when debugging g:Profiler-specific parameters.

Input contract:
    feature_to_genes.jsonl records must contain real gene symbols in the
    "genes" list, which holds for gene_features.py runs over a gene vocabulary.

Output (written to --results by default):
    feature_go_terms.jsonl  — one line per selected feature, containing the
                            feature index and its significant GO terms sorted
                            by adjusted p-value ascending.

Usage:
    python src/evaluate/enrich/go_enrich.py \\
        --results results/gene_features/<sae>/<dataset>/<model>
    python src/evaluate/enrich/go_enrich.py --results <dir> --top-features 100 \\
        --fdr 0.05 --min-genes 5 --top-terms 10 --sources GO:BP GO:MF GO:CC
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import requests

try:
    from .enrich_utils import (
        count_qualifying,
        gene_bounds_label,
        meets_min_genes,
        select_feature_records,
        truncate_genes,
    )
except ImportError:  # pragma: no cover - direct script execution
    from enrich_utils import (
        count_qualifying,
        gene_bounds_label,
        meets_min_genes,
        select_feature_records,
        truncate_genes,
    )

_GPROFILER_URL = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"
_DEFAULT_SOURCES = ["GO:BP", "GO:MF", "GO:CC"]
_RETRY_DELAYS = [2, 5, 10]


def save_go_terms(records: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")


def load_background(path: Path) -> list[str]:
    """
    Read a gene vocabulary to use as the enrichment background.

    Accepts the genes.json written by gene_features.py (a list of symbols) or a
    plain text file with one symbol per line. The vocabulary is the only set of
    genes a feature could ever activate on, so scoring against it instead of the
    whole genome stops broadly expressed terms from looking enriched.
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        genes = json.loads(text)
        if isinstance(genes, dict):
            genes = list(genes)
    else:
        genes = [line.strip() for line in text.splitlines()]
    genes = [str(g) for g in genes if str(g).strip()]
    if not genes:
        raise ValueError(f"No genes found in background file: {path}")
    return genes


# ---------------------------------------------------------------------------
# g:Profiler query
# ---------------------------------------------------------------------------

def query_gprofiler(
    genes: list[str],
    organism: str = "hsapiens",
    sources: list[str] | None = None,
    user_threshold: float = 0.05,
    background: list[str] | None = None,
) -> list[dict]:
    """
    Query g:Profiler for GO enrichment of a gene list.

    background restricts the statistical universe to that gene list, e.g. the
    dataset gene vocabulary; the default None scores against every annotated
    gene in the genome. term_size in the returned dicts is then the term size
    within the background, not genome-wide.

    Returns a list of term dicts with keys:
        id, name, source, p_value, term_size, intersection_size
    sorted by p_value ascending.  Returns [] on failure or no results.
    """
    sources = sources or _DEFAULT_SOURCES
    payload = {
        "organism": organism,
        "query": genes,
        "sources": sources,
        "significance_threshold_method": "fdr",
        "user_threshold": user_threshold,
        "all_results": False,
        "no_iea": False,
        "numeric_ns": "ENTREZGENE_ACC",
        # custom_annotated keeps the background to genes that carry an
        # annotation, so unannotated vocabulary entries do not inflate it.
        "domain_scope": "custom_annotated" if background else "annotated",
    }
    if background:
        payload["background"] = background

    for attempt, delay in enumerate([0] + _RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            resp = requests.post(_GPROFILER_URL, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            raw_results = data.get("result", [])
            terms = [
                {
                    "id": r["native"],
                    "name": r["name"],
                    "source": r["source"],
                    "p_value": r["p_value"],
                    "term_size": r["term_size"],
                    "intersection_size": r["intersection_size"],
                }
                for r in raw_results
                if r.get("source") in sources
            ]
            terms.sort(key=lambda t: t["p_value"])
            return terms
        except (requests.RequestException, KeyError, ValueError) as exc:
            if attempt < len(_RETRY_DELAYS):
                print(f"    [retry {attempt + 1}] {exc}")
            else:
                print(f"    [failed] {exc}")
    return []


# ---------------------------------------------------------------------------
# Main enrichment loop
# ---------------------------------------------------------------------------

def enrich_features(
    feature_to_genes: list[dict],
    organism: str = "hsapiens",
    sources: list[str] | None = None,
    user_threshold: float = 0.05,
    min_genes: int = 3,
    max_genes: int | None = None,
    top_terms: int | None = None,
    request_delay: float = 0.25,
    background: list[str] | None = None,
) -> list[dict]:
    """
    Run GO enrichment for every selected feature with at least ``min_genes`` genes.

    Gene sets longer than ``max_genes`` are truncated to their highest-activating
    ``max_genes`` genes and enriched on that head, rather than being skipped.

    Returns a list of records:
        {"feature": int, "n_genes": int, "n_genes_total": int, "truncated": bool,
         "go_terms": [{id, name, source, p_value, ...}]}
    one entry per input record, in the order the records were selected.
    ``n_genes`` is the number of genes actually queried and ``n_genes_total`` the
    size before truncation. Only features below ``min_genes`` go unqueried; they
    carry ``"skipped": "min_genes"`` so an empty ``go_terms`` elsewhere always
    means the query ran and found nothing significant.
    """
    sources = sources or _DEFAULT_SOURCES
    results = []
    n_total = len(feature_to_genes)

    for i, rec in enumerate(feature_to_genes):
        feat_idx = rec["feature"]
        all_genes = rec["genes"]
        # select_feature_records may have truncated already and stamped the
        # pre-truncation size; len(all_genes) would then report the cap itself.
        n_genes_total = int(rec.get("n_genes_total") or len(all_genes))
        genes = truncate_genes(all_genes, max_genes)
        n_genes = len(genes)

        record = {
            "feature": feat_idx,
            "n_genes": n_genes,
            "n_genes_total": n_genes_total,
            "truncated": n_genes < n_genes_total,
            "go_terms": [],
        }

        if not meets_min_genes(n_genes, min_genes):
            record["skipped"] = "min_genes"
            results.append(record)
            continue

        terms = query_gprofiler(genes, organism, sources, user_threshold, background)
        if top_terms is not None:
            terms = terms[:top_terms]
        record["go_terms"] = terms
        results.append(record)

        n_sig = len(terms)
        if (i + 1) % 50 == 0 or (i + 1) == n_total:
            shown = f"{n_genes} genes" + (
                f" (of {n_genes_total})" if record["truncated"] else ""
            )
            print(f"  [{i + 1:,}/{n_total:,}] Example: feature {feat_idx}: {shown} → {n_sig} GO terms")

        if i + 1 < n_total:
            time.sleep(request_delay)

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GO enrichment for SAE concept features via g:Profiler."
    )
    parser.add_argument(
        "--results", required=True,
        help="gene_features.py result directory containing feature_to_genes.jsonl.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Input JSONL path (default: <results>/feature_to_genes.jsonl).",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help=(
            "feature_activation_summary.csv used to rank features "
            "(default: <results>/feature_activation_summary.csv)."
        ),
    )
    parser.add_argument(
        "--top-features", type=int, default=None, dest="top_features",
        help="Enrich only the top N ranked features (default: all features).",
    )
    parser.add_argument(
        "--rank-by", default="n_genes", dest="rank_by",
        help="feature_activation_summary.csv column to rank by (default: n_genes).",
    )
    parser.add_argument(
        "--features", nargs="+", default=None,
        help="Explicit feature IDs to enrich, e.g. 0 5,9 10-20. Overrides --top-features.",
    )
    parser.add_argument(
        "--fdr", type=float, default=0.05, dest="user_threshold",
        help="FDR significance threshold for GO terms (default: 0.05).",
    )
    parser.add_argument(
        "--min-genes", type=int, default=3, dest="min_genes",
        help="Skip features with fewer than this many genes (default: 3).",
    )
    parser.add_argument(
        "--max-genes", type=int, default=None, dest="max_genes",
        help="Skip features with more than this many genes (default: no cap).",
    )
    parser.add_argument(
        "--top-terms", type=int, default=None, dest="top_terms",
        help="Keep only the top N most-significant GO terms per feature (default: all).",
    )
    parser.add_argument(
        "--sources", nargs="+", default=_DEFAULT_SOURCES,
        help=f"GO sources to query (default: {' '.join(_DEFAULT_SOURCES)}).",
    )
    parser.add_argument(
        "--organism", default="hsapiens",
        help="g:Profiler organism identifier (default: hsapiens).",
    )
    parser.add_argument(
        "--background", nargs="?", default=None, const="genes.json",
        help=(
            "Gene vocabulary file to score against instead of the whole genome. "
            "Bare --background uses <results>/genes.json."
        ),
    )
    parser.add_argument(
        "--delay", type=float, default=0.25, dest="request_delay",
        help="Seconds to wait between API requests (default: 0.25).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output file path (default: <results>/feature_go_terms.jsonl).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results)
    input_path = Path(args.input) if args.input else results_dir / "feature_to_genes.jsonl"
    summary_path = (
        Path(args.summary)
        if args.summary
        else results_dir / "feature_activation_summary.csv"
    )
    output_path = Path(args.output) if args.output else results_dir / "feature_go_terms.jsonl"

    if not input_path.exists():
        raise FileNotFoundError(f"Not found: {input_path}")

    background = None
    if args.background:
        background_path = Path(args.background)
        if not background_path.is_absolute():
            background_path = results_dir / background_path
        if not background_path.exists():
            raise FileNotFoundError(f"Not found: {background_path}")
        background = load_background(background_path)

    if args.features:
        mode = "features"
    elif args.top_features is not None:
        mode = "top_k"
    else:
        mode = "all"
    feature_to_genes: list[dict[str, Any]] = select_feature_records(
        input_path,
        mode=mode,
        top_k=args.top_features,
        rank_by=args.rank_by,
        features=args.features,
        summary_path=summary_path,
        min_genes=args.min_genes,
        max_genes=args.max_genes,
    )
    n_features = len(feature_to_genes)
    n_qualifying = count_qualifying(feature_to_genes, args.min_genes)
    bounds_label = gene_bounds_label(args.min_genes, args.max_genes)

    print(f"\n{'='*60}")
    print(f"Input      : {input_path}")
    print(f"Selection  : {mode}" + (f" (rank_by={args.rank_by})" if mode == "top_k" else ""))
    print(f"Features   : {n_features:,}  ({n_qualifying:,} with {bounds_label})")
    print(f"Organism   : {args.organism}")
    print(f"Background : {f'{len(background):,} genes' if background else 'whole genome'}")
    print(f"Sources    : {', '.join(args.sources)}")
    print(f"FDR cutoff : {args.user_threshold}")
    print(f"Top terms  : {args.top_terms if args.top_terms else 'all'}")
    print(f"Output     : {output_path}")
    print(f"{'='*60}\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched = enrich_features(
        feature_to_genes,
        organism=args.organism,
        sources=args.sources,
        user_threshold=args.user_threshold,
        min_genes=args.min_genes,
        max_genes=args.max_genes,
        top_terms=args.top_terms,
        request_delay=args.request_delay,
        background=background,
    )

    save_go_terms(enriched, output_path)

    n_with_terms = sum(1 for r in enriched if r["go_terms"])
    total_terms = sum(len(r["go_terms"]) for r in enriched)
    print(f"\nDone. {n_with_terms:,}/{n_qualifying:,} qualifying features have significant GO terms.")
    print(f"Total GO term annotations : {total_terms:,}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
