#!/usr/bin/env python3
"""
ToppGene Gene Ontology enrichment for SAE features.

Reads a feature_to_genes.jsonl file produced by src/evaluate/gene_features.py,
maps gene symbols to Human Entrez IDs with the ToppGene lookup API, then
queries ToppGene functional enrichment for GO terms.

Prefer src/evaluate/enrich/enrich.py for config-driven runs where the provider,
the GO sources, and the feature selection should be set in YAML.  Use this
script directly when debugging ToppGene lookup, caching, or request parameters.

Input contract:
    feature_to_genes.jsonl records must contain human gene symbols in the
    "genes" list, which holds for gene_features.py runs over a gene vocabulary.

Unlike go_enrich.py, this script writes one JSONL result record as soon as each
feature finishes. Interrupted runs therefore keep partial output and can resume
from the existing output file.

Outputs, by default in --results:
    feature_toppgene_go_terms.jsonl        streamed per-feature enrichment
    toppgene_symbol_lookup_cache.jsonl     streamed symbol-to-Entrez cache

Usage:
    python src/evaluate/enrich/toppgene_enrich.py \\
        --results results/gene_features/<sae>/<dataset>/<model>
    python src/evaluate/enrich/toppgene_enrich.py --results <dir> --top-features 100 \\
        --p-value 0.05 --min-genes 5 --max-results 50 --sources GO:BP GO:MF GO:CC
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import requests

try:
    from .enrich_utils import (
        chunks,
        count_qualifying,
        gene_bounds_label,
        iter_jsonl,
        list_field,
        load_completed_features,
        maybe_float,
        maybe_int,
        select_feature_records,
        truncate_genes,
        unique_preserving_order,
        write_jsonl_record,
    )
except ImportError:  # pragma: no cover - direct script execution
    from enrich_utils import (
        chunks,
        count_qualifying,
        gene_bounds_label,
        iter_jsonl,
        list_field,
        load_completed_features,
        maybe_float,
        maybe_int,
        select_feature_records,
        truncate_genes,
        unique_preserving_order,
        write_jsonl_record,
    )

_TOPPGENE_API = "https://toppgene.cchmc.org/API"
_LOOKUP_URL = f"{_TOPPGENE_API}/lookup"
_ENRICH_URL = f"{_TOPPGENE_API}/enrich"
_HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}
_RETRY_DELAYS = [2, 5, 10]

_SOURCE_TO_CATEGORY = {
    "GO:BP": "GeneOntologyBiologicalProcess",
    "GO:MF": "GeneOntologyMolecularFunction",
    "GO:CC": "GeneOntologyCellularComponent",
}
_CATEGORY_TO_SOURCE = {v: k for k, v in _SOURCE_TO_CATEGORY.items()}
_DEFAULT_SOURCES = ["GO:BP", "GO:MF", "GO:CC"]


# ---------------------------------------------------------------------------
# Resume and lookup cache
# ---------------------------------------------------------------------------


def load_lookup_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return cache

    for rec in iter_jsonl(path):
        submitted = rec.get("submitted")
        if submitted:
            cache[str(submitted)] = rec
    return cache


def append_lookup_cache(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", buffering=1) as f:
        for rec in records:
            write_jsonl_record(f, rec)


# ---------------------------------------------------------------------------
# ToppGene API
# ---------------------------------------------------------------------------

def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    for attempt, delay in enumerate([0] + _RETRY_DELAYS):
        if delay:
            time.sleep(delay)

        try:
            resp = requests.post(
                url,
                data=json.dumps(payload),
                headers=_HEADERS,
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            if data is None:
                return {}
            if not isinstance(data, dict):
                raise ValueError(f"Expected JSON object, got {type(data).__name__}")
            return data
        except (requests.RequestException, ValueError) as exc:
            if attempt < len(_RETRY_DELAYS):
                print(f"    [retry {attempt + 1}] {url}: {exc}")
            else:
                raise RuntimeError(f"{url}: {exc}") from exc

    raise RuntimeError(f"{url}: request failed")


def lookup_symbols(
    symbols: list[str],
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
    batch_size: int,
    timeout: float,
    request_delay: float,
) -> None:
    missing = [s for s in unique_preserving_order(symbols) if s not in cache]
    if not missing:
        return

    for batch in chunks(missing, batch_size):
        data = post_json(_LOOKUP_URL, {"Symbols": batch}, timeout)
        found_by_submitted: dict[str, dict[str, Any]] = {}

        for gene in list_field(data, "Genes"):
            if not isinstance(gene, dict):
                continue

            submitted_raw = gene.get("Submitted")
            if submitted_raw is None:
                continue

            submitted = str(submitted_raw).strip()
            if not submitted:
                continue

            found_by_submitted[submitted] = {
                "submitted": submitted,
                "official_symbol": gene.get("OfficialSymbol"),
                "entrez": maybe_int(gene.get("Entrez")),
            }

        records = []
        for symbol in batch:
            rec = found_by_submitted.get(
                symbol,
                {
                    "submitted": symbol,
                    "official_symbol": None,
                    "entrez": None,
                },
            )
            cache[symbol] = rec
            records.append(rec)

        append_lookup_cache(cache_path, records)
        if request_delay > 0:
            time.sleep(request_delay)


def build_entrez_query(
    genes: list[str],
    cache: dict[str, dict[str, Any]],
) -> tuple[list[int], list[str]]:
    entrez_ids: list[int] = []
    unmapped: list[str] = []
    seen_entrez: set[int] = set()

    for gene in genes:
        entrez = maybe_int((cache.get(gene) or {}).get("entrez"))
        if entrez is None:
            unmapped.append(gene)
            continue

        if entrez not in seen_entrez:
            seen_entrez.add(entrez)
            entrez_ids.append(entrez)

    return entrez_ids, unmapped


def normalize_sources(values: list[str]) -> list[str]:
    categories = []
    for value in values:
        if value in _SOURCE_TO_CATEGORY:
            categories.append(_SOURCE_TO_CATEGORY[value])
        elif value in _CATEGORY_TO_SOURCE:
            categories.append(value)
        else:
            valid = ", ".join(sorted(_SOURCE_TO_CATEGORY))
            raise ValueError(f"Unknown GO source/category: {value}. Valid sources: {valid}")
    return unique_preserving_order(categories)


def build_categories(
    category_names: list[str],
    p_value: float,
    correction: str,
    term_min_genes: int,
    term_max_genes: int,
    max_results: int,
) -> list[dict[str, Any]]:
    categories = []
    for category_name in category_names:
        category = {
            "Type": category_name,
            "PValue": p_value,
            "MinGenes": term_min_genes,
            "MaxGenes": term_max_genes,
            "Correction": correction,
        }
        if max_results > 0:
            category["MaxResults"] = max_results
        categories.append(category)
    return categories


def term_sort_value(term: dict[str, Any], correction: str) -> tuple[float, float]:
    if correction == "Bonferroni":
        primary = term.get("q_value_bonferroni")
    elif correction == "none":
        primary = term.get("p_value")
    else:
        primary = term.get("q_value_fdr_bh")

    secondary = term.get("p_value")
    return (
        float("inf") if primary is None else float(primary),
        float("inf") if secondary is None else float(secondary),
    )


def query_toppgene_enrichment(
    entrez_ids: list[int],
    categories: list[dict[str, Any]],
    correction: str,
    timeout: float,
    include_term_genes: bool,
) -> list[dict[str, Any]]:
    payload = {"Genes": entrez_ids, "Categories": categories}
    data = post_json(_ENRICH_URL, payload, timeout)

    terms = []
    for ann in list_field(data, "Annotations"):
        if not isinstance(ann, dict):
            continue

        category = ann.get("Category")
        if category not in _CATEGORY_TO_SOURCE:
            continue

        term = {
            "id": ann.get("ID"),
            "name": ann.get("Name"),
            "source": _CATEGORY_TO_SOURCE[category],
            "category": category,
            "p_value": maybe_float(ann.get("PValue")),
            "q_value_fdr_bh": maybe_float(ann.get("QValueFDRBH")),
            "q_value_fdr_by": maybe_float(ann.get("QValueFDRBY")),
            "q_value_bonferroni": maybe_float(ann.get("QValueBonferroni")),
            "total_genes": maybe_int(ann.get("TotalGenes")),
            "term_size": maybe_int(ann.get("GenesInTerm")),
            "query_size": maybe_int(ann.get("GenesInQuery")),
            "intersection_size": maybe_int(ann.get("GenesInTermInQuery")),
        }

        if include_term_genes:
            term_genes = []
            for gene in list_field(ann, "Genes"):
                if not isinstance(gene, dict):
                    continue
                term_genes.append(
                    {
                        "symbol": gene.get("Symbol"),
                        "entrez": maybe_int(gene.get("Entrez")),
                    }
                )
            term["genes"] = term_genes

        terms.append(term)

    terms.sort(key=lambda term: term_sort_value(term, correction))
    return terms


# ---------------------------------------------------------------------------
# Main enrichment loop
# ---------------------------------------------------------------------------

def enrich_one_feature(
    rec: dict[str, Any],
    lookup_cache: dict[str, dict[str, Any]],
    lookup_cache_path: Path,
    categories: list[dict[str, Any]],
    correction: str,
    min_genes: int,
    lookup_batch_size: int,
    timeout: float,
    lookup_delay: float,
    include_term_genes: bool,
    include_unmapped: bool,
    max_genes: int | None = None,
) -> dict[str, Any]:
    feat_idx = rec["feature"]
    all_genes = unique_preserving_order(
        str(gene) for gene in list_field(rec, "genes") if gene
    )
    # select_feature_records may have truncated already and stamped the
    # pre-truncation size; len(all_genes) would then report the cap itself.
    n_genes_total = int(rec.get("n_genes_total") or len(all_genes))
    genes = truncate_genes(all_genes, max_genes)
    n_input_genes = len(genes)

    result = {
        "feature": feat_idx,
        "provider": "ToppGene",
        "n_input_genes": n_input_genes,
        "n_genes_total": n_genes_total,
        "truncated": n_input_genes < n_genes_total,
        "n_mapped_genes": 0,
        "n_unmapped_genes": 0,
        "status": "pending",
        "go_terms": [],
    }

    if n_input_genes < min_genes:
        result["status"] = "skipped_min_genes"
        return result

    try:
        lookup_symbols(
            genes,
            lookup_cache,
            lookup_cache_path,
            lookup_batch_size,
            timeout,
            lookup_delay,
        )
        entrez_ids, unmapped = build_entrez_query(genes, lookup_cache)
        result["n_mapped_genes"] = len(entrez_ids)
        result["n_unmapped_genes"] = len(unmapped)
        if include_unmapped:
            result["unmapped_genes"] = unmapped

        if len(entrez_ids) < min_genes:
            result["status"] = "skipped_mapped_min_genes"
            return result

        terms = query_toppgene_enrichment(
            entrez_ids,
            categories,
            correction,
            timeout,
            include_term_genes,
        )
        result["go_terms"] = terms
        result["status"] = "ok" if terms else "no_terms"
        return result
    except RuntimeError as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        return result


def run_streaming_enrichment(
    records: Iterable[dict[str, Any]],
    output_path: Path,
    lookup_cache_path: Path,
    categories: list[dict[str, Any]],
    correction: str,
    min_genes: int,
    lookup_batch_size: int,
    timeout: float,
    request_delay: float,
    lookup_delay: float,
    include_term_genes: bool,
    include_unmapped: bool,
    overwrite: bool,
    max_genes: int | None = None,
) -> dict[str, int]:
    if overwrite and output_path.exists():
        output_path.unlink()

    completed_features = load_completed_features(output_path)
    lookup_cache = load_lookup_cache(lookup_cache_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if output_path.exists() else "w"

    stats = {
        "seen": 0,
        "written": 0,
        "skipped_existing": 0,
        "with_terms": 0,
        "total_terms": 0,
        "failed": 0,
    }

    with open(output_path, mode, buffering=1) as out:
        for rec in records:
            stats["seen"] += 1
            feat_idx = int(rec["feature"])

            if feat_idx in completed_features:
                stats["skipped_existing"] += 1
                continue

            result = enrich_one_feature(
                rec,
                lookup_cache,
                lookup_cache_path,
                categories,
                correction,
                min_genes,
                lookup_batch_size,
                timeout,
                lookup_delay,
                include_term_genes,
                include_unmapped,
                max_genes,
            )
            write_jsonl_record(out, result)

            n_terms = len(result["go_terms"])
            stats["written"] += 1
            stats["total_terms"] += n_terms
            if n_terms:
                stats["with_terms"] += 1
            if result["status"] == "failed":
                stats["failed"] += 1

            if stats["seen"] % 50 == 0:
                print(
                    f"  [{stats['seen']:,}] feature {feat_idx}: "
                    f"{result['n_input_genes']} genes, "
                    f"{result['n_mapped_genes']} mapped -> "
                    f"{n_terms} ToppGene GO terms ({result['status']})"
                )

            if request_delay > 0:
                time.sleep(request_delay)

    return stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GO enrichment for SAE concept features via the ToppGene API."
    )
    parser.add_argument(
        "--results",
        required=True,
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
        "--top-features",
        type=int,
        default=None,
        dest="top_features",
        help="Enrich only the top N ranked features (default: all features).",
    )
    parser.add_argument(
        "--rank-by",
        default="n_genes",
        dest="rank_by",
        help="feature_activation_summary.csv column to rank by (default: n_genes).",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        help="Explicit feature IDs to enrich, e.g. 0 5,9 10-20. Overrides --top-features.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (default: <results>/feature_toppgene_go_terms.jsonl).",
    )
    parser.add_argument(
        "--lookup-cache",
        default=None,
        help="Symbol lookup cache path (default: <results>/toppgene_symbol_lookup_cache.jsonl).",
    )
    parser.add_argument(
        "--p-value",
        type=float,
        default=0.05,
        dest="p_value",
        help="ToppGene p-value cutoff before correction handling (default: 0.05).",
    )
    parser.add_argument(
        "--correction",
        choices=["FDR", "Bonferroni", "none"],
        default="FDR",
        help="ToppGene multiple-testing correction method (default: FDR).",
    )
    parser.add_argument(
        "--min-genes",
        type=int,
        default=3,
        dest="min_genes",
        help="Skip features with fewer than this many input or mapped genes (default: 3).",
    )
    parser.add_argument(
        "--max-genes",
        type=int,
        default=None,
        dest="max_genes",
        help="Skip features with more than this many input genes (default: no cap).",
    )
    parser.add_argument(
        "--term-min-genes",
        type=int,
        default=1,
        dest="term_min_genes",
        help="ToppGene MinGenes per annotation term (default: 1).",
    )
    parser.add_argument(
        "--term-max-genes",
        type=int,
        default=1500,
        dest="term_max_genes",
        help="ToppGene MaxGenes per annotation term (default: 1500).",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=100,
        dest="max_results",
        help="Max ToppGene results per GO category; use 0 to omit this field (default: 100).",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=_DEFAULT_SOURCES,
        help=f"GO sources to query (default: {' '.join(_DEFAULT_SOURCES)}).",
    )
    parser.add_argument(
        "--lookup-batch-size",
        type=int,
        default=200,
        dest="lookup_batch_size",
        help="Number of new symbols per ToppGene lookup request (default: 200).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.25,
        dest="request_delay",
        help="Seconds to wait after each feature enrichment request (default: 0.25).",
    )
    parser.add_argument(
        "--lookup-delay",
        type=float,
        default=0.05,
        dest="lookup_delay",
        help="Seconds to wait after each lookup batch request (default: 0.05).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds (default: 30).",
    )
    parser.add_argument(
        "--include-term-genes",
        action="store_true",
        help="Include the intersecting genes returned for each GO term.",
    )
    parser.add_argument(
        "--include-unmapped",
        action="store_true",
        help="Include unmapped input gene symbols in each output record.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the ToppGene output file instead of resuming/appending.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results)
    input_path = Path(args.input) if args.input else results_dir / "feature_to_genes.jsonl"
    summary_path = (
        Path(args.summary)
        if args.summary
        else results_dir / "feature_activation_summary.csv"
    )
    output_path = (
        Path(args.output)
        if args.output
        else results_dir / "feature_toppgene_go_terms.jsonl"
    )
    lookup_cache_path = (
        Path(args.lookup_cache)
        if args.lookup_cache
        else results_dir / "toppgene_symbol_lookup_cache.jsonl"
    )

    if not input_path.exists():
        raise FileNotFoundError(f"Not found: {input_path}")

    category_names = normalize_sources(args.sources)
    categories = build_categories(
        category_names,
        args.p_value,
        args.correction,
        args.term_min_genes,
        args.term_max_genes,
        args.max_results,
    )

    if args.features:
        mode = "features"
    elif args.top_features is not None:
        mode = "top_k"
    else:
        mode = "all"
    records = select_feature_records(
        input_path,
        mode=mode,
        top_k=args.top_features,
        rank_by=args.rank_by,
        features=args.features,
        summary_path=summary_path,
        min_genes=args.min_genes,
        max_genes=args.max_genes,
    )
    n_qualifying = count_qualifying(records, args.min_genes)
    bounds_label = gene_bounds_label(args.min_genes, args.max_genes)
    existing = 0 if args.overwrite else len(load_completed_features(output_path))

    print(f"\n{'='*60}")
    print("Provider       : ToppGene")
    print(f"Input          : {input_path}")
    print(f"Output         : {output_path}")
    print(f"Lookup cache   : {lookup_cache_path}")
    print(f"Selection      : {mode}" + (f" (rank_by={args.rank_by})" if mode == "top_k" else ""))
    print(f"Features       : {len(records):,}  ({n_qualifying:,} with {bounds_label})")
    print(f"Resume skip    : {existing:,} existing feature records")
    print(f"Sources        : {', '.join(_CATEGORY_TO_SOURCE[c] for c in category_names)}")
    print(f"P-value cutoff : {args.p_value}")
    print(f"Correction     : {args.correction}")
    print(f"Max results    : {args.max_results if args.max_results > 0 else 'server default'}")
    print(f"{'='*60}\n")

    stats = run_streaming_enrichment(
        records,
        output_path,
        lookup_cache_path,
        categories,
        args.correction,
        args.min_genes,
        args.lookup_batch_size,
        args.timeout,
        args.request_delay,
        args.lookup_delay,
        args.include_term_genes,
        args.include_unmapped,
        args.overwrite,
        max_genes=args.max_genes,
    )

    print("\nDone.")
    print(f"Feature records seen     : {stats['seen']:,}")
    print(f"Skipped existing records : {stats['skipped_existing']:,}")
    print(f"New records written      : {stats['written']:,}")
    print(f"New records with GO terms: {stats['with_terms']:,}")
    print(f"New GO term annotations  : {stats['total_terms']:,}")
    print(f"Failed records           : {stats['failed']:,}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
