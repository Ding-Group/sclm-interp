#!/usr/bin/env python3
"""
Unified gene-set enrichment entrypoint for SAE feature gene sets.

This is the preferred launcher for enrichment after ``src/evaluate/gene_features.py``.
It reads that script's per-feature gene sets, selects which features to enrich,
picks one backend from the YAML config, and writes the results under
``results/enrich/`` mirroring the gene-features directory layout.

Input contract (a gene_features.py result directory):
    feature_to_genes.jsonl
        ``{"feature": int, "genes": [gene_symbol, ...]}`` per activated feature.
    feature_activation_summary.csv
        Per-feature statistics used to rank features for top-k selection.

Feature selection (``select`` section):
    mode: "all"       enrich every feature in feature_to_genes.jsonl
    mode: "top_k"     enrich the ``top_k`` features ranked by ``rank_by``
    mode: "features"  enrich the explicitly listed feature IDs

Output layout:
    Every run lands in its own numbered folder so different feature selections
    over the same dataset never overwrite each other::

        results/enrich/<sae>/<dataset>/<model>/
            runs.json                            index of all runs here
            toppgene_symbol_lookup_cache.jsonl   shared symbol cache
            1/
                metadata.json                    the selection this run enriched
                feature_toppgene_go_terms.jsonl
                toppgene_summary.json
            2/
                ...

    A run whose selection matches an existing run reuses that folder instead of
    allocating a new number, so re-running the same config resumes or overwrites
    in place. Set ``output.run`` (or ``--run``) to target one explicitly.

Provider choices:
    toppgene
        Remote ToppGene over-representation analysis with symbol lookup/cache.
    gprofiler
        Remote g:Profiler GO over-representation analysis.

Usage:
    python src/evaluate/enrich/enrich.py
    python src/evaluate/enrich/enrich.py --config configs/enrich.yaml
    python src/evaluate/enrich/enrich.py --provider gprofiler --top-k 200
    python src/evaluate/enrich/enrich.py --features 0 5,9 100-120 --label b_cells
    python src/evaluate/enrich/enrich.py --run 2 --provider gprofiler
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from . import go_enrich, toppgene_enrich
    from .enrich_utils import (
        RANK_COLUMNS,
        count_qualifying,
        gene_bounds_label,
        load_completed_features,
        select_feature_records,
    )
except ImportError:  # pragma: no cover - direct script execution
    import go_enrich
    import toppgene_enrich
    from enrich_utils import (
        RANK_COLUMNS,
        count_qualifying,
        gene_bounds_label,
        load_completed_features,
        select_feature_records,
    )

from evaluate.model_loading import project_path

_DEFAULT_CONFIG = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "configs"
    / "enrich.yaml"
)
_PROVIDERS = {"toppgene", "gprofiler"}
_SELECT_MODES = {"all", "top_k", "features"}
_DEFAULT_DATABASES = ["GO:BP", "GO:MF", "GO:CC"]
_GENE_FEATURES_ROOT = "gene_features"


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected config section '{name}' to be a mapping.")
    return value


def _as_list(value: Any, default: list[str] | None = None) -> list[str]:
    if value is None:
        return list(default or [])
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    raise ValueError(f"Expected a string or list, got {type(value).__name__}.")


def _config_value(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def _optional_int(value: Any, name: str) -> int | None:
    if value is None or value == "":
        return None
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative or null.")
    return parsed


def _resolve(path: str | Path | None) -> Path | None:
    return project_path(path) if path is not None else None


def _resolve_child(base_dir: Path | None, value: str | Path | None) -> Path | None:
    """Resolve a config path, treating relative values as children of base_dir."""

    if value is None:
        return None
    path_value = Path(value).expanduser()
    if path_value.is_absolute():
        return path_value
    if base_dir is None:
        return project_path(path_value)
    return base_dir / path_value


def mirror_subdir(input_dir: Path) -> Path:
    """
    Map a gene-features result directory to its results/enrich counterpart.

    ``results/gene_features/<sae>/<dataset>/<model>`` becomes
    ``<sae>/<dataset>/<model>`` so both result trees stay aligned.
    """
    parts = input_dir.parts
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == _GENE_FEATURES_ROOT and index + 1 < len(parts):
            return Path(*parts[index + 1 :])
    if len(parts) >= 3:
        return Path(*parts[-3:])
    return Path(input_dir.name)


def load_config(path: str | Path | None) -> SimpleNamespace:
    config_path = Path(path) if path is not None else _DEFAULT_CONFIG
    if not config_path.is_file():
        raise FileNotFoundError(f"Enrichment config not found: {config_path}")
    with config_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML mapping in {config_path}")

    cfg_raw = _section(raw, "enrich") if "enrich" in raw else raw
    input_raw = _section(raw, "input") or _section(cfg_raw, "input")
    output_raw = _section(raw, "output") or _section(cfg_raw, "output")
    select = _section(cfg_raw, "select") or _section(raw, "select")
    common = _section(cfg_raw, "common")
    gprofiler = _section(cfg_raw, "gprofiler")
    toppgene = _section(cfg_raw, "toppgene")

    input_dir = _resolve(input_raw.get("dir"))
    if input_dir is None:
        raise ValueError("Set input.dir to a src/evaluate/gene_features.py result directory.")

    input_path = _resolve_child(input_dir, input_raw.get("feature_to_genes"))
    if input_path is None:
        input_path = input_dir / "feature_to_genes.jsonl"
    summary_path = _resolve_child(input_dir, input_raw.get("feature_summary"))
    if summary_path is None:
        summary_path = input_dir / "feature_activation_summary.csv"

    output_root = _resolve(output_raw.get("dir"))
    if output_root is None:
        raise ValueError("Set output.dir, e.g. results/enrich.")
    subdir = output_raw.get("subdir")
    results_dir = (
        output_root / Path(subdir) if subdir else output_root / mirror_subdir(input_dir)
    )
    run = _optional_int(output_raw.get("run"), "output.run")
    if run is not None and run < 1:
        raise ValueError("output.run must be at least 1 or null.")
    label = output_raw.get("label")

    provider = str(
        _config_value(cfg_raw.get("provider"), common.get("provider"), default="toppgene")
    ).lower()
    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Valid: {', '.join(sorted(_PROVIDERS))}.")

    mode = str(select.get("mode", "top_k")).lower()
    if mode not in _SELECT_MODES:
        raise ValueError(
            f"Unknown select.mode '{mode}'. Valid: {', '.join(sorted(_SELECT_MODES))}."
        )
    top_k = _optional_int(select.get("top_k"), "select.top_k")
    if mode == "top_k" and not top_k:
        raise ValueError("Set select.top_k to at least 1 when select.mode='top_k'.")
    rank_by = str(select.get("rank_by", "n_genes"))
    if rank_by not in RANK_COLUMNS:
        raise ValueError(f"Unknown select.rank_by '{rank_by}'. Valid: {', '.join(RANK_COLUMNS)}.")
    features = select.get("features") or []
    if not isinstance(features, list):
        features = [features]
    if mode == "features" and not features:
        raise ValueError("Set select.features to at least one ID when select.mode='features'.")

    database_value = _config_value(
        cfg_raw.get("databases"),
        cfg_raw.get("database"),
        common.get("databases"),
        common.get("database"),
        default=_DEFAULT_DATABASES,
    )
    databases = _as_list(database_value, _DEFAULT_DATABASES)

    return SimpleNamespace(
        config_path=config_path,
        provider=provider,
        databases=databases,
        input_dir=input_dir,
        input_path=input_path,
        summary_path=summary_path,
        output_root=output_root,
        output_subdir=subdir,
        results_dir=results_dir,
        run=run,
        label=None if label is None else str(label),
        run_dir=None,
        output_path=_resolve(common.get("output")),
        mode=mode,
        top_k=top_k,
        rank_by=rank_by,
        features=features,
        min_genes=int(common.get("min_genes", 3)),
        max_genes=_optional_int(common.get("max_genes"), "enrich.common.max_genes"),
        overwrite=bool(common.get("overwrite", False)),
        gprofiler=gprofiler,
        toppgene=toppgene,
    )


def apply_cli_overrides(cfg: SimpleNamespace, args: argparse.Namespace) -> SimpleNamespace:
    if args.provider is not None:
        cfg.provider = args.provider
    if args.input_dir is not None:
        cfg.input_dir = project_path(args.input_dir)
        cfg.input_path = cfg.input_dir / "feature_to_genes.jsonl"
        cfg.summary_path = cfg.input_dir / "feature_activation_summary.csv"
        if cfg.output_subdir is None:
            # Keep results/enrich mirroring the new gene-features directory.
            cfg.results_dir = cfg.output_root / mirror_subdir(cfg.input_dir)
    if args.input is not None:
        cfg.input_path = project_path(args.input)
    if args.summary is not None:
        cfg.summary_path = project_path(args.summary)
    if args.results is not None:
        cfg.results_dir = project_path(args.results)
    if args.run is not None:
        if args.run < 1:
            raise ValueError("--run must be at least 1.")
        cfg.run = args.run
    if args.label is not None:
        cfg.label = args.label
    if args.output is not None:
        cfg.output_path = project_path(args.output)
    if args.databases is not None:
        cfg.databases = args.databases
    if args.features:
        cfg.mode = "features"
        cfg.features = args.features
    elif args.top_k is not None:
        cfg.mode = "top_k"
        cfg.top_k = args.top_k
    elif args.all_features:
        cfg.mode = "all"
    if args.rank_by is not None:
        cfg.rank_by = args.rank_by
    if args.min_genes is not None:
        cfg.min_genes = args.min_genes
    if args.max_genes is not None:
        cfg.max_genes = _optional_int(args.max_genes, "--max-genes")
    if args.overwrite:
        cfg.overwrite = True
    return cfg


def default_output(cfg: SimpleNamespace, name: str) -> Path:
    if cfg.output_path is not None:
        return cfg.output_path
    return cfg.run_dir / name


def select_records(cfg: SimpleNamespace) -> list[dict[str, Any]]:
    if not cfg.input_path.exists():
        raise FileNotFoundError(f"Not found: {cfg.input_path}")
    return select_feature_records(
        cfg.input_path,
        mode=cfg.mode,
        top_k=cfg.top_k,
        rank_by=cfg.rank_by,
        features=cfg.features,
        summary_path=cfg.summary_path,
        min_genes=cfg.min_genes,
        max_genes=cfg.max_genes,
    )


def selection_label(cfg: SimpleNamespace) -> str:
    if cfg.mode == "top_k":
        return f"top {cfg.top_k:,} by {cfg.rank_by}"
    if cfg.mode == "features":
        return "explicit feature list"
    return "all features"


# ---------------------------------------------------------------------------
# Run directories
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_label(cfg: SimpleNamespace) -> str:
    name = cfg.run_dir.name
    return f"{name} ({cfg.label})" if cfg.label else name


def selection_fingerprint(cfg: SimpleNamespace, records: list[dict[str, Any]]) -> str:
    """
    Identify a run by what it actually enriched.

    Two invocations share a fingerprint when they enrich the same feature IDs
    from the same gene sets against the same GO sources, regardless of how the
    selection was expressed (top_k versus an explicit list) or which provider
    ran, so both providers write into one run folder.
    """
    payload = {
        "feature_to_genes": str(cfg.input_path),
        "features": [rec["feature"] for rec in records],
        "min_genes": cfg.min_genes,
        "max_genes": cfg.max_genes,
        "databases": sorted(cfg.databases),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def iter_run_dirs(base_dir: Path) -> list[tuple[int, Path]]:
    """List ``(run_id, path)`` for the numbered run folders under base_dir."""

    if not base_dir.is_dir():
        return []
    runs = [
        (int(child.name), child)
        for child in base_dir.iterdir()
        if child.is_dir() and child.name.isdigit()
    ]
    return sorted(runs)


def load_run_metadata(run_dir: Path) -> dict[str, Any]:
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.is_file():
        return {}
    with metadata_path.open(encoding="utf-8") as f:
        metadata = json.load(f)
    return metadata if isinstance(metadata, dict) else {}


def resolve_run_dir(cfg: SimpleNamespace, records: list[dict[str, Any]]) -> Path:
    """
    Pick the run folder for this invocation.

    An explicit ``output.run`` always wins. Otherwise a run whose recorded
    fingerprint matches this selection is reused, and a genuinely new selection
    gets the next free number.
    """
    base_dir = cfg.results_dir
    if cfg.run is not None:
        return base_dir / str(cfg.run)

    fingerprint = selection_fingerprint(cfg, records)
    existing = iter_run_dirs(base_dir)
    for run_id, run_dir in existing:
        if load_run_metadata(run_dir).get("fingerprint") == fingerprint:
            print(f"Reusing run {run_id} with a matching feature selection.")
            return run_dir

    next_id = max((run_id for run_id, _ in existing), default=0) + 1
    return base_dir / str(next_id)


def write_run_metadata(
    cfg: SimpleNamespace,
    records: list[dict[str, Any]],
    *,
    provider: str | None = None,
) -> Path:
    """
    Record which features this run enriched.

    Called before the provider runs so an interrupted run still documents its
    selection, then again afterwards to note the provider that completed.
    """
    metadata_path = cfg.run_dir / "metadata.json"
    previous = load_run_metadata(cfg.run_dir)
    providers = list(previous.get("providers") or [])
    if provider is not None and provider not in providers:
        providers.append(provider)

    metadata = {
        "run": int(cfg.run_dir.name) if cfg.run_dir.name.isdigit() else cfg.run_dir.name,
        "label": cfg.label if cfg.label is not None else previous.get("label"),
        "fingerprint": selection_fingerprint(cfg, records),
        "created": previous.get("created", _now()),
        "updated": _now(),
        "providers": providers,
        "inputs": {
            "config": str(cfg.config_path),
            "gene_features_dir": str(cfg.input_dir),
            "feature_to_genes": str(cfg.input_path),
            "feature_summary": str(cfg.summary_path),
        },
        "selection": {
            "mode": cfg.mode,
            "label": selection_label(cfg),
            "top_k": cfg.top_k,
            "rank_by": cfg.rank_by,
            "requested_features": cfg.features,
            "min_genes": cfg.min_genes,
            "max_genes": cfg.max_genes,
            "databases": cfg.databases,
            "n_selected": len(records),
            "n_qualifying": count_qualifying(records, cfg.min_genes),
            "features": [rec["feature"] for rec in records],
            "feature_gene_counts": {
                str(rec["feature"]): len(rec["genes"]) for rec in records
            },
        },
    }
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    return metadata_path


def update_run_index(cfg: SimpleNamespace) -> Path:
    """Rebuild ``runs.json`` from the run folders so the index self-heals."""

    index_path = cfg.results_dir / "runs.json"
    entries = []
    for run_id, run_dir in iter_run_dirs(cfg.results_dir):
        metadata = load_run_metadata(run_dir)
        if not metadata:
            continue
        selection = metadata.get("selection") or {}
        entries.append(
            {
                "run": run_id,
                "dir": run_dir.name,
                "label": metadata.get("label"),
                "fingerprint": metadata.get("fingerprint"),
                "created": metadata.get("created"),
                "updated": metadata.get("updated"),
                "providers": metadata.get("providers") or [],
                "mode": selection.get("mode"),
                "selection": selection.get("label"),
                "n_selected": selection.get("n_selected"),
            }
        )
    with index_path.open("w", encoding="utf-8") as f:
        json.dump({"results_dir": str(cfg.results_dir), "runs": entries}, f, indent=2)
        f.write("\n")
    return index_path


def write_summary(
    cfg: SimpleNamespace,
    records: list[dict[str, Any]],
    output_path: Path,
    stats: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    summary_path = cfg.run_dir / f"{cfg.provider}_summary.json"
    summary = {
        "run": int(cfg.run_dir.name) if cfg.run_dir.name.isdigit() else cfg.run_dir.name,
        "label": cfg.label,
        "inputs": {
            "config": str(cfg.config_path),
            "gene_features_dir": str(cfg.input_dir),
            "feature_to_genes": str(cfg.input_path),
            "feature_summary": str(cfg.summary_path),
        },
        "provider": cfg.provider,
        "settings": {
            "databases": cfg.databases,
            "min_genes": cfg.min_genes,
            "max_genes": cfg.max_genes,
            "overwrite": cfg.overwrite,
            "select": {
                "mode": cfg.mode,
                "top_k": cfg.top_k,
                "rank_by": cfg.rank_by,
                "features": cfg.features,
            },
            **(extra or {}),
        },
        "selection": {
            "label": selection_label(cfg),
            "n_selected": len(records),
            "n_qualifying": count_qualifying(records, cfg.min_genes),
            "features": [rec["feature"] for rec in records],
        },
        "stats": stats,
        "results": {
            "results_dir": str(cfg.results_dir),
            "run_dir": str(cfg.run_dir),
            "metadata": str(cfg.run_dir / "metadata.json"),
            "enrichment": str(output_path),
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    return summary_path


def run_gprofiler(cfg: SimpleNamespace, records: list[dict[str, Any]]) -> None:
    output_path = default_output(cfg, "feature_go_terms.jsonl")

    section = cfg.gprofiler
    top_terms = section.get("top_terms")
    request_delay = float(section.get("delay", section.get("request_delay", 0.25)))
    threshold = float(section.get("fdr", section.get("user_threshold", 0.05)))
    organism = section.get("organism", "hsapiens")

    # true reuses the gene vocabulary of the gene_features run that produced the
    # feature gene sets; a string names another file. null keeps the genome.
    background_value = section.get("background")
    background = None
    background_path = None
    if background_value:
        if background_value is True:
            background_path = cfg.input_dir / "genes.json"
        else:
            background_path = _resolve_child(cfg.input_dir, str(background_value))
        if background_path is None or not background_path.exists():
            raise FileNotFoundError(f"Background gene list not found: {background_path}")
        background = go_enrich.load_background(background_path)

    n_qualifying = count_qualifying(records, cfg.min_genes)
    bounds_label = gene_bounds_label(cfg.min_genes, cfg.max_genes)

    print(f"\n{'='*60}")
    print("Provider   : g:Profiler")
    print(f"Input      : {cfg.input_path}")
    print(f"Run        : {run_label(cfg)}")
    print(f"Output     : {output_path}")
    print(f"Selection  : {selection_label(cfg)}")
    print(f"Features   : {len(records):,} ({n_qualifying:,} with {bounds_label})")
    print(f"Databases  : {', '.join(cfg.databases)}")
    print(f"Organism   : {organism}")
    print(f"Background : {f'{len(background):,} genes' if background else 'whole genome'}")
    print(f"FDR cutoff : {threshold}")
    print(f"{'='*60}\n")

    enriched = go_enrich.enrich_features(
        records,
        organism=organism,
        sources=cfg.databases,
        user_threshold=threshold,
        min_genes=cfg.min_genes,
        max_genes=cfg.max_genes,
        top_terms=None if top_terms is None else int(top_terms),
        request_delay=request_delay,
        background=background,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    go_enrich.save_go_terms(enriched, output_path)

    stats = {
        "selected": len(records),
        "with_terms": sum(1 for rec in enriched if rec["go_terms"]),
        "total_terms": sum(len(rec["go_terms"]) for rec in enriched),
    }
    summary_path = write_summary(
        cfg,
        records,
        output_path,
        stats,
        extra={
            "organism": organism,
            "fdr": threshold,
            "top_terms": top_terms,
            "background": None if background is None else str(background_path),
            "background_size": None if background is None else len(background),
        },
    )

    print(f"\nDone. {stats['with_terms']:,}/{n_qualifying:,} qualifying features have terms.")
    print(f"Total term annotations : {stats['total_terms']:,}")
    print(f"Saved to {output_path}")
    print(f"Summary  {summary_path}")


def run_toppgene(cfg: SimpleNamespace, records: list[dict[str, Any]]) -> None:
    output_path = default_output(cfg, "feature_toppgene_go_terms.jsonl")
    # The symbol cache lives above the run folders so every run on this dataset
    # shares one set of resolved Entrez IDs.
    lookup_cache_path = _resolve_child(cfg.results_dir, cfg.toppgene.get("lookup_cache"))
    if lookup_cache_path is None:
        lookup_cache_path = cfg.results_dir / "toppgene_symbol_lookup_cache.jsonl"

    section = cfg.toppgene
    correction = section.get("correction", "FDR")
    categories = toppgene_enrich.build_categories(
        toppgene_enrich.normalize_sources(cfg.databases),
        float(section.get("p_value", 0.05)),
        correction,
        int(section.get("term_min_genes", 1)),
        int(section.get("term_max_genes", 1500)),
        int(section.get("max_results", 100)),
    )

    n_qualifying = count_qualifying(records, cfg.min_genes)
    existing = 0 if cfg.overwrite else len(load_completed_features(output_path))
    bounds_label = gene_bounds_label(cfg.min_genes, cfg.max_genes)

    print(f"\n{'='*60}")
    print("Provider     : ToppGene")
    print(f"Input        : {cfg.input_path}")
    print(f"Run          : {run_label(cfg)}")
    print(f"Output       : {output_path}")
    print(f"Lookup cache : {lookup_cache_path}")
    print(f"Selection    : {selection_label(cfg)}")
    print(f"Features     : {len(records):,} ({n_qualifying:,} with {bounds_label})")
    print(f"Resume skip  : {existing:,} existing feature records")
    print(f"Databases    : {', '.join(cfg.databases)}")
    print(f"Correction   : {correction}")
    print(f"{'='*60}\n")

    stats = toppgene_enrich.run_streaming_enrichment(
        records,
        output_path,
        lookup_cache_path,
        categories,
        correction,
        cfg.min_genes,
        int(section.get("lookup_batch_size", 200)),
        float(section.get("timeout", 30.0)),
        float(section.get("delay", section.get("request_delay", 0.25))),
        float(section.get("lookup_delay", 0.05)),
        bool(section.get("include_term_genes", False)),
        bool(section.get("include_unmapped", False)),
        cfg.overwrite,
        max_genes=cfg.max_genes,
    )
    summary_path = write_summary(
        cfg,
        records,
        output_path,
        stats,
        extra={
            "correction": correction,
            "p_value": float(section.get("p_value", 0.05)),
            "max_results": int(section.get("max_results", 100)),
            "lookup_cache": str(lookup_cache_path),
        },
    )

    print("\nDone.")
    print(f"Skipped existing records : {stats['skipped_existing']:,}")
    print(f"New records written      : {stats['written']:,}")
    print(f"New records with terms   : {stats['with_terms']:,}")
    print(f"New term annotations     : {stats['total_terms']:,}")
    print(f"Failed records           : {stats['failed']:,}")
    print(f"Saved to {output_path}")
    print(f"Summary  {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified SAE feature gene-set enrichment.")
    parser.add_argument(
        "--config",
        default=None,
        help=f"YAML config path (default: {_DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--provider",
        choices=sorted(_PROVIDERS),
        default=None,
        help="Backend override.",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        dest="input_dir",
        help="gene_features.py result directory override.",
    )
    parser.add_argument("--input", default=None, help="feature_to_genes.jsonl override.")
    parser.add_argument(
        "--summary",
        default=None,
        help="feature_activation_summary.csv override.",
    )
    parser.add_argument("--results", default=None, help="Output directory override.")
    parser.add_argument(
        "--run",
        type=int,
        default=None,
        help=(
            "Numbered run folder to write into. Omit to reuse the run with a "
            "matching selection, or allocate the next number."
        ),
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Human-readable name for this run, recorded in metadata.json.",
    )
    parser.add_argument("--output", default=None, help="Output JSONL override.")
    parser.add_argument(
        "--databases",
        nargs="+",
        default=None,
        help="GO source override, e.g. GO:BP GO:MF.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        dest="top_k",
        help="Enrich only the top K ranked features.",
    )
    parser.add_argument(
        "--rank-by",
        choices=sorted(RANK_COLUMNS),
        default=None,
        dest="rank_by",
        help="feature_activation_summary.csv column used for top-K ranking.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        help="Explicit feature IDs to enrich, e.g. 0 5,9 10-20. Overrides --top-k.",
    )
    parser.add_argument(
        "--all-features",
        action="store_true",
        dest="all_features",
        help="Enrich every feature in feature_to_genes.jsonl.",
    )
    parser.add_argument(
        "--min-genes",
        type=int,
        default=None,
        help="Minimum genes per feature override.",
    )
    parser.add_argument(
        "--max-genes",
        type=int,
        default=None,
        help="Maximum genes per feature override; null in config disables the cap.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite instead of resuming an existing streamed output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = apply_cli_overrides(load_config(args.config), args)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the selection first: the run folder is chosen by what it enriches.
    records = select_records(cfg)
    cfg.run_dir = resolve_run_dir(cfg, records)
    if cfg.label is None:
        # Keep the name an earlier invocation gave this run.
        cfg.label = load_run_metadata(cfg.run_dir).get("label")
    metadata_path = write_run_metadata(cfg, records)

    if cfg.provider == "toppgene":
        run_toppgene(cfg, records)
    elif cfg.provider == "gprofiler":
        run_gprofiler(cfg, records)
    else:  # pragma: no cover - guarded during config loading
        raise ValueError(f"Unknown provider: {cfg.provider}")

    write_run_metadata(cfg, records, provider=cfg.provider)
    index_path = update_run_index(cfg)
    print(f"Metadata {metadata_path}")
    print(f"Index    {index_path}")


if __name__ == "__main__":
    main()
