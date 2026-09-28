"""
C2S data preparation — multi-dataset

Preprocess a single-cell RNA-seq dataset and produce extraction-ready
Arrow train/validation/test datasets for Cell2Sentence (C2S) inference.

Supported datasets (configured via configs/data_prep.yaml):
    cross_tissue_immune_cell_atlas — full cross-tissue immune cell atlas
                                    (329 k cells, 17 organs, 12 donors)
                                    cell_type col: 'Manually_curated_celltype'
                                    donor col: 'Donor'
                                    donors: 582C, 621B, 637C, 640C, A29, A31,
                                            A35, A36, A37, A52, D496, D503
    purified_pbmc                  — 10x sorted PBMC populations
                                    (~105 k cells, blood only)
                                    cell_type col: 'cell_types'
    pbmc                           — scVI PBMC dataset
                                    (~12 k cells, 2 batches)
                                    cell_type col: 'str_labels', whose 10x
                                    display labels are renamed to subset-style
                                    names (B_cells, cd14_monocytes, NK_cells, ...)
                                    var_names are Ensembl IDs; remapped to
                                    symbols from var['gene_symbols']

Pipeline:
    1. Load raw counts, remapping var_names to gene symbols when needed
    2. Filter cells / genes (scarcity filter)
    3. Mitochondrial-content filter (QC)
    4. Count-normalize and log1p-transform (base 10 — required for C2S)
    5. Optional preview plots / UMAP / summary text
    6. Save preprocessed AnnData
    7. Convert to C2S Arrow dataset via `cell2sentence`
    8. Split into train/validation/test datasets
    9. Build vocab.json from train and remove validation/test cells with unseen genes
    10. Optionally write preview/ folder (plots + summary text)
    11. Save metadata.json with generation settings and final split counts

Usage:
    python -m src.data.data_prep
    python -m src.data.data_prep --dataset cross_tissue_immune_cell_atlas
    python -m src.data.data_prep --dataset purified_pbmc
    python -m src.data.data_prep --dataset pbmc
    python -m src.data.data_prep --config configs/data_prep.yaml
    python -m src.data.data_prep --no-umap
"""

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import anndata
import numpy as np
import scipy.sparse

from src.data.cross_tissue_immune_cell_dataset import (
    DATASET_NAMES as CROSS_TISSUE_DATASET_NAMES,
    make_dataset_config as make_cross_tissue_dataset_config,
    prepare_raw_adata as prepare_cross_tissue_raw_adata,
)
from src.data.dataset_config import DatasetConfig, apply_gene_symbols
from src.data.pbmc_dataset import (
    DATASET_NAMES as PBMC_DATASET_NAMES,
    make_dataset_config as make_pbmc_dataset_config,
    prepare_raw_adata as prepare_pbmc_raw_adata,
)
from src.data.purified_pbmc_dataset import (
    DATASET_NAMES as PURIFIED_PBMC_DATASET_NAMES,
    make_dataset_config as make_purified_pbmc_dataset_config,
    prepare_raw_adata as prepare_purified_pbmc_raw_adata,
)

# ── reproducibility ────────────────────────────────────────────────────────

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)

# ── project root ───────────────────────────────────────────────────────────

_FILE_DIR    = Path(__file__).parent
PROJECT_ROOT = _FILE_DIR.parents[1]

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "data_prep.yaml"
DATASET_CONFIGS_KEY = "dataset_configs"
LEGACY_DATASET_CONFIGS_KEY = "paths"


# ── Dataset registry ───────────────────────────────────────────────────────

SUPPORTED_DATASETS = (
    CROSS_TISSUE_DATASET_NAMES
    + PURIFIED_PBMC_DATASET_NAMES
    + PBMC_DATASET_NAMES
)


# ── Config loading ─────────────────────────────────────────────────────────

def load_config(config_path: Path) -> dict:
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_dataset_configs(cfg: dict) -> dict:
    if DATASET_CONFIGS_KEY in cfg:
        return cfg[DATASET_CONFIGS_KEY]
    if LEGACY_DATASET_CONFIGS_KEY in cfg:
        print(
            f"Warning: top-level '{LEGACY_DATASET_CONFIGS_KEY}' is deprecated; "
            f"use '{DATASET_CONFIGS_KEY}' instead."
        )
        return cfg[LEGACY_DATASET_CONFIGS_KEY]
    raise KeyError(
        f"Config must contain a top-level '{DATASET_CONFIGS_KEY}' mapping."
    )


def make_dataset_config(cfg: dict, project_root: Path) -> DatasetConfig:
    dataset = cfg["dataset"]
    dataset_configs = get_dataset_configs(cfg)
    if dataset in CROSS_TISSUE_DATASET_NAMES:
        return make_cross_tissue_dataset_config(dataset, dataset_configs, project_root)
    if dataset in PURIFIED_PBMC_DATASET_NAMES:
        return make_purified_pbmc_dataset_config(dataset_configs, project_root)
    if dataset in PBMC_DATASET_NAMES:
        return make_pbmc_dataset_config(dataset_configs, project_root)
    raise ValueError(
        f"Unknown dataset: {dataset!r}. Choose one of: "
        f"{', '.join(SUPPORTED_DATASETS)}."
    )


# ── 1. Load raw data ───────────────────────────────────────────────────────

def prepare_raw_dataset(ds_cfg: DatasetConfig, seed: int) -> anndata.AnnData:
    if ds_cfg.name in CROSS_TISSUE_DATASET_NAMES:
        return prepare_cross_tissue_raw_adata(ds_cfg, seed)
    if ds_cfg.name in PURIFIED_PBMC_DATASET_NAMES:
        return prepare_purified_pbmc_raw_adata(ds_cfg, seed)
    if ds_cfg.name in PBMC_DATASET_NAMES:
        return prepare_pbmc_raw_adata(ds_cfg, seed)
    raise ValueError(f"No raw-data loader registered for {ds_cfg.name!r}.")


def log_raw_summary(adata: anndata.AnnData, ds_cfg: DatasetConfig) -> None:
    print(f"  shape       : {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")
    print(f"  obs columns : {list(adata.obs.columns)}")
    print(f"  var columns : {list(adata.var.columns)}")
    print(f"  X dtype     : {adata.X.dtype}")
    X = adata.X
    if scipy.sparse.issparse(X):
        X = X.toarray()
    print(f"  X max value : {X.max():.1f}")

    adata.var_names_make_unique()

    ct_col = ds_cfg.cell_type_col
    print(f"\nCell-type breakdown ('{ct_col}'):")
    for ct, n in Counter(adata.obs[ct_col]).most_common():
        print(f"  {n:>5}  {ct}")


# ── 2. Scarcity filtering ──────────────────────────────────────────────────

def filter_scarcity(
    adata: anndata.AnnData,
    min_genes: int = 200,
    max_genes: int = 2000,
    min_cells: int = 3,
) -> anndata.AnnData:
    import scanpy as sc

    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_cells(adata, max_genes=max_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    print(f"After scarcity filtering: {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")
    return adata


# ── 3. QC metrics + mitochondrial filter ──────────────────────────────────

def add_qc_metrics(adata: anndata.AnnData) -> anndata.AnnData:
    import scanpy as sc

    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    n_mt = int(adata.var["mt"].sum())
    print(f"Mitochondrial genes detected: {n_mt}")
    if n_mt == 0:
        print(
            "  Warning: no 'MT-' genes in var_names, so pct_counts_mt is 0 for "
            "every cell and the mitochondrial filter cannot remove anything."
        )
    print(adata.obs[["n_genes_by_counts", "total_counts", "pct_counts_mt"]].describe())
    return adata


def filter_mitochondrial(
    adata: anndata.AnnData,
    mt_threshold: float = 20.0,
) -> anndata.AnnData:
    adata = adata[adata.obs.pct_counts_mt < mt_threshold, :].copy()
    print(f"After MT filter (pct_counts_mt < {mt_threshold}): {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")
    return adata


# ── 4. Normalization ───────────────────────────────────────────────────────

def normalize(adata: anndata.AnnData) -> anndata.AnnData:
    import scanpy as sc

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata, base=10)
    X = adata.X
    if scipy.sparse.issparse(X):
        X = X.toarray()
    print(f"Max value after log10 normalization: {X.max():.4f}  (expected ~3-4)")
    return adata


# ── 5b. Canonical metadata columns ────────────────────────────────────────

def add_standard_metadata_columns(
    adata: anndata.AnnData,
    ds_cfg: DatasetConfig,
) -> anndata.AnnData:
    """Add stable metadata columns expected by downstream data workflows."""

    if ds_cfg.cell_type_col not in adata.obs.columns:
        raise ValueError(
            f"{ds_cfg.name} is missing configured cell-type column "
            f"{ds_cfg.cell_type_col!r}."
        )

    cell_type = adata.obs[ds_cfg.cell_type_col].astype("string").str.strip()
    missing_mask = cell_type.isna() | (cell_type == "")
    if bool(missing_mask.any()):
        n_missing = int(missing_mask.sum())
        raise ValueError(
            f"{ds_cfg.name} has {n_missing:,} empty cell-type value(s) in "
            f"{ds_cfg.cell_type_col!r}."
        )

    adata.obs["cell_type"] = cell_type.astype(str)
    print(f"Standard metadata: cell_type <- {ds_cfg.cell_type_col}")
    return adata


# ── 6. Save preprocessed AnnData ──────────────────────────────────────────

def save_preprocessed(adata: anndata.AnnData, out_path: Path) -> None:
    adata.write_h5ad(out_path)
    print(f"Saved preprocessed AnnData: {out_path}")


# ── 7. C2S Arrow conversion ────────────────────────────────────────────────

def to_arrow(adata: anndata.AnnData, ds_cfg: DatasetConfig, seed: int):
    import cell2sentence as cs

    obs_cols = []
    for col in ["cell_type", *ds_cfg.c2s_label_cols]:
        if col in adata.obs.columns and col not in obs_cols:
            obs_cols.append(col)
    arrow_ds, vocabulary = cs.CSData.adata_to_arrow(
        adata=adata,
        random_state=seed,
        sentence_delimiter=" ",
        label_col_names=obs_cols,
    )

    def _add_num_genes(ex):
        return {"num_genes": len(ex["cell_sentence"].split())}

    arrow_ds = arrow_ds.map(_add_num_genes, desc="Adding num_genes")

    print(f"\nArrow dataset: {arrow_ds}")
    print(f"Vocabulary size: {len(vocabulary):,}")
    return arrow_ds, vocabulary


def validate_cell_sentence_dataset(dataset, dataset_name: str, sample_n: int = 3) -> None:
    """Validate the C2S Arrow dataset contract used by extraction.py."""

    column_names = set(getattr(dataset, "column_names", []))
    if "cell_sentence" not in column_names:
        raise ValueError(
            f"{dataset_name} does not contain a 'cell_sentence' column. "
            "extraction.py requires this column."
        )
    if "cell_type" not in column_names:
        raise ValueError(
            f"{dataset_name} does not contain a 'cell_type' column. "
            "data_prep.py should create this canonical column for every dataset."
        )
    if len(dataset) == 0:
        raise ValueError(f"{dataset_name} is empty after preprocessing.")

    for i, value in enumerate(dataset["cell_type"]):
        if value is None or not str(value).strip():
            raise ValueError(
                f"{dataset_name} sample {i} has an empty or invalid cell_type."
            )

    for i in range(min(sample_n, len(dataset))):
        cell_sentence = dataset[i].get("cell_sentence")
        if not isinstance(cell_sentence, str) or not cell_sentence.strip():
            raise ValueError(
                f"{dataset_name} sample {i} has an empty or invalid cell_sentence."
            )

    print(
        f"Validated extraction-ready cell_sentence column "
        f"({len(dataset):,} cells)."
    )


def validate_saved_cell_sentence_dataset(dataset_path: Path) -> None:
    from datasets import load_from_disk

    saved_dataset = load_from_disk(str(dataset_path))
    validate_cell_sentence_dataset(saved_dataset, str(dataset_path))


def configure_preview_environment() -> None:
    """Ensure preview plotting libraries can write their runtime cache."""

    import os

    os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))


def split_arrow_dataset(
    arrow_ds,
    val_fraction: float,
    test_fraction: float,
    seed: int,
):
    """Return train/val/test splits using stable suffix names."""

    if not 0.0 < val_fraction < 1.0:
        raise ValueError(
            f"split.val_fraction must be between 0 and 1, got {val_fraction}."
        )
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(
            f"split.test_fraction must be between 0 and 1, got {test_fraction}."
        )
    heldout_fraction = val_fraction + test_fraction
    if heldout_fraction >= 1.0:
        raise ValueError(
            "split.val_fraction + split.test_fraction must be less than 1, "
            f"got {heldout_fraction}."
        )

    train_heldout = arrow_ds.train_test_split(
        test_size=heldout_fraction,
        seed=seed,
        shuffle=True,
    )
    train_ds = train_heldout["train"]
    heldout_ds = train_heldout["test"]

    val_test = heldout_ds.train_test_split(
        test_size=test_fraction / heldout_fraction,
        seed=seed + 1,
        shuffle=True,
    )
    val_ds = val_test["train"]
    test_ds = val_test["test"]

    print(
        f"Train/val/test split: train={len(train_ds):,} cells, "
        f"val={len(val_ds):,} cells (val_fraction={val_fraction}), "
        f"test={len(test_ds):,} cells (test_fraction={test_fraction})"
    )
    return {"train": train_ds, "val": val_ds, "test": test_ds}


def build_vocab_from_cell_sentences(dataset, split_name: str) -> list[str]:
    """Build a sorted gene vocabulary from one split's cell sentences."""

    genes: set[str] = set()
    for sample in dataset:
        genes.update(sample["cell_sentence"].split())

    vocab = sorted(genes)
    print(f"{split_name} vocabulary: {len(vocab):,} unique genes")
    return vocab


def filter_cells_to_vocab(dataset, vocab: set[str], split_name: str):
    """Discard cells containing any gene outside the provided vocabulary."""

    keep_indices: list[int] = []
    oov_genes: set[str] = set()
    for i in range(len(dataset)):
        genes = set(dataset[i]["cell_sentence"].split())
        unknown = genes - vocab
        if unknown:
            oov_genes.update(unknown)
            continue
        keep_indices.append(i)

    discarded = len(dataset) - len(keep_indices)
    if discarded:
        if not keep_indices:
            raise ValueError(
                f"{split_name} has no cells left after filtering against train vocab."
            )
        dataset = dataset.select(keep_indices)
        preview = ", ".join(sorted(oov_genes)[:10])
        suffix = "..." if len(oov_genes) > 10 else ""
        print(
            f"{split_name}: kept {len(keep_indices):,}/{len(keep_indices) + discarded:,} "
            f"cells; discarded {discarded:,} cell(s) with {len(oov_genes):,} "
            f"gene(s) absent from train vocab."
        )
        print(f"  OOV examples: {preview}{suffix}")
    else:
        print(f"{split_name}: all {len(dataset):,} cells are covered by train vocab")

    return dataset


def validate_dataset_genes_in_vocab(dataset, vocab: set[str], split_name: str) -> None:
    """Assert every cell sentence only contains genes present in vocab.json."""

    for i in range(len(dataset)):
        unknown = set(dataset[i]["cell_sentence"].split()) - vocab
        if unknown:
            examples = ", ".join(sorted(unknown)[:10])
            raise ValueError(
                f"{split_name} cell {i} contains gene(s) absent from train vocab: "
                f"{examples}"
            )

    print(f"Validated {split_name}: all {len(dataset):,} cells use train-vocab genes.")


def save_c2s_split(split_ds, vocabulary, dataset_dir: Path, split_name: str) -> Path:
    import cell2sentence as cs

    save_name = f"data_{split_name}"
    cs.CSData.csdata_from_arrow(
        arrow_dataset=split_ds,
        vocabulary=vocabulary,
        save_dir=str(dataset_dir),
        save_name=save_name,
        dataset_backend="arrow",
    )
    saved_path = dataset_dir / save_name
    validate_saved_cell_sentence_dataset(saved_path)
    print(f"Saved {split_name} dataset: {saved_path}")
    return saved_path


# ── Vocabulary file ────────────────────────────────────────────────────────

def save_vocab(vocab: list[str], dataset_dir: Path) -> list[str]:
    vocab_path = dataset_dir / "vocab.json"
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, indent=2)

    print(f"Saved train vocab.json: {len(vocab):,} unique genes → {vocab_path}")
    return vocab


# ── Metadata file ──────────────────────────────────────────────────────────

def _project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def _jsonable(value):
    if isinstance(value, Path):
        return _project_relative(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, set):
        return [_jsonable(v) for v in sorted(value)]
    return value


def _shape_metadata(adata: anndata.AnnData) -> dict[str, int]:
    return {
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
    }


def _cli_override_metadata(args: argparse.Namespace) -> dict:
    overrides = {}
    if args.dataset:
        overrides["dataset"] = args.dataset
    if args.data_path:
        overrides["data_path"] = args.data_path
    if args.save_dir:
        overrides["save_dir"] = args.save_dir
    if args.save_name:
        overrides["save_name"] = args.save_name
    if args.no_umap:
        overrides["run_umap"] = False
    return _jsonable(overrides)


def save_metadata(
    *,
    cfg: dict,
    args: argparse.Namespace,
    ds_cfg: DatasetConfig,
    dataset_dir: Path,
    preprocessed_path: Path,
    saved_dataset_paths: dict[str, Path],
    preview_dir: Path | None,
    adata_raw: anndata.AnnData,
    adata_filtered: anndata.AnnData,
    adata_pre_arrow: anndata.AnnData,
    arrow_ds,
    split_datasets: dict,
    vocab: list[str],
) -> Path:
    dataset_configs = get_dataset_configs(cfg)
    configured_dataset = dataset_configs.get(ds_cfg.name, {})
    metadata = {
        "metadata_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_file": args.config,
        "cli_overrides": _cli_override_metadata(args),
        "dataset": ds_cfg.name,
        "dataset_config": configured_dataset,
        "effective_paths": {
            "data_path": ds_cfg.data_path,
            "save_dir": ds_cfg.save_dir,
            "save_name": ds_cfg.save_name,
            "dataset_dir": dataset_dir,
        },
        "selection": {
            "source_cell_type_column": ds_cfg.cell_type_col,
            "canonical_cell_type_column": "cell_type",
            "cell_subset_column": ds_cfg.subset_col,
            "cell_subset": ds_cfg.cell_subset,
            "cell_subset_n": ds_cfg.cell_subset_n,
            "metadata_subset_column": ds_cfg.dataset_subset_col,
            "metadata_subset": ds_cfg.dataset_subset,
            "metadata_subset_label": ds_cfg.dataset_subset_label,
        },
        "seed": cfg["seed"],
        "split": cfg["split"],
        "filtering": cfg["filtering"],
        "processing": cfg["processing"],
        "counts": {
            "raw_after_configured_selection": _shape_metadata(adata_raw),
            "after_filtering": _shape_metadata(adata_filtered),
            "before_arrow_conversion": _shape_metadata(adata_pre_arrow),
            "c2s_cells_before_split": int(len(arrow_ds)),
            "saved_split_cells": {
                split_name: int(len(split_ds))
                for split_name, split_ds in split_datasets.items()
            },
            "vocab_size": int(len(vocab)),
        },
        "artifacts": {
            "preprocessed_h5ad": preprocessed_path,
            "splits": saved_dataset_paths,
            "vocab": dataset_dir / "vocab.json",
            "preview_dir": preview_dir,
        },
    }

    metadata_path = dataset_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(_jsonable(metadata), f, indent=2)
        f.write("\n")

    print(f"Saved metadata.json: {metadata_path}")
    return metadata_path


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess single-cell data and produce extraction-ready C2S Arrow train/val/test datasets."
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help="Path to YAML config file (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset", choices=SUPPORTED_DATASETS,
        help="Override config dataset selection",
    )
    parser.add_argument(
        "--data-path", type=Path,
        help="Override data path from config",
    )
    parser.add_argument(
        "--save-dir", type=Path,
        help="Override save directory from config",
    )
    parser.add_argument(
        "--save-name",
        help="Override save name from config",
    )
    parser.add_argument(
        "--no-umap", action="store_true",
        help="Skip UMAP computation",
    )
    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Load YAML config
    cfg = load_config(args.config)
    processing = cfg["processing"]

    # Apply CLI overrides
    if args.dataset:
        cfg["dataset"] = args.dataset
    if args.no_umap:
        processing["run_umap"] = False

    ds_cfg = make_dataset_config(cfg, PROJECT_ROOT)

    # CLI path overrides (applied after ds_cfg so they can replace defaults)
    if args.data_path:
        ds_cfg.data_path = args.data_path
    if args.save_dir:
        ds_cfg.save_dir = args.save_dir
    if args.save_name:
        ds_cfg.save_name = args.save_name

    seed         = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    val_fraction = cfg["split"]["val_fraction"]
    test_fraction = cfg["split"]["test_fraction"]
    filt         = cfg["filtering"]
    generate_preview = processing["generate_preview"]
    run_umap_opt = processing["run_umap"]

    dataset_dir  = ds_cfg.save_dir / ds_cfg.save_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = dataset_dir / "preview" if generate_preview else None
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
        configure_preview_environment()
        from src.data import data_preview
        data_preview.clear_managed_preview_outputs(preview_dir)

    print(f"Dataset : {ds_cfg.name}")
    print(f"Input   : {ds_cfg.data_path}")
    print(f"Output  : {dataset_dir}")
    print(f"Preview : {preview_dir if generate_preview else 'disabled'}")
    print()

    # ── 1. Load ───────────────────────────────────────────────────────────
    print("=" * 60)
    print("Step 1: Load raw data")
    print("=" * 60)
    adata = prepare_raw_dataset(ds_cfg, seed)
    adata = apply_gene_symbols(adata, ds_cfg)
    adata = add_standard_metadata_columns(adata, ds_cfg)
    log_raw_summary(adata, ds_cfg)
    adata_raw = adata.copy()

    # ── 2. Scarcity filter ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: Scarcity filtering")
    print("=" * 60)
    adata = filter_scarcity(
        adata,
        min_genes=filt["min_genes"],
        max_genes=filt["max_genes"],
        min_cells=filt["min_cells"],
    )

    # ── 3. QC + MT filter ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3: QC metrics + mitochondrial filter")
    print("=" * 60)
    adata = add_qc_metrics(adata)
    adata = filter_mitochondrial(adata, mt_threshold=filt["mt_threshold"])
    adata_filtered = adata.copy()
    if generate_preview:
        data_preview.save_qc_violin(adata_filtered, preview_dir)

    # ── 4. Preview status ─────────────────────────────────────────────────
    if generate_preview:
        print("\nStep 4: Preview generation enabled")
    else:
        print("\nStep 4: Preview generation disabled (generate_preview=false)")

    # ── 5. Normalize ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 5: Normalize (total counts + log10)")
    print("=" * 60)
    adata = normalize(adata)

    # ── 6. Optional UMAP ──────────────────────────────────────────────────
    if generate_preview and run_umap_opt:
        print("\nStep 6: UMAP delayed until after splitting")
    elif not generate_preview:
        print("\nStep 6: UMAP skipped (generate_preview=false)")
    else:
        print("\nStep 6: UMAP skipped (run_umap=false in config)")

    # ── 7. Save preprocessed AnnData ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 7: Save preprocessed AnnData")
    print("=" * 60)
    preprocessed_path = PROJECT_ROOT / "datasets" / f"{ds_cfg.save_name}_preprocessed.h5ad"
    save_preprocessed(adata, preprocessed_path)

    # ── 8. C2S Arrow conversion ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 8: C2S Arrow conversion")
    print("=" * 60)
    # C2S uppercases var_names internally; duplicates after uppercasing
    # shorten enc_map below adata.n_vars, causing an IndexError.
    adata.var_names = adata.var_names.str.upper()
    dupes = adata.var_names[adata.var_names.duplicated()]
    if len(dupes):
        print(f"Dropping {len(dupes):,} duplicate gene(s) after uppercasing: {dupes.tolist()}")
        adata = adata[:, ~adata.var_names.duplicated()].copy()
    arrow_ds, _ = to_arrow(adata, ds_cfg, seed)
    validate_cell_sentence_dataset(arrow_ds, ds_cfg.name)
    split_datasets = split_arrow_dataset(arrow_ds, val_fraction, test_fraction, seed)

    # ── 9. Train vocabulary + held-out filtering ──────────────────────────
    print("\n" + "=" * 60)
    print("Step 9: Build train vocab and filter validation/test")
    print("=" * 60)
    vocab = build_vocab_from_cell_sentences(split_datasets["train"], "train")
    vocab_set = set(vocab)
    for split_name in ("val", "test"):
        split_datasets[split_name] = filter_cells_to_vocab(
            split_datasets[split_name], vocab_set, split_name
        )
    for split_name, split_ds in split_datasets.items():
        validate_dataset_genes_in_vocab(split_ds, vocab_set, split_name)

    # ── 10. Save Arrow datasets ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 10: Save C2S Arrow train/val/test datasets")
    print("=" * 60)
    saved_dataset_paths = {
        split_name: save_c2s_split(split_ds, vocab, dataset_dir, split_name)
        for split_name, split_ds in split_datasets.items()
    }

    # ── 11. Vocabulary file ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 11: Save train vocab.json")
    print("=" * 60)
    vocab = save_vocab(vocab, dataset_dir)

    # ── 12. Preview outputs ───────────────────────────────────────────────
    if generate_preview:
        print("\n" + "=" * 60)
        print("Step 12: Preview outputs")
        print("=" * 60)
        data_preview.save_distribution_plots(split_datasets, ds_cfg, preview_dir)
        if run_umap_opt:
            data_preview.run_umap(
                adata,
                ds_cfg,
                preview_dir,
                seed,
                split_datasets=split_datasets,
            )
        data_preview.save_num_genes_histogram(split_datasets, preview_dir)
        data_preview.save_text_summary(
            adata_raw,
            adata_filtered,
            arrow_ds,
            split_datasets,
            vocab,
            ds_cfg,
            preview_dir,
        )
    else:
        print("\nStep 12: Preview outputs skipped (generate_preview=false)")

    # ── 13. Run metadata ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 13: Save metadata.json")
    print("=" * 60)
    metadata_path = save_metadata(
        cfg=cfg,
        args=args,
        ds_cfg=ds_cfg,
        dataset_dir=dataset_dir,
        preprocessed_path=preprocessed_path,
        saved_dataset_paths=saved_dataset_paths,
        preview_dir=preview_dir,
        adata_raw=adata_raw,
        adata_filtered=adata_filtered,
        adata_pre_arrow=adata,
        arrow_ds=arrow_ds,
        split_datasets=split_datasets,
        vocab=vocab,
    )

    # ── Done ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Done")
    print("=" * 60)
    print(f"Train Arrow dataset  : {saved_dataset_paths['train']}")
    print(f"Val Arrow dataset    : {saved_dataset_paths['val']}")
    print(f"Test Arrow dataset   : {saved_dataset_paths['test']}")
    print(f"Vocab file           : {dataset_dir / 'vocab.json'}")
    print(f"Metadata file        : {metadata_path}")
    if preview_dir is not None:
        print(f"Preview folder       : {preview_dir}")
        print("  " + "\n  ".join(p.name for p in sorted(preview_dir.iterdir())))
    else:
        print("Preview folder       : disabled")
    return saved_dataset_paths


if __name__ == "__main__":
    main()
