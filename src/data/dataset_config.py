"""Shared dataset configuration helpers."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetConfig:
    """Parameters that differ between raw single-cell datasets."""

    name: str
    data_path: Path
    save_dir: Path
    save_name: str
    cell_type_col: str
    c2s_label_cols: list[str]
    umap_color_cols: list[str]
    distribution_plot_specs: list[tuple[str, str, str]]
    gene_symbol_col: str | None = None
    subset_col: str = "batch_condition"
    cell_subset: list[str | int] | None = None
    cell_subset_n: int | None = None
    dataset_subset_col: str | None = None
    dataset_subset: list[str | int] | None = None
    dataset_subset_label: str = "Dataset subset"


def apply_gene_symbols(adata, ds_cfg: DatasetConfig):
    """Replace var_names with gene symbols for datasets keyed by gene IDs.

    C2S builds its vocabulary and cell sentences directly from ``var_names``, so
    a dataset indexed by Ensembl IDs yields ENSG-based sentences: the pretrained
    tokenizer never saw those strings, and every downstream interpretability step
    reports unreadable IDs. Datasets that store symbols in a ``var`` column
    declare it via ``gene_symbol_col`` and are remapped here, before QC (which
    detects mitochondrial genes by an ``MT-`` prefix) and normalization run.

    The original identifiers are retained in ``var['gene_id']``.
    """

    col = ds_cfg.gene_symbol_col
    if col is None:
        return adata
    if col not in adata.var.columns:
        raise ValueError(
            f"{ds_cfg.name} is missing configured gene-symbol column {col!r}. "
            f"Available var columns: {list(adata.var.columns)}."
        )

    symbols = adata.var[col].astype("string").str.strip()
    unusable = (symbols.isna() | (symbols == "") | (symbols.str.lower() == "nan")).to_numpy()
    if unusable.any():
        preview = ", ".join(adata.var_names[unusable][:5])
        print(
            f"Dropping {int(unusable.sum()):,} gene(s) with no {col!r} value "
            f"(e.g. {preview})"
        )
        adata = adata[:, ~unusable].copy()
        symbols = symbols[~unusable]

    adata.var["gene_id"] = adata.var_names.to_numpy()
    adata.var_names = symbols.astype(str).to_numpy()

    duplicated = adata.var_names.duplicated()
    if duplicated.any():
        dupes = sorted(set(adata.var_names[duplicated]))
        preview = ", ".join(dupes[:10])
        suffix = "..." if len(dupes) > 10 else ""
        print(
            f"Dropping {int(duplicated.sum()):,} gene(s) whose {col!r} value "
            f"duplicates an earlier gene: {preview}{suffix}"
        )
        adata = adata[:, ~duplicated].copy()

    print(f"Gene identifiers: var_names <- var[{col!r}]  ({adata.n_vars:,} genes)")
    return adata


def subset_raw_adata(adata, ds_cfg: DatasetConfig, seed: int):
    """Apply configured metadata subsets and optional per-group sample."""

    if ds_cfg.dataset_subset is not None:
        if ds_cfg.dataset_subset_col is None:
            raise ValueError(
                f"{ds_cfg.name} configured dataset_subset without dataset_subset_col."
            )
        mask = adata.obs[ds_cfg.dataset_subset_col].isin(ds_cfg.dataset_subset)
        adata = adata[mask].copy()
        print(
            f"{ds_cfg.dataset_subset_label} {ds_cfg.dataset_subset} "
            f"({ds_cfg.dataset_subset_col}): {adata.shape[0]:,} cells retained"
        )

    if ds_cfg.cell_subset is not None:
        mask = adata.obs[ds_cfg.subset_col].isin(ds_cfg.cell_subset)
        adata = adata[mask].copy()
        print(
            f"Cell subset {ds_cfg.cell_subset} "
            f"({ds_cfg.subset_col}): {adata.shape[0]:,} cells retained"
        )

    if ds_cfg.cell_subset_n is not None:
        import numpy as np

        rng = np.random.default_rng(seed)
        keep = []
        for _, group in adata.obs.groupby(ds_cfg.subset_col, observed=True):
            n = min(ds_cfg.cell_subset_n, len(group))
            keep.extend(rng.choice(group.index, size=n, replace=False).tolist())
        adata = adata[keep].copy()
        print(
            f"Per-{ds_cfg.subset_col} subsample "
            f"({ds_cfg.cell_subset_n:,}/group): {adata.shape[0]:,} cells retained"
        )

    return adata


def path_has_arrow_dataset(path: Path) -> bool:
    """Return true when a path looks like a HuggingFace dataset saved to disk."""

    return (path / "state.json").exists() or (path / "dataset_info.json").exists()
