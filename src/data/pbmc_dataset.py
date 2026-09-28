"""scVI PBMC dataset configuration and raw-data loading."""

from pathlib import Path

from src.data.dataset_config import (
    DatasetConfig,
    subset_raw_adata,
)


DATASET_NAMES = ("pbmc",)

CELL_TYPE_COL = "str_labels"

# obs['str_labels'] ships 10x display labels ("CD14+ Monocytes"), which reach
# downstream code as cell-type names containing spaces and '+'. Rewrite them as
# subset-style identifiers so pbmc cell types match the purified_pbmc
# convention and stay usable as run labels and file names.
CELL_TYPE_NAMES = {
    "B cells": "B_cells",
    "CD4 T cells": "CD4_T_cells",
    "CD8 T cells": "CD8_T_cells",
    "CD14+ Monocytes": "cd14_monocytes",
    "FCGR3A+ Monocytes": "fcgr3a_monocytes",
    "NK cells": "NK_cells",
    "Dendritic Cells": "dendritic_cells",
    "Megakaryocytes": "megakaryocytes",
    "Other": "other",
}


def canonical_cell_type(label) -> str:
    """Map one raw obs['str_labels'] value to its subset-style name."""

    label = str(label).strip()
    return CELL_TYPE_NAMES.get(label, label)


def rename_cell_types(adata):
    """Rewrite obs['str_labels'] display labels as subset-style names."""

    labels = adata.obs[CELL_TYPE_COL].astype(str).str.strip()
    unknown = sorted(set(labels) - set(CELL_TYPE_NAMES))
    if unknown:
        print(
            f"  Warning: no subset name registered for {', '.join(unknown)}; "
            "keeping the raw label."
        )

    adata.obs[CELL_TYPE_COL] = (
        labels.map(CELL_TYPE_NAMES).fillna(labels).astype("category")
    )
    renamed = ", ".join(sorted(set(adata.obs[CELL_TYPE_COL])))
    print(f"Cell types renamed to subset names ({CELL_TYPE_COL}): {renamed}")
    return adata


def make_dataset_config(dataset_configs: dict, project_root: Path) -> DatasetConfig:
    ds_paths = dataset_configs["pbmc"]
    # Accept either spelling in cell_subset so older configs keep working.
    cell_subset = ds_paths.get("cell_subset") or None
    if cell_subset is not None:
        cell_subset = [canonical_cell_type(value) for value in cell_subset]
    return DatasetConfig(
        name="pbmc",
        data_path=project_root / ds_paths["data_path"],
        save_dir=project_root / ds_paths["save_dir"],
        save_name=ds_paths["save_name"],
        cell_type_col=CELL_TYPE_COL,
        # var_names are Ensembl IDs; symbols live in var['gene_symbols'].
        gene_symbol_col="gene_symbols",
        c2s_label_cols=[CELL_TYPE_COL, "labels", "batch"],
        umap_color_cols=[CELL_TYPE_COL],
        distribution_plot_specs=[
            (CELL_TYPE_COL, "cell_type", "Cell type"),
        ],
        subset_col=CELL_TYPE_COL,
        cell_subset=cell_subset,
        cell_subset_n=ds_paths.get("cell_subset_n") or None,
        dataset_subset_col="batch",
        dataset_subset=ds_paths.get("dataset_subset") or None,
        dataset_subset_label="Source dataset subset",
    )


def load_raw(ds_cfg: DatasetConfig):
    """Load scvi.data.pbmc_dataset, caching the returned AnnData as h5ad."""

    path = ds_cfg.data_path
    if not path.exists():
        print(f"scVI PBMC not found at {path} -- downloading via scvi-tools...")
        import scvi

        path.parent.mkdir(parents=True, exist_ok=True)
        adata = scvi.data.pbmc_dataset(save_path=str(path.parent) + "/")
        adata.write_h5ad(path)
        print(f"Saved downloaded data to {path}")
        return adata

    print(f"Loading: {path}")
    import anndata

    return anndata.read_h5ad(path)


def prepare_raw_adata(ds_cfg: DatasetConfig, seed: int):
    """Load and apply PBMC-specific dataset and cell-type selection."""

    adata = load_raw(ds_cfg)
    # Rename before subsetting so cell_subset selects on the subset-style names.
    adata = rename_cell_types(adata)
    return subset_raw_adata(adata, ds_cfg, seed)
