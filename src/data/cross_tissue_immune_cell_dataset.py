"""Cross-tissue immune cell atlas dataset configuration and raw-data loading."""

from pathlib import Path

from src.data.dataset_config import (
    DatasetConfig,
    subset_raw_adata,
)


DATASET_NAMES = ("cross_tissue_immune_cell_atlas",)


def _full_atlas_config(dataset_configs: dict, project_root: Path) -> DatasetConfig:
    ds_paths = dataset_configs["cross_tissue_immune_cell_atlas"]
    return DatasetConfig(
        name="cross_tissue_immune_cell_atlas",
        data_path=project_root / ds_paths["data_path"],
        save_dir=project_root / ds_paths["save_dir"],
        save_name=ds_paths["save_name"],
        cell_type_col="Manually_curated_celltype",
        c2s_label_cols=["Manually_curated_celltype", "Organ", "Donor"],
        umap_color_cols=["Manually_curated_celltype", "Organ", "Donor"],
        distribution_plot_specs=[
            ("Manually_curated_celltype", "cell_type", "Cell type"),
            ("Organ", "organ", "Organ"),
            ("Donor", "donor", "Donor"),
        ],
        subset_col="Manually_curated_celltype",
        cell_subset=ds_paths.get("cell_subset") or None,
        cell_subset_n=ds_paths.get("cell_subset_n") or None,
        dataset_subset_col="Donor",
        dataset_subset=ds_paths.get("donor_subset") or None,
        dataset_subset_label="Donor subset",
    )


def make_dataset_config(dataset_name: str, dataset_configs: dict, project_root: Path) -> DatasetConfig:
    if dataset_name == "cross_tissue_immune_cell_atlas":
        return _full_atlas_config(dataset_configs, project_root)
    raise ValueError(
        f"Unknown cross-tissue dataset: {dataset_name!r}. "
        "Choose 'cross_tissue_immune_cell_atlas'."
    )


def load_raw(ds_cfg: DatasetConfig):
    """Load a cross-tissue AnnData file from disk."""

    print(f"Loading: {ds_cfg.data_path}")
    import anndata

    return anndata.read_h5ad(ds_cfg.data_path)


def prepare_raw_adata(ds_cfg: DatasetConfig, seed: int):
    """Load and apply configured donor and cell-type selection."""

    adata = load_raw(ds_cfg)
    return subset_raw_adata(adata, ds_cfg, seed)
