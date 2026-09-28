"""Purified PBMC dataset configuration and raw-data loading."""

from pathlib import Path

from src.data.dataset_config import (
    DatasetConfig,
    subset_raw_adata,
)


DATASET_NAMES = ("purified_pbmc",)


def make_dataset_config(dataset_configs: dict, project_root: Path) -> DatasetConfig:
    ds_paths = dataset_configs["purified_pbmc"]
    return DatasetConfig(
        name="purified_pbmc",
        data_path=project_root / ds_paths["data_path"],
        save_dir=project_root / ds_paths["save_dir"],
        save_name=ds_paths["save_name"],
        cell_type_col="cell_types",
        c2s_label_cols=["cell_types", "batch"],
        umap_color_cols=["cell_types"],
        distribution_plot_specs=[
            ("cell_types", "cell_types", "Cell type"),
        ],
        subset_col="batch",
        cell_subset=ds_paths.get("cell_subset") or None,
        cell_subset_n=ds_paths.get("cell_subset_n") or None,
    )


def load_raw(ds_cfg: DatasetConfig):
    """Load the purified PBMC AnnData, downloading it with scvi-tools if needed."""

    path = ds_cfg.data_path
    if not path.exists():
        print(f"Purified PBMC not found at {path} -- downloading via scvi-tools...")
        import scvi

        adata = scvi.data.purified_pbmc_dataset(save_path=str(path.parent) + "/")
        path.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(path)
        print(f"Saved downloaded data to {path}")
        return adata

    print(f"Loading: {path}")
    import anndata

    return anndata.read_h5ad(path)


def prepare_raw_adata(ds_cfg: DatasetConfig, seed: int):
    """Load and apply purified-PBMC-specific batch selection/subsampling."""

    adata = load_raw(ds_cfg)
    return subset_raw_adata(adata, ds_cfg, seed)
