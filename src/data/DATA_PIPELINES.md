# Data Processing Pipelines

This directory has two related data workflows:

1. `data_prep.py` turns raw single-cell data into a Cell2Sentence Arrow dataset.
2. `extraction.py` reads that Arrow dataset and extracts model activations and gene vocab embeddings.

Preview plots and summaries live in `data_preview.py` and are controlled by `processing.generate_preview` in `configs/data_prep.yaml`.

Run commands from the repository root using the project Python environment.
YAML dataset/output paths resolve against the repository root. Data-prep CLI
`--config`, `--data-path`, and `--save-dir` paths are used as supplied, so
relative overrides are relative to the working directory.

The handoff between the two is a prepared dataset folder containing `data_train/`, `data_val/`, `data_test/`, `vocab.json`, and `metadata.json`. `data_prep.py` adds a canonical non-empty `cell_type` column for every supported dataset, builds `vocab.json` from `data_train` only, removes validation/test cells with genes outside that train vocabulary, records generation metadata, and validates all saved splits. `extraction.py` checks the same invariant before loading the base model.

## Supported Datasets

| Dataset key | Module | Raw source | Main labels | Default prepared output |
| --- | --- | --- | --- | --- |
| `cross_tissue_immune_cell_atlas` | `cross_tissue_immune_cell_dataset.py` | `raw_datasets/cross-tissue_immune_cell_atlas.h5ad` | `Manually_curated_celltype`, `Organ`, `Donor` | `datasets/c2s_datasets/cross-tissue-immune-cell-atlas/` |
| `purified_pbmc` | `purified_pbmc_dataset.py` | `raw_datasets/PurifiedPBMCDataset.h5ad`, downloaded from `scvi.data.purified_pbmc_dataset` if missing | `cell_types`, `batch` | `datasets/c2s_datasets/purified-pbmc-subset-full/` |
| `pbmc` | `pbmc_dataset.py` | `raw_datasets/PBMCDataset.h5ad`, downloaded from `scvi.data.pbmc_dataset` if missing | `str_labels`, `labels`, `batch` | `datasets/c2s_datasets/pbmc-subset-full/` |

The output names above come from the checked-in `save_name` values; they do
not imply all cells are selected. The current default is `pbmc`, restricted to
`B_cells`, `cd14_monocytes`, and `NK_cells` from both source batches.

PBMC display labels are canonicalized before subsetting (for example,
`CD14+ Monocytes` becomes `cd14_monocytes`); either spelling is accepted in
`cell_subset`. Its Ensembl IDs are remapped from `var['gene_symbols']` before
QC, retaining original IDs in `var['gene_id']` and dropping missing or duplicate
symbols. Its restricted gene panel has no `MT-` genes, so mitochondrial
filtering does not remove cells on that basis.

Use `dataset_configs.cross_tissue_immune_cell_atlas.donor_subset` in `configs/data_prep.yaml` for donor subsets, and `cell_subset` for atlas cell types from `obs['Manually_curated_celltype']`. For `pbmc`, `cell_subset` selects cell types from `obs['str_labels']`, while `dataset_subset` selects the source PBMC dataset batch (`0` for PBMC 8K, `1` for PBMC 4K). `cell_subset_n` samples a seeded number of cells per selected cell type for these datasets. `dominguez_conde` is not a separate supported dataset key because it is a subset of `cross_tissue_immune_cell_atlas`.

## Preprocessing Pipeline

Configuration: `configs/data_prep.yaml`

Run the default dataset:

```bash
python -m src.data.data_prep
```

Run a specific supported dataset:

```bash
python -m src.data.data_prep --dataset purified_pbmc
python -m src.data.data_prep --dataset pbmc
python -m src.data.data_prep --dataset cross_tissue_immune_cell_atlas
```

Steps:

1. Resolve the dataset-specific config and loader.
2. Load raw counts from disk, or download scVI PBMC datasets if missing.
3. Apply configured donor/batch/source-dataset/cell-type subset and optional per-group subsampling.
4. Remap gene symbols when configured, add canonical metadata columns, and filter cells and genes with scarcity filters.
5. Compute QC metrics and filter high mitochondrial-content cells.
6. Save the filtered QC violin plot when `processing.generate_preview=true`.
7. Normalize total counts and apply `log1p(base=10)`.
8. Defer optional PCA/neighbors/UMAP previews until the final splits are available.
9. Save the preprocessed AnnData to `datasets/<save_name>_preprocessed.h5ad`.
10. Uppercase gene names and drop duplicates after uppercasing.
11. Convert AnnData to a Cell2Sentence Arrow dataset with canonical `cell_type` labels.
12. Validate the in-memory Arrow dataset has non-empty `cell_sentence`.
13. Split the Arrow dataset with `split.val_fraction`, `split.test_fraction`, and the shared top-level `seed`.
14. Build the gene vocabulary from `data_train` only.
15. Remove any `data_val` or `data_test` cell containing a gene outside the train vocabulary.
16. Validate all splits only contain genes in the train vocabulary.
17. Save extraction-ready splits under `<save_dir>/<save_name>/data_train`, `<save_dir>/<save_name>/data_val`, and `<save_dir>/<save_name>/data_test`.
18. Reload each saved split and validate `cell_sentence` again.
19. Save `<save_dir>/<save_name>/vocab.json` from the train vocabulary.
20. Write split-aware distribution plots, `preview/num_genes_histogram.png`, and `preview/dataset_summary.txt` when `processing.generate_preview=true`; compute the UMAP preview here if `processing.run_umap=true`.
21. Save `<save_dir>/<save_name>/metadata.json` with the selected config, split fractions, filtering thresholds, final split sizes, and artifact paths.

When `processing.generate_preview=false`, `data_prep.py` skips the `preview/` folder entirely. The train/val/test Arrow datasets, preprocessed AnnData, `vocab.json`, and `metadata.json` are still written. When preview generation is enabled, stale preview files from older versions are removed before writing the current preview set.

The extraction-ready split paths are always:

```text
<save_dir>/<save_name>/data_train
<save_dir>/<save_name>/data_val
<save_dir>/<save_name>/data_test
<save_dir>/<save_name>/vocab.json
<save_dir>/<save_name>/metadata.json
```

`vocab.json` is intentionally train-only. Validation and test filtering happen during data prep so extraction never has to decide which held-out cells to keep.

## Extraction Pipeline

Configuration: `configs/extraction.yaml`

Run both activation and vocab extraction for configured datasets:

```bash
python -m src.data.extraction --model gemma
```

Run only one mode:

```bash
python -m src.data.extraction --model gemma --mode activations
python -m src.data.extraction --model gemma --mode vocab
```

Override the dataset without editing YAML:

```bash
python -m src.data.extraction --model gemma --dataset datasets/c2s_datasets/purified-pbmc-subset-full
python -m src.data.extraction --model pythia --dataset datasets/c2s_datasets/pbmc-subset-full
```

Dataset entries in `configs/extraction.yaml` are prepared dataset folder paths:

```yaml
datasets:
  - path: "datasets/c2s_datasets/purified-pbmc-subset-full"
```

Each folder must contain `data_train/`, `data_val/`, `data_test/`, and `vocab.json`. Before model loading, extraction verifies all splits only contain genes listed in `vocab.json`. Vocab embeddings are extracted from `vocab.json`; activations are extracted for all splits.

The two mode settings control different things: top-level `mode` / `--mode`
selects the task (`activations`, `vocab`, or both when unset), while
`extraction.mode` selects the activation representation (`gene` or `cell`).
`--model gemma` selects `models.gemma` in this config, currently the 27B model
at layer 18. Set `model_id` and `layer_idx` there for the model/layer being studied.

For a short activation run, use `--n_cells 50` (per split).
`--skip_validation` skips token-mapping validation but not train-vocabulary
checks. Both flags use underscores.

Outputs:

```text
datasets/activations/<model_name>/layer{L}/<dataset_name>/gene_activations_{pool}_{prefix}/data_train/
datasets/activations/<model_name>/layer{L}/<dataset_name>/gene_activations_{pool}_{prefix}/data_val/
datasets/activations/<model_name>/layer{L}/<dataset_name>/gene_activations_{pool}_{prefix}/data_test/
datasets/cells/<model_name>/layer{L}/<dataset_name>/cell_activations_last_{prefix}/data_train/
datasets/cells/<model_name>/layer{L}/<dataset_name>/cell_activations_last_{prefix}/data_val/
datasets/cells/<model_name>/layer{L}/<dataset_name>/cell_activations_last_{prefix}/data_test/
datasets/vocabs/<model_name>/layer{L}/<dataset_name>/
```

Set `extraction.mode` to `gene` for gene-token representations or `cell` for
one final-state representation per cell sentence. In gene mode,
`gene_pooling: last` writes one row per gene (its final subtoken), while `all`
writes every gene subtoken as a separate row; it does not average subtokens. Cell-mode safetensors store
row-aligned `activations` and `cell_type_ids` tensors. Each split directory also
contains `cell_types.json`; index this list with `cell_type_ids` to recover the
cell type for every activation row.

Vocabulary outputs are `vocab_last.safetensors` (tensor key `embeddings`) and
`genes.json`, with one aligned row per gene. Each gene is encoded separately
using its last token; failed genes receive zero rows. Existing vocabulary
outputs are reused when both files exist; a vocab-only run with complete
outputs skips model loading. Activation extraction instead removes existing
`shard_*.safetensors` in each selected output split before writing new shards.
Vocabulary output paths do not encode prompt-prefix settings, so use a distinct
`output.vocabs_dir` when comparing extraction setups.

## Run Prep And Extraction

Run the Python entry points in sequence, passing the same prepared dataset to
extraction explicitly:

```bash
python -m src.data.data_prep --dataset pbmc
python -m src.data.extraction --model gemma --dataset datasets/c2s_datasets/pbmc-subset-full
```

To change the preparation output name and run a short activation extraction:

```bash
python -m src.data.data_prep --dataset pbmc --save-name pbmc-example --no-umap
python -m src.data.extraction --model gemma --mode activations --dataset datasets/c2s_datasets/pbmc-example --n_cells 50
```

The entry points use `configs/data_prep.yaml` and `configs/extraction.yaml`
respectively. Their checked-in dataset selections differ: extraction currently
points to `purified-pbmc-subset-2`. It does not automatically discover the last
data-prep output.

## Adding A Dataset

1. Add a dataset module in `src/data/` with `DATASET_NAMES`, `make_dataset_config`, and `prepare_raw_adata`.
2. Register it in `data_prep.py`.
3. Add its `dataset_configs` entry to `configs/data_prep.yaml`.
4. Run `python -m src.data.data_prep --dataset <dataset_key>`.
5. Run `python -m src.data.extraction --model gemma --dataset <save_dir>/<save_name>`, or add that prepared folder to `configs/extraction.yaml`.
