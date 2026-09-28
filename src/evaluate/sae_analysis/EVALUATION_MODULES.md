# SAE Analysis Modules

This package (`src/evaluate/sae_analysis/`) holds the SAE evaluation suite.
`eval_sae.py` runs it from `configs/eval_sae.yaml`. The unified config shares the
checkpoint, activation data, model settings, device, and seed across modules.
Individual evaluations live under `evaluations.<name>` and can be disabled with
`enabled: false`.

Run commands from the repository root with the project dependencies installed.
Inputs are extracted safetensor shards with an `activations` tensor and a trained
SAE checkpoint. Keep the run's `model_metadata.yaml` beside the checkpoint: it
supplies model/training metadata and can supply TopK `k` when the checkpoint
does not store `k_buffer`. These evaluations do not load the C2S language model.

Run all enabled evaluations (`--config` defaults to `configs/eval_sae.yaml`):

```bash
python src/evaluate/sae_analysis/eval_sae.py
python src/evaluate/sae_analysis/eval_sae.py --config configs/eval_sae.yaml
```

Run or skip a subset from the command line. The subset names are
`feature_analysis`, `reconstruction`, `pca`, and `pca_vs_sae`:

```bash
python src/evaluate/sae_analysis/eval_sae.py --only feature_analysis,pca
python src/evaluate/sae_analysis/eval_sae.py --skip reconstruction
```

`--only` filters enabled evaluations; it does not re-enable an evaluation with
`enabled: false`. The driver accepts only `--config`, `--only`, and `--skip`.

Each module can also be run standalone with its own CLI overrides:

```bash
# Quick feature run on a sampled subset, skipping decoder similarity.
python src/evaluate/sae_analysis/feature_analysis.py --max-samples 50000 --max-shards 2 --skip-feature-similarity

# PCA with a custom component count, skipping the activation-correlation pass.
python src/evaluate/sae_analysis/pca_analysis.py --n-components 25 --skip-activation-corr

# Reconstruction on two splits plus their pooled combination.
python src/evaluate/sae_analysis/reconstruction_eval.py --splits data_test,data_val --combined --max-samples 50000

# SAE-versus-PCA comparison at a custom set of direction budgets.
python src/evaluate/sae_analysis/pca_vs_sae.py --ranks 1,8,64,512
```

All modules accept sampling and output CLI overrides. `feature_analysis.py` and
`reconstruction_eval.py` also accept `--splits` / `--combined` / `--no-combined`
to override the configured splits.

## Configuration precedence

Each evaluation merges the shared `checkpoint`, `data`, `model`, `output`,
`infrastructure`, `eval`, and `pca` mappings with its own
`evaluations.<name>.<section>` mapping. Per-evaluation keys override shared keys
within each section; standalone CLI overrides are applied afterward. Paths in
these configs resolve against the repository root; absolute paths are accepted.

| Evaluation | Sample-cap key under `evaluations.<name>` | Splits |
| --- | --- | --- |
| `feature_analysis` | `eval.max_samples` | `data.splits`, optional `data.combined` |
| `reconstruction` | `data.max_samples`; `data.max_unseen_samples` for unseen corpora | `data.splits`, optional `data.combined` and `data.unseen_dirs` |
| `pca` | `data.max_samples` | One `data.split` |
| `pca_vs_sae` | `data.max_samples` | One `data.split` |

All use `data.max_shards` and `output.seed`; the checked-in seed is 42.
`train_config` / `use_train_config` references are rejected. Specify checkpoint
and activation paths directly instead of deriving them from a training config.

## Output layout

Results are written to
`results/<evaluation>/<experiment_name>/<checkpoint_name>/<dataset_name>/`:

- `experiment_name` is the checkpoint's containing directory, matching how
  training names runs.
- `checkpoint_name` is the individual `.ckpt` within that run, taken from
  `checkpoint.name` when set and otherwise from the checkpoint file's own stem
  (`last.ckpt` becomes `last/`). Several checkpoints of one run therefore keep
  their results apart instead of overwriting each other.
- `dataset_name` comes from the `data.dir` being evaluated, read off its
  `<base_model>/layer<N>/<dataset>/` layout (falling back to `dataset.name` in
  the checkpoint's `model_metadata.yaml`). Results therefore follow the data a
  run actually read, so evaluating one checkpoint on several datasets keeps the
  results separate; the training dataset stays recorded in the checkpoint
  metadata.

Setting `output.dir`, globally or per evaluation, overrides the whole layout.

- `results/feature_analysis/<experiment>/<checkpoint>/<dataset>/`
- `results/reconstruction/<experiment>/<checkpoint>/<dataset>/`
- `results/pca_analysis/<experiment>/<checkpoint>/<dataset>/`
- `results/pca_vs_sae/<experiment>/<checkpoint>/<dataset>/`

For example, a `checkpoint.path` of
`checkpoints/gemma-2b/layer10_topk_exp8_last_no_prefix_22/last.ckpt` evaluated
on `pbmc-subset-full` writes feature analysis to
`results/feature_analysis/layer10_topk_exp8_last_no_prefix_22/last/pbmc-subset-full/`.

`checkpoint.name` is also the file selector when `checkpoint.dir` is used in
place of `checkpoint.path`: `{dir: <run>, name: epoch009-step7490}` loads
`<run>/epoch009-step7490.ckpt`. A trailing `.ckpt` is accepted and stripped from
the directory name. Set it explicitly to keep long Lightning filenames out of
result paths.

## Package layout

| File | Role |
| --- | --- |
| `eval_sae.py` | Driver: selects and runs the evaluations named in the config. |
| `feature_analysis.py` | Per-feature firing rates and decoder redundancy. |
| `reconstruction_eval.py` | Reconstruction error and variance explained across splits x checkpoints. |
| `pca_analysis.py` | Activation-space PCA and SAE activation/PCA-score correlation. |
| `pca_vs_sae.py` | Matched-budget SAE-versus-PCA variance explained, plus decoder/PC overlap. |
| `activation_data.py` | Shared split resolution, shard listing, row sampling, batching. |
| `reconstruction.py` | Shared reconstruction sums, pooling, and metric derivation. |
| `reporting.py` | Shared fixed-width summary-table rendering. |

SAE checkpoint loading and config-path helpers live one level up in
`src/evaluate/model_loading.py`, which is shared with the steering, gene-feature,
and downstream-task code.

## Splits

`feature_analysis` and `reconstruction` evaluate a list of activation splits and
share the same layout. Each split named in `evaluations.<name>.data.splits` is
resolved as a subdirectory of the shared `data.dir` and written to
`<output_dir>/<split>/`, with the `data_` prefix stripped (`data_test` becomes
`test/`). With `data.combined: true`, the splits are also pooled into
`<output_dir>/combined/`.

Pooling is exact, not an average of per-split numbers: `reconstruction`
accumulates plain sums (`sum_x`, `sum_xx`, `sum_r`, `sum_rr`, row counts) per
split and adds them, so the combined metrics are identical to what a single pass
over the concatenated corpus would produce. Combined variance therefore includes
any between-split mean shift. Combining requires at least two splits; otherwise
it is skipped with a message.

`reconstruction` additionally writes a top-level `<output_dir>/summary_table.txt`
and JSON comparing every split and checkpoint in one table.

Row sampling (`max_samples`, in the section listed above) is seeded per split,
so a split's sampled rows do not change based on which other splits are evaluated
alongside it. In a
checkpoint sweep, all checkpoints see exactly the same sampled rows.

`pca` and `pca_vs_sae` are the exceptions: each analyzes a single split, because
the covariance and its eigenbasis describe one activation distribution.
Configure them with `evaluations.<name>.data.split`; listing more than one split
raises an error rather than silently analyzing the first.

## Feature Activation

Module: `feature_analysis.py`

This module streams activation shards through the SAE encoder and summarizes
how often each latent feature fires.

Primary outputs:

- `stats.json`: scalar feature-activation summary and top features.
- `activation_rates.npz`: per-feature activation-rate arrays.
- `summary_table.txt`: compact text summary.
- `figures/`: activation-rate histograms, CDFs, and rate/value plots.
- When decoder similarity is enabled:
  - `decoder_neighbors.npz`
  - `decoder_neighbor_pairs.json`
  - decoder-neighbor diagnostic figures

Metrics:

- `activation_rate[i]`: fraction of evaluated samples where feature `i` has
  activation value greater than zero.
- `mean_value_when_active[i]`: average value of feature `i`, conditioned on
  feature `i` being active.
- `mean_l0`: expected number of active features per sample. This is computed as
  `sum_i P(feature_i fires)`.
- `n_dead` / `frac_dead`: number and fraction of features with activation rate
  below `dead_threshold`.
- `n_common` / `frac_common`: number and fraction of features with activation
  rate at or above `common_threshold`.
- `n_universal` / `frac_universal`: number and fraction of features with
  activation rate at or above `universal_threshold`.
- `mean_activation_rate`: mean of all per-feature activation rates.
- `gini`: Gini coefficient over activation rates. Values near 0 mean firing is
  distributed evenly across features; values near 1 mean firing is concentrated
  in a smaller set of features.
- `percentiles`: activation-rate percentiles at `p1`, `p5`, `p10`, `p25`,
  `p50`, `p75`, `p90`, `p95`, `p99`, and `p99.9`.
- `top_features`: highest-rate features, including activation rate and mean
  value while active.

Decoder similarity metrics, when `compute_feature_similarity: true`:

- `nearest_abs_cosine_mean` / `nearest_abs_cosine_median`: summary of each
  feature decoder's nearest neighbor by absolute cosine similarity.
- `nearest_abs_cosine_percentiles`: percentile summary of nearest-neighbor
  absolute cosine similarity.
- `n_features_with_similar_neighbor` / `frac_features_with_similar_neighbor`:
  count and fraction of features whose nearest decoder neighbor has absolute
  cosine similarity at or above `similar_threshold`.
- `n_features_with_duplicate_neighbor` / `frac_features_with_duplicate_neighbor`:
  count and fraction of features whose nearest decoder neighbor has absolute
  cosine similarity at or above `duplicate_threshold`.
- `n_unique_similar_pairs` / `n_unique_duplicate_pairs`: deduplicated pair counts
  after treating `(i, j)` and `(j, i)` as the same pair.
- `top_pairs`: strongest unique decoder-neighbor pairs, including absolute
  cosine, signed cosine, absolute cosine distance, and signed cosine distance.

`decoder_chunk_size` affects only memory/performance for decoder-neighbor
search. It changes how many decoder rows are compared at once and should not
change the metric values.

## Reconstruction

Module: `reconstruction_eval.py`

This module measures how well the SAE reconstructs activation vectors. It
evaluates each configured split (see [Splits](#splits)), their pooled
combination, and optional unseen activation directories, against every
configured checkpoint. Outputs are indexed by split and then by checkpoint.

It replaces the former `variance_explained.py` and `recon_error.py`, which
derived identical metrics from the same accumulator in `reconstruction.py` and
differed only in the axis they swept. For checkpoint sweeps, set
`checkpoint.paths` to a list of `.ckpt` files; checkpoints are labelled by file
stem, disambiguated by parent directory when stems collide. A sweep spanning
several checkpoint directories requires an explicit `output.dir`, since no
single checkpoint can name the result directory. A sweep over several
checkpoints of one run likewise needs `checkpoint.name` to label the shared
result bucket (or `output.dir`), since no single file stem names it.

`evaluations.reconstruction.data.unseen_dirs` takes a list of path strings or
`{name, dir}` mappings. Unseen directories are held-out corpora rather than
splits of the configured data, so they are reported separately and excluded from
the `combined` pooling. An unseen directory whose name would collide with a
split is written as `unseen__<name>`.

Primary outputs:

- `reconstruction.json`: scalar metrics per checkpoint in each split directory;
  a top-level overview combines all splits and checkpoints.
- `reconstruction_per_dim.npz`: per-dimension MSE, FVE, and variance arrays in
  each split directory, including `combined/` when enabled. No top-level NPZ
  is written.
- `summary_table.txt`: compact text summary, both per split and at the top level.

Metrics:

- `fve`: aggregate fraction of variance explained:

  ```text
  1 - sum_d Var(x_d - xhat_d) / sum_d Var(x_d)
  ```

  This is the main reconstruction-variance metric. It is aggregated over all
  activation dimensions.

- `mean_per_dim_fve`: computes FVE per activation dimension first, then averages
  uniformly across dimensions:

  ```text
  mean_d(1 - Var(x_d - xhat_d) / Var(x_d))
  ```

  This can differ from `fve` because low-variance and high-variance dimensions
  receive equal weight. Dimensions with no variance to explain (constant across
  the evaluated rows) would make the ratio numerical noise, so they are excluded
  from the average and counted in `n_degenerate_dims`; their `per_dim_fve`
  entries are `NaN`.

- `mse`: mean per-dimension reconstruction error `E[(x - xhat)^2]`. This is
  uncentered and includes reconstruction bias.
- `rmse`: square root of `mse`.
- `mean_squared_l2_error`: average per-sample squared L2 reconstruction error.
- `root_mean_squared_l2_error`: square root of `mean_squared_l2_error`.
- `nmse`: normalized reconstruction error:

  ```text
  E[||x - xhat||^2] / E[||x||^2]
  ```

  This is based on uncentered second moments, so it is not simply `1 - fve`
  when activations have non-zero mean.

- `relative_l2_error`: square root of `nmse`.
- `mean_l0`: average number of active SAE features per sample during
  reconstruction.
- `total_samples`: number of activation rows evaluated for the split.
- `num_shards`: number of safetensor activation shards used.
- `d_model`: activation dimensionality.
- `n_degenerate_dims`: dimensions excluded from `mean_per_dim_fve`.
- `per_dim_fve`: per-dimension FVE array.
- `mse_per_dim`: per-dimension uncentered mean squared reconstruction error.
- `var_x`: per-dimension input activation variance.
- `var_r`: per-dimension residual variance.

### Scale-free metrics

Raw-scale error and variance metrics can be dominated by high-norm rows:
`E[||x||^2]` and `sum_d Var(x_d)` can be concentrated in a minority of rows.
A high `fve` therefore does not establish that a *typical* row is reconstructed
well. These metrics weight each row
equally and should be reported alongside, not instead of, the raw-scale ones.

- `mean_cosine_similarity`: mean over rows of `cos(x, xhat)`. Ignores magnitude
  entirely — it asks only whether the reconstruction points the right way.
- `std_cosine_similarity`, `min_cosine_similarity`: spread and worst case of the
  same per-row cosine. The minimum is the useful adversarial number: a high mean
  with a very low minimum means some rows are reconstructed in the wrong
  direction outright.
- `fve_normalized`, `mean_per_dim_fve_normalized`, `per_dim_fve_normalized`,
  `var_x_normalized`, `var_r_normalized`, `n_degenerate_dims_normalized`: the
  identical FVE derivation after dividing both input and residual by the input
  norm: `x / ||x||` and `(x - xhat) / ||x||`. The reconstruction is not
  independently normalized, so magnitude error remains measurable without
  high-norm inputs dominating.

  **`fve` minus `fve_normalized` is the diagnostic.** A large positive gap means
  the fit is carried by a high-norm minority and typical rows are reconstructed
  worse than `fve` suggests.

- `nmse_normalized`: `E[||r||^2 / ||x||^2]`, the *unweighted* mean per-row
  relative error. Compare against `nmse`, which is
  `E[||r||^2] / E[||x||^2]` — the same quantity weighted by row norm.
- `relative_l2_error_normalized`: square root of `nmse_normalized`.
- `n_normalized_rows`, `n_cosine_rows`: rows contributing to each. A zero row
  has neither direction nor scale, so it is excluded from the normalized sums;
  a row the SAE maps exactly to the origin is additionally excluded from the
  cosine. These counts equal `total_samples` unless such rows are present.

All of these pool exactly across splits, like the raw sums: cosine accumulates
as a sum over rows, the normalized statistics are ordinary per-dimension sums
over rescaled rows, and the minimum pools by taking the minimum.

## PCA Alignment

Module: `pca_analysis.py`

This module computes PCA directions of the activation rows used for SAE
evaluation and correlates SAE feature activations with the PCA scores of those
rows.

Decoder-direction cosine against the PCA basis lives in `pca_vs_sae.py`, which
reports the same statistic against a random-direction null; without that null a
raw cosine is not interpretable.

Primary outputs:

- `pca_summary.json`: PCA variance summary and top features per component.
- `pca_alignment_arrays.npz`: PCA components, activation mean, and activation
  correlation matrices.
- `feature_pca_alignment.csv`: one row per SAE feature with its strongest PCA
  score correlation. Written only when activation correlations are enabled.
- `figures/`: explained variance and activation-correlation diagnostics.

PCA metrics:

- `total_variance`: total variance in the activation covariance matrix.
- `explained_variance[j]`: eigenvalue for PCA component `j`.
- `explained_variance_ratio[j]`: fraction of total activation variance captured
  by component `j`.
- `cumulative_explained_variance_ratio[j]`: cumulative sum of explained
  variance ratios through component `j`.
- `components`: PCA directions in activation space.
- `mean`: activation mean subtracted before computing PCA scores when
  `center: true`. With `center: false`, PCA is computed from the uncentered
  second-moment matrix and `mean` is an all-zero vector.

Activation-to-PCA alignment metrics, when
`compute_activation_correlations: true`:

- `activation_component_corr[i, j]`: Pearson correlation between SAE feature
  activation `i` and PCA score `j` across evaluated activation rows.
- `best_activation_component[i]`: PCA component with the largest absolute
  activation correlation for feature `i`.
- `best_activation_abs_corr[i]`: absolute correlation value for that best
  component.
- `best_activation_signed_corr[i]`: signed correlation value for that best
  component.
- `top_features_by_activation_correlation`: per-component feature rankings by
  absolute activation correlation, with signed values retained.

## SAE versus PCA

Module: `pca_vs_sae.py`

This module compares SAE and PCA variance explained at matched direction
budgets, reports the full SAE reconstruction score, and compares decoder/PC
alignment with a random-direction reference. Whether the SAE outperforms PCA
is an empirical result of the run; the plot does not assume that outcome or
establish that the representations are unrelated.

Both methods are scored with the same centered `fve` defined above, measured on
the same rows in the same pass. The x-axis is the number of directions each may
spend per token:

- **PCA rank-k**: one global k-dimensional subspace, shared by every row.
- **SAE top-j**: the j strongest active features of that row, chosen per row
  from a `d_hidden`-atom dictionary.

These are equal-budget but not equal-capacity. Per-row atom selection is
strictly more expressive than a fixed subspace, and the SAE carries far more
parameters, so an SAE advantage at matched k is expected rather than surprising.
PCA is also fit on the same split it is scored on, which favours PCA. Both
caveats are recorded in `summary.meta.comparison_note`.

A TopK SAE trained at `k` is off-distribution when truncated to top-j for
`j < k`, so points well below the operating point understate it; the meaningful
head-to-head is at the SAE's actual mean L0.

Primary outputs:

- `pca_vs_sae_summary.json`: curve, matched-budget row, overlap stats, metadata.
- `variance_explained_curve.csv`: one row per direction budget.
- `decoder_component_overlap.npz`: observed and null best-absolute cosines.
- `summary_table.txt`: compact text summary.
- `figures/sae_vs_pca_variance_explained.png`: the three-panel combined figure.

Metrics:

- `curve[k].pca_fve` / `curve[k].sae_fve`: FVE at direction budget `k`.
- `curve[k].sae_minus_pca`: the SAE advantage at that budget.
- `sae.full_fve` / `sae.mean_l0`: the SAE's untruncated operating point.
- `overlap.observed_median` / `overlap.null_median`: median best absolute cosine
  from decoder directions, and from random unit vectors, to any leading PC. Only
  the gap between the two is evidence.
- `overlap.fraction_above_null_p95`: fraction of decoder directions whose best
  absolute cosine exceeds the 95th percentile of the null.

## Choosing Modules

Use the modules together when comparing SAE quality from complementary angles:

- Feature activation shows sparsity, dead/common/universal features, and decoder
  redundancy.
- Reconstruction shows reconstruction quality in the original activation space,
  across splits, held-out corpora, and checkpoints.
- PCA alignment shows whether individual SAE features track dominant dense
  directions in the activation distribution.
- SAE versus PCA answers whether the SAE buys anything over a dense linear basis
  at a matched direction budget.

For quick checks, reduce `max_shards` or sample counts in `configs/eval_sae.yaml`.
