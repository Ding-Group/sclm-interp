# Natural Language Interpretation Settings

This downstream task asks a C2S model to generate a short biological abstract
for one or more cells from their ordered gene-expression sentences. It is a
qualitative generation task: the evaluator reports output lengths and metadata,
not biological correctness or annotation accuracy. Run commands from the
repository root using the project Python environment.

## Prompt

The prompt template is:

```text
src/prompts/natural_language_interpretation.txt
```

It receives:

- `num_genes`: number of top ordered genes included per cell.
- `num_cells`: number of cells in the prompt group.
- `organism`: organism name, usually `Homo sapiens`.
- `cell_noun`: `cell` or `cells`.
- `cell_reference`: `this cell` or `these cells`.
- `cells`: formatted cell blocks with truncated gene sentences.

`cell_word` and `cell` are also accepted aliases for `cell_noun` and `cells`.

The evaluator extracts the first `num_genes` genes from each full
`cell_sentence` before formatting the prompt.

## Config

The main settings live under `natural_language_interpretation` in:

```text
configs/eval_tasks.yaml
```

Shared `model` settings select `mode`, `base_model`, device/dtype, attention
implementation, and the output root. The checked-in config selects
`mode: reconstruct` with `base_model: gemma-27b`; use `--mode base` explicitly
for a base run. Task sections are `natural_language_interpretation.data`,
`.inference`, and `.output`. Configured paths resolve against the repository root.

The checked-in task uses the test split of `purified-pbmc-subset-2`, 50 prompt
groups, two cells per group, and 200 genes per cell.

Important data settings:

- `c2s_dataset_dir`: prepared C2S dataset directory.
- `split`: dataset split, such as `train`, `val`, or `test`.
- `max_examples`: number of prompt groups to generate; `null` uses all groups.
- `cells_per_prompt`: number of cells included in each prompt group.
- `drop_last`: whether to omit a final incomplete group; default `false`.
- `shuffle_cells`: permute the whole split before grouping; default `true`.
- `num_genes`: number of top ordered genes extracted per cell.
- `cell_sentence_column`: source column for full ordered gene sentences.
- `cell_name_column`, `cell_type_column`, `organism_column`: metadata columns
  recorded in predictions when present.
- `default_organism`: fallback organism when no organism column is configured.

Important inference settings:

- `max_new_tokens`: generation length budget.
- `temperature`: sampling temperature.
- `do_sample`: whether to sample instead of greedy decoding.
- `top_k`, `top_p`: sampling filters.
- `stop_at_control_token`: trims generated text before C2S control tokens such as
  `<ctrl100>`.
- `prompt_prefix`: when `true`, use the prompt template; when `false`, send only
  the formatted cell blocks.

The checked-in decoding settings are:

```yaml
max_new_tokens: 1024
temperature: 0.7
do_sample: true
top_k: 30
top_p: 0.9
stop_at_control_token: true
seed: 42
```

`stop_at_control_token` trims decoded text at the first `<ctrl` occurrence;
it does not itself terminate token generation early.

## Cell selection and reproducibility

`shuffle_cells: true` uses a dedicated RNG seeded from `inference.seed` to
permute the entire split before applying the group count. Changing decoding
settings does not change cell selection. With the same seed, a shorter run
uses a prefix of the cells selected by a longer run. Set `shuffle_cells: false`
to group cells in dataset order.

Generation is seeded separately with `seed + group_index`. Use the same seed,
data/grouping, prompt, and decoding settings for base/reconstruction comparisons.
This controls the random seeds, but does not guarantee identical generated
sequences after the intervention changes token probabilities. Set YAML
`inference.seed: null` or pass `--seed -1` to disable seeding. With shuffling
still enabled, cell selection is then non-reproducible too.

## SAE reconstruction

`--mode reconstruct` reads `model.reconstruct.name` and its `interventions`
list, shared with the cell-type task. Each intervention specifies a layer and
checkpoint, with optional SAE type/device. `sae_device: model` uses the language
model's device. `--checkpoint`, `--layer`, `--sae-type`, and `--sae-device`
rebuild one intervention from the first configured entry; use YAML edits to
preserve multiple interventions.

Hooks replace gene hidden states with SAE reconstructions only during prompt
prefill, across every cell block in the group. Checkpoint pooling `last` selects
each gene's final subtoken; `all` (or absent pooling metadata) selects all gene
subtokens. Instruction text and generated tokens are not directly rewritten.
Keep the checkpoint's `model_metadata.yaml` beside it to recover the training
settings. The run records intervention metadata and hook counts.

## Outputs

Results are written to:

```text
results/natural_language_interpretation/<model-output-name>/<dataset>/<model>/
├── runs.json
├── 1/
│   ├── predictions.jsonl
│   ├── summary.json
│   └── model_metadata.yaml
└── 2/
    └── ...
```

`<model-output-name>` comes from `model.base_output_name` (default `base`) or
`model.reconstruct.name`; `<model>` is the model ID's final path component.
`--output-dir` changes the task output root. `--output-tag` is recorded as run
metadata and does not replace the numbered run directory.

Every invocation atomically reserves the next numbered run directory. This
mirrors the `results/enrich` hierarchy and prevents a repeated experiment from
overwriting an earlier one. The evaluated split is recorded in each run's
summary and in `runs.json`.

The main files are:

- `predictions.jsonl`: one row per prompt group, including cell metadata,
  generated text, and optionally the prompt.
- `summary.json`: run metadata, prompt/data settings, decoding settings, and
  aggregate generation lengths.
- `model_metadata.yaml`: model and prompt metadata for the run.
- `runs.json`: compact index of all completed numbered runs for the same
  experiment, dataset, and model.

`output.save_predictions: false` or `--no-save-predictions` disables the JSONL
file; metadata, summary, and the run index are still written. The checked-in
`output.include_prompt: true` includes the rendered prompt in prediction rows.

Each prediction row records `num_genes_available` and `num_genes_in_prompt` per
cell, so it is easy to verify whether a cell had fewer genes than requested.

## Commands

Base model:

```bash
python src/evaluate/downstream_tasks/natural_language_interpretation/natural_language_interpretation.py --mode base --max-examples 20
```

Override the number of cells and genes:

```bash
python src/evaluate/downstream_tasks/natural_language_interpretation/natural_language_interpretation.py --mode base --max-examples 20 --cells-per-prompt 2 --num-genes 300
```

SAE reconstruction mode:

```bash
python src/evaluate/downstream_tasks/natural_language_interpretation/natural_language_interpretation.py --mode reconstruct --max-examples 20
```

Compare the same cell groups and random seeds in the two modes:

```bash
python src/evaluate/downstream_tasks/natural_language_interpretation/natural_language_interpretation.py --mode base --max-examples 20 --seed 42
python src/evaluate/downstream_tasks/natural_language_interpretation/natural_language_interpretation.py --mode reconstruct --max-examples 20 --seed 42
```
