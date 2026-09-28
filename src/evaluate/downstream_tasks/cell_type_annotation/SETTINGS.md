# Cell Type Annotation Settings

This downstream task asks a C2S model to predict the cell type for each cell
from its ordered gene-expression sentence, then scores the generated answer
against the dataset label. Run the commands below from the repository root
using the project Python environment. Configured data, prompt, checkpoint, and
output paths resolve against that root.

## Prompt

The prompt template is:

```text
src/prompts/cell_type_annotation_template.txt
```

It receives:

- `organism`: organism name, usually `Homo sapiens`.
- `cell_sentence`: full gene sentence ordered by descending expression.

When `prompt_prefix` is `true`, the evaluator wraps each `cell_sentence` with
the template. When `prompt_prefix` is `false`, it sends only the bare
`cell_sentence`.

## Config

The main settings live under `cell_type_annotation` in:

```text
configs/eval_tasks.yaml
```

Shared model settings live under `model`: `mode` (`base` or `reconstruct`),
`base_model`, `device`, `dtype`, `attn_implementation`, and `output_dir`.
The checked-in config currently selects `reconstruct` and `gemma-27b`; pass
`--mode base` explicitly for a base-model comparison. Model aliases include
`gemma` / `gemma-2b`, `gemma-27b`, and `pythia` / `pythia-1b`.

The task-specific sections are `cell_type_annotation.data`, `.inference`,
`.scoring`, and `.output`. Current data settings select `pbmc-subset-full`,
`train`, and the full split. `max_examples` takes the first N rows in dataset
order, rather than drawing a random sample.

Important data settings:

- `c2s_dataset_dir`: prepared C2S dataset directory.
- `split`: dataset split, such as `train`, `val`, or `test`.
- `cell_type_column`: gold-label column used for scoring.
- `max_examples`: number of cells to evaluate; `null` uses the full split.
- `organism_column`: optional organism column for prompt formatting.
- `default_organism`: fallback organism when no organism column is configured.

Important inference settings:

- `max_new_tokens`: generation length budget. Cell-type annotation usually uses
  a small value, such as `32`.
- `temperature`: generation temperature. The default `0.0` uses greedy decoding.
- `prompt_prefix`: whether to use the prompt template.
- `prompt_template_path`: path to the cell-type prompt template.

Important scoring settings:

- `acceptable_answers_path`: JSON file mapping gold labels to acceptable aliases.
  Set this to `null` or pass `--acceptable-answers ""` to disable aliases while
  retaining the normalized gold-label matcher.

The matcher normalizes generated text and labels into lowercase alphanumeric
tokens. A prediction is correct when it contains the gold label or one of the
configured aliases as a contiguous token sequence, or contains every token of
that label anywhere in the answer, allowing simple singular/plural variants.
This is a label-presence check, not exact-string or semantic correctness; a
long answer naming multiple cell types can match the gold label.

## SAE reconstruction

`--mode reconstruct` delegates to `sae_reconstruction_inference.py` and reads
`model.reconstruct.name` plus `model.reconstruct.interventions`. Each
intervention sets `layer`, `checkpoint.path`, optionally `checkpoint.sae_type`,
and `sae_device` (`model` uses the language model's device). Multiple configured
layers can be reconstructed in the same forward pass.

At each layer the hook replaces selected prompt hidden states with
`sae.decode(sae.encode(hidden))`. It acts only during prompt prefill on gene
tokens: checkpoint pooling `last` selects each gene's last subtoken, while
`all` selects all gene subtokens. Instruction and generated-token positions are
not directly rewritten. Missing pooling metadata falls back to all gene
subtokens. Keep `model_metadata.yaml` next to each checkpoint so pooling and
training settings can be recovered. Layer/prompt-prefix mismatches produce
warnings; incompatible hidden widths raise errors.

`--checkpoint`, `--layer`, `--sae-type`, and `--sae-device` rebuild one
intervention from the first configured entry. Use YAML edits to preserve
multiple interventions.
The output records hook call/token counts and checkpoint metadata.

## Outputs

Results are written to:

```text
results/cell_type_annotation/<model-output-name>/<dataset>/<model>/data_<split>/
```

`<model-output-name>` is `model.base_output_name` (default `base`) or
`model.reconstruct.name`. `<model>` is the model ID's final path component.
Reconstruction `--output-tag` adds a directory below `data_<split>`;
`--output-dir` overrides the task output root. Reusing a resolved output
location overwrites its current run files; this task has no numbered runs.

The main files are:

- `predictions.jsonl`: one row per cell, including generated text, gold label,
  correctness, and matched answer.
- `incorrect_predictions.jsonl`: incorrect rows only, useful for inspection.
- `summary.json`: run metadata, accuracy, per-cell-type accuracy, and output
  paths.
- `model_metadata.yaml`: model and prompt metadata for the run.
- `figures/per_cell_type_accuracy.png`: accuracy by cell type when possible.

`output.save_predictions: false` or `--no-save-predictions` disables both
prediction JSONL files while retaining summary, metadata, and figures.
`output.include_prompt` controls whether prediction rows contain the prompt.

## Commands

Base model:

```bash
python src/evaluate/downstream_tasks/cell_type_annotation/cell_type_annotation.py --mode base --max-examples 50
```

Configured SAE reconstruction mode:

```bash
python src/evaluate/downstream_tasks/cell_type_annotation/cell_type_annotation.py --mode reconstruct --max-examples 50
```

Override dataset and split:

```bash
python src/evaluate/downstream_tasks/cell_type_annotation/cell_type_annotation.py --mode base --dataset datasets/c2s_datasets/pbmc-subset-full --split test --max-examples 50
```

Use a different model family:

```bash
python src/evaluate/downstream_tasks/cell_type_annotation/cell_type_annotation.py --mode base --model pythia --max-examples 50
```

Disable acceptable-answer aliases:

```bash
python src/evaluate/downstream_tasks/cell_type_annotation/cell_type_annotation.py --acceptable-answers "" --max-examples 50
```

## Related downstream evaluation

`src/evaluate/downstream_tasks/qa/generate.py` builds multiple-choice datasets
from prepared C2S splits using `configs/qa_generation.yaml`.
`src/evaluate/downstream_tasks/qa/cell_type_multiple_choice.py` consumes them
using the separate `cell_type_multiple_choice` section of `configs/eval_tasks.yaml`.
Its `scoring.method` selects `likelihood` or `generate`; the checked-in value is
`generate`. These settings do not change the free-generation annotation task
in this document.
