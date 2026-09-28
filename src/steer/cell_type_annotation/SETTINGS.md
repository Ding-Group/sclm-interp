# SAE-Steered Cell-Type Annotation

This task tests whether SAE steering changes the cell type the model *writes*
for a cell. It uses the same generation and scoring path as the base cell-type
annotation downstream task: by default, the model receives the instruction template
and its generated text is matched against the dataset's labels through
`cell_type_acceptable_answers.json`. Run commands from the repository root
using the project Python environment. The task entry point delegates to
`src/steer/sae_steer_inference.py`; both accept the same arguments and use
`configs/steer.yaml` by default.

What the run is asking is which cells the intervention moved and where they
moved to, so the answer is `changed_predictions.jsonl` -- one row per cell whose
steered generation differs from its unsteered one -- rather than a single score.

## Evaluation

For every cell, back to back in a single pass, the evaluator:

1. Generates a baseline answer (skip with `baseline_generation: false` or
   `--no-baseline`).
2. Generates a second answer with the interventions active.
3. Scores both texts for the gold dataset label, and assigns each to the most
   specific dataset class it names (`none` when no class matches).

The two generations for a cell use the same model weights and prompt, with
hooks registered for both calls. The checked-in temperature is `0.0` (greedy
decoding). This evaluator does not reset a per-cell random seed between the two
calls, so stochastic decoding can also change answers independently of steering.

Which model the baseline answer comes from is `evaluation.baseline`, and its
`auto` default matches it to the steering base (see below), so the measured
difference is the steering itself under either steering model:

- `base_model`: the hooks stay inert — registered but never armed for the
  prompt. The evaluator checks their call counters around the generation and
  aborts if they fire, so the baseline cannot be silently contaminated.
- `reconstruction`: the hooks are armed but write the plain SAE reconstruction
  with no feature rescaled, i.e. this run's `alpha = 0`. The steered counters
  and delivery stats are held back under the same check, and the baseline
  passes are counted separately as `baseline_hook_calls` and
  `baseline_hook_tokens`. Requires an intervention with `base: reconstruction`.

With prediction saving enabled, a cell whose two generation strings differ is
written to `changed_predictions.jsonl` with both texts and both predicted
classes. `num_generation_changed` counts exact text changes, which may leave
the predicted class unchanged. `class_transitions` aggregates all baseline /
steered class pairs, including unchanged classes. These are label-presence
matches, not a semantic assessment of the answer.

## Feature groups and directions

Features are steered in **groups**. Each group has its own features, its own
strength, and its own `direction`, and all of an intervention's groups are
applied in the same forward pass, so one set can be pushed up while another is
pushed down. For normalized fraction-scaled groups, the signed injection is:

```python
injected = sum(
    (activations[:, list(group.feature_indices)] * activations.new_tensor(group.coefficients))
    @ sae.w_dec[list(group.feature_indices)]
    for group in groups
)
```

`direction` supplies the sign:

- `up`: add the group's push.
- `down`: subtract it.

`toward`/`away` and `+`/`-` are accepted as direction aliases. No feature may
appear in two groups — overlapping groups would collapse to one net
coefficient, so the config is rejected instead.

Three config forms, in precedence order:

```yaml
# 1. Any number of groups, each naming its own direction.
groups:
  - name: "cd14_markers"
    direction: up
    features: [1146, 2397, 5465, 17613, 15694]
    alpha: 1.0
  - name: "b_cell_markers"
    direction: down
    features: [9419, 16995, 18331, 16617, 5600]
    alpha: 0.99

# 2. The same thing for the common two-set case. Each takes a bare feature
#    list or a mapping with its own alpha and name.
up:
  features: [1146, 2397, 5465, 17613, 15694]
  alpha: 1.0
down: [9419, 16995, 18331, 16617, 5600]

# 3. One set, with its own direction.
direction: down
features: [9419, 16995, 18331, 16617, 5600]
alpha: 0.99
```

A group without its own strength inherits the intervention's (or the module's);
one of the three has to set it, since a group that steers nothing must say
`alpha: 0` outright rather than reach a default. Direction is never inherited:
every group names its own, either through the `up:`/`down:` key it is written
under or with its own `direction:`, and a group that names none is rejected at
config load.

## How hard a group pushes: `alpha` vs `magnitude`

A group says how hard it pushes in one of two ways, in either direction and
under either steering base. `scaling` names which one is in force; writing
`magnitude:` instead of `alpha:` selects `additive` on its own, so a fixed
magnitude can never be read as a fraction by accident.

### `scaling: fraction` (default) — `alpha`, a share of the activation

```yaml
up:
  features: [1146, 2397, 5465]
  alpha: 1.0            # one fraction for every feature in the group
```

`alpha` is a non-negative fraction of each feature's own activation at each
steered token, so relative feature strengths come from the cell itself and no
per-feature magnitude has to be supplied. With a shared feature/reconstruction
SAE under `base: reconstruction`, `up` with `1.0` doubles the selected decoder
contribution and `down` with `1.0` removes it exactly. Under `base: activations`,
the same vector is added/subtracted while the SAE reconstruction residual stays
in the hidden state; this is not guaranteed to remove all information about
the feature. A feature that does not fire at a token gets no push there, which is why a feature that is off in the steered
cells cannot be steered this way at all — check the delivery diagnostics'
`firing_rate` before reading anything into a null result.

### `scaling: additive` — `magnitude`, a fixed amount per token

```yaml
up:
  features: [1146, 2397, 5465]
  magnitude: 30         # one magnitude for every feature in the group
down:
  features: [9419, 16995, 18331]
  magnitude: [40, 25, 10]   # or one per feature, in the order listed above
```

`magnitude` is added (`up`) or subtracted (`down`) at every steered token
whether or not the feature fires, in the same units as the activations
themselves — comparable to `mean_activation_when_active` in the delivery
diagnostics, which is the number to size it against. This is the knob for a
feature the fractional form cannot move, and the one place where relative
strengths have to be chosen rather than inherited from the cell.

Either form takes a single number for the whole group or a list with one number
per feature, matched to `features` in the order it is written, so
`features: [a, b, c]` with `magnitude: [40, 25, 10]` pushes `a` by 40, `b` by 25
and `c` by 10. The sign always comes from `direction`, never from the numbers.
An explicit `scaling:` overrides the key-based inference, so `alpha: 20` plus
`scaling: additive` means the same as `magnitude: 20`.

Both keys inherit the same way as everything else — group, then intervention,
then `model.steer` — and mixing scalings across groups in one intervention is
allowed: a fraction-scaled group and an additive group are applied together in
the same pass.

## What the run reports

There is no run-level steering mode and no target class: direction lives on the
group, and nothing is scored for or against a configured class. What the
intervention did is read off the changed generations —
`changed_predictions.jsonl`, `num_generation_changed`, and `class_transitions`
(`unsteered -> steered`, with the largest transitions also printed at the end of
the run) — alongside the gold-label accuracy before and after.

Configure a run under `model.steer` in `configs/steer.yaml` (`base` at the
`model.steer` level, the rest under `model.steer.evaluation`; the legacy key
`model.steer.classification` is still read in place of `evaluation`):

```yaml
base: activations       # or: reconstruction
evaluation:
  baseline_generation: true
  baseline: auto        # or: base_model, reconstruction
interventions:
  - layer: 20
    intervene_checkpoint: {path: "checkpoints/example/features.ckpt", sae_type: "topk"}
    reconstruction_checkpoint: null  # reuse the feature SAE
    up: [1146, 2397]    # groups; each names its own direction
    down: [9419, 16995]
    alpha: 0.99         # default strength for groups that set none
    # magnitude: 30     # ... or steer additively instead (scaling: additive)
```

## Feature and reconstruction checkpoints

Each intervention now has two checkpoint roles:

- `intervene_checkpoint`: the SAE that supplies feature activations, feature
  indices, and decoder directions. The older `checkpoint` key remains accepted.
- `reconstruction_checkpoint`: the SAE used to rebuild the hidden state when
  `base: reconstruction`. Omit it or set it to `null` to reuse the feature SAE.

Both mappings accept `path` and `sae_type`. `layer` specifies where the hook
writes, independently of which layer trained the feature SAE. This permits a
cross-layer feature-direction experiment while keeping reconstruction fixed at
the intervention layer. Both SAE decoder widths must match the hidden state;
layer and prompt-prefix metadata mismatches are reported as warnings. Feature
indices always belong to `intervene_checkpoint`.

Use `--intervene-checkpoint` (alias `--checkpoint`) and
`--reconstruction-checkpoint` to change the roles independently. An explicitly
configured reconstruction checkpoint stays fixed when only the feature
checkpoint is overridden. Under `base: activations`, the reconstruction
checkpoint is not loaded or applied.

## Steering models: what the steered vector lands on

`base` picks which activations the steered vector is written on top of. Set it
under `model.steer` for every intervention, per intervention, or on the command
line with `--steer-base`. The two values are two different models.

### `base: activations` — additive steering on the base activations

The base-model hidden state is preserved and only the selected features' own
contribution to it is rescaled:

```python
activations = sae.encode(hidden_state)
replacement = hidden_state + effective_alpha * (
    activations[:, feature_indices] @ sae.w_dec[feature_indices]
)
```

No SAE reconstruction replaces the hidden state, so `alpha == 0` is an exact
no-op and the SAE's reconstruction error never reaches the residual stream. The
matched baseline is the untouched base model.

### `base: reconstruction` — steering with reconstructed activations

The hidden state is replaced by the SAE's reconstruction of it, carrying the
same rescaling:

```python
activations = feature_sae.encode(hidden_state)
reconstructed = reconstruction_sae.decode(reconstruction_sae.encode(hidden_state))
replacement = reconstructed + effective_alpha * (
    activations[:, feature_indices] @ feature_sae.w_dec[feature_indices]
)
```

When the two SAEs are the same and scaling is fractional, this is equivalent
to scaling the selected latents by `1 + effective_alpha` before decoding.
With separate SAEs, the feature SAE's vector is added to the reconstruction
SAE's output; it is not an exact ablation of a reconstruction latent.
`alpha == 0` gives the plain reconstruction, and the matched baseline uses
that same reconstruction checkpoint.

The price is the SAE's reconstruction error. Under `baseline: auto` it sits on
both sides of the comparison, and it is reported per run as
`mean_reconstruction_to_residual_ratio` in `summary.json` and in the
`[Intervention delivery]` block. Automatic naming adds `_recon` when any
intervention reconstructs, unless the configured name already contains `recon`.
Explicit names containing `recon` or a shared `--output-dir` can therefore
reuse a location; use distinct names/tags for comparisons.

### What both models share

Under `scaling: fraction` the coefficient is read from the token's own
activation, so the steered vector differs per token and a token where a feature
does not fire gets no push from that feature — with a TopK SAE that is every
token where the feature falls outside the top `k`. A fraction-scaled `up` group
therefore cannot turn on a feature that is silent on a cell; a `down` group is
unaffected, since it only removes what is already there. Under
`scaling: additive` the same magnitude is written at every steered position
regardless, which is how an `up` group turns on a feature that never fires.

Steered positions are:

- only at the configured intervention layer,
- only at gene-token positions of the prompt, selected by the checkpoint's
  `pooling_method` (`last` steers each gene's final subtoken; `all` steers every
  gene subtoken). Token selection uses the feature checkpoint for
  `base: activations`, and the reconstruction checkpoint for
  `base: reconstruction`; missing pooling metadata selects all gene subtokens,
- only during prompt prefill. Instruction-template tokens and every generated
  token are left untouched, so later tokens are affected only through attention
  over the modified gene positions.

`summary.json` records `hook_calls` and `hook_tokens`
(total steered positions) per intervention, which is the fastest way to confirm
the intervention actually ran.

## Commands

Intervention CLI overrides (including feature/strength, checkpoint, layer,
device, and `--steer-base`) rebuild a single intervention from the first YAML
entry. Use YAML edits to preserve multiple interventions. `--features` replaces
all groups with one group using the first configured group's direction and
settings; use `--up-features` / `--down-features` for explicitly directed sets.

Run the configured experiment:

```bash
python src/steer/cell_type_annotation/cell_type_annotation.py \
  --config configs/steer.yaml
```

Override features and strength. `--features` creates one group with the new
features and the first configured group's direction; use `--up-features`/`--down-features`
to set the direction from the command line:

```bash
python src/steer/cell_type_annotation/cell_type_annotation.py \
  --features 12 34 56 \
  --alpha 1.0
```

Steer two sets in opposite directions in one pass. `--alpha` sets every group's
strength; `--up-alpha`/`--down-alpha` override their own direction:

```bash
python src/steer/cell_type_annotation/cell_type_annotation.py \
  --up-features 1146 2397 5465 \
  --down-features 9419 16995 18331 \
  --up-alpha 1.0 --down-alpha 0.99
```

Steer additively instead, with one magnitude for every feature or one per
feature. `--magnitude` implies `--scaling additive`, and the
`--up-`/`--down-` forms override their own direction:

```bash
# one fixed magnitude across all features, in both directions
python src/steer/cell_type_annotation/cell_type_annotation.py \
  --up-features 1146 2397 5465 --down-features 9419 16995 \
  --magnitude 30

# a different magnitude per feature, listed in feature order
python src/steer/cell_type_annotation/cell_type_annotation.py \
  --up-features 1146 2397 5465 --down-features 9419 16995 \
  --up-magnitude 40 25 10 --down-magnitude 12 6
```

Run the same features as either steered model, into separate output
directories, without editing the config:

```bash
python src/steer/cell_type_annotation/cell_type_annotation.py --steer-base activations
python src/steer/cell_type_annotation/cell_type_annotation.py --steer-base reconstruction
```

Results are written under:

```text
results/steer/cell_type_annotation/<steer-name>/<dataset>/<model>/data_<split>/
```

The output includes:

- `predictions.jsonl`: baseline and steered generations, matched answers, and
  predicted classes, for every cell.
- `changed_predictions.jsonl`: the subset of rows whose steered generation
  differs from its unsteered one -- the cells the intervention actually moved.
  Written only when baseline generation and prediction saving are both on.
- `comparison.tsv`: one row per cell with the unsteered and steered text side
  by side, whether they differ, and both predicted classes. Written only when
  baseline generation and prediction saving are both on.
- `summary.json`: how many generations changed and the class transitions
  behind them, gold-label accuracy before and after, and per-source-class
  results. `accuracy` is the steered run's gold-label accuracy; the steering
  effect is `num_generation_changed`, not that number.
- `model_metadata.yaml`: model, SAE, and evaluation configuration.

Group metadata records `direction`, `feature_indices`, `scaling`, per-feature
`magnitudes`, and signed `effective_coefficients`. Uniform groups additionally
record `alpha_fraction` / `effective_alpha_fraction` or `magnitude` /
`effective_magnitude`. Mixed-direction runs are flagged as `bidirectional`.
Reconstruction metadata identifies the separate reconstruction checkpoint.
Per-feature delivery diagnostics include group/direction, firing rates, and
activation magnitudes; per-intervention diagnostics include injection and
reconstruction norm ratios.

`--output-tag` appends a directory below `data_<split>`; `--output-dir` overrides
the task output root. Reusing the same resolved location overwrites run files.
`--no-save-predictions` disables all three per-cell files while preserving the
summary, metadata, and accuracy figure. `--no-baseline` disables before/after
comparisons. The task also writes `figures/per_cell_type_accuracy.png`.

## Related analysis entry points

- `src/steer/select_features.py` reads `all_cell_type_features.csv` from
  `src/evaluate/cell_feature_analysis.py` and selects marker, housekeeping,
  dead, and stratified sets. Feature IDs are checkpoint-specific; a feature
  that is dead in a saved cell table can still fire under a different prompt
  or at gene-token positions. Additive scaling can move even a silent feature.
- `src/steer/intervention_delivery.py` uses the steering config for forward-only
  delivery diagnostics, including `--prompt-prefix both` to compare prompt
  distributions. It does not generate or score answers.
