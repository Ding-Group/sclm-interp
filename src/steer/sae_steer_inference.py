#!/usr/bin/env python3
"""Generate cell-type annotations while steering selected SAE features.

An intervention steers any number of feature groups at one layer. Each group
has its own features, its own strength, and its own direction, so a single pass
can push one set of features up while pushing another down::

    activations = sae.encode(hidden_state)
    injected = sum(
        group.alpha * activations[:, group.features] @ sae.w_dec[group.features]
        for group in groups
    )

Two steering models are available, chosen per intervention with ``base``, and
they differ in what that vector is written on top of.

``base: activations`` -- additive steering on the base activations. The
base-model hidden state is kept and only the selected features' own
contribution to it is rescaled::

    replacement = hidden_state + injected

No SAE reconstruction is substituted, so ``alpha = 0`` is exactly the original
hidden state and the SAE's reconstruction error never enters the residual
stream.

``base: reconstruction`` -- steering on top of the SAE reconstruction. The
hidden state is replaced by its reconstruction plus the selected feature
directions::

    reconstructed = reconstruction_sae.decode(reconstruction_sae.encode(hidden_state))
    replacement = reconstructed + injected

``reconstruction_checkpoint`` selects that SAE independently of ``intervene_checkpoint``,
which supplies the feature activations and decoder directions. If omitted, both
use ``intervene_checkpoint``. Fractional steering always uses the feature SAE's encoding
of the original hidden state, even when reconstruction uses another SAE.

Here ``alpha = 0`` is the plain SAE reconstruction rather than the base model,
so the matched comparison generation is the unsteered reconstruction and not
the base model. ``evaluation.baseline`` selects that, and its ``auto`` default
already picks the reconstruction for a reconstruction-base run, which keeps the
SAE's reconstruction error out of the measured steering effect.

Each group also picks how hard it pushes, with ``scaling`` choosing what its
number means.

``scaling: fraction`` (the default) -- ``alpha`` is a fraction of each feature's
original activation at each token, so no per-feature magnitude has to be
configured: a feature that fires twice as hard as another is steered twice as
hard, and ``alpha = 1.0`` down removes that group's contribution exactly when
the feature and reconstruction SAEs are the same. Tokens
where a feature does not fire receive no push from it, which is why a feature
that is off in the steered cells cannot be steered this way at all.

``scaling: additive`` -- ``magnitude`` is a fixed amount added at every steered
token, firing or not, in the same units as the activations themselves::

    injected = sum(
        group.magnitude * w_dec[group.features] for group in groups
    )

Either number takes one of two forms: a single value applied to every feature in
the group, or a list with one value per feature, matched to ``features`` in the
order it is written, so each feature can be pushed by its own amount. Both forms
work in both directions and under both scalings.

An ``up`` group adds its scaled directions and a ``down`` group subtracts them.
Every group names its own direction, either through the ``up:``/``down:`` key it
is written under or with its own ``direction:``. Only gene-token positions of
the prompt are modified, and only during prefill.

Evaluation mirrors the cell-type annotation downstream task: the model
generates text for every cell, once as the baseline and once with the
interventions installed, and both texts are scored against the gold label and
assigned to the dataset class they name, using the shared acceptable-answers
matcher. The cells the run actually moved are the rows of
``changed_predictions.jsonl``.

Usage:
    python src/steer/sae_steer_inference.py
    python src/steer/sae_steer_inference.py --config configs/steer.yaml
    python src/steer/sae_steer_inference.py \
        --features 12 34 --alpha 1.0
    python src/steer/sae_steer_inference.py \
        --steer-base reconstruction --baseline reconstruction
    python src/steer/sae_steer_inference.py \
        --up-features 1146 2397 --down-features 9419 16995 --alpha 1.0
    python src/steer/sae_steer_inference.py \
        --features 12 34 --magnitude 40
    python src/steer/sae_steer_inference.py \
        --features 12 34 56 --magnitude 40 25 10
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import torch
from datasets import load_from_disk
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from src.data.inference import (
    InferenceConfig,
    generate,
    load_model,
)
from src.evaluate.downstream_tasks import sae_reconstruction_inference as reconstruction
from src.evaluate.downstream_tasks.cell_type_annotation import (
    cell_type_annotation as cell_eval,
)
from src.evaluate.model_loading import load_sae, resolve_checkpoint_path

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "steer.yaml"
CONFIG_SECTION = "model.steer"
_OUTPUT_SAFE_RE = re.compile(r"[^A-Za-z0-9_.=-]+")
# Scoring criterion for a run, and the default direction for a feature group
# that does not name one.
# Direction of one feature group inside an intervention. Groups are independent,
# so one set can be pushed up while another is pushed down in the same pass.
STEERING_DIRECTIONS = {"up", "down"}
# How a group's configured number is turned into the vector that is injected:
#   "fraction"  alpha * acts[:, f] * w_dec[f] -- a fraction of the activation
#               the feature already has at that token, so relative strengths
#               come from the cell and a feature that does not fire is untouched.
#   "additive"  magnitude * w_dec[f] -- a fixed amount added at every steered
#               token whether or not the feature fires, in activation units.
STEERING_SCALINGS = {"fraction", "additive"}
DEFAULT_STEERING_SCALING = "fraction"
SCALING_LABELS = {
    "fraction": "fraction_of_each_feature_activation_per_token",
    "additive": "fixed_magnitude_per_feature_per_token",
}
# Which activations the steered vector is written on top of:
#   "activations"    keep the base-model hidden state and add to it
#   "reconstruction" replace it with the SAE reconstruction, steered inside it
STEERING_BASES = {"activations", "reconstruction"}
DEFAULT_STEERING_BASE = "activations"
# Which model the per-cell comparison generation comes from. "auto" matches the
# baseline to the steering base (see _resolve_baseline_mode).
BASELINE_MODES = {"auto", "base_model", "reconstruction"}
BASELINE_LABELS = {
    "base_model": "unsteered base model",
    "reconstruction": "unsteered SAE reconstruction",
}
BASELINE_DEFINITIONS = {
    "base_model": "generation with every intervention inert (the base model)",
    "reconstruction": (
        "generation with the hidden state replaced by the unsteered SAE "
        "reconstruction (alpha = 0), so the comparison isolates the steering "
        "from the SAE's reconstruction error"
    ),
}
INTERVENTION_TYPES = {
    "activations": (
        "base_hidden_plus_fraction_of_selected_sae_feature_contributions"
    ),
    "reconstruction": (
        "sae_reconstruction_with_selected_sae_feature_contributions_rescaled"
    ),
}
# The same two bases when a group writes a fixed magnitude instead of a share
# of each feature's own activation. Fraction-only runs keep the strings above,
# so they stay comparable with every run recorded before additive steering.
ADDITIVE_INTERVENTION_TYPES = {
    "activations": (
        "base_hidden_plus_fixed_magnitude_of_selected_sae_feature_directions"
    ),
    "reconstruction": (
        "sae_reconstruction_plus_fixed_magnitude_of_selected_sae_feature_"
        "directions"
    ),
}
# Appended to the steer model name so a reconstruction-base run cannot
# overwrite the activations-base run configured under the same name. The
# earlier spelling is still stripped, so a name carried over from it resolves
# to the same directory as one written fresh.
RECONSTRUCTION_NAME_SUFFIX = "recon"
LEGACY_RECONSTRUCTION_NAME_SUFFIXES = ("recon_base",)
NONE_CLASS = "none"
# Distinct "unsteered -> steered" transitions shown in the terminal summary.
# Individual rows always go to changed_predictions.jsonl and comparison.tsv.
MAX_REPORTED_TRANSITIONS = 8


@dataclass
class FeatureGroupConfig:
    """One set of features steered in one direction.

    ``alpha`` is a magnitude; ``direction`` supplies the sign, and every group
    names its own: there is no run-level default to fall back on.

    ``alpha`` is either one number applied to every feature in the group, or a
    list carrying one number per feature in ``feature_indices`` order.
    ``scaling`` says what that number means: ``"fraction"`` scales each
    feature's own activation at each token, ``"additive"`` adds a fixed amount
    per token regardless of whether the feature fires.
    """

    feature_indices: list[int]
    alpha: float | list[float]
    direction: str | None = None
    name: str | None = None
    scaling: str = DEFAULT_STEERING_SCALING


@dataclass(frozen=True)
class SteeredGroup:
    """A resolved feature group as the hook applies it.

    Unlike ``FeatureGroupConfig`` the coefficients here are signed and always
    per feature: positive pushes the group up, negative pushes it down.
    ``scaling`` decides whether each coefficient multiplies the feature's own
    activation or is added outright.
    """

    feature_indices: list[int]
    coefficients: list[float]
    direction: str
    scaling: str = DEFAULT_STEERING_SCALING
    name: str | None = None

    @property
    def label(self) -> str:
        return self.name or self.direction

    @property
    def uniform_coefficient(self) -> float | None:
        """The one signed coefficient the group applies, when it has one."""

        distinct = set(self.coefficients)
        return distinct.pop() if len(distinct) == 1 else None


@dataclass
class SteeringInterventionConfig:
    """Configuration for one layer-level SAE steering intervention.

    One intervention carries any number of feature groups at a single layer, so
    a single pass can push one set of features up while pushing another down.
    """

    layer_idx: int
    checkpoint_path: Path
    groups: list[FeatureGroupConfig] = field(default_factory=list)
    sae_type: str | None = None
    sae_device: str = "model"
    base: str = DEFAULT_STEERING_BASE
    reconstruction_checkpoint_path: Path | None = None
    reconstruction_sae_type: str | None = None

    @property
    def effective_reconstruction_checkpoint_path(self) -> Path:
        return self.reconstruction_checkpoint_path or self.checkpoint_path

    @property
    def effective_reconstruction_sae_type(self) -> str | None:
        if self.reconstruction_checkpoint_path is None:
            return self.sae_type
        return self.reconstruction_sae_type

    @property
    def feature_indices(self) -> list[int]:
        """Every steered feature, in group order."""

        return [
            feature for group in self.groups for feature in group.feature_indices
        ]


@dataclass
class GenerationEvalConfig:
    """Generation-scoring settings for evaluating a steering run."""

    baseline_generation: bool
    baseline_mode: str = "auto"


@dataclass
class SAESteerEvalConfig:
    cell_type: cell_eval.EvalConfig
    interventions: list[SteeringInterventionConfig]
    evaluation: GenerationEvalConfig
    output_dir: Path
    save_predictions: bool
    include_prompt: bool
    steer_model_name: str
    output_tag: str | None


def _validate_feature_indices(
    feature_indices: list[int],
    d_hidden: int | None = None,
) -> list[int]:
    if not feature_indices:
        raise ValueError("At least one SAE feature index must be configured.")

    normalized: list[int] = []
    seen: set[int] = set()
    for feature in feature_indices:
        if isinstance(feature, bool):
            raise ValueError("SAE feature indices must be integers, not booleans.")
        if not isinstance(feature, int):
            raise ValueError(f"Invalid SAE feature index: {feature!r}")
        feature_idx = feature
        if feature_idx < 0:
            raise ValueError("SAE feature indices must be non-negative.")
        if d_hidden is not None and feature_idx >= d_hidden:
            raise ValueError(
                f"SAE feature index {feature_idx} is out of range for "
                f"d_hidden={d_hidden}."
            )
        if feature_idx not in seen:
            normalized.append(feature_idx)
            seen.add(feature_idx)
    return normalized


def _validate_alpha(value: Any) -> float:
    try:
        alpha = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("SAE steering alpha must be a finite number.") from exc
    if not math.isfinite(alpha):
        raise ValueError("SAE steering alpha must be a finite number.")
    return alpha


def _validate_magnitudes(
    value: Any,
    num_features: int,
    context: str = "steering group",
) -> list[float]:
    """One unsigned magnitude per feature. A single number covers them all.

    Both config forms land here: a scalar is broadcast across the group, and a
    list is matched against the group's features in the order they are listed,
    so ``features: [a, b, c]`` with ``magnitude: [1, 2, 3]`` pushes ``a`` by 1,
    ``b`` by 2 and ``c`` by 3. The sign comes from the group's direction, never
    from the numbers, so a negative entry is read as its magnitude.
    """

    if isinstance(value, (list, tuple)):
        magnitudes = [abs(_validate_alpha(item)) for item in value]
        if not magnitudes:
            raise ValueError(f"{context} has an empty per-feature magnitude list.")
        if len(magnitudes) == 1:
            return magnitudes * num_features
        if len(magnitudes) != num_features:
            raise ValueError(
                f"{context} lists {len(magnitudes)} magnitudes for "
                f"{num_features} features. Give one per feature in the order "
                "the features are listed, or a single number for all of them."
            )
        return magnitudes
    return [abs(_validate_alpha(value))] * num_features


def _uniform_magnitude(value: Any) -> float | None:
    """The one magnitude a group applies to every feature, when it has one."""

    if isinstance(value, (list, tuple)):
        distinct = {abs(_validate_alpha(item)) for item in value}
        return distinct.pop() if len(distinct) == 1 else None
    return abs(_validate_alpha(value))


def _normalize_steering_base(value: Any) -> str:
    """Resolve which activations the steered vector is written on top of."""

    base = str(value or DEFAULT_STEERING_BASE).strip().lower().replace("-", "_")
    aliases = {
        "activation": "activations",
        "additive": "activations",
        "base": "activations",
        "base_activations": "activations",
        "base_hidden": "activations",
        "hidden": "activations",
        "recon": "reconstruction",
        "reconstruct": "reconstruction",
        "reconstructed": "reconstruction",
        "reconstructed_activations": "reconstruction",
        "sae_reconstruction": "reconstruction",
    }
    base = aliases.get(base, base)
    if base not in STEERING_BASES:
        valid = ", ".join(sorted(STEERING_BASES))
        raise ValueError(f"Unknown steering base {value!r}. Valid bases: {valid}.")
    return base


def _normalize_baseline_mode(value: Any) -> str:
    """Resolve which model the comparison generation comes from."""

    mode = str(value or "auto").strip().lower().replace("-", "_")
    aliases = {
        "base": "base_model",
        "base_activations": "base_model",
        "base_hidden": "base_model",
        "model": "base_model",
        "none": "base_model",
        "unsteered": "base_model",
        "recon": "reconstruction",
        "reconstruct": "reconstruction",
        "reconstructed": "reconstruction",
        "sae_reconstruction": "reconstruction",
    }
    mode = aliases.get(mode, mode)
    if mode not in BASELINE_MODES:
        valid = ", ".join(sorted(BASELINE_MODES))
        raise ValueError(
            f"Unknown baseline mode {value!r}. Valid baselines: {valid}."
        )
    return mode


def _normalize_steering_direction(value: Any) -> str:
    """Resolve which way one feature group is pushed."""

    direction = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "+": "up",
        "amplify": "up",
        "boost": "up",
        "increase": "up",
        "toward": "up",
        "towards": "up",
        "-": "down",
        "ablate": "down",
        "away": "down",
        "decrease": "down",
        "remove": "down",
        "suppress": "down",
    }
    direction = aliases.get(direction, direction)
    if direction not in STEERING_DIRECTIONS:
        valid = ", ".join(sorted(STEERING_DIRECTIONS))
        raise ValueError(
            f"Unknown steering direction {value!r}. Valid directions: {valid}."
        )
    return direction


def _normalize_steering_scaling(value: Any) -> str:
    """Resolve what a group's configured number means.

    ``"fraction"`` is the original behavior -- alpha is a fraction of each
    feature's own activation, so nothing is pushed where the feature does not
    fire. ``"additive"`` injects a fixed magnitude per steered token instead,
    which is the knob to reach for when a feature is off in the cells being
    steered and the fractional form therefore has nothing to scale.
    """

    scaling = (
        str(value or DEFAULT_STEERING_SCALING).strip().lower().replace("-", "_")
    )
    aliases = {
        "activation_fraction": "fraction",
        "fractional": "fraction",
        "proportional": "fraction",
        "relative": "fraction",
        "scale": "fraction",
        "absolute": "additive",
        "add": "additive",
        "constant": "additive",
        "direct": "additive",
        "direct_additive": "additive",
        "fixed": "additive",
        "magnitude": "additive",
        "vector": "additive",
    }
    scaling = aliases.get(scaling, scaling)
    if scaling not in STEERING_SCALINGS:
        valid = ", ".join(sorted(STEERING_SCALINGS))
        raise ValueError(
            f"Unknown steering scaling {value!r}. Valid scalings: {valid}."
        )
    return scaling


def _format_steer_coefficient(coefficient: float, scaling: str) -> str:
    """Fractions read as percentages; additive magnitudes are activation units."""

    if _normalize_steering_scaling(scaling) == "additive":
        return f"{coefficient:+.4g}"
    return f"{coefficient:+.0%}"


def _resolve_direction(
    direction: str | None,
    context: str = "steering group",
) -> str:
    """Every group names its own direction; there is no run-level default."""

    if direction is None:
        raise ValueError(
            f"{context} does not say which way it is steered. Declare the "
            "features under 'up:' or 'down:', or set 'direction: up'/"
            "'direction: down' on the group."
        )
    return _normalize_steering_direction(direction)


def _effective_alpha(alpha: float, direction: str) -> float:
    """Use alpha as a multiplier: add going up, subtract going down."""

    magnitude = abs(_validate_alpha(alpha))
    return magnitude if _normalize_steering_direction(direction) == "up" else -magnitude


def _effective_coefficients(
    alpha: Any,
    direction: str,
    num_features: int,
    context: str = "steering group",
) -> list[float]:
    """One signed coefficient per feature: add going up, subtract going down."""

    sign = 1.0 if _normalize_steering_direction(direction) == "up" else -1.0
    return [
        sign * magnitude
        for magnitude in _validate_magnitudes(alpha, num_features, context)
    ]


def _steered_group(group: FeatureGroupConfig) -> SteeredGroup:
    """Resolve one configured group into the signed form the hook applies."""

    direction = _resolve_direction(group.direction, group.name or "steering group")
    features = list(group.feature_indices)
    return SteeredGroup(
        feature_indices=features,
        coefficients=_effective_coefficients(
            group.alpha,
            direction,
            len(features),
            group.name or f"{direction} group",
        ),
        direction=direction,
        scaling=_normalize_steering_scaling(group.scaling),
        name=group.name,
    )


def _normalize_steered_groups(
    groups: SteeredGroup | list[SteeredGroup],
    d_hidden: int | None = None,
) -> list[SteeredGroup]:
    """Validate the groups of one intervention against each other.

    A feature steered up in one group and down in another would net out to a
    single unintended coefficient, so overlapping groups are rejected rather
    than silently summed.
    """

    if isinstance(groups, SteeredGroup):
        groups = [groups]
    if not groups:
        raise ValueError("At least one steering feature group must be configured.")

    normalized: list[SteeredGroup] = []
    owner: dict[int, str] = {}
    for group in groups:
        indices = _validate_feature_indices(group.feature_indices, d_hidden)
        for feature in indices:
            if feature in owner:
                raise ValueError(
                    f"SAE feature {feature} appears in steering groups "
                    f"{owner[feature]!r} and {group.label!r}; steer each "
                    "feature in one direction only."
                )
            owner[feature] = group.label
        coefficients = [
            _validate_alpha(coefficient) for coefficient in group.coefficients
        ]
        if len(coefficients) != len(indices):
            raise ValueError(
                f"Steering group {group.label!r} carries {len(coefficients)} "
                f"coefficients for {len(indices)} features. Give one magnitude "
                "per feature, or a single number for the whole group."
            )
        normalized.append(
            SteeredGroup(
                feature_indices=indices,
                coefficients=coefficients,
                direction=_normalize_steering_direction(group.direction),
                scaling=_normalize_steering_scaling(group.scaling),
                name=group.name,
            )
        )
    return normalized


def steer_sae_features(
    sae,
    hidden: torch.Tensor,
    groups: SteeredGroup | list[SteeredGroup],
    stats: dict[str, Any] | None = None,
    base: str = DEFAULT_STEERING_BASE,
    reconstruction_sae=None,
) -> torch.Tensor:
    """Write each group's steered feature directions into each hidden state.

    ``groups`` may hold any number of ``SteeredGroup``s, each with its own
    features and its own signed coefficients, so a single call can push one set
    up while pushing another down. A group's ``scaling`` picks how its
    coefficients reach the residual stream::

        fraction: injected += coefs * activations[:, features] @ w_dec[features]
        additive: injected += coefs @ w_dec[features]

    A fraction-scaled group is the original behavior: each coefficient is a
    fraction of the activation its features already have at that token, so
    relative strengths come from the cell and a feature that does not fire is
    left alone. An additive group instead writes a fixed magnitude at every
    steered token, firing or not, in the same units as the activations
    themselves. Both scalings may be mixed in one call, and each carries one
    coefficient per feature, so a group can push each of its features by a
    different amount.

    ``base`` picks which activations that vector is written on top of. With
    ``"activations"`` the base-model hidden state is kept and the injected
    vector is added to it, so an all-zero alpha returns the input unchanged.
    With ``"reconstruction"`` the hidden state is replaced by
    ``reconstruction_sae.decode(reconstruction_sae.encode(hidden))`` plus the
    feature SAE's steering vector (defaults to using ``sae`` for both), so a zero
    alpha is the plain SAE reconstruction and everything the SAE cannot express
    is dropped from the residual stream.

    Under fraction scaling ``alpha = -1.0`` removes a feature's contribution
    exactly and ``alpha = 1.0`` doubles it when both SAEs are the same. Tokens
    where a feature does not fire receive no push from it. Under additive scaling every steered token is
    pushed by the configured magnitude, which is what makes a feature that
    almost never fires steerable at all.

    When ``stats`` is supplied it is updated in place with the per-feature
    firing counts and the injected-to-residual norm ratio at the steered
    positions. With a TopK SAE a feature outside the top ``k`` at a token has
    activation exactly zero, so steering it there is an exact no-op; these
    counters are what distinguish "the feature did nothing" from "the feature
    was never on".
    """

    w_dec = getattr(sae, "w_dec", None)
    if not isinstance(w_dec, torch.Tensor) or w_dec.ndim != 2:
        raise ValueError("SAE must expose a 2D w_dec tensor for feature steering.")
    steered_groups = _normalize_steered_groups(groups, int(w_dec.shape[0]))
    if int(hidden.shape[-1]) != int(w_dec.shape[1]):
        raise ValueError(
            f"Hidden width {hidden.shape[-1]} does not match SAE decoder width "
            f"{w_dec.shape[1]}."
        )
    steering_base = _normalize_steering_base(base)
    # Only the activations base is a no-op at zero; the reconstruction base
    # still has to replace the hidden state with the SAE's reconstruction.
    if steering_base == "activations" and all(
        coefficient == 0.0
        for group in steered_groups
        for coefficient in group.coefficients
    ):
        return hidden

    activations = sae.encode(hidden.to(device=w_dec.device, dtype=w_dec.dtype))
    # Groups never overlap (_normalize_steered_groups rejects that), so the
    # whole injection is one matmul over the concatenated feature set with a
    # per-feature coefficient carrying its group's sign and scaling.
    indices = [
        feature for group in steered_groups for feature in group.feature_indices
    ]
    coefficients = [
        coefficient
        for group in steered_groups
        for coefficient in group.coefficients
    ]
    scalings = [
        group.scaling for group in steered_groups for _ in group.feature_indices
    ]

    def coefficients_for(scaling: str) -> torch.Tensor:
        """The coefficient vector with every other scaling zeroed out."""

        return torch.tensor(
            [
                coefficient if own == scaling else 0.0
                for coefficient, own in zip(coefficients, scalings)
            ],
            device=activations.device,
            dtype=activations.dtype,
        )

    # A fraction-scaled coefficient multiplies the activation the feature
    # already has; an additive one is written whether the feature fired or not.
    # Splitting the coefficients between the two terms keeps both scalings
    # inside one matmul, so a mixed group costs no more than either alone.
    selected = activations[..., indices]
    gains = selected * coefficients_for("fraction")
    if any(scaling == "additive" for scaling in scalings):
        gains = gains + coefficients_for("additive")
    injected = gains @ w_dec[indices]
    reconstructed = None
    if steering_base == "reconstruction":
        if reconstruction_sae is None or reconstruction_sae is sae:
            reconstructed = sae.decode(activations)
        else:
            recon_param = next(reconstruction_sae.parameters())
            recon_hidden = hidden.to(
                device=recon_param.device, dtype=recon_param.dtype,
            )
            reconstructed = reconstruction_sae.decode(
                reconstruction_sae.encode(recon_hidden)
            )
        if reconstructed.shape != hidden.shape:
            raise ValueError("Reconstruction SAE output shape must match the hidden state.")

    if stats is not None:
        _accumulate_steering_stats(
            stats,
            selected=selected,
            injected=injected,
            hidden=hidden,
            num_features=len(indices),
            reconstructed=reconstructed,
        )

    if reconstructed is None:
        return hidden + injected.to(device=hidden.device, dtype=hidden.dtype)
    return (reconstructed + injected.to(reconstructed)).to(
        device=hidden.device, dtype=hidden.dtype
    )


def _accumulate_steering_stats(
    stats: dict[str, Any],
    *,
    selected: torch.Tensor,
    injected: torch.Tensor,
    hidden: torch.Tensor,
    num_features: int,
    reconstructed: torch.Tensor | None = None,
) -> None:
    """Fold one steered batch into the running delivery counters.

    Everything is reduced to scalars and per-feature vectors here so the
    diagnostic never retains a token-by-feature matrix. ``reconstructed`` is
    supplied only by the reconstruction base, where the hidden state is also
    displaced by the SAE's reconstruction error; that displacement is tracked
    separately from the steering itself.
    """

    flat = selected.reshape(-1, num_features).float()
    num_tokens = int(flat.shape[0])
    if num_tokens == 0:
        return

    if not stats:
        stats["num_tokens"] = 0
        stats["active_tokens"] = [0] * num_features
        stats["activation_sum"] = [0.0] * num_features
        stats["injected_norm_sum"] = 0.0
        stats["hidden_norm_sum"] = 0.0
        stats["norm_ratio_sum"] = 0.0
        stats["recon_tokens"] = 0
        stats["recon_shift_sum"] = 0.0
        stats["recon_ratio_sum"] = 0.0

    active = (flat > 0).sum(dim=0).tolist()
    totals = flat.sum(dim=0).tolist()
    stats["num_tokens"] += num_tokens
    stats["active_tokens"] = [
        prior + int(count) for prior, count in zip(stats["active_tokens"], active)
    ]
    stats["activation_sum"] = [
        prior + float(total)
        for prior, total in zip(stats["activation_sum"], totals)
    ]

    hidden_flat = hidden.reshape(-1, hidden.shape[-1]).float()
    # injected already carries every group's signed alpha, so its norm is the
    # size of the whole perturbation, up and down contributions together.
    injected_norm = injected.reshape(-1, injected.shape[-1]).float().norm(dim=-1)
    residual = hidden_flat.norm(dim=-1).to(injected_norm.device)
    stats["injected_norm_sum"] += float(injected_norm.sum())
    stats["hidden_norm_sum"] += float(residual.sum())
    stats["norm_ratio_sum"] += float(
        (injected_norm / residual.clamp_min(1e-6)).sum()
    )

    if reconstructed is not None:
        recon_flat = reconstructed.reshape(-1, reconstructed.shape[-1]).float()
        shift = (recon_flat - hidden_flat.to(recon_flat.device)).norm(dim=-1)
        stats["recon_tokens"] = int(stats.get("recon_tokens", 0)) + num_tokens
        stats["recon_shift_sum"] = float(
            stats.get("recon_shift_sum", 0.0)
        ) + float(shift.sum())
        stats["recon_ratio_sum"] = float(
            stats.get("recon_ratio_sum", 0.0)
        ) + float((shift / residual.to(shift.device).clamp_min(1e-6)).sum())


class SAESteeringHook(reconstruction.SAEReconstructionHook):
    """Rescale selected SAE features inside one layer's gene-token states.

    Position selection is inherited from ``SAEReconstructionHook``: the parent's
    ``_reconstruct_selected`` only touches the token indices supplied by
    ``_prepare_hooks_for_prompt`` -- the cell sentence's gene tokens -- and only
    on the prefill pass. This subclass only changes what is written there.

    ``base`` selects the steering model: ``"activations"`` adds the scaled
    feature directions to the base-model hidden state, ``"reconstruction"``
    writes the SAE reconstruction carrying the same rescaling.

    ``groups`` holds one or more ``SteeredGroup``s, each with its own features,
    scaling and signed per-feature coefficients, so one set can be pushed up
    while another is pushed down in the same forward pass.

    Setting ``baseline_pass`` switches a reconstruction-base hook to writing the
    plain, unsteered SAE reconstruction, which is the matched comparison
    generation for that model. An activations-base hook stays inert in a
    baseline pass, since its own baseline is the untouched base model.
    """

    def __init__(
        self,
        model,
        layer_idx: int,
        sae,
        *,
        name: str,
        groups: SteeredGroup | list[SteeredGroup],
        base: str = DEFAULT_STEERING_BASE,
        expected_d_model: int | None = None,
        pooling_method: str | None = None,
        reconstruction_sae=None,
    ):
        super().__init__(
            model,
            layer_idx,
            reconstruction_sae if reconstruction_sae is not None else sae,
            name=name,
            expected_d_model=expected_d_model,
            pooling_method=pooling_method,
        )
        self.feature_sae = sae
        d_hidden = getattr(sae, "d_hidden", None)
        self.groups = _normalize_steered_groups(
            groups,
            int(d_hidden) if d_hidden is not None else None,
        )
        # Flat views in group order: the delivery counters are indexed by
        # position, and _summarize in intervention_delivery.py reads them.
        self.feature_indices = [
            feature for group in self.groups for feature in group.feature_indices
        ]
        self.feature_groups = [
            group for group in self.groups for _ in group.feature_indices
        ]
        self.feature_coefficients = [
            coefficient
            for group in self.groups
            for coefficient in group.coefficients
        ]
        self.base = _normalize_steering_base(base)
        self.stats: dict[str, Any] = {}
        self.baseline_pass = False
        self.num_baseline_calls = 0
        self.num_baseline_tokens = 0

    def _reconstruct_selected(self, hidden: torch.Tensor) -> torch.Tensor | None:
        # An activations-base hook must not touch its own baseline, and the
        # steered counters must only ever count steered passes.
        if self.baseline_pass and self.base != "reconstruction":
            return None

        calls_before = self.num_calls
        tokens_before = self.num_tokens
        updated = super()._reconstruct_selected(hidden)
        if self.baseline_pass:
            self.num_baseline_calls += self.num_calls - calls_before
            self.num_baseline_tokens += self.num_tokens - tokens_before
            self.num_calls = calls_before
            self.num_tokens = tokens_before
        return updated

    def _reconstruct_flat(self, flat: torch.Tensor) -> torch.Tensor:
        d_model = int(flat.shape[-1])
        if self.expected_d_model is not None and d_model != self.expected_d_model:
            raise RuntimeError(
                f"{self.name} expected hidden dim {self.expected_d_model}, "
                f"got {d_model}."
            )

        if self.baseline_pass:
            # The parent writes sae.decode(sae.encode(hidden)) with no feature
            # rescaled: exactly this run's alpha = 0 reconstruction.
            return super()._reconstruct_flat(flat)

        with torch.inference_mode():
            return steer_sae_features(
                self.feature_sae,
                flat,
                self.groups,
                stats=self.stats,
                base=self.base,
                reconstruction_sae=self.sae,
            )

    def diagnostics(self) -> dict[str, Any]:
        """Per-feature firing rates and injected-norm ratio at steered tokens.

        ``firing_rate`` is the fraction of steered tokens where the feature was
        inside the SAE's active set. Under fraction scaling a near-zero rate
        means steering was an exact no-op at almost every position, which is a
        distribution problem (for example an SAE trained without the instruction
        prefix but applied with it) rather than evidence about whether the
        feature is causal. An additive group still writes its magnitude at every
        steered token, so a low firing rate there says the feature is off in
        these cells, not that the intervention did nothing.
        """

        stats = self.stats
        num_tokens = int(stats.get("num_tokens", 0))
        if num_tokens == 0:
            return {
                "steering_base": self.base,
                "steering_directions": sorted(
                    {group.direction for group in self.groups}
                ),
                "steering_scalings": sorted(
                    {group.scaling for group in self.groups}
                ),
                "steered_tokens": 0,
                "note": "hook never wrote; check token selection and prompt length",
            }

        per_feature = []
        for position, feature_id in enumerate(self.feature_indices):
            active = int(stats["active_tokens"][position])
            total_activation = float(stats["activation_sum"][position])
            group = self.feature_groups[position]
            coefficient = self.feature_coefficients[position]
            per_feature.append(
                {
                    "feature_id": feature_id,
                    "group": group.label,
                    "direction": group.direction,
                    "scaling": group.scaling,
                    "effective_coefficient": coefficient,
                    # Only meaningful for fraction scaling; an additive group
                    # pushes by a magnitude, not by a share of the activation.
                    "effective_alpha_fraction": (
                        coefficient if group.scaling == "fraction" else None
                    ),
                    "firing_rate": active / num_tokens,
                    "mean_activation_when_active": (
                        total_activation / active if active else 0.0
                    ),
                }
            )

        firing_rates = [entry["firing_rate"] for entry in per_feature]
        diagnostics = {
            "steering_base": self.base,
            "steering_directions": sorted(
                {group.direction for group in self.groups}
            ),
            "steering_scalings": sorted({group.scaling for group in self.groups}),
            "steered_tokens": num_tokens,
            "per_feature": per_feature,
            "min_firing_rate": min(firing_rates),
            "mean_firing_rate": sum(firing_rates) / len(firing_rates),
            "mean_injected_norm": stats["injected_norm_sum"] / num_tokens,
            "mean_residual_norm": stats["hidden_norm_sum"] / num_tokens,
            "mean_injected_to_residual_ratio": (
                stats["norm_ratio_sum"] / num_tokens
            ),
        }
        recon_tokens = int(stats.get("recon_tokens", 0))
        if recon_tokens:
            # How far the reconstruction alone moves the hidden state, i.e. the
            # SAE error this base introduces on top of the steering.
            diagnostics["mean_reconstruction_shift_norm"] = (
                stats["recon_shift_sum"] / recon_tokens
            )
            diagnostics["mean_reconstruction_to_residual_ratio"] = (
                stats["recon_ratio_sum"] / recon_tokens
            )
        if self.num_baseline_tokens:
            diagnostics["baseline_reconstructed_tokens"] = self.num_baseline_tokens
        return diagnostics


SAESteeringPatcher = reconstruction.SAEReconstructionPatcher


def _project_path(value: str | Path) -> Path:
    return reconstruction._project_path(value)


def _mapping(value: Any, context: str) -> dict[str, Any]:
    return reconstruction._mapping(value, context)


def _model_steer_config(raw: dict[str, Any]) -> dict[str, Any]:
    model = _mapping(raw.get("model"), "model")
    return _mapping(model.get("steer"), CONFIG_SECTION)


def _safe_output_name(value: Any, fallback: str = "steer") -> str:
    candidate = fallback if value is None or str(value).strip() == "" else str(value)
    safe = _OUTPUT_SAFE_RE.sub("_", candidate).strip("_")
    return safe or fallback


def _configured_steer_model_name(raw: dict[str, Any]) -> str:
    steer = _model_steer_config(raw)
    for key in ("name", "output_name", "model_name"):
        if key in steer:
            return _safe_output_name(steer.get(key))

    output = _mapping(steer.get("output"), f"{CONFIG_SECTION}.output")
    if "tag" in output:
        return _safe_output_name(output.get("tag"))

    return "steer"


def _resolve_steer_model_name(
    configured_name: str,
    interventions: list[SteeringInterventionConfig],
) -> str:
    """Keep the two steering models in separate output directories.

    The reconstruction-base run of a given config is a different model from the
    activations-base one, so it gets its own name unless the configured name
    already says so. Stripping first makes this idempotent under a CLI override
    that switches the base back.
    """

    suffix = f"_{RECONSTRUCTION_NAME_SUFFIX}"
    name = configured_name
    for known in (
        *(f"_{legacy}" for legacy in LEGACY_RECONSTRUCTION_NAME_SUFFIXES),
        suffix,
    ):
        if name.endswith(known):
            name = name[: -len(known)]
            break
    uses_reconstruction = any(
        intervention.base == "reconstruction" for intervention in interventions
    )
    if uses_reconstruction and "recon" not in name.lower():
        return f"{name}{suffix}"
    return name


def _steer_output_dir(
    raw: dict[str, Any],
    steer_model_name: str,
) -> Path:
    model = _mapping(raw.get("model"), "model")
    output_root = _project_path(
        model.get(
            "output_dir",
            model.get("results_dir", "results"),
        )
    )
    return output_root / "steer" / cell_eval.TASK_NAME / steer_model_name


def _load_evaluation_config(module: dict[str, Any]) -> GenerationEvalConfig:
    evaluation = _mapping(
        module.get("evaluation", module.get("classification")),
        f"{CONFIG_SECTION}.evaluation",
    )
    baseline_mode = _normalize_baseline_mode(
        evaluation.get(
            "baseline",
            evaluation.get("baseline_mode", module.get("baseline", "auto")),
        )
    )

    return GenerationEvalConfig(
        baseline_generation=bool(evaluation.get("baseline_generation", True)),
        baseline_mode=baseline_mode,
    )


def _resolve_baseline_mode(cfg: "SAESteerEvalConfig") -> str:
    """Pick which model the per-cell comparison generation comes from.

    ``auto`` matches the baseline to the steering model. An activations-base run
    is compared against the untouched base model; a reconstruction-base run is
    compared against the same SAE reconstruction with nothing rescaled, so the
    difference measures the steering rather than the SAE's reconstruction error.
    An explicit ``base_model`` on a reconstruction-base run is still allowed --
    it just folds that error into the measured effect.
    """

    mode = _normalize_baseline_mode(cfg.evaluation.baseline_mode)
    has_reconstruction = any(
        intervention.base == "reconstruction" for intervention in cfg.interventions
    )
    if mode == "auto":
        return "reconstruction" if has_reconstruction else "base_model"
    if mode == "reconstruction" and not has_reconstruction:
        raise ValueError(
            "evaluation.baseline='reconstruction' needs at least one "
            "intervention with base='reconstruction'; with additive steering "
            "the baseline is the base model."
        )
    return mode


def _feature_list(item: dict[str, Any], inherited: Any = None) -> list[int]:
    value = item.get(
        "features",
        item.get("feature_indices", item.get("feature_numbers", inherited)),
    )
    if not isinstance(value, list):
        raise ValueError(
            "Steering features must be a YAML list under 'features' "
            "(or 'feature_indices')."
        )
    return _validate_feature_indices(value)


# Config keys that declare feature groups. "up"/"down" name a direction
# outright; "groups" carries an explicit list, each entry free to set its own.
DIRECTED_GROUP_KEYS = {
    "up": "up",
    "steer_up": "up",
    "toward": "up",
    "down": "down",
    "steer_down": "down",
    "away": "down",
}
GROUP_LIST_KEYS = ("groups", "feature_groups")
GROUP_KEYS = (*GROUP_LIST_KEYS, *DIRECTED_GROUP_KEYS)
FLAT_FEATURE_KEYS = ("features", "feature_indices", "feature_numbers")
# The two ways to say how hard to push, and the scaling each implies.
# "alpha" is a fraction of each feature's own activation; "magnitude" is a
# fixed amount added at every steered token. Either takes one number for the
# whole group or one per feature, listed in the group's feature order.
ALPHA_KEYS = ("alpha", "alphas", "alpha_fraction")
MAGNITUDE_KEYS = ("magnitude", "magnitudes", "additive_magnitude")
SCALING_KEYS = ("scaling", "steer_scaling", "alpha_scaling")
# Every key a config level may use to declare how hard to push, so an
# intervention that names its own can drop what it inherited.
STRENGTH_KEYS = (*ALPHA_KEYS, *MAGNITUDE_KEYS, *SCALING_KEYS)


def _declared_value(source: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """First of ``keys`` this config level actually sets, else ``None``."""

    for key in keys:
        value = source.get(key)
        if value is not None:
            return value
    return None


def _declared_strength(
    source: dict[str, Any],
    *,
    context: str,
    default_alpha: Any,
    default_scaling: str,
) -> tuple[Any, str]:
    """Read how hard one config level pushes, and how that number is applied.

    ``alpha:`` keeps the fractional scaling; ``magnitude:`` switches the level
    to additive steering, since a fixed magnitude read as a fraction would be a
    silent hundredfold mistake. An explicit ``scaling:`` still wins over that
    inference, which is how an additive run can be written as ``alpha:`` plus
    ``scaling: additive``. A level that names no number at all inherits both
    the number and the scaling from the level above.
    """

    magnitude = _declared_value(source, MAGNITUDE_KEYS)
    alpha = _declared_value(source, ALPHA_KEYS)
    if magnitude is not None and alpha is not None:
        raise ValueError(
            f"{context} sets both {'/'.join(ALPHA_KEYS)} and "
            f"{'/'.join(MAGNITUDE_KEYS)}. Use 'alpha' for a fraction of each "
            "feature's own activation, or 'magnitude' for a fixed amount added "
            "at every steered token."
        )
    declared_scaling = _declared_value(source, SCALING_KEYS)
    if declared_scaling is not None:
        scaling = _normalize_steering_scaling(declared_scaling)
    elif magnitude is not None:
        scaling = "additive"
    else:
        scaling = _normalize_steering_scaling(default_scaling)
    value = magnitude if magnitude is not None else alpha
    return (default_alpha if value is None else value), scaling


def _group_alpha(
    value: Any,
    context: str,
    num_features: int,
    scaling: str = DEFAULT_STEERING_SCALING,
) -> float | list[float]:
    """Magnitudes for one group. A missing one is an error, not a zero.

    An explicit ``alpha: 0`` stays legal -- it is the exact-no-op control on the
    activations base, and the plain reconstruction on the reconstruction base.
    But a group that never names an alpha would otherwise silently steer nothing
    while still showing up in the config, the run printout, and the delivery
    diagnostics as though it were live.

    A list is validated here, against this group's own feature count, so a
    mismatched per-feature list fails at config load rather than after the
    model is on the GPU.
    """

    if value is None:
        knob = "magnitude" if scaling == "additive" else "alpha"
        raise ValueError(
            f"{context} has no {knob}. Set one on the group, on the "
            f"intervention, or on model.steer. A group that steers nothing must "
            f"say '{knob}: 0' outright."
        )
    if isinstance(value, (list, tuple)):
        return _validate_magnitudes(value, num_features, context)
    return abs(_validate_alpha(value))


def _feature_group_from_entry(
    value: Any,
    *,
    context: str,
    direction: str | None,
    default_alpha: Any,
    default_scaling: str = DEFAULT_STEERING_SCALING,
) -> FeatureGroupConfig:
    """Build one group from a bare feature list or a mapping."""

    item = {"features": value} if isinstance(value, list) else _mapping(value, context)
    own_direction = item.get("direction", item.get("mode"))
    if (
        own_direction is not None
        and direction is not None
        and _normalize_steering_direction(own_direction) != direction
    ):
        raise ValueError(
            f"{context} sets direction {own_direction!r}, which contradicts the "
            f"{direction!r} key it is declared under."
        )
    raw_direction = own_direction if own_direction is not None else direction
    features = _feature_list(item)
    alpha, scaling = _declared_strength(
        item,
        context=context,
        default_alpha=default_alpha,
        default_scaling=default_scaling,
    )
    return FeatureGroupConfig(
        feature_indices=features,
        alpha=_group_alpha(alpha, context, len(features), scaling),
        direction=(
            _normalize_steering_direction(raw_direction)
            if raw_direction is not None
            else None
        ),
        name=item.get("name"),
        scaling=scaling,
    )


def _reject_overlapping_groups(
    groups: list[FeatureGroupConfig],
    context: str,
) -> list[FeatureGroupConfig]:
    """Fail at config load, not after the model is on the GPU.

    A feature in two groups would collapse to one unintended net coefficient,
    which is a config mistake in every case worth supporting.
    """

    owner: dict[int, str] = {}
    for position, group in enumerate(groups):
        label = group.name or group.direction or f"group {position}"
        for feature in group.feature_indices:
            if feature in owner:
                raise ValueError(
                    f"{context}: SAE feature {feature} appears in steering "
                    f"groups {owner[feature]!r} and {label!r}; steer each "
                    "feature in one direction only."
                )
            owner[feature] = label
    return groups


def _load_feature_groups(
    item: dict[str, Any],
    *,
    context: str,
    inherited_groups: dict[str, Any],
    inherited_features: Any,
    default_alpha: Any,
    default_scaling: str = DEFAULT_STEERING_SCALING,
) -> list[FeatureGroupConfig]:
    """Read the feature groups of one intervention.

    Three config forms, in precedence order:

    1. ``groups: [...]``          -- any number of sets, each with its own
                                     ``direction`` and ``alpha``/``magnitude``.
    2. ``up:`` and/or ``down:``   -- the same thing for the common two-set case;
                                     each takes a feature list or a mapping.
    3. ``features:`` + ``direction:`` + ``alpha:`` -- one set, the original
                                     single-set form.

    Every group takes its strength as ``alpha:`` (a fraction of each feature's
    own activation) or ``magnitude:`` (a fixed amount added per steered token),
    each of them either one number for the group or one per feature.

    Group keys set on the intervention win outright; otherwise group keys
    inherited from the module level are merged under the intervention's own
    keys.
    """

    if any(key in item for key in GROUP_KEYS) and any(
        key in item for key in FLAT_FEATURE_KEYS
    ):
        raise ValueError(
            f"{context} sets both a feature group ({'/'.join(GROUP_KEYS)}) and a "
            f"flat feature list ({'/'.join(FLAT_FEATURE_KEYS)}). Use one form."
        )
    source = (
        item
        if any(key in item for key in GROUP_KEYS)
        else {**inherited_groups, **item}
    )

    for key in GROUP_LIST_KEYS:
        entries = source.get(key)
        if entries is None:
            continue
        if not isinstance(entries, list) or not entries:
            raise ValueError(
                f"{context}.{key} must be a non-empty list of feature groups."
            )
        return _reject_overlapping_groups(
            [
                _feature_group_from_entry(
                    entry,
                    context=f"{context}.{key}[{idx}]",
                    direction=None,
                    default_alpha=default_alpha,
                    default_scaling=default_scaling,
                )
                for idx, entry in enumerate(entries)
            ],
            f"{context}.{key}",
        )

    groups = [
        _feature_group_from_entry(
            source[key],
            context=f"{context}.{key}",
            direction=direction,
            default_alpha=default_alpha,
            default_scaling=default_scaling,
        )
        for key, direction in DIRECTED_GROUP_KEYS.items()
        if key in source
    ]
    if groups:
        return _reject_overlapping_groups(groups, context)

    features = _feature_list(source, inherited_features)
    return [
        FeatureGroupConfig(
            feature_indices=features,
            alpha=_group_alpha(
                default_alpha,
                context,
                len(features),
                default_scaling,
            ),
            direction=_resolve_direction(
                source.get("direction", source.get("steer_direction")),
                f"{context}.features",
            ),
            scaling=default_scaling,
        )
    ]


def _intervention_from_item(
    item: dict[str, Any],
    *,
    context: str,
    inherited_sae_type: str | None,
    inherited_sae_device: str,
    inherited_groups: dict[str, Any],
    inherited_features: Any,
    inherited_alpha: Any,
    inherited_scaling: str,
    inherited_base: Any,
) -> SteeringInterventionConfig:
    if "intervene_checkpoint" in item:
        checkpoint_cfg = _mapping(
            item["intervene_checkpoint"], f"{context}.intervene_checkpoint",
        )
    else:
        # Retain support for steering configs written before the explicit name.
        checkpoint_cfg = reconstruction._checkpoint_cfg_from_item(item)
    checkpoint_path = resolve_checkpoint_path(checkpoint_cfg)
    recon_cfg = item.get("reconstruction_checkpoint")
    if recon_cfg is not None:
        recon_cfg = _mapping(recon_cfg, f"{context}.reconstruction_checkpoint")
        recon_path = resolve_checkpoint_path(recon_cfg)
        recon_type = recon_cfg.get("sae_type")
    else:
        recon_path = None
        recon_type = None

    layer_idx = item.get("layer_idx", item.get("layer", checkpoint_cfg.get("layer")))
    if layer_idx is None:
        raise ValueError("Each SAE steering intervention must set layer_idx or layer.")
    layer_idx = int(layer_idx)
    if layer_idx < 0:
        raise ValueError("SAE steering layer index must be non-negative.")

    default_alpha, default_scaling = _declared_strength(
        item,
        context=context,
        default_alpha=inherited_alpha,
        default_scaling=inherited_scaling,
    )
    groups = _load_feature_groups(
        item,
        context=context,
        inherited_groups=inherited_groups,
        inherited_features=inherited_features,
        default_alpha=default_alpha,
        default_scaling=default_scaling,
    )

    sae_type = item.get(
        "sae_type",
        checkpoint_cfg.get("sae_type", inherited_sae_type),
    )
    sae_device = item.get(
        "sae_device",
        item.get("device", inherited_sae_device),
    )
    base = _normalize_steering_base(
        item.get(
            "base",
            item.get("steer_base", item.get("intervention_base", inherited_base)),
        )
    )
    return SteeringInterventionConfig(
        layer_idx=layer_idx,
        checkpoint_path=checkpoint_path,
        groups=groups,
        sae_type=str(sae_type) if sae_type is not None else None,
        sae_device=str(sae_device),
        base=base,
        reconstruction_checkpoint_path=recon_path,
        reconstruction_sae_type=str(recon_type) if recon_type is not None else None,
    )


def _load_interventions(module: dict[str, Any]) -> list[SteeringInterventionConfig]:
    model_cfg = _mapping(module.get("model"), f"{CONFIG_SECTION}.model")
    inherited_sae_type = module.get("sae_type", model_cfg.get("sae_type"))
    inherited_sae_device = str(module.get("sae_device", "model"))
    inherited_features = module.get(
        "features",
        module.get("feature_indices", module.get("feature_numbers")),
    )
    # No default: a group with no alpha anywhere is rejected rather than
    # silently steering nothing (see _group_alpha). A module-level "magnitude:"
    # sets additive scaling for every group that does not override it.
    inherited_alpha, inherited_scaling = _declared_strength(
        module,
        context=CONFIG_SECTION,
        default_alpha=None,
        default_scaling=DEFAULT_STEERING_SCALING,
    )
    inherited_base = module.get(
        "base",
        module.get("steer_base", module.get("intervention_base", DEFAULT_STEERING_BASE)),
    )
    inherited_groups = {
        key: module[key] for key in GROUP_KEYS if key in module
    }

    entries = module.get("interventions")
    if entries is None:
        intervention_cfg = _mapping(
            module.get("intervention"),
            f"{CONFIG_SECTION}.intervention",
        ).copy()
        if "checkpoint" in module:
            intervention_cfg["checkpoint"] = module["checkpoint"]
        for key in (
            "intervene_checkpoint",
            "checkpoint_path",
            "checkpoint_dir",
            "reconstruction_checkpoint",
            "layer",
            "layer_idx",
            "sae_type",
            "sae_device",
            "device",
            *FLAT_FEATURE_KEYS,
            *GROUP_KEYS,
            *STRENGTH_KEYS,
            "base",
            "steer_base",
            "intervention_base",
        ):
            if key in module:
                intervention_cfg[key] = module[key]
        entries = [intervention_cfg]

    if not isinstance(entries, list) or not entries:
        raise ValueError(
            f"{CONFIG_SECTION}.interventions must be a non-empty list, or set "
            f"{CONFIG_SECTION}.intervene_checkpoint plus layer, features, and an alpha "
            "or magnitude."
        )

    interventions: list[SteeringInterventionConfig] = []
    for idx, entry in enumerate(entries):
        item = _mapping(entry, f"{CONFIG_SECTION}.interventions[{idx}]")
        interventions.append(
            _intervention_from_item(
                item,
                context=f"{CONFIG_SECTION}.interventions[{idx}]",
                inherited_sae_type=inherited_sae_type,
                inherited_sae_device=inherited_sae_device,
                inherited_groups=inherited_groups,
                inherited_features=inherited_features,
                inherited_alpha=inherited_alpha,
                inherited_scaling=inherited_scaling,
                inherited_base=inherited_base,
            )
        )
    return interventions


def load_eval_config(path: str | Path | None = None) -> SAESteerEvalConfig:
    path = _project_path(path or DEFAULT_CONFIG)
    raw = reconstruction._load_yaml(path)
    module = _model_steer_config(raw)
    if not module:
        raise ValueError(f"Config section '{CONFIG_SECTION}' is required.")

    interventions = _load_interventions(module)
    steer_model_name = _resolve_steer_model_name(
        _configured_steer_model_name(raw),
        interventions,
    )
    base_cfg = cell_eval.load_eval_config(
        path,
        mode_override="reconstruct",
        output_name_override=steer_model_name,
    )
    base_cfg.mode = "steer"
    base_cfg.output_dir = _steer_output_dir(raw, steer_model_name)
    output = _mapping(module.get("output"), f"{CONFIG_SECTION}.output")
    return SAESteerEvalConfig(
        cell_type=base_cfg,
        interventions=interventions,
        evaluation=_load_evaluation_config(module),
        output_dir=base_cfg.output_dir,
        save_predictions=bool(output.get("save_predictions", base_cfg.save_predictions)),
        include_prompt=bool(output.get("include_prompt", base_cfg.include_prompt)),
        steer_model_name=steer_model_name,
        output_tag=output.get("tag"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate cell-type annotations after rescaling selected SAE "
            "features inside a transformer layer's gene-token hidden states. "
            "Feature groups are steered independently, so one set can go up "
            "while another goes down."
        )
    )
    parser.add_argument(
        "--config",
        default=None,
        help=f"YAML config path (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument("--model", default=None, help="Override model.base_model.")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Override cell_type_annotation.data.c2s_dataset_dir.",
    )
    parser.add_argument("--split", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--prompt-template", default=None)
    parser.add_argument(
        "--acceptable-answers",
        default=None,
        help=(
            "Override cell_type_annotation.scoring.acceptable_answers_path. "
            "Use an empty string to disable."
        ),
    )
    parser.add_argument(
        "--intervene-checkpoint",
        "--checkpoint",
        dest="checkpoint",
        default=None,
        help="Override the intervention feature SAE checkpoint path.",
    )
    parser.add_argument(
        "--reconstruction-checkpoint",
        default=None,
        help="Override the reconstruction SAE checkpoint path independently of features.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Override the layer index.",
    )
    parser.add_argument("--sae-type", default=None)
    parser.add_argument("--sae-device", default=None)
    parser.add_argument(
        "--features",
        "--feature-indices",
        dest="features",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Override the SAE feature indices with a single group, keeping the "
            "direction of the configured group. Cannot be combined with "
            "--up-features/--down-features."
        ),
    )
    parser.add_argument(
        "--up-features",
        dest="up_features",
        type=int,
        nargs="+",
        default=None,
        help="Replace the groups with an up-steered set of these features.",
    )
    parser.add_argument(
        "--down-features",
        dest="down_features",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Replace the groups with a down-steered set of these features. "
            "Combine with --up-features to steer two sets in opposite "
            "directions in one pass."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Fraction of each feature's own activation to add (up) or subtract "
            "(down) at every steered token, applied to every group. 1.0 is "
            "100%%. One value covers every feature; pass one value per feature "
            "to scale them differently, in the order they are listed."
        ),
    )
    parser.add_argument(
        "--up-alpha",
        dest="up_alpha",
        type=float,
        nargs="+",
        default=None,
        help="Alpha for the up-steered groups only; overrides --alpha there.",
    )
    parser.add_argument(
        "--down-alpha",
        dest="down_alpha",
        type=float,
        nargs="+",
        default=None,
        help="Alpha for the down-steered groups only; overrides --alpha there.",
    )
    parser.add_argument(
        "--magnitude",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Steer additively instead: add (up) or subtract (down) this fixed "
            "amount at every steered token whether or not the feature fires, "
            "in activation units. One value covers every feature; pass one "
            "value per feature to push each by its own amount, in the order "
            "they are listed. Implies --scaling additive."
        ),
    )
    parser.add_argument(
        "--up-magnitude",
        dest="up_magnitude",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Additive magnitude for the up-steered groups only; overrides "
            "--magnitude there."
        ),
    )
    parser.add_argument(
        "--down-magnitude",
        dest="down_magnitude",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Additive magnitude for the down-steered groups only; overrides "
            "--magnitude there."
        ),
    )
    parser.add_argument(
        "--scaling",
        default=None,
        help=(
            "How the configured numbers are applied: 'fraction' scales each "
            "feature's own activation at each token, 'additive' adds a fixed "
            "magnitude at every steered token. Applies to every group."
        ),
    )
    parser.add_argument(
        "--up-scaling",
        dest="up_scaling",
        default=None,
        help="Scaling for the up-steered groups only; overrides --scaling there.",
    )
    parser.add_argument(
        "--down-scaling",
        dest="down_scaling",
        default=None,
        help=(
            "Scaling for the down-steered groups only; overrides --scaling "
            "there."
        ),
    )
    parser.add_argument(
        "--steer-base",
        dest="steer_base",
        default=None,
        help=(
            "Override the steering model: 'activations' adds the scaled "
            "feature directions to the base-model hidden state, "
            "'reconstruction' writes the SAE reconstruction carrying the same "
            "rescaling."
        ),
    )
    parser.add_argument(
        "--baseline",
        dest="baseline_mode",
        default=None,
        help=(
            "Override model.steer.evaluation.baseline: auto (match the "
            "steering base), base_model, or reconstruction."
        ),
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip the unsteered generation pass.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-tag", default=None)
    parser.add_argument("--no-save-predictions", action="store_true")
    return parser.parse_args()


def _override_feature_groups(
    groups: list[FeatureGroupConfig],
    args: argparse.Namespace,
) -> list[FeatureGroupConfig]:
    """Apply the feature and alpha flags to one intervention's groups.

    ``--up-features``/``--down-features`` replace the group list outright, which
    is how a bidirectional run is set up from the command line. ``--features``
    keeps the single-set behavior. Strength flags then apply on top: ``--alpha``
    or ``--magnitude`` to every group, and the ``--up-``/``--down-`` forms to
    the matching direction only. ``--alpha`` keeps the fractional scaling and
    ``--magnitude`` switches to additive; each takes one value for the whole
    group or one per feature. ``--scaling`` overrides the scaling on its own.
    """

    if args.features is not None and (
        args.up_features is not None or args.down_features is not None
    ):
        raise ValueError(
            "--features sets a single group; use --up-features and/or "
            "--down-features instead of combining the two forms."
        )

    def flag(per_direction: dict[str, Any], shared: Any, direction: str) -> Any:
        value = per_direction[direction]
        return shared if value is None else value

    def scalar(value: Any) -> Any:
        """A one-element list is a group-wide number, not a per-feature one."""

        if isinstance(value, list) and len(value) == 1:
            return value[0]
        return value

    def configured(direction: str) -> FeatureGroupConfig | None:
        """The group whose strength a rebuilt group of this direction reuses."""

        for group in groups:
            if _resolve_direction(group.direction) == direction:
                return group
        return groups[0] if groups else None

    def refit(alpha: Any, num_features: int, context: str) -> Any:
        """Carry a configured strength over to a different feature count."""

        if isinstance(alpha, (list, tuple)) and len(alpha) != num_features:
            uniform = _uniform_magnitude(alpha)
            if uniform is None:
                raise ValueError(
                    f"{context} configures {len(alpha)} per-feature magnitudes, "
                    f"but {num_features} features were given on the command "
                    "line. Pass --alpha/--magnitude with one value per feature."
                )
            return uniform
        return alpha

    if args.up_features is not None or args.down_features is not None:
        rebuilt = []
        for direction, features in (
            ("up", args.up_features),
            ("down", args.down_features),
        ):
            if features is None:
                continue
            indices = _validate_feature_indices(features)
            template = configured(direction)
            rebuilt.append(
                FeatureGroupConfig(
                    feature_indices=indices,
                    alpha=refit(
                        template.alpha if template else 1.0,
                        len(indices),
                        f"--{direction}-features",
                    ),
                    direction=direction,
                    scaling=(
                        template.scaling if template else DEFAULT_STEERING_SCALING
                    ),
                )
            )
    elif args.features is not None:
        if not groups:
            raise ValueError(
                "--features replaces the features of the configured group, and "
                "there is none to take a direction from. Use --up-features "
                "and/or --down-features instead."
            )
        template = groups[0]
        indices = _validate_feature_indices(args.features)
        rebuilt = [
            replace(
                template,
                feature_indices=indices,
                alpha=refit(template.alpha, len(indices), "--features"),
            )
        ]
    else:
        rebuilt = list(groups)

    alpha_flags = {"up": args.up_alpha, "down": args.down_alpha}
    magnitude_flags = {"up": args.up_magnitude, "down": args.down_magnitude}
    scaling_flags = {"up": args.up_scaling, "down": args.down_scaling}
    resolved = []
    for group in rebuilt:
        direction = _resolve_direction(group.direction, group.name or "group")
        alpha_override = flag(alpha_flags, args.alpha, direction)
        magnitude_override = flag(magnitude_flags, args.magnitude, direction)
        if alpha_override is not None and magnitude_override is not None:
            raise ValueError(
                f"--alpha and --magnitude both set the {direction}-steered "
                "groups. Use --alpha for a fraction of each feature's own "
                "activation, or --magnitude for a fixed amount added at every "
                "steered token."
            )
        scaling_override = flag(scaling_flags, args.scaling, direction)
        if scaling_override is not None:
            scaling = _normalize_steering_scaling(scaling_override)
        elif magnitude_override is not None:
            scaling = "additive"
        else:
            scaling = _normalize_steering_scaling(group.scaling)

        override = (
            magnitude_override if magnitude_override is not None else alpha_override
        )
        alpha = group.alpha if override is None else scalar(override)
        context = f"the {direction}-steered groups"
        if isinstance(alpha, (list, tuple)):
            # Fail here rather than once the model is on the GPU.
            alpha = _validate_magnitudes(
                alpha,
                len(group.feature_indices),
                context,
            )
        else:
            alpha = abs(_validate_alpha(alpha))
        resolved.append(replace(group, alpha=alpha, scaling=scaling))
    return resolved


def apply_cli_overrides(cfg: SAESteerEvalConfig, args: argparse.Namespace) -> None:
    cell_eval.apply_cli_overrides(cfg.cell_type, args)

    if args.output_dir is not None:
        cfg.output_dir = _project_path(args.output_dir)
    if args.output_tag is not None:
        cfg.output_tag = args.output_tag
    if args.no_save_predictions:
        cfg.save_predictions = False
    if args.no_baseline:
        cfg.evaluation.baseline_generation = False
    if args.baseline_mode is not None:
        cfg.evaluation.baseline_mode = _normalize_baseline_mode(args.baseline_mode)

    values = (
        args.checkpoint,
        getattr(args, "reconstruction_checkpoint", None),
        args.layer,
        args.sae_type,
        args.sae_device,
        args.features,
        args.up_features,
        args.down_features,
        args.alpha,
        args.up_alpha,
        args.down_alpha,
        args.magnitude,
        args.up_magnitude,
        args.down_magnitude,
        args.scaling,
        args.up_scaling,
        args.down_scaling,
        args.steer_base,
    )
    if any(value is not None for value in values):
        first = cfg.interventions[0]
        cfg.interventions = [
            SteeringInterventionConfig(
                layer_idx=args.layer if args.layer is not None else first.layer_idx,
                checkpoint_path=_project_path(args.checkpoint)
                if args.checkpoint is not None
                else first.checkpoint_path,
                reconstruction_checkpoint_path=(
                    _project_path(args.reconstruction_checkpoint)
                    if getattr(args, "reconstruction_checkpoint", None) is not None
                    else first.reconstruction_checkpoint_path
                ),
                reconstruction_sae_type=(
                    None if getattr(args, "reconstruction_checkpoint", None) is not None
                    else first.reconstruction_sae_type
                ),
                groups=_override_feature_groups(first.groups, args),
                sae_type=args.sae_type if args.sae_type is not None else first.sae_type,
                sae_device=args.sae_device
                if args.sae_device is not None
                else first.sae_device,
                base=_normalize_steering_base(args.steer_base)
                if args.steer_base is not None
                else first.base,
            )
        ]

    # A base switched on the command line is a different model, so move the
    # output with it rather than overwriting the other model's results.
    if args.output_dir is None:
        resolved_name = _resolve_steer_model_name(
            cfg.steer_model_name,
            cfg.interventions,
        )
        if resolved_name != cfg.steer_model_name:
            if cfg.output_dir.name == cfg.steer_model_name:
                cfg.output_dir = cfg.output_dir.parent / resolved_name
            cfg.cell_type.output_dir = cfg.output_dir
            cfg.steer_model_name = resolved_name


def make_inference_config(cfg: SAESteerEvalConfig) -> InferenceConfig:
    return cell_eval.make_inference_config(cfg.cell_type)


def _checkpoint_tag(path: Path) -> str:
    return reconstruction._checkpoint_tag(path)


def _intervention_tag(intervention: SteeringInterventionConfig) -> str:
    tag = f"layer{intervention.layer_idx}_{_checkpoint_tag(intervention.checkpoint_path)}"
    if intervention.base == "reconstruction":
        tag = f"{tag}_recon"
    return tag


def _group_metadata(group: FeatureGroupConfig) -> dict[str, Any]:
    direction = _resolve_direction(group.direction, group.name or "steering group")
    scaling = _normalize_steering_scaling(group.scaling)
    context = group.name or f"{direction} group"
    num_features = len(group.feature_indices)
    metadata = {
        "name": group.name,
        "direction": direction,
        "feature_indices": list(group.feature_indices),
        "scaling": scaling,
        "scaling_description": SCALING_LABELS[scaling],
        "magnitudes": _validate_magnitudes(group.alpha, num_features, context),
        "effective_coefficients": _effective_coefficients(
            group.alpha,
            direction,
            num_features,
            context,
        ),
    }
    uniform = _uniform_magnitude(group.alpha)
    if uniform is not None:
        # Keep the scalar shape earlier runs recorded, under the key that says
        # what the number means: a fraction of the activation, or a magnitude.
        key = "alpha_fraction" if scaling == "fraction" else "magnitude"
        metadata[key] = uniform
        metadata[f"effective_{key}"] = _effective_alpha(uniform, direction)
    return metadata


def _intervention_type(base: str, scalings: list[str]) -> str | list[str]:
    """What one intervention writes, as the run metadata records it."""

    types = [
        (INTERVENTION_TYPES if scaling == "fraction" else ADDITIVE_INTERVENTION_TYPES)[
            base
        ]
        for scaling in (scalings or [DEFAULT_STEERING_SCALING])
    ]
    return types[0] if len(types) == 1 else types


def _intervention_types(cfg: SAESteerEvalConfig) -> list[str]:
    """Every distinct base/scaling pair the run installs, in a stable order."""

    types: list[str] = []
    for intervention in cfg.interventions:
        scalings = sorted(
            {
                _normalize_steering_scaling(group.scaling)
                for group in intervention.groups
            }
        )
        entry = _intervention_type(intervention.base, scalings)
        for value in entry if isinstance(entry, list) else [entry]:
            if value not in types:
                types.append(value)
    return types


def _intervention_scaling_metadata(
    intervention: SteeringInterventionConfig,
) -> dict[str, Any]:
    groups = [_group_metadata(group) for group in intervention.groups]
    directions = sorted({group["direction"] for group in groups})
    scalings = sorted({group["scaling"] for group in groups})
    metadata = {
        "groups": groups,
        "steering_directions": directions,
        "bidirectional": len(directions) > 1,
        "steering_scalings": scalings,
        "scaling": (
            SCALING_LABELS[scalings[0]]
            if len(scalings) == 1
            else [SCALING_LABELS[scaling] for scaling in scalings]
        ),
        "steering_base": intervention.base,
        "intervention_type": _intervention_type(intervention.base, scalings),
        "replaces_hidden_with_sae_reconstruction": (
            intervention.base == "reconstruction"
        ),
    }
    if len(groups) == 1:
        # Keep the single-set shape earlier runs recorded. A per-feature group
        # has no single number, and an additive one records a magnitude rather
        # than a fraction, so only copy the keys the group actually carries.
        for key in ("alpha_fraction", "magnitude"):
            if key in groups[0]:
                metadata[key] = groups[0][key]
                metadata[f"effective_{key}"] = groups[0][f"effective_{key}"]
    return metadata


def _is_bidirectional(cfg: SAESteerEvalConfig) -> bool:
    directions = {
        _resolve_direction(group.direction)
        for intervention in cfg.interventions
        for group in intervention.groups
    }
    return len(directions) > 1


def _steering_bases(cfg: SAESteerEvalConfig) -> list[str]:
    return sorted({intervention.base for intervention in cfg.interventions})


def _steering_scalings(cfg: SAESteerEvalConfig) -> list[str]:
    return sorted(
        {
            _normalize_steering_scaling(group.scaling)
            for intervention in cfg.interventions
            for group in intervention.groups
        }
    )


def _steer_model_metadata_card(
    cfg: SAESteerEvalConfig,
    inf_cfg: InferenceConfig,
) -> dict[str, Any]:
    bases = _steering_bases(cfg)
    scalings = _steering_scalings(cfg)
    types = _intervention_types(cfg)
    baseline_mode = _resolve_baseline_mode(cfg)
    return {
        "task": "cell_type_annotation",
        "mode": "steer",
        "steering_bases": bases,
        "steering_scalings": scalings,
        "bidirectional": _is_bidirectional(cfg),
        "intervention_type": types[0] if len(types) == 1 else types,
        "base_model_id": cfg.cell_type.model_id,
        "base_model_short_name": inf_cfg.model_short_name,
        "base_model_family": inf_cfg.family,
        "steer_model_name": cfg.steer_model_name,
        "evaluation": {
            "type": "generation",
            "baseline_generation": cfg.evaluation.baseline_generation,
            "baseline_mode": (
                baseline_mode if cfg.evaluation.baseline_generation else None
            ),
            "baseline_definition": (
                BASELINE_DEFINITIONS[baseline_mode]
                if cfg.evaluation.baseline_generation
                else None
            ),
            "max_new_tokens": cfg.cell_type.max_new_tokens,
            "temperature": cfg.cell_type.temperature,
        },
        "interventions": [
            {
                "name": _intervention_tag(intervention),
                "checkpoint_path": str(intervention.checkpoint_path),
                "checkpoint_metadata_path": str(
                    intervention.checkpoint_path.parent
                    / cell_eval.MODEL_METADATA_FILENAME
                ),
                "layer_idx": intervention.layer_idx,
                "sae_type": intervention.sae_type,
                "reconstruction_checkpoint_path": (
                    str(intervention.effective_reconstruction_checkpoint_path)
                    if intervention.base == "reconstruction" else None
                ),
                "reconstruction_sae_type": intervention.effective_reconstruction_sae_type,
                "feature_indices": intervention.feature_indices,
                **_intervention_scaling_metadata(intervention),
            }
            for intervention in cfg.interventions
        ],
    }


def load_steering_hooks(
    model,
    cfg: SAESteerEvalConfig,
    inf_cfg: InferenceConfig,
) -> tuple[list[SAESteeringHook], list[dict[str, Any]]]:
    hooks: list[SAESteeringHook] = []
    metadata: list[dict[str, Any]] = []
    expected_d_model = inf_cfg.arch.get("d_model") if inf_cfg.arch else None

    for idx, intervention in enumerate(cfg.interventions):
        sae_device = reconstruction._resolve_sae_device(
            intervention.sae_device,
            inf_cfg,
        )
        checkpoint_model = reconstruction._checkpoint_model_metadata(
            intervention.checkpoint_path
        )
        checkpoint_layer = checkpoint_model.get("layer")
        checkpoint_pooling_method = checkpoint_model.get("pooling_method")
        if (
            checkpoint_layer is not None
            and int(checkpoint_layer) != intervention.layer_idx
        ):
            print(
                "[Warning] SAE checkpoint metadata says layer "
                f"{checkpoint_layer}, but intervention is configured for "
                f"layer {intervention.layer_idx}."
            )
        checkpoint_prompt_prefix = checkpoint_model.get("prompt_prefix")
        if (
            isinstance(checkpoint_prompt_prefix, bool)
            and checkpoint_prompt_prefix != cfg.cell_type.prompt_prefix
        ):
            print(
                "[Warning] SAE checkpoint metadata says prompt_prefix="
                f"{checkpoint_prompt_prefix}, but evaluation prompt_prefix="
                f"{cfg.cell_type.prompt_prefix}."
            )

        sae, sae_meta = load_sae(
            intervention.checkpoint_path,
            intervention.sae_type,
            device=sae_device,
        )
        if expected_d_model is not None and sae_meta["d_model"] != expected_d_model:
            raise ValueError(
                f"Checkpoint {intervention.checkpoint_path} has d_model="
                f"{sae_meta['d_model']}, but {inf_cfg.model_id} expects "
                f"d_model={expected_d_model}."
            )
        steered_groups = _normalize_steered_groups(
            [_steered_group(group) for group in intervention.groups],
            sae_meta["d_hidden"],
        )
        reconstruction_sae = None
        reconstruction_metadata = None
        pooling_method = checkpoint_pooling_method
        if intervention.base == "reconstruction":
            recon_path = intervention.effective_reconstruction_checkpoint_path
            recon_type = intervention.effective_reconstruction_sae_type
            recon_model = reconstruction._checkpoint_model_metadata(recon_path)
            if (
                recon_path == intervention.checkpoint_path
                and recon_type == intervention.sae_type
            ):
                reconstruction_sae, recon_meta = sae, sae_meta
            else:
                reconstruction_sae, recon_meta = load_sae(
                    recon_path, recon_type, device=sae_device,
                )
            if recon_meta["d_model"] != sae_meta["d_model"]:
                raise ValueError(
                    f"Reconstruction checkpoint {recon_path} has d_model="
                    f"{recon_meta['d_model']}, but the feature SAE has "
                    f"d_model={sae_meta['d_model']}."
                )
            for key, expected in (
                ("layer", intervention.layer_idx),
                ("prompt_prefix", cfg.cell_type.prompt_prefix),
            ):
                actual = recon_model.get(key)
                if actual is not None and actual != expected:
                    print(
                        f"[Warning] Reconstruction SAE checkpoint metadata says "
                        f"{key}={actual}, but evaluation uses {expected}."
                    )
            # Token selection follows the reconstruction being evaluated, even
            # when feature directions come from a differently pooled SAE.
            pooling_method = recon_model.get("pooling_method")
            reconstruction_metadata = {
                **recon_meta,
                "checkpoint_path": str(recon_path),
                "checkpoint_metadata_path": str(
                    recon_path.parent / cell_eval.MODEL_METADATA_FILENAME
                ),
                "configured_sae_type": recon_type,
                "checkpoint_layer": recon_model.get("layer"),
                "checkpoint_pooling_method": pooling_method,
                "checkpoint_prompt_prefix": recon_model.get("prompt_prefix"),
            }
        feature_indices = [
            feature for group in steered_groups for feature in group.feature_indices
        ]

        name = _intervention_tag(intervention)
        hooks.append(
            SAESteeringHook(
                model,
                intervention.layer_idx,
                sae,
                name=name,
                groups=steered_groups,
                base=intervention.base,
                reconstruction_sae=reconstruction_sae,
                expected_d_model=sae_meta["d_model"],
                pooling_method=str(pooling_method)
                if pooling_method
                else None,
            )
        )
        meta = dict(sae_meta)
        meta.update(
            {
                "name": name,
                "layer_idx": intervention.layer_idx,
                "configured_sae_type": intervention.sae_type,
                "reconstruction_checkpoint": reconstruction_metadata,
                "sae_device": str(sae_device),
                "feature_indices": feature_indices,
                **_intervention_scaling_metadata(intervention),
                "steered_token_positions": (
                    "gene tokens selected by the reconstruction checkpoint pooling "
                    "method (feature checkpoint for activations base), "
                    "prompt prefill only"
                ),
                "order": idx,
                "checkpoint_layer": checkpoint_layer,
                "checkpoint_pooling_method": checkpoint_pooling_method,
                "checkpoint_prompt_prefix": checkpoint_prompt_prefix,
            }
        )
        metadata.append(meta)

    return hooks, metadata


def mentions_class(
    label: str,
    generated_text: str,
    acceptable_answers: dict[str, list[str]] | None = None,
) -> bool:
    """Return true when generated text names the label or one of its aliases."""

    return (
        cell_eval.find_encoded_answer(label, generated_text, acceptable_answers)
        is not None
    )


def predicted_class_from_text(
    generated_text: str,
    classes: list[str],
    acceptable_answers: dict[str, list[str]] | None = None,
) -> str:
    """Assign generated text to the most specific dataset class it names."""

    best_label = NONE_CLASS
    best_specificity = -1
    for label in classes:
        if not mentions_class(label, generated_text, acceptable_answers):
            continue
        specificity = len(cell_eval._label_tokens(label))
        if specificity > best_specificity:
            best_label = label
            best_specificity = specificity
    return best_label


def _generate_for_sample(
    sample: dict[str, Any],
    *,
    cfg: SAESteerEvalConfig,
    inf_cfg: InferenceConfig,
    tokenizer,
    model,
    steering_hooks: list[SAESteeringHook] | None = None,
) -> tuple[str, str]:
    """Generate an answer for one cell, optionally with steering installed."""

    prompt = cell_eval.build_prompt(sample, cfg.cell_type)
    hooks = steering_hooks or []
    if hooks:
        reconstruction._prepare_hooks_for_prompt(
            hooks,
            tokenizer,
            prompt,
            str(sample["cell_sentence"]),
        )
    try:
        generated_text = generate(prompt, tokenizer, model, inf_cfg)
    finally:
        if hooks:
            reconstruction._clear_hook_prompt_state(hooks)
    return generated_text, prompt


def _generate_unsteered_with_hooks_installed(
    sample: dict[str, Any],
    *,
    cfg: SAESteerEvalConfig,
    inf_cfg: InferenceConfig,
    tokenizer,
    model,
    hooks: list[SAESteeringHook],
) -> tuple[str, str]:
    """Generate the base-model answer while the steering hooks stay registered.

    A registered hook is inert until ``_prepare_hooks_for_prompt`` arms it, so
    skipping that call leaves the base model untouched. The call counters are
    compared before and after to guarantee that.
    """

    calls_before = [hook.num_calls for hook in hooks]
    generated_text, prompt = _generate_for_sample(
        sample,
        cfg=cfg,
        inf_cfg=inf_cfg,
        tokenizer=tokenizer,
        model=model,
    )
    if [hook.num_calls for hook in hooks] != calls_before:
        raise RuntimeError(
            "Steering hooks fired during the unsteered generation; the "
            "baseline would not be comparable."
        )
    return generated_text, prompt


def _generate_reconstruction_baseline(
    sample: dict[str, Any],
    *,
    cfg: SAESteerEvalConfig,
    inf_cfg: InferenceConfig,
    tokenizer,
    model,
    hooks: list[SAESteeringHook],
) -> tuple[str, str]:
    """Generate the matched baseline for reconstruction-base steering.

    The hooks are armed the same way as for the steered pass, but write the
    plain SAE reconstruction with no feature rescaled. The difference between
    this and the steered generation is therefore the steering alone, with the
    SAE's reconstruction error present on both sides.
    """

    calls_before = [hook.num_calls for hook in hooks]
    for hook in hooks:
        hook.baseline_pass = True
    try:
        generated_text, prompt = _generate_for_sample(
            sample,
            cfg=cfg,
            inf_cfg=inf_cfg,
            tokenizer=tokenizer,
            model=model,
            steering_hooks=hooks,
        )
    finally:
        for hook in hooks:
            hook.baseline_pass = False
    if [hook.num_calls for hook in hooks] != calls_before:
        raise RuntimeError(
            "The baseline pass advanced the steered hook counters; the "
            "reconstruction baseline would not be comparable."
        )
    return generated_text, prompt


def _generate_baseline(
    sample: dict[str, Any],
    *,
    cfg: SAESteerEvalConfig,
    inf_cfg: InferenceConfig,
    tokenizer,
    model,
    hooks: list[SAESteeringHook],
    baseline_mode: str,
) -> tuple[str, str]:
    generator = (
        _generate_reconstruction_baseline
        if baseline_mode == "reconstruction"
        else _generate_unsteered_with_hooks_installed
    )
    return generator(
        sample,
        cfg=cfg,
        inf_cfg=inf_cfg,
        tokenizer=tokenizer,
        model=model,
        hooks=hooks,
    )


STALE_OUTPUT_FILENAMES = ("incorrect_predictions.jsonl",)


def _output_paths(
    cfg: SAESteerEvalConfig,
    inf_cfg: InferenceConfig,
) -> tuple[Path, Path, Path]:
    """Return the predictions, changed-predictions, and summary paths."""

    out_dir = (
        cfg.output_dir
        / cfg.cell_type.c2s_dataset_dir.name
        / inf_cfg.model_short_name
        / f"data_{cfg.cell_type.split}"
    )
    if cfg.output_tag:
        out_dir = out_dir / (_OUTPUT_SAFE_RE.sub("_", cfg.output_tag).strip("_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename in STALE_OUTPUT_FILENAMES:
        stale = out_dir / filename
        if stale.exists():
            stale.unlink()
    return (
        out_dir / "predictions.jsonl",
        out_dir / "changed_predictions.jsonl",
        out_dir / "summary.json",
    )


def _collapse_whitespace(text: str) -> str:
    return " ".join(str(text).split())


def _write_comparison_row(handle, values: list[Any]) -> None:
    handle.write("\t".join(_collapse_whitespace(value) for value in values) + "\n")


def run_eval(cfg: SAESteerEvalConfig) -> dict[str, Any]:
    cell_eval.validate_inputs(cfg.cell_type)
    split_dir = cfg.cell_type.c2s_dataset_dir / f"data_{cfg.cell_type.split}"
    dataset = load_from_disk(str(split_dir))

    required = {"cell_sentence", cfg.cell_type.cell_type_column}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"{split_dir} is missing required column(s): {sorted(missing)}")

    classes = sorted(
        set(str(label) for label in dataset[cfg.cell_type.cell_type_column])
    )
    baseline_mode = _resolve_baseline_mode(cfg)
    total_available = len(dataset)
    n_examples = (
        min(cfg.cell_type.max_examples, total_available)
        if cfg.cell_type.max_examples
        else total_available
    )
    inf_cfg = make_inference_config(cfg)
    predictions_path, changed_predictions_path, summary_path = _output_paths(
        cfg,
        inf_cfg,
    )
    comparison_path = summary_path.parent / "comparison.tsv"
    model_metadata_path = cell_eval.write_model_metadata_card(
        summary_path.parent,
        _steer_model_metadata_card(cfg, inf_cfg),
    )
    acceptable_answers = cell_eval.load_acceptable_answers(
        cfg.cell_type.acceptable_answers_path
    )

    print("[Config]")
    print(f"  steer_model: {cfg.steer_model_name}")
    print(f"  model     : {cfg.cell_type.model_id}")
    print(f"  device    : {cfg.cell_type.device}")
    print(f"  dtype     : {cfg.cell_type.dtype}")
    print(f"  dataset   : {cfg.cell_type.c2s_dataset_dir}")
    print(f"  split     : data_{cfg.cell_type.split} ({total_available:,} available)")
    print(f"  examples  : {n_examples:,}")
    print(
        f"  prompt    : "
        f"{'template' if cfg.cell_type.prompt_prefix else 'bare cell_sentence'}"
    )
    if cfg.cell_type.prompt_prefix:
        print(f"  prompt tpl: {cfg.cell_type.prompt_template_path}")
    print(
        f"  generation: max_new_tokens={cfg.cell_type.max_new_tokens}, "
        f"temperature={cfg.cell_type.temperature}"
    )
    print(
        "  baseline  : "
        + (
            BASELINE_LABELS[baseline_mode]
            if cfg.evaluation.baseline_generation
            else "none"
        )
    )
    if cfg.cell_type.acceptable_answers_path:
        print(
            f"  answers   : {cfg.cell_type.acceptable_answers_path} "
            f"({len(acceptable_answers):,} label entries)"
        )
    else:
        print("  answers   : disabled")
    print("  interventions:")
    for intervention in cfg.interventions:
        print(
            f"    - layer {intervention.layer_idx}: base={intervention.base}, "
            f"intervene_checkpoint={intervention.checkpoint_path}"
        )
        if intervention.base == "reconstruction":
            print(
                "      reconstruction_checkpoint="
                f"{intervention.effective_reconstruction_checkpoint_path}"
            )
        for group in intervention.groups:
            direction = _resolve_direction(group.direction, group.name or "group")
            scaling = _normalize_steering_scaling(group.scaling)
            coefficients = _effective_coefficients(
                group.alpha,
                direction,
                len(group.feature_indices),
                group.name or f"{direction} group",
            )
            formatted = [
                _format_steer_coefficient(coefficient, scaling)
                for coefficient in coefficients
            ]
            amount = (
                formatted[0]
                if len(set(formatted)) == 1
                else "[" + ", ".join(formatted) + "]"
            )
            key = (
                "activation_fraction"
                if scaling == "fraction"
                else "added_magnitude"
            )
            label = f" {group.name}" if group.name else ""
            print(
                f"        {direction:<4}{label}: "
                f"features={group.feature_indices}, "
                f"{key}={amount}"
            )

    tokenizer, model = load_model(inf_cfg)
    hooks, intervention_meta = load_steering_hooks(model, cfg, inf_cfg)
    compare = cfg.evaluation.baseline_generation

    baseline_gold_correct = 0
    steered_gold_correct = 0
    changed_generation = 0
    baseline_class_counts: Counter[str] = Counter()
    steered_class_counts: Counter[str] = Counter()
    transition_counts: Counter[str] = Counter()
    per_cell_type: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "correct": 0}
    )
    prediction_writer_context = (
        open(predictions_path, "w", encoding="utf-8")
        if cfg.save_predictions
        else nullcontext(None)
    )
    changed_writer_context = (
        open(changed_predictions_path, "w", encoding="utf-8")
        if cfg.save_predictions and compare
        else nullcontext(None)
    )
    comparison_writer_context = (
        open(comparison_path, "w", encoding="utf-8")
        if cfg.save_predictions and compare
        else nullcontext(None)
    )
    changed_transitions: Counter[tuple[str, str, str]] = Counter()

    with (
        SAESteeringPatcher(hooks),
        prediction_writer_context as writer,
        changed_writer_context as changed_writer,
        comparison_writer_context as comparison_writer,
    ):
        if comparison_writer is not None:
            _write_comparison_row(
                comparison_writer,
                [
                    "index",
                    "gold_cell_type",
                    "unsteered_text",
                    "steered_text",
                    "changed",
                    "unsteered_class",
                    "steered_class",
                ],
            )
        desc = (
            f"Generating baseline ({BASELINE_LABELS[baseline_mode]}) vs "
            "steered answers"
            if compare
            else "Generating steered answers"
        )
        for idx in tqdm(range(n_examples), desc=desc):
            sample = dataset[idx]
            gold = str(sample[cfg.cell_type.cell_type_column])
            baseline_text = (
                _generate_baseline(
                    sample,
                    cfg=cfg,
                    inf_cfg=inf_cfg,
                    tokenizer=tokenizer,
                    model=model,
                    hooks=hooks,
                    baseline_mode=baseline_mode,
                )[0]
                if compare
                else None
            )
            generated_text, prompt = _generate_for_sample(
                sample,
                cfg=cfg,
                inf_cfg=inf_cfg,
                tokenizer=tokenizer,
                model=model,
                steering_hooks=hooks,
            )
            matched_answer = cell_eval.find_encoded_answer(
                gold,
                generated_text,
                acceptable_answers,
            )
            predicted_class = predicted_class_from_text(
                generated_text,
                classes,
                acceptable_answers,
            )

            steered_gold_correct += int(matched_answer is not None)
            steered_class_counts[predicted_class] += 1
            per_cell_type[gold]["total"] += 1
            per_cell_type[gold]["correct"] += int(matched_answer is not None)

            row = {
                "index": idx,
                "cell_name": sample.get("cell_name"),
                "cell_type": gold,
                "source_cell_type": gold,
                "generated_text": generated_text,
                "predicted_class": predicted_class,
                "matched_answer": matched_answer,
                "gold_correct": matched_answer is not None,
                "correct": matched_answer is not None,
            }

            if baseline_text is not None:
                baseline_matched = cell_eval.find_encoded_answer(
                    gold,
                    baseline_text,
                    acceptable_answers,
                )
                baseline_class = predicted_class_from_text(
                    baseline_text,
                    classes,
                    acceptable_answers,
                )
                changed = baseline_text != generated_text
                baseline_gold_correct += int(baseline_matched is not None)
                baseline_class_counts[baseline_class] += 1
                transition_counts[f"{baseline_class} -> {predicted_class}"] += 1
                changed_generation += int(changed)
                row.update(
                    {
                        "baseline_text": baseline_text,
                        "baseline_class": baseline_class,
                        "baseline_matched_answer": baseline_matched,
                        "baseline_gold_correct": baseline_matched is not None,
                        "generation_changed": changed,
                    }
                )
                if changed:
                    changed_transitions[
                        (
                            gold,
                            _collapse_whitespace(baseline_text),
                            _collapse_whitespace(generated_text),
                        )
                    ] += 1
                if comparison_writer is not None:
                    _write_comparison_row(
                        comparison_writer,
                        [
                            str(idx),
                            gold,
                            baseline_text,
                            generated_text,
                            "yes" if changed else "no",
                            baseline_class,
                            predicted_class,
                        ],
                    )

            if cfg.include_prompt:
                row["prompt"] = prompt
            if writer is not None:
                writer.write(json.dumps(row) + "\n")
            if changed_writer is not None and row.get("generation_changed"):
                changed_writer.write(json.dumps(row) + "\n")

    by_cell_type = {
        cell_type: {
            "total": stats["total"],
            "correct": stats["correct"],
            "accuracy": reconstruction._safe_accuracy(
                stats["correct"],
                stats["total"],
            ),
        }
        for cell_type, stats in sorted(per_cell_type.items())
    }
    for meta, hook in zip(intervention_meta, hooks):
        meta["hook_calls"] = hook.num_calls
        meta["hook_tokens"] = hook.num_tokens
        meta["baseline_hook_calls"] = hook.num_baseline_calls
        meta["baseline_hook_tokens"] = hook.num_baseline_tokens
        meta["diagnostics"] = hook.diagnostics()

    steered_gold_accuracy = reconstruction._safe_accuracy(
        steered_gold_correct,
        n_examples,
    )
    baseline_gold_accuracy = reconstruction._safe_accuracy(
        baseline_gold_correct,
        n_examples,
    )

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "cell_type_annotation",
        "mode": "steer",
        "evaluation": "generation",
        "steering_bases": _steering_bases(cfg),
        "bidirectional": _is_bidirectional(cfg),
        "steer_model_name": cfg.steer_model_name,
        "model_id": cfg.cell_type.model_id,
        "dataset_dir": str(cfg.cell_type.c2s_dataset_dir),
        "split": f"data_{cfg.cell_type.split}",
        "cell_type_column": cfg.cell_type.cell_type_column,
        "num_available": total_available,
        "num_examples": n_examples,
        "num_correct": steered_gold_correct,
        "num_incorrect": n_examples - steered_gold_correct,
        "accuracy": steered_gold_accuracy,
        "accuracy_definition": (
            "steered generation names the gold dataset label; the steering "
            "effect itself is num_generation_changed and the rows of "
            "changed_predictions.jsonl"
        ),
        "classes": classes,
        "none_class": NONE_CLASS,
        "steered_class_counts": dict(sorted(steered_class_counts.items())),
        "steered_gold_annotation_accuracy": steered_gold_accuracy,
        "prompt_prefix": cfg.cell_type.prompt_prefix,
        "prompt_template_path": str(cfg.cell_type.prompt_template_path)
        if cfg.cell_type.prompt_prefix
        else None,
        "max_new_tokens": cfg.cell_type.max_new_tokens,
        "temperature": cfg.cell_type.temperature,
        "acceptable_answers_path": str(cfg.cell_type.acceptable_answers_path)
        if cfg.cell_type.acceptable_answers_path
        else None,
        "num_acceptable_answer_entries": len(acceptable_answers),
        "label_matcher": (
            "gold label plus configured acceptable answers; normalized token sequence "
            "or all normalized label tokens with simple singular/plural variants"
        ),
        "interventions": intervention_meta,
        "model_metadata_path": str(model_metadata_path),
        "predictions_path": str(predictions_path) if cfg.save_predictions else None,
        "changed_predictions_path": str(changed_predictions_path)
        if cfg.save_predictions and compare
        else None,
        "by_cell_type": by_cell_type,
        "by_source_cell_type": by_cell_type,
    }

    if compare:
        summary.update(
            {
                "baseline_generation": True,
                "baseline_mode": baseline_mode,
                "baseline_definition": BASELINE_DEFINITIONS[baseline_mode],
                "comparison_path": str(comparison_path)
                if cfg.save_predictions
                else None,
                "baseline_class_counts": dict(sorted(baseline_class_counts.items())),
                "class_transitions": dict(sorted(transition_counts.items())),
                "baseline_gold_annotation_accuracy": baseline_gold_accuracy,
                "num_generation_changed": changed_generation,
                "generation_change_rate": reconstruction._safe_accuracy(
                    changed_generation,
                    n_examples,
                ),
            }
        )
    else:
        summary["baseline_generation"] = False
        summary["baseline_mode"] = None

    summary["figures"] = cell_eval.plot_accuracy_figures(summary, summary_path.parent)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print("\n[Results]")
    if compare:
        print(
            f"  changed   : {changed_generation:,}/{n_examples:,} steered "
            f"generations differ from their baseline "
            f"({BASELINE_LABELS[baseline_mode]}) answer"
        )
    print(f"  gold acc  : {steered_gold_accuracy:.3f} (steered vs dataset label)")
    if compare:
        print(
            f"  baseline  : {baseline_gold_accuracy:.3f} "
            f"({BASELINE_LABELS[baseline_mode]} vs dataset label)"
        )
    for meta in intervention_meta:
        diagnostics = meta.get("diagnostics") or {}
        if not diagnostics.get("steered_tokens"):
            continue
        print(
            f"\n[Intervention delivery] {meta['name']} "
            f"(base={diagnostics.get('steering_base', DEFAULT_STEERING_BASE)})"
        )
        print(
            f"  steered tokens : {diagnostics['steered_tokens']:,}  "
            f"|delta|/|h| = {diagnostics['mean_injected_to_residual_ratio']:.3%} "
            f"(|delta|={diagnostics['mean_injected_norm']:.1f}, "
            f"|h|={diagnostics['mean_residual_norm']:.1f})"
        )
        if "mean_reconstruction_to_residual_ratio" in diagnostics:
            print(
                "  reconstruction : "
                f"|recon-h|/|h| = "
                f"{diagnostics['mean_reconstruction_to_residual_ratio']:.3%} "
                f"(|recon-h|={diagnostics['mean_reconstruction_shift_norm']:.1f})"
                + (
                    "  [matched in the baseline]"
                    if baseline_mode == "reconstruction"
                    else "  [also in the measured effect]"
                )
            )
        for entry in diagnostics["per_feature"]:
            scaling = entry.get("scaling", DEFAULT_STEERING_SCALING)
            coefficient = entry.get(
                "effective_coefficient",
                entry.get("effective_alpha_fraction"),
            )
            print(
                f"    F{entry['feature_id']:<6} "
                f"{entry['direction']:<4} "
                f"{_format_steer_coefficient(coefficient, scaling):>8}  "
                f"fires on "
                f"{entry['firing_rate']:7.2%} of steered tokens, "
                f"mean act when active = "
                f"{entry['mean_activation_when_active']:.2f}"
            )
        # Only fraction scaling is silent where a feature does not fire; an
        # additive group writes its magnitude at every steered token anyway.
        if any(
            entry.get("scaling", DEFAULT_STEERING_SCALING) == "fraction"
            and entry["firing_rate"] < 0.10
            for entry in diagnostics["per_feature"]
        ):
            print(
                "    [Warning] a fraction-scaled feature fires on <10% of "
                "steered tokens. The rescaling contributes nothing where it "
                "does not fire, so a null result here is uninformative about "
                "causality. Check that the SAE's prompt_prefix matches the "
                "evaluation prompt, or steer that group with an additive "
                "magnitude instead."
            )
    if compare:
        if changed_transitions:
            changed_fraction = changed_generation / n_examples if n_examples else 0.0
            print(
                f"\n[Changed generations] {changed_generation:,}/{n_examples:,} "
                f"({changed_fraction:.1%}), "
                f"{len(changed_transitions):,} distinct transitions"
            )
            reported = changed_transitions.most_common(MAX_REPORTED_TRANSITIONS)
            gold_width = max(len(gold) for (gold, _, _), _ in reported)
            for (gold, before, after), count in reported:
                print(
                    f"  x{count:<4} {gold:<{gold_width}}  "
                    f"{before[:45]} -> {after[:45]}"
                )
            hidden = len(changed_transitions) - len(reported)
            if hidden > 0:
                print(f"  ... {hidden:,} more transition types")
        else:
            print(
                "\n[Changed generations] none - every steered generation matched "
                f"its baseline ({BASELINE_LABELS[baseline_mode]}) answer exactly."
            )
    if cfg.save_predictions:
        print(f"  generated : {predictions_path}")
        if compare:
            print(
                f"  changed   : {changed_predictions_path} "
                f"({changed_generation:,} rows)"
            )
            print(f"  comparison: {comparison_path}")
    print(f"  summary   : {summary_path}")
    print(f"  metadata  : {model_metadata_path}")
    if summary["figures"]:
        print(f"  figures   : {summary_path.parent / 'figures'}")
    return summary


def main() -> None:
    args = parse_args()
    cfg = load_eval_config(args.config)
    apply_cli_overrides(cfg, args)
    run_eval(cfg)


if __name__ == "__main__":
    main()
