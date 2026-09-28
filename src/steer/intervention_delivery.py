#!/usr/bin/env python3
"""Intervention delivery check

Usage:
    .venv/bin/python src/steer/intervention_delivery.py
    .venv/bin/python src/steer/intervention_delivery.py --max-examples 50 --prompt-prefix both
    .venv/bin/python src/steer/intervention_delivery.py --config configs/steer.yaml --alpha 1.0
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
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

from src.data.inference import load_model
from src.evaluate.downstream_tasks import sae_reconstruction_inference as reconstruction
from src.evaluate.downstream_tasks.cell_type_annotation import (
    cell_type_annotation as cell_eval,
)
from src.steer import sae_steer_inference as steer_eval

FIRING_RATE_FLOOR = 0.10
NORM_RATIO_FLOOR = 0.01
FEATURE_TABLE_NAME = "all_cell_type_features.csv"
DEFAULT_FEATURE_ROOT = PROJECT_ROOT / "results" / "cell_type_features"


def _blank_accumulator(num_features: int) -> dict[str, Any]:
    return {
        "num_tokens": 0,
        "num_cells": 0,
        "active_tokens": [0] * num_features,
        "activation_sum": [0.0] * num_features,
        "injected_norm_sum": 0.0,
        "hidden_norm_sum": 0.0,
        "norm_ratio_sum": 0.0,
        # Only the reconstruction steering base fills these in: how far the SAE
        # reconstruction alone displaces the hidden state.
        "recon_tokens": 0,
        "recon_shift_sum": 0.0,
        "recon_ratio_sum": 0.0,
    }


def _fold(target: dict[str, Any], source: dict[str, Any]) -> None:
    """Add one cell's raw hook counters into a running accumulator."""

    if not source:
        return
    target["num_tokens"] += int(source["num_tokens"])
    target["num_cells"] += 1
    target["active_tokens"] = [
        prior + int(count)
        for prior, count in zip(target["active_tokens"], source["active_tokens"])
    ]
    target["activation_sum"] = [
        prior + float(total)
        for prior, total in zip(target["activation_sum"], source["activation_sum"])
    ]
    target["injected_norm_sum"] += float(source["injected_norm_sum"])
    target["hidden_norm_sum"] += float(source["hidden_norm_sum"])
    target["norm_ratio_sum"] += float(source["norm_ratio_sum"])
    target["recon_tokens"] += int(source.get("recon_tokens", 0))
    target["recon_shift_sum"] += float(source.get("recon_shift_sum", 0.0))
    target["recon_ratio_sum"] += float(source.get("recon_ratio_sum", 0.0))


def _summarize(
    accumulator: dict[str, Any],
    feature_indices: list[int],
    feature_groups: list | None = None,
    feature_coefficients: list[float] | None = None,
) -> dict[str, Any]:
    """Reduce one accumulator to rates. ``feature_groups`` is the hook's flat
    per-feature group view, so each row can name the direction it was steered,
    and ``feature_coefficients`` is the matching flat view of the signed number
    each feature was steered by -- a fraction of its own activation under
    fraction scaling, a fixed magnitude under additive scaling.
    """

    num_tokens = accumulator["num_tokens"]
    if num_tokens == 0:
        return {"num_cells": accumulator["num_cells"], "steered_tokens": 0}

    per_feature = []
    for position, feature_id in enumerate(feature_indices):
        active = accumulator["active_tokens"][position]
        activation_total = accumulator["activation_sum"][position]
        group = feature_groups[position] if feature_groups else None
        coefficient = (
            feature_coefficients[position] if feature_coefficients else None
        )
        scaling = group.scaling if group else None
        per_feature.append(
            {
                "feature_id": feature_id,
                "group": group.label if group else None,
                "direction": group.direction if group else None,
                "scaling": scaling,
                "effective_coefficient": coefficient,
                # A fraction of the activation only under fraction scaling; an
                # additive group pushes by a magnitude instead.
                "effective_alpha_fraction": (
                    coefficient if scaling == "fraction" else None
                ),
                "firing_rate": active / num_tokens,
                "mean_activation_when_active": (
                    activation_total / active if active else 0.0
                ),
            }
        )
    rates = [entry["firing_rate"] for entry in per_feature]
    summary = {
        "num_cells": accumulator["num_cells"],
        "steered_tokens": num_tokens,
        "per_feature": per_feature,
        "min_firing_rate": min(rates),
        "mean_firing_rate": sum(rates) / len(rates),
        "mean_injected_norm": accumulator["injected_norm_sum"] / num_tokens,
        "mean_residual_norm": accumulator["hidden_norm_sum"] / num_tokens,
        "mean_injected_to_residual_ratio": accumulator["norm_ratio_sum"] / num_tokens,
    }
    recon_tokens = int(accumulator.get("recon_tokens", 0))
    if recon_tokens:
        summary["mean_reconstruction_shift_norm"] = (
            accumulator["recon_shift_sum"] / recon_tokens
        )
        summary["mean_reconstruction_to_residual_ratio"] = (
            accumulator["recon_ratio_sum"] / recon_tokens
        )
    return summary


def _verdict(summary: dict[str, Any]) -> tuple[str, str]:
    if not summary.get("steered_tokens"):
        return (
            "NOT RUN",
            "The hook never wrote. Check token selection and prompt length.",
        )
    # Only fraction-scaled features are silent where they do not fire. An
    # additive group writes its magnitude at every steered token, so a low
    # firing rate there says the feature is off in these cells, not that the
    # intervention was a no-op.
    fraction_rates = [
        entry["firing_rate"]
        for entry in summary["per_feature"]
        if (entry.get("scaling") or steer_eval.DEFAULT_STEERING_SCALING)
        == "fraction"
    ]
    if fraction_rates and min(fraction_rates) < FIRING_RATE_FLOOR:
        return (
            "NO-OP",
            "At least one fraction-scaled feature fires on under "
            f"{FIRING_RATE_FLOOR:.0%} of steered tokens, so steering is an "
            "exact no-op at most positions. A null steering result is "
            "uninformative about causality. Align the SAE's prompt_prefix "
            "with the evaluation prompt, rebuild the feature table on the "
            "steered distribution, or steer that group with an additive "
            "magnitude, before interpreting any result.",
        )
    if summary["mean_injected_to_residual_ratio"] < NORM_RATIO_FLOOR:
        return (
            "UNDERPOWERED",
            "Features fire, but the injected vector is under "
            f"{NORM_RATIO_FLOOR:.0%} of the residual norm. Run the alpha "
            "ladder before concluding anything from a null.",
        )
    return (
        "DELIVERED",
        "Features fire and the perturbation is a material fraction of the "
        "residual stream. A null output result is now a real measurement, so "
        "the matched controls and the dose-response curve are worth running.",
    )


def _load_feature_table(checkpoint_path: Path, feature_root: Path) -> dict[int, dict]:
    """Cell-level stats for the configured features, for a distribution check.

    Returns the best (highest specificity margin) row per feature id. Missing
    tables are not an error; the comparison column is simply omitted.
    """

    sae_name = checkpoint_path.parent.name
    matches = sorted(feature_root.glob(f"{sae_name}/*/{FEATURE_TABLE_NAME}"))
    if not matches:
        return {}

    best: dict[int, dict] = {}
    with open(matches[0], encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            feature_id = int(row["feature_id"])
            margin = float(row["specificity_margin"])
            if feature_id not in best or margin > best[feature_id]["margin"]:
                best[feature_id] = {
                    "margin": margin,
                    "cell_type": row["cell_type"],
                    "activation_rate": float(row["activation_rate"]),
                    "mean_activation_when_active": float(
                        row["mean_activation_when_active"]
                    ),
                }
    return best


def _run_variant(
    *,
    dataset,
    n_examples: int,
    cfg,
    inf_cfg,
    tokenizer,
    model,
    hooks,
    prompt_prefix: bool,
) -> dict[str, Any]:
    """Prefill every cell once with the hooks installed and collect counters."""

    cfg.cell_type.prompt_prefix = prompt_prefix

    overall = [_blank_accumulator(len(hook.feature_indices)) for hook in hooks]
    by_cell_type: list[dict[str, dict]] = [defaultdict(dict) for _ in hooks]

    label_column = cfg.cell_type.cell_type_column
    description = f"prompt_prefix={prompt_prefix}"
    for idx in tqdm(range(n_examples), desc=f"Prefill ({description})"):
        sample = dataset[idx]
        gold = str(sample[label_column])
        prompt = cell_eval.build_prompt(sample, cfg.cell_type)

        for hook in hooks:
            hook.stats = {}
        reconstruction._prepare_hooks_for_prompt(
            hooks,
            tokenizer,
            prompt,
            str(sample["cell_sentence"]),
        )
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(inf_cfg.device)
            with torch.inference_mode():
                model(**inputs, use_cache=False)
        finally:
            reconstruction._clear_hook_prompt_state(hooks)

        for position, hook in enumerate(hooks):
            _fold(overall[position], hook.stats)
            bucket = by_cell_type[position]
            if gold not in bucket:
                bucket[gold] = _blank_accumulator(len(hook.feature_indices))
            _fold(bucket[gold], hook.stats)

    results = []
    for position, hook in enumerate(hooks):
        summary = _summarize(
            overall[position],
            hook.feature_indices,
            hook.feature_groups,
            hook.feature_coefficients,
        )
        status, explanation = _verdict(summary)
        results.append(
            {
                "name": hook.name,
                "layer_idx": hook.layer_idx,
                "steering_base": hook.base,
                "groups": [
                    {
                        "name": group.name,
                        "direction": group.direction,
                        "scaling": group.scaling,
                        "effective_coefficients": list(group.coefficients),
                        "effective_alpha_fraction": (
                            group.uniform_coefficient
                            if group.scaling == "fraction"
                            else None
                        ),
                        "feature_indices": group.feature_indices,
                    }
                    for group in hook.groups
                ],
                "feature_indices": hook.feature_indices,
                "prompt_prefix": prompt_prefix,
                "overall": summary,
                "by_cell_type": {
                    label: _summarize(
                        accumulator,
                        hook.feature_indices,
                        hook.feature_groups,
                        hook.feature_coefficients,
                    )
                    for label, accumulator in sorted(by_cell_type[position].items())
                },
                "verdict": status,
                "verdict_detail": explanation,
            }
        )
    return {"prompt_prefix": prompt_prefix, "interventions": results}


def _format_group_strength(group: dict[str, Any]) -> str:
    """One group's steering strength, as a fraction or an added magnitude."""

    scaling = group.get("scaling") or steer_eval.DEFAULT_STEERING_SCALING
    coefficients = group.get("effective_coefficients")
    if not coefficients:
        coefficient = group.get("effective_alpha_fraction")
        return (
            "?"
            if coefficient is None
            else steer_eval._format_steer_coefficient(coefficient, scaling)
        )
    formatted = [
        steer_eval._format_steer_coefficient(coefficient, scaling)
        for coefficient in coefficients
    ]
    if len(set(formatted)) == 1:
        return formatted[0]
    return "[" + ", ".join(formatted) + "]"


def _print_variant(variant: dict[str, Any], feature_tables: dict[str, dict]) -> None:
    print(f"\n{'=' * 74}")
    print(f"prompt_prefix = {variant['prompt_prefix']}")
    print("=" * 74)

    for entry in variant["interventions"]:
        summary = entry["overall"]
        alphas = "  ".join(
            f"{group['direction']} {_format_group_strength(group)}"
            for group in entry["groups"]
        )
        print(f"\n[{entry['name']}]  layer {entry['layer_idx']}  {alphas}")
        if not summary.get("steered_tokens"):
            print("  hook never wrote.")
            continue

        print(
            f"  {summary['num_cells']:,} cells, "
            f"{summary['steered_tokens']:,} steered tokens"
        )
        print(
            f"  |alpha*delta|/|h| = "
            f"{summary['mean_injected_to_residual_ratio']:.3%}   "
            f"(|delta|={summary['mean_injected_norm']:.1f}, "
            f"|h|={summary['mean_residual_norm']:.1f})"
        )

        table = feature_tables.get(entry["name"], {})
        header = (
            f"    {'feature':>8}  {'dir':>5}  {'fires':>8}  {'act|fire':>9}"
        )
        if table:
            header += f"  {'table act':>10}  {'table rate':>10}"
        print(header)
        for feature in summary["per_feature"]:
            line = (
                f"    {feature['feature_id']:>8}  "
                f"{feature.get('direction') or '-':>5}  "
                f"{feature['firing_rate']:>7.2%}  "
                f"{feature['mean_activation_when_active']:>9.2f}"
            )
            reference = table.get(feature["feature_id"])
            if table:
                if reference:
                    line += (
                        f"  {reference['mean_activation_when_active']:>10.2f}"
                        f"  {reference['activation_rate']:>10.2%}"
                    )
                else:
                    line += f"  {'-':>10}  {'-':>10}"
            print(line)

        if entry["by_cell_type"]:
            print("    by gold cell type (mean firing rate):")
            for label, stats in entry["by_cell_type"].items():
                if not stats.get("steered_tokens"):
                    continue
                print(
                    f"      {label:<18} n={stats['num_cells']:>4}  "
                    f"mean={stats['mean_firing_rate']:>7.2%}  "
                    f"min={stats['min_firing_rate']:>7.2%}  "
                    f"ratio={stats['mean_injected_to_residual_ratio']:>7.3%}"
                )

        print(f"\n  VERDICT: {entry['verdict']}")
        print(f"  {entry['verdict_detail']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether the configured steering features "
            "fire at the steered positions, and how large the injected vector "
            "is relative to the residual stream. Prefill only; no generation."
        )
    )
    parser.add_argument(
        "--config",
        default=None,
        help="YAML config path (default: configs/steer.yaml).",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=50,
        help="Cells to prefill. 50 is plenty; firing rates converge quickly.",
    )
    parser.add_argument(
        "--prompt-prefix",
        choices=("config", "true", "false", "both"),
        default="config",
        help=(
            "Which prompt to diagnose. 'both' runs the templated and bare "
            "prompts over the same cells to measure the train/eval shift."
        ),
    )
    parser.add_argument(
        "--features",
        type=int,
        nargs="+",
        default=None,
        help="Override the configured feature indices.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Override alpha, the fraction of each feature's own activation. "
            "One value for every feature, or one per feature. Only scales the "
            "norm ratio, not the firing rates."
        ),
    )
    parser.add_argument(
        "--magnitude",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Override the strength as a fixed additive magnitude per steered "
            "token instead of a fraction of the activation. One value for "
            "every feature, or one per feature."
        ),
    )
    parser.add_argument(
        "--feature-root",
        default=str(DEFAULT_FEATURE_ROOT),
        help="Root holding all_cell_type_features.csv for the reference columns.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Where to write the JSON report "
            "(default: results/steer/intervention_delivery/<steer-name>.json)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = steer_eval.load_eval_config(args.config)

    if args.alpha is not None and args.magnitude is not None:
        raise ValueError(
            "--alpha and --magnitude set the same knob two ways. Use --alpha "
            "for a fraction of each feature's own activation, or --magnitude "
            "for a fixed amount added at every steered token."
        )
    if (
        args.features is not None
        or args.alpha is not None
        or args.magnitude is not None
    ):
        intervention = cfg.interventions[0]
        if args.features is not None:
            features = steer_eval._validate_feature_indices(args.features)
            group = intervention.groups[0]
            # A per-feature strength no longer lines up with a new feature
            # list, so keep it only when every feature carried the same number.
            alpha = group.alpha
            if isinstance(alpha, list) and len(alpha) != len(features):
                uniform = steer_eval._uniform_magnitude(alpha)
                if uniform is None:
                    raise ValueError(
                        "--features replaces a group with per-feature "
                        "magnitudes; pass --alpha/--magnitude with one value "
                        "per feature as well."
                    )
                alpha = uniform
            intervention.groups = [
                replace(group, feature_indices=features, alpha=alpha)
            ]
        override = args.alpha if args.alpha is not None else args.magnitude
        if override is not None:
            strength = override[0] if len(override) == 1 else override
            scaling = "fraction" if args.alpha is not None else "additive"
            intervention.groups = [
                replace(
                    group,
                    alpha=steer_eval._group_alpha(
                        strength,
                        "--alpha" if args.alpha is not None else "--magnitude",
                        len(group.feature_indices),
                        scaling,
                    ),
                    scaling=scaling,
                )
                for group in intervention.groups
            ]
        cfg.interventions = [intervention]

    cell_eval.validate_inputs(cfg.cell_type)
    split_dir = cfg.cell_type.c2s_dataset_dir / f"data_{cfg.cell_type.split}"
    dataset = load_from_disk(str(split_dir))
    n_examples = min(args.max_examples, len(dataset))

    inf_cfg = steer_eval.make_inference_config(cfg)
    tokenizer, model = load_model(inf_cfg)
    hooks, _meta = steer_eval.load_steering_hooks(model, cfg, inf_cfg)

    feature_root = Path(args.feature_root)
    if not feature_root.is_absolute():
        feature_root = PROJECT_ROOT / feature_root
    feature_tables = {
        hook.name: _load_feature_table(
            intervention.checkpoint_path,
            feature_root,
        )
        for hook, intervention in zip(hooks, cfg.interventions)
    }

    if args.prompt_prefix == "config":
        variants = [cfg.cell_type.prompt_prefix]
    elif args.prompt_prefix == "both":
        variants = [True, False]
    else:
        variants = [args.prompt_prefix == "true"]

    print("[Intervention delivery]")
    print(f"  config    : {args.config or steer_eval.DEFAULT_CONFIG}")
    print(f"  model     : {cfg.cell_type.model_id}")
    print(f"  dataset   : {cfg.cell_type.c2s_dataset_dir.name} / data_{cfg.cell_type.split}")
    print(f"  cells     : {n_examples:,}")
    print(f"  variants  : prompt_prefix in {variants}")
    for intervention in cfg.interventions:
        print(f"  intervene : layer {intervention.layer_idx}, base={intervention.base}")
        for group in intervention.groups:
            direction = steer_eval._resolve_direction(
                group.direction,
                group.name or "group",
            )
            scaling = steer_eval._normalize_steering_scaling(group.scaling)
            key = "alpha" if scaling == "fraction" else "magnitude"
            print(
                f"              {direction:<4} features={group.feature_indices}, "
                f"{key}={group.alpha}"
            )

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "intervention_delivery",
        "model_id": cfg.cell_type.model_id,
        "dataset_dir": str(cfg.cell_type.c2s_dataset_dir),
        "split": f"data_{cfg.cell_type.split}",
        "num_cells": n_examples,
        "firing_rate_floor": FIRING_RATE_FLOOR,
        "norm_ratio_floor": NORM_RATIO_FLOOR,
        "variants": [],
    }

    with steer_eval.SAESteeringPatcher(hooks):
        for prompt_prefix in variants:
            variant = _run_variant(
                dataset=dataset,
                n_examples=n_examples,
                cfg=cfg,
                inf_cfg=inf_cfg,
                tokenizer=tokenizer,
                model=model,
                hooks=hooks,
                prompt_prefix=prompt_prefix,
            )
            report["variants"].append(variant)
            _print_variant(variant, feature_tables)

    if len(report["variants"]) == 2:
        print(f"\n{'=' * 74}")
        print("prompt_prefix comparison (train/eval distribution shift)")
        print("=" * 74)
        templated, bare = report["variants"]
        for left, right in zip(templated["interventions"], bare["interventions"]):
            left_rate = left["overall"].get("mean_firing_rate")
            right_rate = right["overall"].get("mean_firing_rate")
            if left_rate is None or right_rate is None:
                continue
            print(
                f"  {left['name']}: templated {left_rate:.2%} vs "
                f"bare {right_rate:.2%}"
            )
            if right_rate > 0 and left_rate < 0.5 * right_rate:
                print(
                    "    The instruction template roughly halves or worse the "
                    "firing rate. The SAE is being applied off-distribution; "
                    "steering results under the templated prompt are not "
                    "comparable to the feature table."
                )

    output_path = (
        Path(args.output)
        if args.output
        else PROJECT_ROOT
        / "results"
        / "steer"
        / "intervention_delivery"
        / f"{cfg.steer_model_name}.json"
    )
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"\nreport: {output_path}")


if __name__ == "__main__":
    main()
