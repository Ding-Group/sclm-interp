#!/usr/bin/env python3
"""Select per-layer steering feature sets from cell-type feature tables.

Reads ``all_cell_type_features.csv`` written by
``src/evaluate/cell_feature_analysis.py`` and emits, for each SAE and cell
type, the feature sets a steering experiment needs:

``marker``
    Top-N by ``specificity_margin`` -- the interpretable, high-specificity
    candidates.
``housekeeping``
    Low-specificity features matched to the marker set on
    ``mean_activation_when_active``, so the injected norm is comparable and
    only feature identity differs. This is the control that makes a
    marker-vs-null comparison meaningful.
``dead``
    Features with zero activation rate in every cell type. Steering these must
    do nothing at any alpha; a non-zero effect means the harness is misreading
    feature indices.
``stratified``
    A sample spread evenly across the observed specificity-margin range, for
    the interpretability-vs-steerability scatter. These are meant to be steered
    one feature at a time, not as a bundle.

Because decoder directions live in the residual space of the layer the SAE was
trained on, every set is drawn from that layer's own table and is only valid
for an intervention at that layer.

Example:
    .venv/bin/python src/steer/select_features.py --cell-type cd14_monocytes
    .venv/bin/python src/steer/select_features.py --sae layer20_topk_exp8_last_no_prefix_21 --yaml
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PROJECT_ROOT / "results" / "cell_type_features"
DEFAULT_CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"
TABLE_NAME = "all_cell_type_features.csv"

NUMERIC_FIELDS = (
    "specificity_margin",
    "mean_activation",
    "mean_activation_when_active",
    "activation_rate",
    "specificity_share",
)


def load_table(csv_path: Path) -> list[dict[str, Any]]:
    """Read one all_cell_type_features.csv, coercing the numeric columns."""

    rows: list[dict[str, Any]] = []
    with open(csv_path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            record: dict[str, Any] = {
                "cell_type": row["cell_type"],
                "feature_id": int(row["feature_id"]),
            }
            for field in NUMERIC_FIELDS:
                record[field] = float(row[field])
            rows.append(record)
    if not rows:
        raise ValueError(f"{csv_path} contained no rows.")
    return rows


def discover_tables(root: Path, sae_filter: str | None) -> dict[str, Path]:
    """Map SAE name -> feature table path under the results root."""

    tables: dict[str, Path] = {}
    for path in sorted(root.glob(f"*/*/{TABLE_NAME}")):
        sae_name = path.parents[1].name
        if sae_filter and sae_filter != sae_name:
            continue
        tables[sae_name] = path
    return tables


def infer_layer(sae_name: str) -> int | None:
    head = sae_name.split("_", 1)[0]
    if head.startswith("layer") and head[5:].isdigit():
        return int(head[5:])
    return None


def find_checkpoint(sae_name: str, checkpoint_root: Path) -> Path | None:
    matches = sorted(checkpoint_root.glob(f"*/{sae_name}/last.ckpt"))
    return matches[0] if matches else None


def dead_features(rows: list[dict[str, Any]], count: int) -> list[int]:
    """Features whose activation rate is zero for every cell type."""

    seen: dict[int, bool] = {}
    for row in rows:
        feature_id = row["feature_id"]
        silent = row["activation_rate"] == 0.0
        seen[feature_id] = silent if feature_id not in seen else (seen[feature_id] and silent)
    return sorted(feature_id for feature_id, silent in seen.items() if silent)[:count]


def select_markers(
    rows: list[dict[str, Any]],
    cell_type: str,
    count: int,
) -> list[dict[str, Any]]:
    candidates = [row for row in rows if row["cell_type"] == cell_type]
    candidates.sort(key=lambda row: -row["specificity_margin"])
    return candidates[:count]


def select_housekeeping(
    rows: list[dict[str, Any]],
    cell_type: str,
    markers: list[dict[str, Any]],
    *,
    max_specificity_share: float,
    margin_quantile: float,
) -> list[dict[str, Any]]:
    """Low-specificity features matched to the markers on activation magnitude.

    Each marker is paired with the unused low-specificity feature whose
    ``mean_activation_when_active`` is closest to it, so the two sets carry a
    comparable injected norm and differ only in what they encode.

    The pool is bounded by a *quantile* of the layer's own margin distribution
    rather than a fixed threshold, because absolute margins shrink sharply at
    shallower layers -- at 2B layer 10 the best CD14 feature has a specificity
    share of 0.48, so any fixed share cutoff would admit the markers
    themselves. Features in the marker set are excluded outright.
    """

    marker_ids = {row["feature_id"] for row in markers}
    live = [
        row
        for row in rows
        if row["cell_type"] == cell_type and row["activation_rate"] > 0.0
    ]
    margins = sorted(row["specificity_margin"] for row in live)
    cutoff_index = max(0, min(len(margins) - 1, int(len(margins) * margin_quantile)))
    margin_cutoff = margins[cutoff_index]

    pool = [
        row
        for row in live
        if row["feature_id"] not in marker_ids
        and row["specificity_margin"] <= margin_cutoff
        and row["specificity_share"] <= max_specificity_share
    ]
    if not pool:
        pool = [row for row in live if row["feature_id"] not in marker_ids]
    chosen: list[dict[str, Any]] = []
    used: set[int] = set()
    for marker in markers:
        target = marker["mean_activation_when_active"]
        remaining = [row for row in pool if row["feature_id"] not in used]
        if not remaining:
            break
        best = min(
            remaining,
            key=lambda row: abs(row["mean_activation_when_active"] - target),
        )
        used.add(best["feature_id"])
        chosen.append(best)
    return chosen


def select_stratified(
    rows: list[dict[str, Any]],
    cell_type: str,
    count: int,
) -> list[dict[str, Any]]:
    """Sample evenly across the specificity-margin range, top bin first."""

    candidates = [row for row in rows if row["cell_type"] == cell_type]
    candidates.sort(key=lambda row: -row["specificity_margin"])
    if count >= len(candidates):
        return candidates

    high = candidates[0]["specificity_margin"]
    low = candidates[-1]["specificity_margin"]
    span = high - low
    if span <= 0:
        return candidates[:count]

    chosen: list[dict[str, Any]] = []
    used: set[int] = set()
    for step in range(count):
        edge = high - span * (step / (count - 1)) if count > 1 else high
        best = min(
            (row for row in candidates if row["feature_id"] not in used),
            key=lambda row: abs(row["specificity_margin"] - edge),
        )
        used.add(best["feature_id"])
        chosen.append(best)
    return chosen


def summarize(label: str, selected: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [row["feature_id"] for row in selected]
    activations = [row["mean_activation_when_active"] for row in selected]
    margins = [row["specificity_margin"] for row in selected]
    shares = [row["specificity_share"] for row in selected]
    total_sq = sum(value * value for value in activations)
    return {
        "arm": label,
        "features": ids,
        "specificity_margin": [round(value, 2) for value in margins],
        "specificity_share": [round(value, 3) for value in shares],
        "mean_activation_when_active": [round(value, 2) for value in activations],
        "orthogonal_norm_at_alpha_1": round(total_sq**0.5, 2),
    }


def render_yaml(
    arm: dict[str, Any],
    *,
    sae_name: str,
    cell_type: str,
    layer: int | None,
    checkpoint: Path | None,
    alpha: float,
) -> str:
    layer_text = "<layer>" if layer is None else str(layer)
    checkpoint_text = (
        "<path-to>/last.ckpt"
        if checkpoint is None
        else str(checkpoint.relative_to(PROJECT_ROOT))
    )
    return (
        f"      - name: \"layer{layer_text}_{cell_type}_{arm['arm']}\"\n"
        f"        layer: {layer_text}\n"
        f"        checkpoint:\n"
        f"          path: \"{checkpoint_text}\"\n"
        f"          sae_type: \"topk\"\n"
        f"        sae_device: \"model\"\n"
        f"        # {sae_name} | margins {arm['specificity_margin']}\n"
        f"        # orthogonal injected norm at alpha=1.0: "
        f"{arm['orthogonal_norm_at_alpha_1']}\n"
        f"        features: {arm['features']}\n"
        f"        alpha: {alpha}\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select marker, matched-housekeeping, dead, and stratified feature "
            "sets per layer for steering experiments."
        )
    )
    parser.add_argument("--results-root", default=str(DEFAULT_ROOT))
    parser.add_argument(
        "--checkpoint-root",
        default=str(DEFAULT_CHECKPOINT_ROOT),
    )
    parser.add_argument(
        "--sae",
        default=None,
        help="Restrict to one SAE directory name; default is every table found.",
    )
    parser.add_argument(
        "--cell-type",
        default=None,
        help="Restrict to one cell type; default is every type in the table.",
    )
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument(
        "--stratified-n",
        type=int,
        default=0,
        help=(
            "Emit this many features spread across the margin range for the "
            "interpretability-vs-steerability scatter. Steer these singly."
        ),
    )
    parser.add_argument(
        "--max-specificity-share",
        type=float,
        default=0.5,
        help="Upper bound on specificity_share for housekeeping controls.",
    )
    parser.add_argument(
        "--margin-quantile",
        type=float,
        default=0.5,
        help=(
            "Housekeeping features are drawn from below this quantile of the "
            "layer's own specificity-margin distribution."
        ),
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--yaml",
        action="store_true",
        help="Print configs/steer.yaml intervention blocks.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path for the full manifest as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    if not results_root.is_absolute():
        results_root = PROJECT_ROOT / results_root
    checkpoint_root = Path(args.checkpoint_root)
    if not checkpoint_root.is_absolute():
        checkpoint_root = PROJECT_ROOT / checkpoint_root

    tables = discover_tables(results_root, args.sae)
    if not tables:
        raise SystemExit(f"No {TABLE_NAME} found under {results_root}.")

    manifest: list[dict[str, Any]] = []
    for sae_name, csv_path in tables.items():
        rows = load_table(csv_path)
        layer = infer_layer(sae_name)
        checkpoint = find_checkpoint(sae_name, checkpoint_root)
        cell_types = sorted({row["cell_type"] for row in rows})
        if args.cell_type:
            if args.cell_type not in cell_types:
                continue
            cell_types = [args.cell_type]

        dead = dead_features(rows, args.top_n)
        print(f"\n=== {sae_name}  (layer {layer}, {len(rows) // len(cell_types)} features)")
        if checkpoint is None:
            print("  [warning] no matching last.ckpt found under the checkpoint root")

        for cell_type in cell_types:
            markers = select_markers(rows, cell_type, args.top_n)
            housekeeping = select_housekeeping(
                rows,
                cell_type,
                markers,
                max_specificity_share=args.max_specificity_share,
                margin_quantile=args.margin_quantile,
            )
            arms = [
                summarize("marker", markers),
                summarize("housekeeping", housekeeping),
            ]
            if dead:
                arms.append(
                    summarize(
                        "dead",
                        [
                            row
                            for row in rows
                            if row["cell_type"] == cell_type
                            and row["feature_id"] in set(dead)
                        ],
                    )
                )
            if args.stratified_n:
                arms.append(
                    summarize(
                        "stratified",
                        select_stratified(rows, cell_type, args.stratified_n),
                    )
                )

            print(f"\n  {cell_type}")
            for arm in arms:
                print(
                    f"    {arm['arm']:<13} {arm['features']}\n"
                    f"      margin={arm['specificity_margin']}\n"
                    f"      share ={arm['specificity_share']}\n"
                    f"      act   ={arm['mean_activation_when_active']}"
                    f"  -> ||delta||={arm['orthogonal_norm_at_alpha_1']} at alpha=1.0"
                )
                manifest.append(
                    {
                        "sae_name": sae_name,
                        "layer": layer,
                        "checkpoint": str(checkpoint) if checkpoint else None,
                        "cell_type": cell_type,
                        **arm,
                    }
                )

            if args.yaml:
                print("\n    # --- configs/steer.yaml interventions ---")
                for arm in arms:
                    if arm["arm"] == "stratified":
                        continue
                    print(
                        render_yaml(
                            arm,
                            sae_name=sae_name,
                            cell_type=cell_type,
                            layer=layer,
                            checkpoint=checkpoint,
                            alpha=args.alpha,
                        )
                    )

    if args.json_out:
        out_path = Path(args.json_out)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
        print(f"\nmanifest: {out_path}")


if __name__ == "__main__":
    main()
