#!/usr/bin/env python3
"""
Feature activation frequency evaluation for a trained SAE checkpoint.

Streams one or more activation splits through the SAE encoder and computes
per-feature firing rates, dead/universal feature counts, decoder redundancy, and
summary statistics. Splits are evaluated separately and, when ``data.combined``
is set, also pooled into a single combined assessment.

All paths and hyperparameters are read from eval_sae.yaml.

Usage:
    python src/evaluate/sae_analysis/feature_analysis.py
    python src/evaluate/sae_analysis/feature_analysis.py --config configs/eval_sae.yaml
    python src/evaluate/sae_analysis/feature_analysis.py --max-samples 50000 --max-shards 2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_TOP_K = 50  # features included in the top-features JSON list

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch

# Make src/ importable when run as a script from anywhere in the project
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sae import SAE
from evaluate.model_loading import (
    PROJECT_ROOT,
    evaluation_config,
    load_sae,
    load_yaml_mapping,
    project_path,
    resolve_checkpoint_label,
    resolve_checkpoint_path,
    resolve_output_dir,
)
from evaluate.sae_analysis.activation_data import (
    SplitSource,
    check_d_model,
    describe_source,
    iter_selected_batches,
    prepare_split_source,
    resolve_device,
    resolve_split_specs,
)
from evaluate.sae_analysis.reporting import format_table, write_table

_DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "eval_sae.yaml"
_COMBINED_LABEL = "combined"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str | Path | None = None) -> SimpleNamespace:
    """Load evaluation config from YAML and flatten into a SimpleNamespace."""
    raw = evaluation_config(
        load_yaml_mapping(path or _DEFAULT_CONFIG), "feature_analysis"
    )

    ckpt = raw.get("checkpoint", {})
    data = raw.get("data", {})
    out = raw.get("output", {})
    model = raw.get("model", {})

    checkpoint_path = resolve_checkpoint_path(ckpt)

    flat: dict[str, Any] = {
        # checkpoint
        "checkpoint_path": checkpoint_path,
        "layer": ckpt.get("layer"),
        "pooling": ckpt.get("pooling"),
        "base_model": ckpt.get("base_model"),
        "prompt_prefix": ckpt.get("prompt_prefix"),
        # data — splits resolved against the shared data.dir.
        "data_cfg": dict(data),
        "split_specs": resolve_split_specs(
            data,
            legacy_split_key="split",
            default_splits=("data_train",),
        ),
        "combine_splits": bool(data.get("combined", False)),
        "max_shards": data.get("max_shards"),
        # output
        "output_dir": resolve_output_dir(
            out,
            "feature_analysis",
            checkpoint_path=checkpoint_path,
            data_dir=data.get("dir"),
            checkpoint_name=resolve_checkpoint_label(ckpt, checkpoint_path),
        ),
        "seed": out.get("seed", 42),
        # model defaults can be inferred from checkpoint metadata
        "sae_type": model.get("sae_type"),
    }
    for section in ("model", "eval", "infrastructure"):
        flat.update(raw.get(section, {}))

    flat.setdefault("batch_size", 8192)
    flat.setdefault("device", "auto")
    flat.setdefault("max_samples", None)
    flat.setdefault("compute_feature_similarity", True)
    flat.setdefault("decoder_neighbor_top_k", 5)
    flat.setdefault("decoder_chunk_size", 512)
    flat.setdefault("duplicate_threshold", 0.95)
    flat.setdefault("similar_threshold", 0.80)
    flat.setdefault("top_similarity_pairs", 50)

    return SimpleNamespace(**flat)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SAE feature activation frequencies and decoder similarity."
    )
    parser.add_argument("--config", default=None, help="Path to YAML config file.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--splits",
        default=None,
        help="Comma-separated split names to evaluate, e.g. data_train,data_test.",
    )
    combined = parser.add_mutually_exclusive_group()
    combined.add_argument("--combined", dest="combined", action="store_true", default=None)
    combined.add_argument("--no-combined", dest="combined", action="store_false")
    parser.add_argument(
        "--skip-feature-similarity",
        action="store_true",
        help="Skip decoder nearest-neighbor feature comparison.",
    )
    return parser.parse_args()


def apply_cli_overrides(cfg: SimpleNamespace, args: argparse.Namespace) -> None:
    if args.max_samples is not None:
        cfg.max_samples = args.max_samples
    if args.max_shards is not None:
        cfg.max_shards = args.max_shards
    if args.output_dir is not None:
        cfg.output_dir = project_path(args.output_dir)
    if args.combined is not None:
        cfg.combine_splits = args.combined
    if args.splits is not None:
        splits = [s.strip() for s in args.splits.split(",") if s.strip()]
        if not splits:
            raise ValueError("--splits must name at least one split.")
        cfg.split_specs = resolve_split_specs({**cfg.data_cfg, "splits": splits})
    if args.skip_feature_similarity:
        cfg.compute_feature_similarity = False


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(cfg: SimpleNamespace) -> None:
    ckpt_path = Path(cfg.checkpoint_path)
    base_output_dir = Path(cfg.output_dir)
    device = resolve_device(cfg.device)

    layer_s = f"layer{cfg.layer}" if cfg.layer is not None else "layerX"
    pool_s = cfg.pooling or "pool?"
    ckpt_s = ckpt_path.stem or "checkpoint"
    base_label = f"{layer_s}-{pool_s}-{cfg.sae_type}-{ckpt_s}"

    sources = [
        prepare_split_source(
            spec,
            max_shards=cfg.max_shards,
            max_samples=cfg.max_samples,
            seed=cfg.seed,
        )
        for spec in cfg.split_specs
    ]
    combine = bool(cfg.combine_splits) and len(sources) >= 2
    if cfg.combine_splits and not combine:
        print("Skipping combined analysis: fewer than two splits.")

    print(f"\n{'=' * 60}")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Device     : {device}")
    print("Splits     :")
    for source in sources:
        print(f"  {describe_source(source)}")

    sae, sae_meta = load_sae(ckpt_path, cfg.sae_type, device=device)
    sae_meta.update({"layer": cfg.layer, "pooling": cfg.pooling})
    print(
        f"SAE        : {sae_meta['sae_type']}, d_hidden={sae_meta['d_hidden']}, "
        f"expansion={sae_meta['expansion']}"
    )
    if "training_dead_frac" in sae_meta:
        print(f"Training dead frac (ever_fired): {sae_meta['training_dead_frac']:.4%}")

    # Decoder feature similarity depends only on the SAE weights, so compute it
    # once and reuse it across every split.
    decoder = None
    decoder_summary = None
    if cfg.compute_feature_similarity:
        print("Computing decoder feature nearest neighbors ...")
        decoder = compute_decoder_neighbors(
            sae,
            top_k=cfg.decoder_neighbor_top_k,
            chunk_size=cfg.decoder_chunk_size,
            device=device,
        )
        decoder_summary = summarize_decoder_neighbors(
            decoder,
            duplicate_threshold=cfg.duplicate_threshold,
            similar_threshold=cfg.similar_threshold,
            top_pairs=cfg.top_similarity_pairs,
        )

    per_split_raw: list[dict] = []
    for source in sources:
        meta = dict(sae_meta)
        meta["label"] = f"{base_label}-{source.label}"
        meta["split"] = source.spec.name

        print(f"\n{'-' * 60}")
        print(f"Split      : {source.spec.name}")
        print(f"Data dir   : {source.spec.path}")
        raw = compute_activation_stats(
            sae, source, batch_size=cfg.batch_size, device=device
        )
        per_split_raw.append(raw)
        _report(
            cfg, raw, meta, decoder, decoder_summary,
            base_output_dir / source.label, meta["label"],
        )

    if combine:
        meta = dict(sae_meta)
        meta["label"] = f"{base_label}-{_COMBINED_LABEL}"
        meta["split"] = "+".join(s.spec.name for s in sources)

        print(f"\n{'-' * 60}")
        print(f"Split      : combined ({', '.join(s.spec.name for s in sources)})")
        _report(
            cfg, _combine_raw(per_split_raw), meta, decoder, decoder_summary,
            base_output_dir / _COMBINED_LABEL, meta["label"],
        )


def _combine_raw(raw_list: list[dict]) -> dict:
    """Pool per-feature counts across splits into a single stats dict."""
    return _derive_rates(
        fire_count=sum(r["fire_count"] for r in raw_list),
        value_sum=sum(r["value_sum"] for r in raw_list),
        total=sum(r["total_samples"] for r in raw_list),
        num_shards=sum(r["num_shards"] for r in raw_list),
    )


def _report(
    cfg: SimpleNamespace,
    raw: dict,
    meta: dict,
    decoder: dict | None,
    decoder_summary: dict | None,
    output_dir: Path,
    label: str,
) -> None:
    """Summarize raw stats, print diagnostics, and write outputs/figures."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize(
        raw, meta,
        decoder_summary=decoder_summary,
        dead_threshold=cfg.dead_threshold,
        universal_threshold=cfg.universal_threshold,
        common_threshold=cfg.common_threshold,
    )

    print(
        f"  Samples  : {raw['total_samples']:,}  ({raw['num_shards']} shards)\n"
        f"  Mean L0  : {summary['mean_l0']:.1f}\n"
        f"  Dead     : {summary['n_dead']:,} / {summary['d_hidden']:,} ({summary['frac_dead']:.2%})\n"
        f"  Common   : {summary['n_common']:,} ({summary['frac_common']:.2%})\n"
        f"  Universal: {summary['n_universal']:,} ({summary['frac_universal']:.2%})\n"
        f"  Gini     : {summary['gini']:.4f}\n"
        f"  Median rate: {summary['percentiles']['p50']:.2e}"
    )
    if decoder_summary is not None:
        print(
            f"  Median nearest |cos|: {decoder_summary['nearest_abs_cosine_median']:.3f}\n"
            f"  Similar neighbors   : {decoder_summary['n_features_with_similar_neighbor']:,} "
            f"features ({decoder_summary['frac_features_with_similar_neighbor']:.2%})\n"
            f"  Duplicate neighbors : {decoder_summary['n_features_with_duplicate_neighbor']:,} "
            f"features ({decoder_summary['frac_features_with_duplicate_neighbor']:.2%})"
        )
    print("\n" + summary_table(summary))

    print(f"Saving results to {output_dir} ...")
    save_results(summary, raw, output_dir, decoder=decoder)

    print("Generating figures ...")
    plot_activation_rate_histogram(raw, label, summary, fig_dir)
    plot_activation_rate_cdf(raw, label, summary, fig_dir)
    plot_activation_rate_vs_value(raw, label, fig_dir)
    if decoder is not None:
        plot_decoder_neighbor_histograms(decoder, label, fig_dir)
        plot_decoder_distance_cdf(decoder, label, summary, fig_dir)
        plot_decoder_distance_vs_activation_rate(raw, decoder, label, fig_dir)
        plot_decoder_neighbor_rank_profile(decoder, label, fig_dir)

    print(f"Done. Outputs written to {output_dir}/")
    print("  stats.json            - full stats")
    print("  activation_rates.npz  - raw per-feature rate array")
    if decoder is not None:
        print("  decoder_neighbors.npz - nearest decoder-feature arrays")
        print("  decoder_neighbor_pairs.json - top nearest-neighbor feature pairs")
    print("  summary_table.txt     - printable summary")
    print("  figures/              - activation-rate diagnostics")


# ---------------------------------------------------------------------------
# Decoder feature similarity
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_decoder_neighbors(
    sae: SAE,
    top_k: int = 5,
    chunk_size: int = 512,
    device: str = "cpu",
) -> dict:
    """
    Compute nearest-neighbor decoder similarities between SAE features.

    Decoder rows are normalized before cosine computation. The full d_hidden x
    d_hidden matrix can be large, so similarities are computed in row chunks.

    Returns:
        nearest_idx / nearest_cos:
            top-k signed cosine neighbors by raw cosine.
        nearest_abs_idx / nearest_abs_cos / nearest_abs_signed_cos:
            top-k neighbors by absolute cosine, with the signed cosine retained.
    """
    w_dec = sae.w_dec.detach().float().to(device)
    w_dec = w_dec / w_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)

    d_hidden = w_dec.shape[0]
    top_k = min(top_k, max(1, d_hidden - 1))

    def empty(dtype: torch.dtype) -> torch.Tensor:
        return torch.empty((d_hidden, top_k), dtype=dtype, device=device)

    nearest_idx = empty(torch.int64)
    nearest_cos = empty(torch.float32)
    nearest_abs_idx = empty(torch.int64)
    nearest_abs_cos = empty(torch.float32)
    nearest_abs_signed_cos = empty(torch.float32)

    for start in range(0, d_hidden, chunk_size):
        end = min(start + chunk_size, d_hidden)
        sim = w_dec[start:end] @ w_dec.T
        diag_rows = torch.arange(end - start, device=sim.device)
        diag_cols = torch.arange(start, end, device=sim.device)
        # Mask self-similarity before ranking; note abs() of -inf is +inf, so the
        # absolute ranking below has to re-apply the mask.
        sim[diag_rows, diag_cols] = -math.inf

        vals, idx = torch.topk(sim, k=top_k, dim=1)
        nearest_cos[start:end] = vals
        nearest_idx[start:end] = idx

        abs_sim = sim.abs()
        abs_sim[diag_rows, diag_cols] = -math.inf
        abs_vals, abs_idx = torch.topk(abs_sim, k=top_k, dim=1)
        nearest_abs_cos[start:end] = abs_vals
        nearest_abs_idx[start:end] = abs_idx
        nearest_abs_signed_cos[start:end] = torch.gather(sim, 1, abs_idx)

        if (end == d_hidden) or ((start // chunk_size + 1) % 10 == 0):
            print(f"  decoder similarity: {end:,}/{d_hidden:,} features processed")

    nearest_cos = nearest_cos.cpu()
    nearest_abs_cos = nearest_abs_cos.cpu()
    return {
        "nearest_idx": nearest_idx.cpu(),
        "nearest_cos": nearest_cos,
        "nearest_abs_idx": nearest_abs_idx.cpu(),
        "nearest_abs_cos": nearest_abs_cos,
        "nearest_abs_signed_cos": nearest_abs_signed_cos.cpu(),
        "nearest_cosine_distance": 1.0 - nearest_cos,
        "nearest_abs_cosine_distance": 1.0 - nearest_abs_cos,
    }


def summarize_decoder_neighbors(
    decoder: dict,
    duplicate_threshold: float = 0.95,
    similar_threshold: float = 0.80,
    top_pairs: int = 50,
) -> dict:
    """Summarize nearest-neighbor decoder similarity as redundancy diagnostics."""
    nearest_abs = decoder["nearest_abs_cos"][:, 0].numpy().astype(np.float64)
    nearest_signed = decoder["nearest_abs_signed_cos"][:, 0].numpy().astype(np.float64)
    nearest_idx = decoder["nearest_abs_idx"][:, 0].numpy().astype(np.int64)
    d_hidden = nearest_abs.shape[0]

    duplicate_mask = nearest_abs >= duplicate_threshold
    similar_mask = nearest_abs >= similar_threshold

    unique_duplicate_pairs: set[tuple[int, int]] = set()
    unique_similar_pairs: set[tuple[int, int]] = set()
    for feature_idx, neighbor_idx in enumerate(nearest_idx.tolist()):
        if neighbor_idx < 0:
            continue
        pair = tuple(sorted((feature_idx, int(neighbor_idx))))
        if duplicate_mask[feature_idx]:
            unique_duplicate_pairs.add(pair)
        if similar_mask[feature_idx]:
            unique_similar_pairs.add(pair)

    order = np.argsort(nearest_abs)[::-1]
    top_unique_pairs = []
    seen: set[tuple[int, int]] = set()
    for feature_idx in order:
        neighbor_idx = int(nearest_idx[feature_idx])
        if neighbor_idx < 0:
            continue
        pair = tuple(sorted((int(feature_idx), neighbor_idx)))
        if pair in seen:
            continue
        seen.add(pair)
        abs_cos = float(nearest_abs[feature_idx])
        signed_cos = float(nearest_signed[feature_idx])
        top_unique_pairs.append(
            {
                "feature_idx": int(feature_idx),
                "neighbor_idx": neighbor_idx,
                "abs_cosine": abs_cos,
                "signed_cosine": signed_cos,
                "abs_cosine_distance": float(1.0 - abs_cos),
                "cosine_distance": float(1.0 - signed_cos),
            }
        )
        if len(top_unique_pairs) >= top_pairs:
            break

    percentiles = {
        f"p{p}": float(np.percentile(nearest_abs, p))
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
    }

    return {
        "nearest_abs_cosine_mean": float(nearest_abs.mean()),
        "nearest_abs_cosine_median": float(np.median(nearest_abs)),
        "nearest_abs_cosine_percentiles": percentiles,
        "n_features_with_similar_neighbor": int(similar_mask.sum()),
        "frac_features_with_similar_neighbor": round(float(similar_mask.mean()), 6),
        "n_features_with_duplicate_neighbor": int(duplicate_mask.sum()),
        "frac_features_with_duplicate_neighbor": round(float(duplicate_mask.mean()), 6),
        "n_unique_similar_pairs": len(unique_similar_pairs),
        "n_unique_duplicate_pairs": len(unique_duplicate_pairs),
        "thresholds": {
            "similar": similar_threshold,
            "duplicate": duplicate_threshold,
        },
        "top_pairs": top_unique_pairs,
        "d_hidden": d_hidden,
    }


# ---------------------------------------------------------------------------
# Activation statistics
# ---------------------------------------------------------------------------

def _derive_rates(
    fire_count: torch.Tensor,
    value_sum: torch.Tensor,
    total: int,
    num_shards: int,
) -> dict:
    """Turn per-feature counts into firing rates and conditional means."""
    d_hidden = fire_count.shape[0]
    activation_rate = (fire_count.double() / max(total, 1)).float()
    mean_active_value = torch.where(
        fire_count > 0,
        (value_sum / fire_count.double().clamp(min=1)).float(),
        torch.zeros(d_hidden),
    )
    return {
        "fire_count": fire_count,
        "value_sum": value_sum,
        "activation_rate": activation_rate,
        "mean_active_value": mean_active_value,
        "total_samples": total,
        "num_shards": num_shards,
    }


@torch.no_grad()
def compute_activation_stats(
    sae: SAE,
    source: SplitSource,
    batch_size: int = 8192,
    device: str = "cpu",
) -> dict:
    """
    Stream a split through the SAE encoder and accumulate:
        - fire_count[i]: how many samples activated feature i (int64)
        - value_sum[i]:  sum of feature i's activation value over all samples (float64)
        - total_samples: total number of samples seen

    Counts are accumulated on the compute device and moved to the host once.
    """
    check_d_model(source, int(sae.d_model))

    d_hidden = sae.d_hidden
    fire_count = torch.zeros(d_hidden, dtype=torch.int64, device=device)
    value_sum = torch.zeros(d_hidden, dtype=torch.float64, device=device)
    total = 0

    for i, batch in enumerate(iter_selected_batches(source, batch_size, device)):
        encoded = sae.encode(batch)
        fire_count += (encoded > 0).sum(dim=0)
        value_sum += encoded.sum(dim=0, dtype=torch.float64)
        total += batch.shape[0]

        if (i + 1) % 50 == 0:
            print(f"    {total:,} rows processed")

    if total == 0:
        raise RuntimeError(f"No activation rows were processed from {source.spec.path}")

    return _derive_rates(
        fire_count.cpu(), value_sum.cpu(), total, len(source.shard_paths)
    )


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

def summarize(
    raw: dict,
    meta: dict,
    decoder_summary: dict | None = None,
    dead_threshold: float = 1e-4,
    universal_threshold: float = 0.5,
    common_threshold: float = 0.05,
) -> dict:
    """Compute scalar summary metrics and top-K feature list from raw stats."""
    rates = raw["activation_rate"].numpy().astype(np.float64)
    values = raw["mean_active_value"].numpy().astype(np.float64)
    d = len(rates)

    n_dead = int((rates < dead_threshold).sum())
    n_universal = int((rates >= universal_threshold).sum())
    n_common = int((rates >= common_threshold).sum())
    mean_l0 = float(rates.sum())  # E[L0] = sum of P(feature fires), by linearity

    # Gini coefficient of activation rates (0 = perfectly equal, 1 = one feature dominates)
    s = np.sort(rates)
    gini = float(
        (2 * (np.arange(1, d + 1) * s).sum() / (d * s.sum() + 1e-12)) - (d + 1) / d
    )

    top_idx = np.argsort(rates)[::-1][:_TOP_K]
    top_features = [
        {
            "feature_idx": int(i),
            "activation_rate": float(rates[i]),
            "mean_value_when_active": float(values[i]),
        }
        for i in top_idx
    ]

    percentiles = {
        f"p{p}": float(np.percentile(rates, p))
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
    }

    summary = {
        "label": meta["label"],
        "meta": {k: v for k, v in meta.items() if not k.startswith("path")},
        "total_samples": raw["total_samples"],
        "num_shards": raw["num_shards"],
        "d_hidden": d,
        "mean_l0": round(mean_l0, 3),
        "n_dead": n_dead,
        "frac_dead": round(float(n_dead / d), 6),
        "n_universal": n_universal,
        "frac_universal": round(float(n_universal / d), 6),
        "n_common": n_common,
        "frac_common": round(float(n_common / d), 6),
        "mean_activation_rate": float(rates.mean()),
        "gini": round(gini, 4),
        "percentiles": percentiles,
        "top_features": top_features,
        "thresholds": {
            "dead": dead_threshold,
            "universal": universal_threshold,
            "common": common_threshold,
        },
    }
    if decoder_summary is not None:
        summary["feature_similarity"] = decoder_summary
    return summary


# ---------------------------------------------------------------------------
# Text reporting
# ---------------------------------------------------------------------------

def summary_table(summary: dict) -> str:
    """Render the one-line summary table for a split."""
    headers = ["Checkpoint", "Samples", "L0", "Dead%", "Common%", "Universal%", "Gini"]
    cells = [
        summary["label"],
        f"{summary['total_samples']:,}",
        f"{summary['mean_l0']:.1f}",
        f"{summary['frac_dead'] * 100:.2f}%",
        f"{summary['frac_common'] * 100:.2f}%",
        f"{summary['frac_universal'] * 100:.2f}%",
        f"{summary['gini']:.4f}",
    ]
    similarity = summary.get("feature_similarity")
    if similarity is not None:
        headers += ["NN|cos|", "Dup%"]
        cells += [
            f"{similarity['nearest_abs_cosine_median']:.3f}",
            f"{similarity['frac_features_with_duplicate_neighbor'] * 100:.2f}%",
        ]
    return format_table(headers, [cells])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

# Shared aesthetic settings applied to every figure in this module.
_PLOT_STYLE = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "figure.dpi": 110,
    "savefig.bbox": "tight",
}

# Colorblind-friendly accents (Wong / Okabe-Ito).
_PRIMARY = "#0072B2"
_DECODER_COLOR = "#009E73"
_ACCENT = "#CC79A7"

# Region shading (dead / common / universal) — desaturated tones.
_DEAD_COLOR = "#B0B0B0"
_COMMON_COLOR = "#9ACD9A"
_UNIVERSAL_COLOR = "#F4A6A6"

# Smooth white -> teal -> deep-indigo colormap for density (hexbin) plots.
_DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "soft_density",
    [
        (0.00, "#FFFFFF"),
        (0.15, "#E1ECF4"),
        (0.35, "#A6CEE3"),
        (0.55, "#5FA8D3"),
        (0.75, "#3A6EA5"),
        (1.00, "#1B264F"),
    ],
)

_LOW_EDGE = 1e-6  # smallest activation rate resolved on the log axes


def _shade_regions(ax: plt.Axes, thresholds: dict, high: float = 1.0) -> None:
    """Shade dead / common / universal activation-rate zones on a log-x axis."""
    ax.axvspan(_LOW_EDGE, thresholds["dead"], color=_DEAD_COLOR, alpha=0.18, zorder=0)
    ax.axvspan(
        thresholds["common"], thresholds["universal"],
        color=_COMMON_COLOR, alpha=0.18, zorder=0,
    )
    ax.axvspan(
        thresholds["universal"], high, color=_UNIVERSAL_COLOR, alpha=0.22, zorder=0
    )


def _threshold_lines(ax: plt.Axes, thresholds: dict) -> None:
    for name in ("dead", "common", "universal"):
        x = thresholds[name]
        ax.axvline(x, color="black", lw=0.8, ls=":", alpha=0.55, zorder=1)
        ax.text(
            x, 1.0,
            f" {name}\n {x:g}",
            transform=ax.get_xaxis_transform(),
            ha="left", va="top", fontsize=8, color="black", alpha=0.7,
        )


def plot_activation_rate_histogram(
    raw: dict, label: str, summary: dict, output_dir: Path
) -> None:
    """Per-feature activation-rate histogram with dead/common/universal zones."""
    with plt.rc_context(_PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        thresholds = summary["thresholds"]
        _shade_regions(ax, thresholds)

        rates = raw["activation_rate"].numpy().astype(np.float64)
        # Features with rate below the lowest bin (incl. exactly 0) are
        # accumulated into the leftmost bin so dead features remain visible.
        ax.hist(
            rates.clip(_LOW_EDGE),
            bins=np.logspace(np.log10(_LOW_EDGE), 0, 60),
            alpha=0.8,
            color=_PRIMARY,
            edgecolor="white",
            linewidth=0.4,
        )
        _threshold_lines(ax, thresholds)

        ax.set_xscale("log")
        ax.set_xlim(_LOW_EDGE, 1.0)
        ax.set_xlabel("Activation rate (fraction of samples)")
        ax.set_ylabel("Number of features per bin")
        ax.set_title(f"Per-feature activation rate distribution\n{label}")
        ax.text(
            0.01, -0.16,
            f"Features with rate < {_LOW_EDGE:.0e} pooled into the leftmost bin.",
            transform=ax.transAxes, fontsize=8, color="gray",
        )
        fig.tight_layout()
        fig.savefig(output_dir / "activation_rate_hist.png")
        plt.close(fig)


def plot_activation_rate_cdf(
    raw: dict, label: str, summary: dict, output_dir: Path
) -> None:
    """CDF of per-feature activation rates with threshold annotations."""
    with plt.rc_context(_PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        thresholds = summary["thresholds"]
        _shade_regions(ax, thresholds)

        rates = np.sort(raw["activation_rate"].numpy().astype(np.float64))
        cdf = np.arange(1, len(rates) + 1) / len(rates)
        ax.plot(
            rates.clip(_LOW_EDGE), cdf,
            label=(
                f"dead {summary['frac_dead']:.1%}, "
                f"common {summary['frac_common']:.1%}, "
                f"universal {summary['frac_universal']:.1%}"
            ),
            color=_PRIMARY, lw=2,
        )
        _threshold_lines(ax, thresholds)

        ax.set_xscale("log")
        ax.set_xlim(_LOW_EDGE, 1.0)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("Activation rate")
        ax.set_ylabel("Fraction of features <= rate")
        ax.set_title(f"CDF of per-feature activation rates\n{label}")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(output_dir / "activation_rate_cdf.png")
        plt.close(fig)


def plot_activation_rate_vs_value(raw: dict, label: str, output_dir: Path) -> None:
    """Hexbin density of activation rate vs mean active value per feature."""
    rates = raw["activation_rate"].numpy().astype(np.float64)
    values = raw["mean_active_value"].numpy().astype(np.float64)
    # Keep only firing features with positive magnitude.
    mask = (rates > 0) & (values > 0)
    if not mask.any():
        return

    with plt.rc_context(_PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(6.6, 4.8))

        y_lo = max(float(values[mask].min()), 1e-3)
        y_hi = float(values[mask].max()) * 1.1

        hb = ax.hexbin(
            np.clip(rates[mask], _LOW_EDGE, 1.0),
            np.clip(values[mask], y_lo, y_hi),
            xscale="log", yscale="log",
            gridsize=45, mincnt=1,
            cmap=_DENSITY_CMAP, bins="log",
            linewidths=0,
        )
        ax.set_xlim(_LOW_EDGE, 1.0)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("Activation rate")
        ax.set_ylabel("Mean activation value (when active)")
        ax.set_title(f"Activation rate vs magnitude per feature\n{label}")
        cb = fig.colorbar(hb, ax=ax, pad=0.02)
        cb.set_label("Features per hex", fontsize=9)
        fig.tight_layout()
        fig.savefig(output_dir / "rate_vs_value.png")
        plt.close(fig)


def plot_decoder_neighbor_histograms(
    decoder: dict, label: str, output_dir: Path
) -> None:
    """Nearest decoder-neighbor cosine and distance distributions."""
    with plt.rc_context(_PLOT_STYLE):
        fig, (ax_cos, ax_dist) = plt.subplots(1, 2, figsize=(12, 4.6))
        bins = np.linspace(0, 1, 70)

        nearest_abs_cos = decoder["nearest_abs_cos"][:, 0].numpy().astype(np.float64)
        ax_cos.hist(
            nearest_abs_cos, bins=bins, alpha=0.8,
            color=_DECODER_COLOR, edgecolor="white", linewidth=0.4,
        )
        ax_dist.hist(
            1.0 - nearest_abs_cos, bins=bins, alpha=0.8,
            color=_DECODER_COLOR, edgecolor="white", linewidth=0.4,
        )

        ax_cos.set_xlabel("Nearest-neighbor |cosine|")
        ax_cos.set_ylabel("Number of features")
        ax_cos.set_title("Nearest decoder similarity")
        ax_dist.set_xlabel("Nearest-neighbor distance (1 - |cosine|)")
        ax_dist.set_ylabel("Number of features")
        ax_dist.set_title("Nearest decoder distance")
        fig.suptitle(label, fontsize=10)
        fig.tight_layout()
        fig.savefig(output_dir / "decoder_neighbor_hist.png")
        plt.close(fig)


def plot_decoder_distance_cdf(
    decoder: dict, label: str, summary: dict, output_dir: Path
) -> None:
    """CDF of nearest decoder distance with similarity thresholds annotated."""
    with plt.rc_context(_PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(8.5, 5))

        dist = np.sort(
            decoder["nearest_abs_cosine_distance"][:, 0].numpy().astype(np.float64)
        )
        cdf = np.arange(1, len(dist) + 1) / len(dist)
        ax.plot(dist, cdf, lw=2, color=_DECODER_COLOR)

        similarity = summary.get("feature_similarity")
        if similarity is not None:
            for name in ("duplicate", "similar"):
                dist_thr = 1.0 - similarity["thresholds"][name]
                ax.axvline(dist_thr, color="black", lw=0.8, ls=":", alpha=0.6)
                ax.text(
                    dist_thr, 1.0,
                    f" {name}\n {dist_thr:.2f}",
                    transform=ax.get_xaxis_transform(),
                    ha="left", va="top", fontsize=8, color="black", alpha=0.7,
                )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Nearest-neighbor distance (1 - |cosine|)")
        ax.set_ylabel("Fraction of features <= distance")
        ax.set_title(f"CDF of nearest decoder distance\n{label}")
        fig.tight_layout()
        fig.savefig(output_dir / "decoder_distance_cdf.png")
        plt.close(fig)


def plot_decoder_distance_vs_activation_rate(
    raw: dict, decoder: dict, label: str, output_dir: Path
) -> None:
    """Density plot of feature firing rate versus nearest decoder distance."""
    with plt.rc_context(_PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(6.8, 4.8))

        rates = raw["activation_rate"].numpy().astype(np.float64)
        dist = decoder["nearest_abs_cosine_distance"][:, 0].numpy().astype(np.float64)

        hb = ax.hexbin(
            np.clip(rates, _LOW_EDGE, 1.0),
            np.clip(dist, 0.0, 1.0),
            xscale="log",
            gridsize=50, mincnt=1,
            cmap=_DENSITY_CMAP, bins="log",
            linewidths=0,
        )
        ax.set_xlim(_LOW_EDGE, 1.0)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("Activation rate")
        ax.set_ylabel("Nearest decoder distance (1 - |cosine|)")
        ax.set_title(f"Feature frequency vs decoder redundancy\n{label}")
        cb = fig.colorbar(hb, ax=ax, pad=0.02)
        cb.set_label("Features per hex", fontsize=9)
        fig.tight_layout()
        fig.savefig(output_dir / "decoder_distance_vs_activation_rate.png")
        plt.close(fig)


def plot_decoder_neighbor_rank_profile(
    decoder: dict, label: str, output_dir: Path
) -> None:
    """Mean nearest-neighbor |cosine| by neighbor rank."""
    with plt.rc_context(_PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(8, 4.8))

        nearest_abs = decoder["nearest_abs_cos"].numpy().astype(np.float64)
        ranks = np.arange(1, nearest_abs.shape[1] + 1)
        ax.plot(
            ranks, nearest_abs.mean(axis=0),
            marker="o", color=_DECODER_COLOR, label="mean",
        )
        ax.plot(
            ranks, np.percentile(nearest_abs, 90, axis=0),
            ls="--", color=_ACCENT, alpha=0.85, label="p90",
        )
        ax.plot(
            ranks, np.percentile(nearest_abs, 99, axis=0),
            ls=":", color=_PRIMARY, alpha=0.85, label="p99",
        )

        ax.set_xlabel("Nearest-neighbor rank")
        ax.set_ylabel("Absolute decoder cosine")
        ax.set_ylim(0, 1)
        ax.set_title(f"Decoder-neighbor similarity by rank\n{label}")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(output_dir / "decoder_neighbor_rank_profile.png")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_results(
    summary: dict,
    raw: dict,
    output_dir: Path,
    decoder: dict | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    label = summary["label"]

    with open(output_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    np.savez_compressed(
        output_dir / "activation_rates.npz",
        **{label: raw["activation_rate"].numpy()},
    )

    if decoder is not None:
        np.savez_compressed(
            output_dir / "decoder_neighbors.npz",
            **{f"{label}__{key}": value.numpy() for key, value in decoder.items()},
        )
        with open(output_dir / "decoder_neighbor_pairs.json", "w", encoding="utf-8") as f:
            json.dump({label: summary["feature_similarity"]["top_pairs"]}, f, indent=2)

    write_table(output_dir, summary_table(summary))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    apply_cli_overrides(cfg, args)
    run(cfg)


if __name__ == "__main__":
    main()
