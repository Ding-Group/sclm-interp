#!/usr/bin/env python3
"""
Head-to-head variance-explained comparison between a trained SAE and PCA.

This evaluation joins what ``reconstruction_eval.py`` and ``pca_analysis.py``
measure separately, so one figure can support three claims:

    1. The SAE reconstructs its activation space well (high FVE).
    2. Its learned features are not a rotation of the dense PCA basis
       (low decoder/component overlap versus a random-direction null).
    3. At a matched budget of directions, the SAE explains more variance
       than PCA.

Matched budget
--------------
Both methods are scored with the *same* centered FVE used by
``reconstruction_eval.py``::

    FVE = 1 - sum_d Var(x_d - xhat_d) / sum_d Var(x_d)

measured on the same rows in the same pass. The x-axis is the number of
directions each method may use per token:

    - PCA rank k  : one *global* k-dimensional subspace, shared by every row.
    - SAE top-j   : the j strongest active features of that row, chosen
                    *per row* from a d_hidden-atom dictionary.

These are equal-budget but not equal-capacity. Per-row atom selection is
strictly more expressive than a fixed subspace, and the SAE carries far more
parameters (d_hidden x d_model versus k x d_model), so an SAE advantage at
matched k is expected rather than surprising. The comparison is reported
because FVE-at-a-sparsity-budget is the operating point that matters for
downstream interpretability work, not as evidence that the SAE is a better
compressor per parameter. Both numbers are printed so the trade is visible.

PCA is fit on the same split it is scored on, which favours PCA (in-sample),
so claim 3 is measured in the conservative direction.

Usage:
    python src/evaluate/sae_analysis/pca_vs_sae.py
    python src/evaluate/sae_analysis/pca_vs_sae.py --config configs/eval_sae.yaml
    python src/evaluate/sae_analysis/pca_vs_sae.py --max-samples 50000
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

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
from evaluate.sae_analysis.pca_analysis import (
    compute_activation_covariance,
    eigendecompose_covariance,
    full_precision_matmul,
)
from evaluate.sae_analysis.reporting import format_table, write_table

_DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "eval_sae.yaml"
_DEFAULT_RANKS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str | Path | None = None) -> SimpleNamespace:
    raw = evaluation_config(load_yaml_mapping(path or _DEFAULT_CONFIG), "pca_vs_sae")

    ckpt = raw.get("checkpoint", {})
    data = raw.get("data", {})
    out = raw.get("output", {})
    model = raw.get("model", {})

    specs = resolve_split_specs(
        data, legacy_split_key="split", default_splits=("data_test",)
    )
    if len(specs) > 1:
        raise ValueError(
            "pca_vs_sae compares one activation split, but "
            f"{len(specs)} were configured: {', '.join(s.name for s in specs)}. "
            "Set evaluations.pca_vs_sae.data.split."
        )

    checkpoint_path = resolve_checkpoint_path(ckpt)
    flat: dict[str, Any] = {
        "checkpoint_path": checkpoint_path,
        "layer": ckpt.get("layer"),
        "pooling": ckpt.get("pooling"),
        "split_spec": specs[0],
        "max_shards": data.get("max_shards"),
        "max_samples": data.get("max_samples"),
        "output_dir": resolve_output_dir(
            out,
            "pca_vs_sae",
            checkpoint_path=checkpoint_path,
            data_dir=data.get("dir"),
            checkpoint_name=resolve_checkpoint_label(ckpt, checkpoint_path),
        ),
        "seed": out.get("seed", 42),
        "sae_type": model.get("sae_type"),
    }
    for section in ("model", "eval", "infrastructure"):
        flat.update(raw.get(section, {}))

    flat.setdefault("batch_size", 8192)
    flat.setdefault("device", "auto")
    flat.setdefault("ranks", list(_DEFAULT_RANKS))
    flat.setdefault("max_rank", None)
    flat.setdefault("overlap_components", 50)
    flat.setdefault("null_samples", 4096)
    return SimpleNamespace(**flat)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SAE and PCA variance explained at a matched budget."
    )
    parser.add_argument("--config", default=None, help="Path to YAML config file.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument(
        "--ranks",
        default=None,
        help="Comma-separated direction budgets, e.g. 1,4,16,64,256.",
    )
    parser.add_argument("--overlap-components", type=int, default=None)
    parser.add_argument("--device", default=None, choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def apply_overrides(cfg: SimpleNamespace, args: argparse.Namespace) -> None:
    if args.max_samples is not None:
        cfg.max_samples = args.max_samples
    if args.max_shards is not None:
        cfg.max_shards = args.max_shards
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.device is not None:
        cfg.device = args.device
    if args.overlap_components is not None:
        cfg.overlap_components = args.overlap_components
    if args.output_dir is not None:
        cfg.output_dir = project_path(args.output_dir)
    if args.ranks is not None:
        cfg.ranks = [int(v) for v in args.ranks.split(",") if v.strip()]
    if args.split is not None:
        cfg.split_spec = resolve_split_specs({"split": args.split})[0]


def resolve_ranks(
    ranks: Any, *, d_model: int, d_hidden: int, max_rank: Any = None
) -> list[int]:
    """Clamp the requested budgets to what both methods can actually spend."""
    try:
        values = sorted({int(rank) for rank in ranks})
    except (TypeError, ValueError) as exc:
        raise ValueError("eval.ranks must be a list of integers.") from exc
    if any(rank < 1 for rank in values):
        raise ValueError("eval.ranks entries must be at least 1.")

    # PCA cannot exceed d_model directions; the SAE cannot exceed d_hidden.
    ceiling = min(int(d_model), int(d_hidden))
    if max_rank is not None:
        ceiling = min(ceiling, int(max_rank))
    values = [rank for rank in values if rank <= ceiling]
    if not values:
        raise ValueError(
            f"No eval.ranks entries fall within the usable budget (<= {ceiling})."
        )
    return values


# ---------------------------------------------------------------------------
# Matched-budget sweep
# ---------------------------------------------------------------------------

def _fve_from_sums(
    sum_r: torch.Tensor,
    sum_rr: torch.Tensor,
    var_x_total: float,
    n: float,
) -> float:
    """Centered FVE, matching reconstruction.reconstruction_metrics."""
    mean_r = sum_r / n
    var_r = (sum_rr / n - mean_r.pow(2)).clamp(min=0.0)
    if var_x_total <= 0.0:
        return math.nan
    return 1.0 - float(var_r.sum()) / var_x_total


@torch.no_grad()
def sweep_variance_explained(
    sae: SAE,
    source: SplitSource,
    *,
    mean: torch.Tensor,
    components: torch.Tensor,
    ranks: list[int],
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    """Measure PCA rank-k and SAE top-j FVE on the same rows in one pass."""
    d_model = int(sae.d_model)
    check_d_model(source, d_model)

    max_rank = ranks[-1]
    if components.shape[0] < max_rank:
        raise ValueError(
            f"Need {max_rank} PCA components for the requested ranks, but only "
            f"{components.shape[0]} were computed. Raise eval.max_rank."
        )

    comps = components[:max_rank].to(device).float()
    mean_d = mean.to(device).float()
    w_dec = sae.w_dec.detach().to(device).float()
    b_dec = sae.b_dec.detach().to(device).float()

    def accumulator() -> torch.Tensor:
        return torch.zeros(d_model, dtype=torch.float64, device=device)

    sum_x, sum_xx = accumulator(), accumulator()
    pca_sums = {rank: (accumulator(), accumulator()) for rank in ranks}
    sae_sums = {rank: (accumulator(), accumulator()) for rank in ranks}
    full_sum_r, full_sum_rr = accumulator(), accumulator()
    sum_l0 = torch.zeros((), dtype=torch.float64, device=device)
    total = 0

    for i, batch in enumerate(iter_selected_batches(source, batch_size, device)):
        x = batch.float()
        sum_x += x.sum(dim=0, dtype=torch.float64)
        sum_xx += x.pow(2).sum(dim=0, dtype=torch.float64)

        # PCA: grow the reconstruction one rank block at a time.
        centered = x - mean_d
        scores = centered @ comps.T
        pca_recon = torch.zeros_like(centered)
        previous = 0
        for rank in ranks:
            pca_recon += scores[:, previous:rank] @ comps[previous:rank]
            residual = centered - pca_recon
            sum_r, sum_rr = pca_sums[rank]
            sum_r += residual.sum(dim=0, dtype=torch.float64)
            sum_rr += residual.pow(2).sum(dim=0, dtype=torch.float64)
            previous = rank

        acts = sae.encode(x).float()
        sum_l0 += (acts > 0).sum(dtype=torch.float64)
        residual = x - sae.decode(acts)
        full_sum_r += residual.sum(dim=0, dtype=torch.float64)
        full_sum_rr += residual.pow(2).sum(dim=0, dtype=torch.float64)

        # SAE: add decoder atoms strongest-first. Accumulating rank-1 updates
        # costs n*d_model per atom, far less than a dense (n, d_hidden) matmul
        # against a mostly-zero code.
        n_atoms = min(max_rank, acts.shape[1])
        values, indices = acts.topk(n_atoms, dim=1)
        sae_recon = b_dec.expand_as(x).clone()
        previous = 0
        for rank in ranks:
            stop = min(rank, n_atoms)
            for atom in range(previous, stop):
                sae_recon.addcmul_(
                    values[:, atom : atom + 1],
                    w_dec.index_select(0, indices[:, atom]),
                )
            previous = stop
            residual = x - sae_recon
            sum_r, sum_rr = sae_sums[rank]
            sum_r += residual.sum(dim=0, dtype=torch.float64)
            sum_rr += residual.pow(2).sum(dim=0, dtype=torch.float64)

        total += x.shape[0]
        if (i + 1) % 20 == 0:
            print(f"  sweep pass: {total:,} rows processed")

    if total == 0:
        raise RuntimeError(f"No activation rows were processed from {source.spec.path}")

    n = float(total)
    mean_x = sum_x / n
    var_x = (sum_xx / n - mean_x.pow(2)).clamp(min=0.0)
    var_x_total = float(var_x.sum())

    curve = [
        {
            "rank": rank,
            "pca_fve": _fve_from_sums(*pca_sums[rank], var_x_total, n),
            "sae_fve": _fve_from_sums(*sae_sums[rank], var_x_total, n),
        }
        for rank in ranks
    ]
    for row in curve:
        row["sae_minus_pca"] = row["sae_fve"] - row["pca_fve"]

    return {
        "curve": curve,
        "sae_full_fve": _fve_from_sums(full_sum_r, full_sum_rr, var_x_total, n),
        "mean_l0": float(sum_l0.item()) / n,
        "total_variance": var_x_total,
        "total_samples": total,
    }


# ---------------------------------------------------------------------------
# Feature/component overlap
# ---------------------------------------------------------------------------

@torch.no_grad()
def decoder_component_overlap(
    sae: SAE,
    components: torch.Tensor,
    *,
    n_components: int,
    null_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Best |cosine| from each decoder direction to any leading PCA component.

    A random direction in high dimensions already has a nonzero best cosine to
    a set of components, so the same statistic is computed for random unit
    vectors and reported alongside. Only the gap between the two is evidence.
    """
    w_dec = sae.w_dec.detach().float().cpu()
    w_dec = w_dec / w_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)

    n_components = min(int(n_components), components.shape[0])
    comps = components[:n_components].float().cpu()
    comps = comps / comps.norm(dim=1, keepdim=True).clamp(min=1e-8)

    observed = (w_dec @ comps.T).abs().max(dim=1).values

    generator = torch.Generator().manual_seed(int(seed))
    random_dirs = torch.randn(
        int(null_samples), w_dec.shape[1], generator=generator
    )
    random_dirs = random_dirs / random_dirs.norm(dim=1, keepdim=True).clamp(min=1e-8)
    null = (random_dirs @ comps.T).abs().max(dim=1).values

    return {
        "observed": observed,
        "null": null,
        "n_components": n_components,
        "observed_median": float(observed.median()),
        "observed_mean": float(observed.mean()),
        "observed_p95": float(observed.quantile(0.95)),
        "null_median": float(null.median()),
        "null_p95": float(null.quantile(0.95)),
        "fraction_above_null_p95": float((observed > null.quantile(0.95)).float().mean()),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def comparison_table(sweep: dict[str, Any]) -> str:
    headers = ("Directions", "PCA FVE", "SAE FVE", "SAE - PCA")
    rows = [
        [
            f"{row['rank']:,}",
            f"{row['pca_fve']:.4f}",
            f"{row['sae_fve']:.4f}",
            f"{row['sae_minus_pca']:+.4f}",
        ]
        for row in sweep["curve"]
    ]
    return format_table(headers, rows)


def write_curve_csv(path: Path, sweep: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["rank", "pca_fve", "sae_fve", "sae_minus_pca"]
        )
        writer.writeheader()
        writer.writerows(sweep["curve"])


_PLOT_STYLE = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.alpha": 0.22,
    "grid.linestyle": "--",
    "savefig.bbox": "tight",
}
_SAE_COLOR = "#0072B2"
_PCA_COLOR = "#D55E00"
_NULL_COLOR = "#999999"


def plot_comparison(
    output_dir: Path,
    sweep: dict[str, Any],
    overlap: dict[str, Any],
    *,
    label: str,
    dpi: int = 150,
) -> Path:
    """One figure carrying all three claims, one panel each."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / "sae_vs_pca_variance_explained.png"

    ranks = np.array([row["rank"] for row in sweep["curve"]], dtype=float)
    pca_fve = np.array([row["pca_fve"] for row in sweep["curve"]], dtype=float)
    sae_fve = np.array([row["sae_fve"] for row in sweep["curve"]], dtype=float)
    delta = sae_fve - pca_fve

    with plt.rc_context(_PLOT_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

        # (a) claims 1 and 3: FVE at a matched direction budget.
        ax = axes[0]
        ax.plot(ranks, sae_fve, marker="o", ms=4, color=_SAE_COLOR, label="SAE (top-j)")
        ax.plot(ranks, pca_fve, marker="s", ms=4, color=_PCA_COLOR, label="PCA (rank-k)")
        full_fve = sweep["sae_full_fve"]
        mean_l0 = sweep["mean_l0"]
        if math.isfinite(full_fve) and math.isfinite(mean_l0) and mean_l0 >= 1:
            ax.plot(
                [mean_l0],
                [full_fve],
                marker="*",
                ms=14,
                color=_SAE_COLOR,
                linestyle="none",
                label=f"SAE operating point (L0={mean_l0:,.0f}, FVE={full_fve:.3f})",
            )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Directions used per token")
        ax.set_ylabel("Fraction of variance explained")
        ax.set_title("(a) Variance explained at matched budget")
        ax.legend(fontsize=8, frameon=False, loc="lower right")

        # (b) claim 3 on its own axis, where small gaps stay visible.
        ax = axes[1]
        ax.axhline(0.0, color="#444444", lw=1)
        ax.plot(ranks, delta, marker="o", ms=4, color=_SAE_COLOR)
        ax.fill_between(ranks, 0.0, delta, where=delta >= 0, color=_SAE_COLOR, alpha=0.18)
        ax.fill_between(ranks, 0.0, delta, where=delta < 0, color=_PCA_COLOR, alpha=0.18)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Directions used per token")
        ax.set_ylabel("FVE(SAE) - FVE(PCA)")
        ax.set_title("(b) SAE advantage over PCA")

        # (c) claim 2: overlap against the random-direction null.
        ax = axes[2]
        observed = overlap["observed"].numpy()
        null = overlap["null"].numpy()
        observed_max = float(observed.max())
        bins = np.linspace(0.0, float(max(observed_max, null.max(), 1e-3)) * 1.05, 200)
        # A handful of strongly aligned features would otherwise squeeze both
        # distributions into the left edge; the tail is annotated instead.
        view_limit = float(
            max(np.quantile(observed, 0.999), null.max(), 1e-3)
        ) * 1.15
        ax.hist(
            observed,
            bins=bins,
            color=_SAE_COLOR,
            alpha=0.75,
            density=True,
            label="SAE decoder directions",
        )
        ax.hist(
            null,
            bins=bins,
            color=_NULL_COLOR,
            alpha=0.6,
            density=True,
            label="random directions (null)",
        )
        ax.axvline(
            overlap["observed_median"],
            color=_SAE_COLOR,
            ls="--",
            lw=1.2,
            label=f"SAE median {overlap['observed_median']:.3f}",
        )
        ax.axvline(
            overlap["null_median"],
            color=_NULL_COLOR,
            ls="--",
            lw=1.2,
            label=f"null median {overlap['null_median']:.3f}",
        )
        ax.set_xlim(0.0, view_limit)
        if observed_max > view_limit:
            ax.annotate(
                f"max {observed_max:.2f}\n(off axis)",
                xy=(0.97, 0.55),
                xycoords="axes fraction",
                ha="right",
                fontsize=8,
                color=_SAE_COLOR,
            )
        ax.set_xlabel(f"Best |cosine| to any of {overlap['n_components']} PCs")
        ax.set_ylabel("Density")
        ax.set_title("(c) Feature/PC overlap vs null")
        ax.legend(fontsize=8, frameon=False)

        fig.suptitle(f"SAE versus PCA on {label}", y=1.02)
        fig.tight_layout()
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
    return path


def save_results(
    output_dir: Path,
    meta: dict[str, Any],
    sweep: dict[str, Any],
    overlap: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    matched = next(
        (
            row
            for row in reversed(sweep["curve"])
            if row["rank"] <= max(sweep["mean_l0"], 1)
        ),
        None,
    )
    summary = {
        "meta": meta,
        "sae": {
            "full_fve": sweep["sae_full_fve"],
            "mean_l0": sweep["mean_l0"],
        },
        "matched_budget": matched,
        "curve": sweep["curve"],
        "overlap": {
            key: value
            for key, value in overlap.items()
            if key not in {"observed", "null"}
        },
        "total_variance": sweep["total_variance"],
        "total_samples": sweep["total_samples"],
    }
    with open(output_dir / "pca_vs_sae_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    write_curve_csv(output_dir / "variance_explained_curve.csv", sweep)
    np.savez_compressed(
        output_dir / "decoder_component_overlap.npz",
        observed_best_abs_cosine=overlap["observed"].numpy(),
        null_best_abs_cosine=overlap["null"].numpy(),
    )
    write_table(output_dir, comparison_table(sweep))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(cfg: SimpleNamespace) -> None:
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    ckpt_path = Path(cfg.checkpoint_path)
    output_dir = Path(cfg.output_dir)
    device = resolve_device(cfg.device)

    layer_s = f"layer{cfg.layer}" if cfg.layer is not None else "layerX"
    pool_s = cfg.pooling or "pool?"
    label = f"{layer_s}-{pool_s}-{cfg.sae_type or 'sae'}"

    source = prepare_split_source(
        cfg.split_spec,
        max_shards=cfg.max_shards,
        max_samples=cfg.max_samples,
        seed=cfg.seed,
    )

    print(f"\n{'=' * 60}")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Device     : {device}")
    print(f"Output     : {output_dir}")
    print(f"Split      : {describe_source(source)}")

    sae, sae_meta = load_sae(ckpt_path, cfg.sae_type, device=device)
    ranks = resolve_ranks(
        cfg.ranks,
        d_model=sae_meta["d_model"],
        d_hidden=sae_meta["d_hidden"],
        max_rank=cfg.max_rank,
    )
    n_components = max(ranks[-1], int(cfg.overlap_components))
    print(
        f"SAE        : {sae_meta['sae_type']}, d_model={sae_meta['d_model']}, "
        f"d_hidden={sae_meta['d_hidden']}"
    )
    print(f"Budgets    : {', '.join(str(rank) for rank in ranks)}")

    with full_precision_matmul():
        print("\nComputing activation covariance ...")
        cov_stats = compute_activation_covariance(
            source=source,
            d_model=sae_meta["d_model"],
            batch_size=cfg.batch_size,
            center=True,
            device=device,
        )
        print(f"Eigendecomposing covariance ({n_components} components) ...")
        pca = eigendecompose_covariance(cov_stats["cov"], n_components)

        print("Sweeping matched-budget variance explained ...")
        sweep = sweep_variance_explained(
            sae,
            source,
            mean=cov_stats["mean"],
            components=pca["components"],
            ranks=ranks,
            batch_size=cfg.batch_size,
            device=device,
        )

        print("Comparing decoder directions to PCA components ...")
        overlap = decoder_component_overlap(
            sae,
            pca["components"],
            n_components=cfg.overlap_components,
            null_samples=cfg.null_samples,
            seed=cfg.seed,
        )

    meta = {
        **sae_meta,
        "label": label,
        "layer": cfg.layer,
        "pooling": cfg.pooling,
        "checkpoint_path": str(ckpt_path),
        "data_dir": str(source.spec.path),
        "split": source.spec.name,
        "ranks": ranks,
        "pca_components": int(pca["components"].shape[0]),
        "pca_fit_split": source.spec.name,
        "comparison_note": (
            "PCA rank-k uses one global subspace for every row; SAE top-j picks "
            "j atoms per row from d_hidden. Equal budget, unequal capacity and "
            "parameter count. PCA is fit and scored on the same split, which "
            "favours PCA."
        ),
    }
    save_results(output_dir, meta, sweep, overlap)
    figure_path = plot_comparison(output_dir, sweep, overlap, label=label)

    print()
    print(comparison_table(sweep))
    print(
        f"SAE operating point : L0={sweep['mean_l0']:.1f}, "
        f"FVE={sweep['sae_full_fve']:.4f}"
    )
    print(
        f"Decoder/PC overlap  : median {overlap['observed_median']:.4f} "
        f"(null {overlap['null_median']:.4f}), "
        f"{overlap['fraction_above_null_p95']:.1%} of features above null p95"
    )
    print(f"\nSaved to {output_dir}")
    print(f"  pca_vs_sae_summary.json        - curve, overlap, and metadata")
    print(f"  variance_explained_curve.csv   - one row per direction budget")
    print(f"  decoder_component_overlap.npz  - observed and null cosines")
    print(f"  {figure_path.name} - combined comparison figure")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    apply_overrides(cfg, args)
    run(cfg)


if __name__ == "__main__":
    main()
