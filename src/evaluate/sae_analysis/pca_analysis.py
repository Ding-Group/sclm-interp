#!/usr/bin/env python3
"""
PCA analysis for SAE-trained gene-token activation spaces.

This script computes principal directions of the same hidden activation rows
used to train an SAE, then correlates SAE feature activations with the PCA
scores of those rows.

Decoder-direction cosine against the PCA basis lives in
``pca_vs_sae.py``, which reports the same statistic against a random-direction
null distribution; without that null a raw cosine is not interpretable.

Because the SAE is trained on gene-token activations, all comparisons here are
token-level by default. Whole-cell analyses should be built by pooling token
scores per cell before comparing to expression PCA.

Unlike the other evaluations in this package, PCA analyzes a single split: the
covariance and its eigenbasis describe one activation distribution.

Usage:
    python src/evaluate/sae_analysis/pca_analysis.py
    python src/evaluate/sae_analysis/pca_analysis.py --config configs/eval_sae.yaml
    python src/evaluate/sae_analysis/pca_analysis.py --max-samples 50000 --n-components 25
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

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

_DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "eval_sae.yaml"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str | Path | None = None) -> SimpleNamespace:
    raw = evaluation_config(load_yaml_mapping(path or _DEFAULT_CONFIG), "pca")

    ckpt = raw.get("checkpoint", {})
    data = raw.get("data", {})
    out = raw.get("output", {})
    model = raw.get("model", {})

    specs = resolve_split_specs(
        data, legacy_split_key="split", default_splits=("data_train",)
    )
    if len(specs) > 1:
        raise ValueError(
            "PCA analyzes a single activation split, but "
            f"{len(specs)} were configured: {', '.join(s.name for s in specs)}. "
            "Set evaluations.pca.data.split (or a one-element data.splits)."
        )

    checkpoint_path = resolve_checkpoint_path(ckpt)

    flat: dict[str, Any] = {
        "checkpoint_path": checkpoint_path,
        "layer": ckpt.get("layer"),
        "pooling": ckpt.get("pooling"),
        "base_model": ckpt.get("base_model"),
        "prompt_prefix": ckpt.get("prompt_prefix"),
        "split_spec": specs[0],
        "max_shards": data.get("max_shards"),
        "max_samples": data.get("max_samples"),
        "output_dir": resolve_output_dir(
            out,
            "pca_analysis",
            checkpoint_path=checkpoint_path,
            data_dir=data.get("dir"),
            checkpoint_name=resolve_checkpoint_label(ckpt, checkpoint_path),
        ),
        "seed": out.get("seed", 42),
        "sae_type": model.get("sae_type"),
    }
    for section in ("model", "pca", "eval", "infrastructure"):
        flat.update(raw.get(section, {}))

    flat.setdefault("batch_size", 8192)
    flat.setdefault("device", "auto")
    flat.setdefault("n_components", 50)
    flat.setdefault("center", True)
    flat.setdefault("compute_activation_correlations", True)
    flat.setdefault("top_features_per_component", 50)
    return SimpleNamespace(**flat)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run activation-space PCA analysis.")
    parser.add_argument("--config", default=None, help="Path to YAML config.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--n-components", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--skip-activation-corr",
        action="store_true",
        help="Compute PCA only, skipping the activation-correlation pass.",
    )
    return parser.parse_args()


def apply_cli_overrides(cfg: SimpleNamespace, args: argparse.Namespace) -> None:
    if args.max_samples is not None:
        cfg.max_samples = args.max_samples
    if args.max_shards is not None:
        cfg.max_shards = args.max_shards
    if args.n_components is not None:
        cfg.n_components = args.n_components
    if args.output_dir is not None:
        cfg.output_dir = project_path(args.output_dir)
    if args.skip_activation_corr:
        cfg.compute_activation_correlations = False


@contextlib.contextmanager
def full_precision_matmul() -> Iterator[None]:
    """Disable TF32 for float32 matmuls inside the block.

    On Ampere+ GPUs the default float32 matmul path carries only ~10 mantissa
    bits, which is too coarse for accumulated second moments and correlations.
    """
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


# ---------------------------------------------------------------------------
# PCA from streaming covariance
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_activation_covariance(
    source: SplitSource,
    d_model: int,
    batch_size: int,
    center: bool,
    device: str = "cpu",
) -> dict:
    """
    Accumulate the activation mean and second-moment matrix in one pass.

    Sums are accumulated in float64 on the compute device; ``batch_size``
    controls the transient memory of the float64 batch copy.
    """
    check_d_model(source, d_model)

    sum_x = torch.zeros(d_model, dtype=torch.float64, device=device)
    sum_xx = torch.zeros((d_model, d_model), dtype=torch.float64, device=device)
    total = 0

    for i, batch in enumerate(iter_selected_batches(source, batch_size, device)):
        x = batch.double()
        sum_x += x.sum(dim=0)
        sum_xx += x.T @ x
        total += x.shape[0]
        if (i + 1) % 100 == 0:
            print(f"  covariance pass: {total:,} rows processed")

    if total == 0:
        raise RuntimeError(f"No activation rows were processed from {source.spec.path}")

    mean = (sum_x / float(total)).cpu()
    second_moment = (sum_xx / float(total)).cpu()
    if center:
        cov = second_moment - torch.outer(mean, mean)
    else:
        cov = second_moment
        mean = torch.zeros_like(mean)
    cov = (cov + cov.T) / 2  # symmetrize away accumulated asymmetry
    return {"cov": cov, "mean": mean.float(), "total_samples": total}


def eigendecompose_covariance(cov: torch.Tensor, n_components: int) -> dict:
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order].clamp(min=0)
    eigvecs = eigvecs[:, order]

    n_components = min(n_components, eigvals.numel())
    components = eigvecs[:, :n_components].T.contiguous().float()
    explained = eigvals[:n_components].float()
    total_var = float(eigvals.sum().item())
    explained_ratio = explained / max(total_var, 1e-12)

    return {
        "components": components,
        "explained_variance": explained,
        "explained_variance_ratio": explained_ratio,
        "cumulative_explained_variance_ratio": explained_ratio.cumsum(dim=0),
        "total_variance": total_var,
    }


# ---------------------------------------------------------------------------
# SAE/PCA comparisons
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_activation_score_correlations(
    sae: SAE,
    source: SplitSource,
    mean: torch.Tensor,
    components: torch.Tensor,
    batch_size: int,
    device: str,
) -> dict:
    """Pearson correlation between each SAE feature and each PCA score."""
    d_hidden = sae.d_hidden
    n_components = components.shape[0]

    sum_a = torch.zeros(d_hidden, dtype=torch.float64, device=device)
    sum_aa = torch.zeros(d_hidden, dtype=torch.float64, device=device)
    sum_s = torch.zeros(n_components, dtype=torch.float64, device=device)
    sum_ss = torch.zeros(n_components, dtype=torch.float64, device=device)
    sum_as = torch.zeros((d_hidden, n_components), dtype=torch.float64, device=device)
    total = 0

    mean_d = mean.to(device).float()
    comps_d = components.to(device).float()

    for i, batch in enumerate(iter_selected_batches(source, batch_size, device)):
        scores = (batch - mean_d) @ comps_d.T
        acts = sae.encode(batch).float()

        sum_a += acts.sum(dim=0, dtype=torch.float64)
        sum_aa += acts.pow(2).sum(dim=0, dtype=torch.float64)
        sum_s += scores.sum(dim=0, dtype=torch.float64)
        sum_ss += scores.pow(2).sum(dim=0, dtype=torch.float64)
        # (d_hidden, n_components); float64 operands would need a full copy of
        # acts, so accumulate the float32 product instead.
        sum_as += (acts.T @ scores).double()
        total += batch.shape[0]

        if (i + 1) % 100 == 0:
            print(f"  correlation pass: {total:,} rows processed")

    if total == 0:
        raise RuntimeError(f"No activation rows were processed from {source.spec.path}")

    n = float(total)
    mean_a = (sum_a / n).cpu()
    mean_s = (sum_s / n).cpu()
    var_a = ((sum_aa / n).cpu() - mean_a.square()).clamp(min=0.0)
    var_s = ((sum_ss / n).cpu() - mean_s.square()).clamp(min=0.0)
    cov_as = (sum_as / n).cpu() - torch.outer(mean_a, mean_s)

    denom = (
        torch.sqrt(var_a.clamp(min=1e-12)).unsqueeze(1)
        * torch.sqrt(var_s.clamp(min=1e-12)).unsqueeze(0)
    )
    corr = torch.nan_to_num(cov_as / denom, nan=0.0, posinf=0.0, neginf=0.0).float()

    best_abs, best_component = corr.abs().max(dim=1)
    signed_at_best = corr[torch.arange(d_hidden), best_component]

    return {
        "activation_component_corr": corr,
        "best_activation_abs_corr": best_abs,
        "best_activation_component": best_component,
        "best_activation_signed_corr": signed_at_best,
    }


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def top_features_by_component(
    matrix: torch.Tensor,
    top_k: int,
    signed: bool = True,
) -> list[dict]:
    """Rank features by absolute value within each component column."""
    rows: list[dict] = []
    values = matrix.cpu()

    for comp_idx in range(values.shape[1]):
        col = values[:, comp_idx]
        k = min(top_k, col.numel())
        top_vals, top_idx = torch.topk(col.abs(), k=k)
        entries = []
        for abs_val, feat_idx in zip(top_vals.tolist(), top_idx.tolist()):
            entry = {"feature": int(feat_idx), "abs_value": float(abs_val)}
            if signed:
                entry["signed_value"] = float(col[feat_idx].item())
            entries.append(entry)
        rows.append({"component": comp_idx, "features": entries})
    return rows


def build_feature_alignment_rows(activation_alignment: dict | None) -> list[dict]:
    """One row per SAE feature, with its strongest PCA score correlation."""
    if activation_alignment is None:
        return []
    columns: dict[str, torch.Tensor] = {
        "best_activation_component": activation_alignment["best_activation_component"],
        "best_activation_abs_corr": activation_alignment["best_activation_abs_corr"],
        "best_activation_signed_corr": activation_alignment["best_activation_signed_corr"],
    }

    listed = {name: tensor.tolist() for name, tensor in columns.items()}
    n_features = len(listed["best_activation_component"])
    return [
        {"feature": i, **{name: values[i] for name, values in listed.items()}}
        for i in range(n_features)
    ]


def save_results(
    output_dir: Path,
    meta: dict,
    pca: dict,
    mean: torch.Tensor,
    activation_alignment: dict | None,
    top_k: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "meta": meta,
        "pca": {
            "n_components": int(pca["components"].shape[0]),
            "total_variance": float(pca["total_variance"]),
            "explained_variance": pca["explained_variance"].tolist(),
            "explained_variance_ratio": pca["explained_variance_ratio"].tolist(),
            "cumulative_explained_variance_ratio": pca[
                "cumulative_explained_variance_ratio"
            ].tolist(),
        },
        "top_features_by_activation_correlation": (
            None
            if activation_alignment is None
            else top_features_by_component(
                activation_alignment["activation_component_corr"],
                top_k=top_k,
                signed=True,
            )
        ),
    }
    with open(output_dir / "pca_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    npz_data = {
        "components": pca["components"].numpy(),
        "mean": mean.numpy(),
        "explained_variance": pca["explained_variance"].numpy(),
        "explained_variance_ratio": pca["explained_variance_ratio"].numpy(),
    }
    if activation_alignment is not None:
        npz_data.update(
            {key: value.numpy() for key, value in activation_alignment.items()}
        )
    np.savez_compressed(output_dir / "pca_alignment_arrays.npz", **npz_data)

    rows = build_feature_alignment_rows(activation_alignment)
    if rows:
        with open(
            output_dir / "feature_pca_alignment.csv", "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

_PLOT_STYLE = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "savefig.bbox": "tight",
}


def _alignment_heatmap(
    matrix: np.ndarray, title: str, colorbar_label: str, path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=min(1.0, max(0.1, matrix.max())))
    ax.set_xlabel("PCA component")
    ax.set_ylabel("SAE feature")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _histogram(values: np.ndarray, xlabel: str, title: str, color: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(values, bins=60, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of SAE features")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_figures(
    output_dir: Path,
    pca: dict,
    activation_alignment: dict | None,
) -> None:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(_PLOT_STYLE):
        evr = pca["explained_variance_ratio"].numpy()
        cum = pca["cumulative_explained_variance_ratio"].numpy()
        xs = np.arange(1, len(evr) + 1)

        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.bar(xs, evr, color="#0072B2", alpha=0.8, label="per component")
        ax.plot(xs, cum, color="#D55E00", marker="o", ms=3, label="cumulative")
        ax.set_xlabel("PCA component")
        ax.set_ylabel("Explained variance ratio")
        ax.set_title("Activation-space PCA explained variance")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "explained_variance.png", dpi=150)
        plt.close(fig)

        if activation_alignment is not None:
            _histogram(
                activation_alignment["best_activation_abs_corr"].numpy(),
                "Best absolute activation correlation to any PCA score",
                "SAE activation alignment to PCA token scores",
                "#CC79A7",
                fig_dir / "activation_best_abs_corr_hist.png",
            )
            _alignment_heatmap(
                activation_alignment["activation_component_corr"].abs().numpy(),
                "Absolute activation correlation: SAE features x PCA scores",
                "abs correlation",
                fig_dir / "activation_component_corr_heatmap.png",
            )


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
    label = f"{layer_s}-{pool_s}-{cfg.sae_type}-{ckpt_path.stem or 'checkpoint'}"

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
    print(
        f"SAE        : {sae_meta['sae_type']}, d_model={sae_meta['d_model']}, "
        f"d_hidden={sae_meta['d_hidden']}, expansion={sae_meta['expansion']}"
    )

    with full_precision_matmul():
        print("\nComputing activation covariance ...")
        cov_stats = compute_activation_covariance(
            source=source,
            d_model=sae_meta["d_model"],
            batch_size=cfg.batch_size,
            center=cfg.center,
            device=device,
        )

        print("Eigendecomposing covariance ...")
        pca = eigendecompose_covariance(cov_stats["cov"], cfg.n_components)
        mean = cov_stats["mean"]

        activation_alignment = None
        if cfg.compute_activation_correlations:
            print("Comparing SAE feature activations to PCA token scores ...")
            activation_alignment = compute_activation_score_correlations(
                sae=sae,
                source=source,
                mean=mean,
                components=pca["components"],
                batch_size=cfg.batch_size,
                device=device,
            )

    meta = {
        **sae_meta,
        "label": label,
        "layer": cfg.layer,
        "pooling": cfg.pooling,
        "checkpoint_path": str(ckpt_path),
        "data_dir": str(source.spec.path),
        "split": source.spec.name,
        "n_samples": int(cov_stats["total_samples"]),
        "num_shards": len(source.shard_paths),
        "centered": bool(cfg.center),
        "analysis_note": (
            "PCA was computed on gene-token hidden activations, the same row "
            "space used to train the SAE. Decoder cosine is a direct direction "
            "comparison; activation correlation compares SAE feature values to "
            "PCA scores on the same token rows."
        ),
    }

    print(f"Saving outputs to {output_dir} ...")
    save_results(
        output_dir=output_dir,
        meta=meta,
        pca=pca,
        mean=mean,
        activation_alignment=activation_alignment,
        top_k=cfg.top_features_per_component,
    )
    plot_figures(output_dir, pca, activation_alignment)

    print("\nDone.")
    print("  pca_summary.json          - explained variance and top features")
    print("  pca_alignment_arrays.npz  - components and alignment matrices")
    if activation_alignment is not None:
        print("  feature_pca_alignment.csv - one feature per row")
    print("  figures/                  - variance and alignment diagnostics")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    apply_cli_overrides(cfg, args)
    run(cfg)


if __name__ == "__main__":
    main()
