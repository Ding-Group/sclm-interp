"""
Reconstruction statistics used by ``reconstruction_eval.py``.

It streams activation shards through an SAE and accumulates the same sufficient
statistics (``sum_x``, ``sum_xx``, ``sum_r``, ``sum_rr``, ``sum_l0``, row count).
Deriving the metrics in one place keeps the two evaluations from drifting apart.

Because the accumulators are plain sums, pooling several splits into a single
"combined" assessment is exact: add the sums and derive metrics once.

Scale-free companions to the raw-scale metrics
----------------------------------------------
``mse``, ``fve`` and ``nmse`` are all dominated by the highest-norm rows: a
handful of outlier tokens can carry a large share of ``E[||x||^2]``, so a good
score does not by itself establish that a typical row is reconstructed well.
Two scale-free metrics are accumulated alongside them.

1. Cosine similarity between ``x`` and its reconstruction, per row. It ignores
   magnitude entirely and answers "does the reconstruction point the right way".

2. Metrics on unit-normalized rows: every row is divided by its own ``||x||``
   before the sums are taken, so each row contributes equally regardless of
   norm. ``fve_normalized`` is the direct scale-free counterpart of ``fve``, and
   ``nmse_normalized`` equals ``E[||r||^2 / ||x||^2]`` — the *unweighted* mean
   per-row relative error, where ``nmse`` is the norm-weighted one.

Both are exactly poolable in the same way as the raw sums: cosine similarity
accumulates as a sum over rows, and the normalized sums are ordinary per-dim
sums over rescaled rows.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import torch

from sae import SAE
from evaluate.sae_analysis.activation_data import (
    SplitSource,
    check_d_model,
    iter_selected_batches,
)

# Relative floor below which an activation dimension is treated as constant and
# excluded from per-dimension FVE (its ratio would be numerical noise).
_DEGENERATE_VAR_RATIO = 1e-12

SUM_KEYS = (
    "sum_x",
    "sum_xx",
    "sum_r",
    "sum_rr",
    # Same four sums over unit-normalized rows (x / ||x||).
    "sum_xn",
    "sum_xxn",
    "sum_rn",
    "sum_rrn",
)

# Scalar accumulators that pool by addition.
COUNT_KEYS = ("sum_l0", "sum_cos", "sum_cos_sq", "n_normalized", "n_cosine")

SCALAR_METRIC_KEYS = (
    "total_samples",
    "num_shards",
    "d_model",
    "mse",
    "rmse",
    "mean_squared_l2_error",
    "root_mean_squared_l2_error",
    "nmse",
    "relative_l2_error",
    "fve",
    "mean_per_dim_fve",
    "n_degenerate_dims",
    "mean_l0",
    # Scale-free metrics.
    "mean_cosine_similarity",
    "std_cosine_similarity",
    "min_cosine_similarity",
    "n_cosine_rows",
    "fve_normalized",
    "mean_per_dim_fve_normalized",
    "n_degenerate_dims_normalized",
    "nmse_normalized",
    "relative_l2_error_normalized",
    "n_normalized_rows",
)

PER_DIM_METRIC_KEYS = (
    "mse_per_dim",
    "per_dim_fve",
    "var_x",
    "var_r",
    "per_dim_fve_normalized",
    "var_x_normalized",
    "var_r_normalized",
)

# Column layout shared by both evaluations' summary tables.
METRIC_HEADERS = (
    "Samples",
    "MSE",
    "RMSE",
    "NMSE",
    "Rel L2",
    "FVE",
    "PerDim FVE",
    "CosSim",
    "NormFVE",
    "NormRelL2",
    "L0",
)


# ---------------------------------------------------------------------------
# Streaming accumulation
# ---------------------------------------------------------------------------

@torch.no_grad()
def accumulate_reconstruction_sums(
    sae: SAE,
    source: SplitSource,
    *,
    batch_size: int = 8192,
    device: str = "cpu",
    progress_every: int = 50,
) -> dict[str, Any]:
    """
    Stream one split through the SAE and accumulate reconstruction sums.

    Reductions are accumulated in float64 on the compute device, so only
    ``d_model``-sized vectors ever cross the device boundary.
    """
    d_model = int(sae.d_model)
    check_d_model(source, d_model)

    def accumulator() -> torch.Tensor:
        return torch.zeros(d_model, dtype=torch.float64, device=device)

    def scalar() -> torch.Tensor:
        return torch.zeros((), dtype=torch.float64, device=device)

    sum_x, sum_xx, sum_r, sum_rr = (accumulator() for _ in range(4))
    sum_xn, sum_xxn, sum_rn, sum_rrn = (accumulator() for _ in range(4))
    sum_l0, sum_cos, sum_cos_sq = (scalar() for _ in range(3))
    n_normalized, n_cosine = (scalar() for _ in range(2))
    min_cos = torch.full((), math.inf, dtype=torch.float64, device=device)
    total = 0

    for i, batch in enumerate(iter_selected_batches(source, batch_size, device)):
        encoded = sae.encode(batch)
        reconstruction = sae.decode(encoded)
        residual = batch - reconstruction

        sum_x += batch.sum(dim=0, dtype=torch.float64)
        sum_xx += batch.pow(2).sum(dim=0, dtype=torch.float64)
        sum_r += residual.sum(dim=0, dtype=torch.float64)
        sum_rr += residual.pow(2).sum(dim=0, dtype=torch.float64)
        sum_l0 += (encoded > 0).sum(dtype=torch.float64)
        total += batch.shape[0]

        # --- scale-free statistics -------------------------------------
        # A zero row has no direction and no scale, so it can contribute to
        # neither the cosine nor the normalized sums; it is dropped from those
        # counts rather than being scored as a perfect or a failed row. The
        # masking is branch-free and the counts stay on-device: reading them
        # back per batch would stall the pipeline on a host sync.
        norm_x = batch.norm(dim=1)
        norm_recon = reconstruction.norm(dim=1)
        has_direction = norm_x > 0
        zero = torch.zeros_like(norm_x)

        inv_norm = torch.where(has_direction, norm_x.reciprocal(), zero).unsqueeze(1)
        normalized_x = batch * inv_norm
        normalized_r = residual * inv_norm

        sum_xn += normalized_x.sum(dim=0, dtype=torch.float64)
        sum_xxn += normalized_x.pow(2).sum(dim=0, dtype=torch.float64)
        sum_rn += normalized_r.sum(dim=0, dtype=torch.float64)
        sum_rrn += normalized_r.pow(2).sum(dim=0, dtype=torch.float64)
        n_normalized += has_direction.sum(dtype=torch.float64)

        # Cosine additionally needs a non-degenerate reconstruction: a row the
        # SAE maps to the origin has no direction to compare against. Excluded
        # rows are set to 0 so they are inert in the sums, and to +inf for the
        # running minimum.
        comparable = has_direction & (norm_recon > 0)
        denominator = torch.where(comparable, norm_x * norm_recon, torch.ones_like(norm_x))
        cosine = torch.where(
            comparable,
            ((batch * reconstruction).sum(dim=1) / denominator).clamp(-1.0, 1.0),
            zero,
        )
        sum_cos += cosine.sum(dtype=torch.float64)
        sum_cos_sq += cosine.pow(2).sum(dtype=torch.float64)
        min_cos = torch.minimum(
            min_cos,
            torch.where(comparable, cosine, torch.full_like(cosine, math.inf))
            .min()
            .double(),
        )
        n_cosine += comparable.sum(dtype=torch.float64)

        if progress_every and (i + 1) % progress_every == 0:
            print(f"    {total:,} rows processed")

    if total == 0:
        raise RuntimeError(f"No activation rows were processed from {source.spec.path}")

    return {
        "sum_x": sum_x.cpu(),
        "sum_xx": sum_xx.cpu(),
        "sum_r": sum_r.cpu(),
        "sum_rr": sum_rr.cpu(),
        "sum_xn": sum_xn.cpu(),
        "sum_xxn": sum_xxn.cpu(),
        "sum_rn": sum_rn.cpu(),
        "sum_rrn": sum_rrn.cpu(),
        "sum_l0": float(sum_l0.item()),
        "sum_cos": float(sum_cos.item()),
        "sum_cos_sq": float(sum_cos_sq.item()),
        "min_cos": float(min_cos.item()),
        "n_normalized": int(n_normalized.item()),
        "n_cosine": int(n_cosine.item()),
        "total_samples": int(total),
        "num_shards": len(source.shard_paths),
        "d_model": d_model,
    }


def combine_sums(sums_list: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pool per-split sums into the statistics of the concatenated corpus."""
    if not sums_list:
        raise ValueError("combine_sums requires at least one set of sums.")

    d_model = sums_list[0]["d_model"]
    if any(s["d_model"] != d_model for s in sums_list):
        raise ValueError("Cannot combine splits with different activation widths.")

    combined: dict[str, Any] = {key: sum(s[key] for s in sums_list) for key in SUM_KEYS}
    for key in COUNT_KEYS:
        combined[key] = sum(s[key] for s in sums_list)
    combined["sum_l0"] = float(combined["sum_l0"])
    combined["n_normalized"] = int(combined["n_normalized"])
    combined["n_cosine"] = int(combined["n_cosine"])
    # A running minimum pools by taking the minimum, not by summing.
    combined["min_cos"] = min(s["min_cos"] for s in sums_list)
    combined["total_samples"] = int(sum(s["total_samples"] for s in sums_list))
    combined["num_shards"] = int(sum(s["num_shards"] for s in sums_list))
    combined["d_model"] = d_model
    return combined


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _variance_explained(
    sum_x: torch.Tensor,
    sum_xx: torch.Tensor,
    sum_r: torch.Tensor,
    sum_rr: torch.Tensor,
    n: float,
) -> dict[str, Any]:
    """
    Centered variance-explained statistics for one set of per-dim sums.

    Shared by the raw-scale metrics and their unit-normalized counterparts, so
    the two cannot drift apart in definition.
    """
    mean_x = sum_x / n
    mean_r = sum_r / n
    var_x = (sum_xx / n - mean_x.pow(2)).clamp(min=0.0)
    var_r = (sum_rr / n - mean_r.pow(2)).clamp(min=0.0)

    total_var_x = float(var_x.sum())
    total_var_r = float(var_r.sum())
    fve = 1.0 - total_var_r / total_var_x if total_var_x > 0.0 else math.nan

    # Dimensions with (near-)zero variance carry no signal to explain; their FVE
    # ratio is numerical noise, so exclude them from the per-dimension average.
    var_floor = float(var_x.max()) * _DEGENERATE_VAR_RATIO
    valid = var_x > var_floor
    per_dim_fve = torch.where(
        valid,
        1.0 - var_r / var_x.clamp(min=torch.finfo(torch.float64).tiny),
        torch.full_like(var_x, math.nan),
    )
    mean_per_dim_fve = float(torch.nanmean(per_dim_fve)) if bool(valid.any()) else math.nan

    return {
        "fve": fve,
        "mean_per_dim_fve": mean_per_dim_fve,
        "n_degenerate_dims": int((~valid).sum()),
        "per_dim_fve": per_dim_fve,
        "var_x": var_x,
        "var_r": var_r,
    }


def _nan_per_dim(d_model: int) -> torch.Tensor:
    return torch.full((d_model,), math.nan, dtype=torch.float64)


def reconstruction_metrics(sums: dict[str, Any]) -> dict[str, Any]:
    """
    Derive reconstruction metrics from accumulated sums.

    Variance-based metrics (``fve``, ``per_dim_fve``) are centered; error-based
    metrics (``mse``, ``nmse``) use uncentered second moments, so ``nmse`` is
    not ``1 - fve`` unless the activations are mean-zero.

    Every raw-scale metric is dominated by the highest-norm rows. The ``*_
    normalized`` metrics repeat the same derivation on rows rescaled to unit
    norm, giving each row equal weight, and ``mean_cosine_similarity`` drops
    magnitude altogether. Report them together: a large gap between ``fve`` and
    ``fve_normalized`` means the fit is carried by a high-norm minority.
    """
    total = int(sums["total_samples"])
    if total <= 0:
        raise ValueError("Cannot derive metrics from zero samples.")

    n = float(total)
    d_model = int(sums["d_model"])
    sum_xx, sum_rr = sums["sum_xx"], sums["sum_rr"]

    raw = _variance_explained(sums["sum_x"], sum_xx, sums["sum_r"], sum_rr, n)

    mse_per_dim = sum_rr / n                       # E[r_d^2]
    second_moment_per_dim = sum_xx / n             # E[x_d^2]

    residual_energy = float(mse_per_dim.sum())     # E[||r||^2]
    signal_energy = float(second_moment_per_dim.sum())

    mse = float(mse_per_dim.mean())
    nmse = residual_energy / signal_energy if signal_energy > 0.0 else math.nan

    # --- scale-free metrics -------------------------------------------------
    n_cosine = int(sums["n_cosine"])
    if n_cosine > 0:
        mean_cos = float(sums["sum_cos"]) / n_cosine
        # Var = E[c^2] - E[c]^2, clamped: the subtraction can go slightly
        # negative when the cosines are tightly clustered near 1.
        var_cos = max(float(sums["sum_cos_sq"]) / n_cosine - mean_cos**2, 0.0)
        std_cos = math.sqrt(var_cos)
        min_cos = float(sums["min_cos"])
    else:
        mean_cos = std_cos = min_cos = math.nan

    n_normalized = int(sums["n_normalized"])
    if n_normalized > 0:
        n_norm = float(n_normalized)
        normalized = _variance_explained(
            sums["sum_xn"], sums["sum_xxn"], sums["sum_rn"], sums["sum_rrn"], n_norm
        )
        # Denominator is 1.0 up to float error (unit-norm rows), so this ratio
        # is E[||r||^2 / ||x||^2]: the mean per-row relative error, unweighted.
        normalized_signal = float((sums["sum_xxn"] / n_norm).sum())
        normalized_residual = float((sums["sum_rrn"] / n_norm).sum())
        nmse_normalized = (
            normalized_residual / normalized_signal
            if normalized_signal > 0.0
            else math.nan
        )
    else:
        normalized = {
            "fve": math.nan,
            "mean_per_dim_fve": math.nan,
            "n_degenerate_dims": d_model,
            "per_dim_fve": _nan_per_dim(d_model),
            "var_x": _nan_per_dim(d_model),
            "var_r": _nan_per_dim(d_model),
        }
        nmse_normalized = math.nan

    return {
        "total_samples": total,
        "num_shards": int(sums["num_shards"]),
        "d_model": d_model,
        "mse": mse,
        "rmse": math.sqrt(max(mse, 0.0)),
        "mean_squared_l2_error": residual_energy,
        "root_mean_squared_l2_error": math.sqrt(max(residual_energy, 0.0)),
        "nmse": nmse,
        "relative_l2_error": math.sqrt(nmse) if nmse >= 0.0 else math.nan,
        "fve": raw["fve"],
        "mean_per_dim_fve": raw["mean_per_dim_fve"],
        "n_degenerate_dims": raw["n_degenerate_dims"],
        "mean_l0": float(sums["sum_l0"]) / n,
        "mean_cosine_similarity": mean_cos,
        "std_cosine_similarity": std_cos,
        "min_cosine_similarity": min_cos,
        "n_cosine_rows": n_cosine,
        "fve_normalized": normalized["fve"],
        "mean_per_dim_fve_normalized": normalized["mean_per_dim_fve"],
        "n_degenerate_dims_normalized": normalized["n_degenerate_dims"],
        "nmse_normalized": nmse_normalized,
        "relative_l2_error_normalized": (
            math.sqrt(nmse_normalized) if nmse_normalized >= 0.0 else math.nan
        ),
        "n_normalized_rows": n_normalized,
        "mse_per_dim": mse_per_dim.float().numpy(),
        "per_dim_fve": raw["per_dim_fve"].float().numpy(),
        "var_x": raw["var_x"].float().numpy(),
        "var_r": raw["var_r"].float().numpy(),
        "per_dim_fve_normalized": normalized["per_dim_fve"].float().numpy(),
        "var_x_normalized": normalized["var_x"].float().numpy(),
        "var_r_normalized": normalized["var_r"].float().numpy(),
    }


def json_scalars(metrics: dict[str, Any]) -> dict[str, Any]:
    """Scalar metrics only, with non-finite values mapped to null."""
    out: dict[str, Any] = {}
    for key in SCALAR_METRIC_KEYS:
        value = metrics[key]
        if isinstance(value, float) and not math.isfinite(value):
            out[key] = None
        else:
            out[key] = value
    return out


def per_dim_arrays(metrics: dict[str, Any], prefix: str) -> dict[str, np.ndarray]:
    return {f"{prefix}__{key}": metrics[key] for key in PER_DIM_METRIC_KEYS}


def metric_cells(metrics: dict[str, Any]) -> list[str]:
    """Formatted cells matching ``METRIC_HEADERS``."""
    return [
        f"{metrics['total_samples']:,}",
        f"{metrics['mse']:.4e}",
        f"{metrics['rmse']:.4e}",
        f"{metrics['nmse']:.4f}",
        f"{metrics['relative_l2_error']:.4f}",
        f"{metrics['fve']:.4f}",
        f"{metrics['mean_per_dim_fve']:.4f}",
        f"{metrics['mean_cosine_similarity']:.4f}",
        f"{metrics['fve_normalized']:.4f}",
        f"{metrics['relative_l2_error_normalized']:.4f}",
        f"{metrics['mean_l0']:.2f}",
    ]
