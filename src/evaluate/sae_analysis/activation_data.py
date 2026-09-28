"""
Shared activation-shard access for the SAE evaluation suite.

Every evaluation in this package streams the same corpus: safetensors shards of
hidden activations, organized as ``<data.dir>/<split>/shard_*.safetensors``.
This module owns split resolution, shard listing, row sampling, and batching so
the evaluations agree on which rows they evaluate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from safetensors import safe_open

from evaluate.model_loading import project_path, resolve_activation_dir


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SplitSpec:
    """One evaluated activation directory."""

    name: str    # config split name, e.g. "data_test", or an unseen-dir name
    label: str   # output subdirectory, e.g. "test"
    path: Path


@dataclass(frozen=True)
class SplitSource:
    """A split with its shard list and sampled row indices resolved."""

    spec: SplitSpec
    shard_paths: tuple[Path, ...]
    row_indices: tuple[np.ndarray | None, ...]
    num_rows: int
    total_rows: int
    d_model: int

    @property
    def label(self) -> str:
        return self.spec.label

    @property
    def sampled(self) -> bool:
        return self.num_rows < self.total_rows


def split_label(split: str) -> str:
    """Short output-directory name for a split (``data_test`` -> ``test``)."""
    return split.removeprefix("data_")


def split_dir(base: Path, split: str) -> Path:
    """Append the split subdirectory unless ``base`` already points at it."""
    if base.name == split or base.parent.name == split:
        return base
    return base / split


def resolve_split_specs(
    data_cfg: dict[str, Any],
    *,
    legacy_split_key: str | None = None,
    default_splits: Sequence[str] = ("data_train",),
) -> list[SplitSpec]:
    """
    Resolve the activation directories to evaluate.

    Precedence:
      1. ``data.splits`` — split names appended to ``data.dir``.
      2. ``data.<legacy_split_key>`` (e.g. ``split``) — a single split, for the
         evaluations that analyse exactly one.
      3. ``default_splits``.
    """
    splits = data_cfg.get("splits")
    if not splits:
        legacy_split = data_cfg.get(legacy_split_key) if legacy_split_key else None
        splits = [legacy_split] if legacy_split else list(default_splits)
    if isinstance(splits, (str, Path)):
        raise ValueError("data.splits must be a list of split names.")

    base = resolve_activation_dir(data_cfg, path_key="dir")
    specs: list[SplitSpec] = []
    seen: set[str] = set()
    for split in splits:
        name = str(split)
        if name in seen:
            continue
        seen.add(name)
        specs.append(SplitSpec(name=name, label=split_label(name), path=split_dir(base, name)))
    if not specs:
        raise ValueError("No activation splits configured.")
    return specs


def normalize_dir_entries(entries: Any) -> list[SplitSpec]:
    """Accept a list of path strings or ``{name, dir}`` mappings."""
    specs: list[SplitSpec] = []
    for entry in entries or []:
        if isinstance(entry, (str, Path)):
            path = project_path(entry)
            name = path.name
        elif isinstance(entry, dict):
            path = project_path(entry.get("dir"))
            if path is None:
                raise ValueError(f"Directory entry is missing 'dir': {entry!r}")
            name = str(entry.get("name") or path.name)
        else:
            raise ValueError(f"Unsupported directory entry: {entry!r}")
        specs.append(SplitSpec(name=name, label=split_label(name), path=path))
    return specs


# ---------------------------------------------------------------------------
# Shard sampling
# ---------------------------------------------------------------------------

def list_activation_shards(data_dir: Path, max_shards: int | None = None) -> list[Path]:
    shard_paths = sorted(Path(data_dir).glob("*.safetensors"))
    if max_shards is not None:
        shard_paths = shard_paths[:max_shards]
    if not shard_paths:
        raise FileNotFoundError(f"No .safetensors shards found in {data_dir}")
    return shard_paths


def read_shard_shapes(shard_paths: Sequence[Path]) -> tuple[list[int], int]:
    """Read per-shard row counts and the activation width from headers only."""
    sizes: list[int] = []
    widths: set[int] = set()
    for path in shard_paths:
        with safe_open(path, framework="pt", device="cpu") as f:
            shape = f.get_slice("activations").get_shape()
        sizes.append(int(shape[0]))
        widths.add(int(shape[1]))
    if len(widths) > 1:
        raise ValueError(
            f"Inconsistent activation widths across shards in "
            f"{shard_paths[0].parent}: {sorted(widths)}"
        )
    return sizes, widths.pop()


def select_rows_per_shard(
    shard_sizes: Sequence[int],
    max_samples: int | None,
    rng: np.random.Generator,
) -> list[np.ndarray | None]:
    """
    Choose which rows to load from each shard.

    Returns one entry per shard: ``None`` means "use every row", otherwise an
    int64 array of shard-local row indices. Sampling is done once over the
    concatenated corpus so every row is equally likely regardless of shard size.
    """
    total = int(sum(shard_sizes))
    if max_samples is None or max_samples >= total:
        return [None] * len(shard_sizes)

    selected = np.sort(rng.choice(total, size=max_samples, replace=False))
    out: list[np.ndarray | None] = []
    offset = 0
    for size in shard_sizes:
        mask = (selected >= offset) & (selected < offset + size)
        out.append((selected[mask] - offset).astype(np.int64))
        offset += size
    return out


def prepare_split_source(
    spec: SplitSpec,
    *,
    max_shards: int | None = None,
    max_samples: int | None = None,
    seed: int = 42,
) -> SplitSource:
    """Resolve shards and sampled rows for one split.

    Row selection is seeded per split, so a split's sampled rows do not depend
    on how many other splits are evaluated alongside it.
    """
    shard_paths = list_activation_shards(spec.path, max_shards)
    shard_sizes, d_model = read_shard_shapes(shard_paths)
    row_indices = select_rows_per_shard(
        shard_sizes, max_samples, np.random.default_rng(seed)
    )
    num_rows = sum(
        size if rows is None else len(rows)
        for size, rows in zip(shard_sizes, row_indices)
    )
    return SplitSource(
        spec=spec,
        shard_paths=tuple(shard_paths),
        row_indices=tuple(row_indices),
        num_rows=int(num_rows),
        total_rows=int(sum(shard_sizes)),
        d_model=int(d_model),
    )


def iter_selected_batches(
    source: SplitSource,
    batch_size: int,
    device: str,
) -> Iterator[torch.Tensor]:
    """Yield the selected rows of a split as batches on ``device``."""
    for shard_path, rows in zip(source.shard_paths, source.row_indices):
        if rows is not None and len(rows) == 0:
            continue
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            data = f.get_tensor("activations").float()
        if rows is not None:
            data = data[rows]
        for start in range(0, data.shape[0], batch_size):
            yield data[start : start + batch_size].to(device)


def check_d_model(source: SplitSource, d_model: int) -> None:
    if source.d_model != d_model:
        raise ValueError(
            f"Activation width {source.d_model} in {source.spec.path} does not "
            f"match SAE d_model {d_model}."
        )


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def describe_source(source: SplitSource) -> str:
    sampled = f" (sampled from {source.total_rows:,})" if source.sampled else ""
    return (
        f"{source.spec.name:<12} {source.num_rows:>12,} rows{sampled}"
        f"  [{len(source.shard_paths)} shards]  {source.spec.path}"
    )
