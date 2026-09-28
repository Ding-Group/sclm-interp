"""
Shared SAE loading and evaluation path helpers.

Evaluation scripts should use this module instead of reimplementing checkpoint
reconstruction. The helpers intentionally avoid importing src/train.py so that
analysis jobs do not need Lightning just to resolve a trained SAE.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
import yaml

from sae import JumpReluSAE, SAE, TopKSAE, VanillaSAE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIRS = {
    "feature_analysis": "feature_analysis",
    "reconstruction": "reconstruction",
    "pca_analysis": "pca_analysis",
    "pca_vs_sae": "pca_vs_sae",
}
DEPRECATED_CHECKPOINT_KEYS = (
    "checkpoint_name",
    "step",
    "train_config",
    "use_train_config",
)
DEPRECATED_TRAIN_CONFIG_KEYS = ("train_config", "use_train_config")
LAYER_DIR_RE = re.compile(r"layer\d+")


def project_path(path: str | Path | None) -> Path | None:
    """Resolve a config path relative to the project root."""
    if path is None:
        return None
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    path = project_path(path)
    if path is None:
        raise ValueError("YAML path must not be None")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return raw


def config_section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    section = raw.get(name, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"Expected '{name}' config section to be a mapping.")
    return section


def _merged_section(
    raw: dict[str, Any],
    override: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    merged = config_section(raw, name).copy()
    merged.update(config_section(override, name))
    return merged


def evaluation_config(raw: dict[str, Any], name: str) -> dict[str, Any]:
    """
    Compose one evaluation's config from shared eval.yaml sections.

    Legacy single-evaluation YAMLs without an ``evaluations`` section are still
    accepted by returning the raw mapping after validation.
    """
    reject_train_config_references(raw)

    evaluations = raw.get("evaluations")
    if evaluations is None:
        return raw
    if not isinstance(evaluations, dict):
        raise ValueError("Expected 'evaluations' config section to be a mapping.")

    override = evaluations.get(name, {}) or {}
    if not isinstance(override, dict):
        raise ValueError(f"Expected evaluations.{name} to be a mapping.")

    merged: dict[str, Any] = {}
    for section in (
        "checkpoint",
        "data",
        "model",
        "output",
        "infrastructure",
        "eval",
        "pca",
    ):
        value = _merged_section(raw, override, section)
        if value:
            merged[section] = value
    merged["enabled"] = override.get("enabled", True)
    return merged


def evaluation_enabled(raw: dict[str, Any], name: str) -> bool:
    evaluations = raw.get("evaluations")
    if evaluations is None:
        return True
    if not isinstance(evaluations, dict):
        raise ValueError("Expected 'evaluations' config section to be a mapping.")

    cfg = evaluations.get(name, {}) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected evaluations.{name} to be a mapping.")
    return bool(cfg.get("enabled", True))


def reject_train_config_references(raw: dict[str, Any]) -> None:
    offenders: list[str] = []
    for section_name in (None, "checkpoint", "data"):
        section = raw if section_name is None else raw.get(section_name, {})
        if not isinstance(section, dict):
            continue
        prefix = "" if section_name is None else f"{section_name}."
        offenders.extend(
            f"{prefix}{key}"
            for key in DEPRECATED_TRAIN_CONFIG_KEYS
            if key in section
        )

    if offenders:
        keys = ", ".join(offenders)
        raise ValueError(
            f"{keys} are no longer supported in evaluation configs. "
            "Set explicit checkpoint, data, and vocab paths instead."
        )


def checkpoint_file_stem(name: Any) -> str:
    """Reduce a ``checkpoint.name`` to a bare file stem (no ``.ckpt`` suffix)."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("checkpoint.name must be a non-empty string.")
    stem = name.strip()
    if stem.endswith(".ckpt"):
        stem = stem[: -len(".ckpt")]
    return safe_path_name(stem, "checkpoint.name")


def resolve_checkpoint_path(checkpoint_cfg: dict[str, Any]) -> Path:
    deprecated = [key for key in DEPRECATED_CHECKPOINT_KEYS if key in checkpoint_cfg]
    if deprecated:
        keys = ", ".join(f"checkpoint.{key}" for key in deprecated)
        raise ValueError(
            f"{keys} are no longer supported. "
            "Set checkpoint.path to the exact .ckpt file instead."
        )

    explicit_path = checkpoint_cfg.get("path")
    if explicit_path:
        return project_path(explicit_path)

    explicit_dir = checkpoint_cfg.get("dir")
    if explicit_dir:
        checkpoint_dir = project_path(explicit_dir)
    else:
        raise ValueError("Set checkpoint.path or checkpoint.dir in the eval config.")

    # checkpoint.name selects the file within the run directory; without it the
    # Lightning-written last.ckpt is the conventional default.
    name = checkpoint_cfg.get("name")
    stem = checkpoint_file_stem(name) if name is not None else "last"
    return checkpoint_dir / f"{stem}.ckpt"


def resolve_checkpoint_label(
    checkpoint_cfg: dict[str, Any],
    checkpoint_path: str | Path,
) -> str:
    """Name the checkpoint level of a result directory.

    ``checkpoint.name`` wins so a run can label its results explicitly; otherwise
    the checkpoint file's own stem names the level (``last.ckpt`` -> ``last``).
    """
    name = checkpoint_cfg.get("name")
    if name is not None:
        return checkpoint_file_stem(name)
    return safe_path_name(Path(checkpoint_path).stem, "checkpoint file stem")


def resolve_activation_dir(
    data_cfg: dict[str, Any],
    *,
    path_key: str = "dir",
    split_key: str | None = None,
) -> Path:
    explicit_path = data_cfg.get(path_key)
    if explicit_path:
        path = project_path(explicit_path)
        split = data_cfg.get(split_key) if split_key else None
        if split and path.name != split and path.parent.name != split:
            return path / str(split)
        return path

    raise ValueError(f"Set data.{path_key} in the eval config.")


def safe_path_name(value: Any, label: str) -> str:
    """Reduce a config or metadata value to a single safe path component."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    if not safe:
        raise ValueError(f"{label} does not contain a safe path component.")
    return safe


def dataset_identity_from_path(path: str | Path) -> tuple[str | None, str | None]:
    """Infer ``(base_model_name, dataset_name)`` from an extraction layout.

    Activation and vocabulary trees are laid out as
    ``<base_model>/layer<N>/<dataset>/...``, so the component before the
    ``layer<N>`` directory names the base model and the one after it names the
    dataset.
    """
    parts = Path(path).parts
    for index, part in enumerate(parts):
        if LAYER_DIR_RE.fullmatch(part) and index > 0 and index + 1 < len(parts):
            return parts[index - 1], parts[index + 1]
    return None, None


def resolve_result_identity(
    checkpoint_path: str | Path,
    *,
    data_dir: str | Path | None = None,
    evaluation_name: str = "evaluation",
    checkpoint_name: str | None = None,
) -> tuple[str, str, str]:
    """Resolve ``(experiment_name, checkpoint_name, dataset_name)``.

    The experiment name is the checkpoint's containing directory, matching how
    training names runs. The checkpoint name is the individual ``.ckpt`` within
    that run, so several checkpoints of one run keep their results apart. The
    dataset name comes from the activation directory being evaluated, so results
    follow the data the run actually read; it falls back to the checkpoint's
    recorded ``dataset.name`` when ``data_dir`` does not follow the extraction
    layout. The training dataset is already recorded in the checkpoint metadata,
    so the path does not need to repeat it.
    """
    experiment_name = safe_path_name(
        Path(checkpoint_path).parent.name, "experiment name"
    )
    if checkpoint_name is None:
        checkpoint_name = Path(checkpoint_path).stem
    checkpoint_name = safe_path_name(checkpoint_name, "checkpoint name")

    dataset_name = None
    if data_dir is not None:
        _, dataset_name = dataset_identity_from_path(data_dir)
    if not dataset_name:
        dataset_meta = load_checkpoint_metadata(checkpoint_path).get("dataset") or {}
        if not isinstance(dataset_meta, dict):
            dataset_meta = {}
        dataset_name = dataset_meta.get("name")
    if not dataset_name:
        raise ValueError(
            f"Cannot resolve a dataset name for the {evaluation_name} evaluation. "
            f"data.dir does not follow <base_model>/layer<N>/<dataset>/ and "
            f"checkpoint {checkpoint_path} records no dataset.name. Set "
            "output.dir explicitly instead."
        )
    return (
        experiment_name,
        checkpoint_name,
        safe_path_name(dataset_name, "dataset name"),
    )


def resolve_output_dir(
    output_cfg: dict[str, Any],
    evaluation_name: str,
    *,
    checkpoint_path: str | Path | None = None,
    data_dir: str | Path | None = None,
    checkpoint_name: str | None = None,
) -> Path:
    """Resolve a result root as ``results/<eval>/<experiment>/<checkpoint>/<dataset>/``.

    ``checkpoint_name`` names the checkpoint level; it defaults to the
    checkpoint file's stem. ``output.dir`` still overrides the layout when an
    exact path is needed.
    """
    explicit_dir = output_cfg.get("dir")
    if explicit_dir:
        return project_path(explicit_dir)

    if checkpoint_path is None:
        raise ValueError(
            f"Set output.dir for the {evaluation_name} evaluation, or pass a "
            "checkpoint path so the result directory can be derived."
        )

    experiment_name, checkpoint_name, dataset_name = resolve_result_identity(
        checkpoint_path,
        data_dir=data_dir,
        evaluation_name=evaluation_name,
        checkpoint_name=checkpoint_name,
    )
    result_dir = DEFAULT_OUTPUT_DIRS.get(evaluation_name, evaluation_name)
    return (
        project_path("results")
        / result_dir
        / experiment_name
        / checkpoint_name
        / dataset_name
    )


def metadata_path_for_checkpoint(ckpt_path: str | Path) -> Path:
    return Path(ckpt_path).expanduser().resolve().parent / "model_metadata.yaml"


def load_checkpoint_metadata(ckpt_path: str | Path) -> dict[str, Any]:
    path = metadata_path_for_checkpoint(ckpt_path)
    if not path.exists():
        return {}
    return load_yaml_mapping(path)


def _metadata_sae_config(metadata: dict[str, Any]) -> dict[str, Any]:
    model = metadata.get("model") or {}
    sae = model.get("sae") or {}
    return sae if isinstance(sae, dict) else {}


def _resolve_topk_k(
    sae_sd: dict[str, torch.Tensor],
    sae_cfg: dict[str, Any],
) -> int | None:
    """Resolve TopK sparsity from checkpoint state, then checkpoint metadata."""
    if "k_buffer" in sae_sd:
        return int(sae_sd["k_buffer"].item())
    if sae_cfg.get("k") is not None:
        return int(sae_cfg["k"])
    return None


def infer_sae_type(
    sae_sd: dict[str, torch.Tensor],
    *,
    configured: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    if configured:
        return str(configured)

    sae_cfg = _metadata_sae_config(metadata or {})
    if sae_cfg.get("sae_type"):
        return str(sae_cfg["sae_type"])

    if "log_threshold" in sae_sd:
        return "jumprelu"
    if (
        "k_buffer" in sae_sd
        or "activation_rate" in sae_sd
        or sae_cfg.get("k") is not None
    ):
        return "topk"
    return "vanilla"


def load_sae(
    ckpt_path: str | Path,
    sae_type: str | None = None,
    device: str = "cpu",
) -> tuple[SAE, dict[str, Any]]:
    """
    Load SAE weights from a Lightning checkpoint.

    For TopK SAEs, k is read from the checkpoint's k_buffer when available,
    then model_metadata.yaml.
    """
    ckpt_path = project_path(ckpt_path)
    if ckpt_path is None:
        raise ValueError("ckpt_path must not be None")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    sae_sd = {
        key[4:]: value
        for key, value in state_dict.items()
        if key.startswith("sae.")
    }
    if not sae_sd:
        raise ValueError(f"No SAE state_dict entries found in checkpoint: {ckpt_path}")

    metadata = load_checkpoint_metadata(ckpt_path)
    sae_cfg = _metadata_sae_config(metadata)
    sae_type = infer_sae_type(sae_sd, configured=sae_type, metadata=metadata)

    ever_fired: torch.Tensor | None = state_dict.get("ever_fired")
    d_hidden, d_model = sae_sd["w_dec"].shape
    weight_tying = "w_enc" not in sae_sd
    expansion = d_hidden // d_model
    k: int | None = None

    if sae_type == "jumprelu":
        sae: SAE = JumpReluSAE(
            d_model=d_model,
            expansion=expansion,
            weight_tying=weight_tying,
        )
    elif sae_type == "vanilla":
        sae = VanillaSAE(
            d_model=d_model,
            expansion=expansion,
            weight_tying=weight_tying,
        )
    elif sae_type == "topk":
        k = _resolve_topk_k(sae_sd, sae_cfg)
        if k is None:
            raise ValueError(
                "TopK k not found in checkpoint k_buffer or model_metadata.yaml"
            )
        sae = TopKSAE(
            d_model=d_model,
            expansion=expansion,
            k=k,
            weight_tying=weight_tying,
        )
    else:
        raise ValueError(f"Unknown SAE type: {sae_type}")

    sae.load_state_dict(sae_sd, strict=False)
    sae = sae.to(device).eval()

    meta: dict[str, Any] = {
        "d_model": int(d_model),
        "d_hidden": int(d_hidden),
        "expansion": int(expansion),
        "sae_type": sae_type,
        "weight_tying": bool(weight_tying),
        "checkpoint_path": str(ckpt_path),
    }
    if k is not None:
        meta["k"] = int(k)
    if ever_fired is not None:
        meta["training_dead_frac"] = float((~ever_fired).float().mean().item())
    if metadata:
        meta["checkpoint_metadata_path"] = str(metadata_path_for_checkpoint(ckpt_path))

    return sae, meta
