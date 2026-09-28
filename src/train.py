"""
SAE training aligned with the reference implementation of 'Scaling and
Evaluating Sparse Autoencoders' (Gao et al. 2024, openai/sparse_autoencoder).

This module supersedes train_legacy.py. Differences from the legacy trainer:
    -   Uses the variance-normalized losses from sae.py; the first training
        batch always initializes b_dec and mse_scale, so losses read as
        fraction of variance unexplained across base models and layers.
    -   Before each optimizer step, decoder rows are renormalized and the
        gradient component parallel to each decoder direction is projected out
        (their unit_norm_decoder_ / unit_norm_decoder_grad_adjustment_);
        rows are renormalized again after the step.
    -   Adam eps defaults to 6.25e-10 (training.adam_eps), per the official
        config, since normalized losses produce small gradients.
    - Gradient clipping is off by default and configurable
        (training.gradient_clip_val), matching the official default.
"""

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import yaml

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from sae import VanillaSAE, TopKSAE, JumpReluSAE
from dataset import ShardedActivationDataset

# Default config path: <project_root>/configs/train.yaml
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "train.yaml"
SAE_BRANCH_KEYS = ("vanilla", "topk", "jumprelu")

# Adam eps used by the official implementation.
DEFAULT_ADAM_EPS = 6.25e-10

# Hidden dimension for each supported base model
MODEL_DIMS: dict[str, int] = {
    "pythia-1b": 2048,
    "gemma-2b": 2304,
    "gemma-27b": 4608,
}


def config_section(raw: dict, name: str) -> dict:
    section = raw.get(name, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"Expected '{name}' config section to be a mapping.")
    return section


def load_config(path: str | Path | None = None) -> SimpleNamespace:
    """
    Load training configuration from a YAML file.

    The YAML is structured into sections (data, model, training, infrastructure).
    Model-specific args live under model.vanilla / model.topk / model.jumprelu;
    only the branch matching model.sae_type is merged into the final flat namespace.
    """
    path = Path(path).expanduser() if path is not None else _DEFAULT_CONFIG
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Training config must be a YAML mapping: {path}")

    flat: dict[str, object] = {}

    # Data section
    flat.update(config_section(raw, "data"))

    # Model section: merge shared keys and the selected SAE-specific branch.
    model_cfg = config_section(raw, "model")
    sae_type = model_cfg.get("sae_type", "topk")
    if sae_type not in SAE_BRANCH_KEYS:
        raise ValueError(
            f"Unknown model.sae_type '{sae_type}'. Expected one of {SAE_BRANCH_KEYS}."
        )

    for k, v in model_cfg.items():
        if k in SAE_BRANCH_KEYS:
            continue
        flat[k] = v

    branch = model_cfg.get(sae_type, {})
    if branch is None:
        branch = {}
    if not isinstance(branch, dict):
        raise ValueError(f"Expected model.{sae_type} config branch to be a mapping.")
    flat.update(branch)

    # Training section
    flat.update(config_section(raw, "training"))

    # Infrastructure section
    flat.update(config_section(raw, "infrastructure"))

    flat["_config_path"] = str(path)
    flat["_raw_config"] = raw

    return SimpleNamespace(**flat)


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------

class SAELitModule(L.LightningModule):
    """
    Lightning wrapper for training Sparse Autoencoders.

    Handles:
        -   forward / training / validation steps with the correct loss signature
        -   decoder renormalization and parallel-gradient projection before each
            optimizer step (official unit-norm decoder handling)
        -   logging of raw/normalized reconstruction loss, total loss, and L0
        -   configurable optimizer and LR scheduler
    """

    def __init__(
        self,
        sae: torch.nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        adam_eps: float = DEFAULT_ADAM_EPS,
        lr_scheduler: str | None = None,
        warmup_fraction: float = 0.0,
        max_steps: int | None = None,
    ):
        super().__init__()
        self.sae = sae
        self.lr = lr
        self.weight_decay = weight_decay
        self.adam_eps = adam_eps
        self.lr_scheduler = lr_scheduler
        self.warmup_fraction = warmup_fraction
        self.max_steps = max_steps

        # Save hyperparameters (excluding the model itself)
        self.save_hyperparameters(ignore=["sae"])

        # Track which features have ever fired across all training steps
        self.register_buffer("ever_fired", torch.zeros(sae.d_hidden, dtype=torch.bool))

    # ------------------------------------------------------------------
    # Shared step
    # ------------------------------------------------------------------

    def _forward_and_loss(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(self.sae, JumpReluSAE):
            encoded, x_hat, pre_acts = self.sae(x)
            loss = self.sae.loss_function(x_hat, x, pre_acts)
        elif isinstance(self.sae, VanillaSAE):
            encoded, x_hat = self.sae(x)
            loss = self.sae.loss_function(x_hat, x, encoded)
        else:
            encoded, x_hat = self.sae(x)
            loss = self.sae.loss_function(x_hat, x)

        return encoded, x_hat, loss

    def _step(self, batch: torch.Tensor, stage: str) -> torch.Tensor:
        x = batch
        encoded, x_hat, loss = self._forward_and_loss(x)

        with torch.no_grad():
            # Raw MSE kept for continuity with legacy runs; recon_fvu is the
            # variance-normalized counterpart actually optimized.
            recon_loss = F.mse_loss(x_hat, x)
            recon_fvu = self.sae.scaled_mse(x_hat, x)
            l0 = (encoded > 0).float().sum(dim=-1).mean()
            if stage == "train":
                self.ever_fired.logical_or_((encoded > 0).any(dim=0))
            dead_frac = 1.0 - self.ever_fired.float().mean()

        self.log(
            f"{stage}/loss",
            loss,
            prog_bar=True,
            on_step=(stage == "train"),
            on_epoch=True,
        )
        self.log(f"{stage}/recon_loss", recon_loss, on_step=False, on_epoch=True)
        self.log(f"{stage}/recon_fvu", recon_fvu, on_step=False, on_epoch=True)
        self.log(f"{stage}/l0", l0, on_step=False, on_epoch=True)
        self.log(f"{stage}/dead_features", dead_frac, on_step=False, on_epoch=True)

        if isinstance(self.sae, VanillaSAE):
            with torch.no_grad():
                sparsity_loss = self.sae.l1_coeff * self.sae.sparsity_loss(encoded, x)
            self.log(f"{stage}/sparsity_loss", sparsity_loss, on_step=False, on_epoch=True)

        return loss

    # ------------------------------------------------------------------
    # Training / Validation
    # ------------------------------------------------------------------

    def training_step(self, batch: torch.Tensor, _batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: torch.Tensor, _batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def on_before_optimizer_step(self, optimizer):
        """
        Official unit-norm decoder handling: renormalize decoder rows, then
        project out the gradient component parallel to each decoder direction
        so the optimizer step stays tangent to the unit-norm constraint.
        """
        self.sae.normalize_decoder_weights()
        self.sae.remove_parallel_decoder_grads()

    def on_train_batch_end(self, _outputs, _batch, _batch_idx):
        """
        Renormalize again after the step so the next forward pass (and any
        checkpoint saved at batch/epoch boundaries) sees exactly unit-norm
        decoder rows, cleaning up the small post-step drift Adam introduces.
        """
        self.sae.normalize_decoder_weights()

    # ------------------------------------------------------------------
    # Optimizer & Scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.sae.parameters(),
            lr=self.lr,
            eps=self.adam_eps,
            weight_decay=self.weight_decay,
        )

        if self.warmup_fraction > 0 and self.max_steps is None:
            raise ValueError("max_steps must be set when warmup_fraction > 0.")
        warmup_steps = (
            int(self.warmup_fraction * self.max_steps)
            if self.warmup_fraction > 0 and self.max_steps is not None
            else 0
        )

        if self.lr_scheduler == "cosine" and self.max_steps is None:
            raise ValueError("max_steps must be set for cosine scheduler.")

        # No scheduler and no warmup: return the bare optimizer.
        if self.lr_scheduler is None and warmup_steps == 0:
            return optimizer

        schedulers = []
        milestones = []

        # Optional linear warmup: ramp LR from near-zero to the base LR.
        if warmup_steps > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-2,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            schedulers.append(warmup)
            milestones.append(warmup_steps)

        # Decay phase
        if self.lr_scheduler == "cosine":
            remaining = self.max_steps - warmup_steps
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, remaining),
            )
            schedulers.append(cosine)
        elif self.lr_scheduler is not None:
            raise ValueError(f"Unknown lr_scheduler: {self.lr_scheduler}")

        # Build composite or single scheduler
        if len(schedulers) == 1:
            scheduler = schedulers[0]
        else:
            # SequentialLR switches from warmup to decay at the milestone.
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=schedulers,
                milestones=milestones,
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def project_path(path: str | Path) -> Path:
    """Resolve relative config paths from the project root."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else _PROJECT_ROOT / path


def prefix_tag(cfg: SimpleNamespace) -> str:
    return "prefix" if getattr(cfg, "prompt_prefix", False) else "no_prefix"


def exp_tag(cfg: SimpleNamespace) -> str:
    value = getattr(cfg, "exp_tag", None)
    if value is None:
        raise ValueError("infrastructure.exp_tag must be set.")

    return str(value)


def run_slug(cfg: SimpleNamespace) -> str:
    return (
        f"layer{cfg.layer}_{cfg.sae_type}_exp{cfg.expansion}_"
        f"{cfg.pooling_method}_{prefix_tag(cfg)}_{exp_tag(cfg)}"
    )


def resolve_save_dir(cfg: SimpleNamespace) -> Path:
    save_dir = project_path("checkpoints") / cfg.base_model / run_slug(cfg)
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def resolve_activation_dir(path: str | Path) -> Path:
    """Resolve a split activation directory from the config."""
    path = project_path(path)
    if not path.is_dir():
        raise NotADirectoryError(f"Activation split path must be a directory: {path}")
    return path


def activation_dir_name(cfg: SimpleNamespace) -> str:
    return f"gene_activations_{cfg.pooling_method}_{prefix_tag(cfg)}"


def activation_parent_dir(cfg: SimpleNamespace) -> Path:
    return project_path(cfg.data_path) / activation_dir_name(cfg)


def split_activation_dir(cfg: SimpleNamespace, split_name: str) -> Path:
    if not getattr(cfg, "data_path", None):
        raise ValueError("data.data_path must be set to the activation split parent directory.")

    return resolve_activation_dir(activation_parent_dir(cfg) / split_name)


def make_activation_loader(
    path: str | Path,
    cfg: SimpleNamespace,
    *,
    shuffle_shards: bool,
    shuffle_within_shard: bool,
    drop_last: bool,
) -> DataLoader:
    dataset = ShardedActivationDataset(
        resolve_activation_dir(path),
        shuffle_shards=shuffle_shards,
        shuffle_within_shard=shuffle_within_shard,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=False,
        persistent_workers=cfg.num_workers > 0,
        drop_last=drop_last,
    )


def build_dataloaders(cfg: SimpleNamespace) -> tuple[DataLoader, DataLoader]:
    """
    Build loaders from extraction outputs:
        .../<dataset>/gene_activations_.../data_train/*.safetensors
        .../<dataset>/gene_activations_.../data_val/*.safetensors
    """
    train_loader = make_activation_loader(
        split_activation_dir(cfg, "data_train"),
        cfg,
        shuffle_shards=True,
        shuffle_within_shard=True,
        drop_last=True,
    )
    val_loader = make_activation_loader(
        split_activation_dir(cfg, "data_val"),
        cfg,
        shuffle_shards=False,
        shuffle_within_shard=False,
        drop_last=False,
    )
    return train_loader, val_loader


def selected_sae_config(cfg: SimpleNamespace) -> dict:
    config = {
        "sae_type": cfg.sae_type,
        "d_model": cfg.d_model,
        "expansion": cfg.expansion,
        "d_hidden": cfg.d_model * cfg.expansion,
        "weight_tying": cfg.weight_tying,
    }

    if cfg.sae_type == "vanilla":
        config["l1_coeff"] = cfg.l1_coeff
    elif cfg.sae_type == "topk":
        config.update({
            "k": cfg.k,
            "use_aux_loss": cfg.use_aux_loss,
            "k_aux": cfg.k_aux,
            "aux_alpha": cfg.aux_alpha,
            "dead_rate_threshold": getattr(cfg, "dead_rate_threshold", 1e-4),
        })
    elif cfg.sae_type == "jumprelu":
        config.update({
            "threshold_init": cfg.threshold_init,
            "bandwidth": cfg.bandwidth,
            "sparsity_lambda": cfg.sparsity_lambda,
        })
    else:
        raise ValueError(f"Unknown SAE type: {cfg.sae_type}")

    return config


def build_model_metadata(cfg: SimpleNamespace, sae: torch.nn.Module | None = None) -> dict:
    sae_config = selected_sae_config(cfg)
    if sae is not None:
        sae_config["class_name"] = type(sae).__name__
        sae_config["d_hidden"] = int(sae.d_hidden)
        sae_config["parameter_count"] = sum(p.numel() for p in sae.parameters())
        # mse_scale is a non-persistent buffer; record it so the training loss
        # scale can be reproduced from the checkpoint.
        sae_config["mse_scale"] = float(sae.mse_scale)
        # Record the resolved aux-loss dead threshold for cross-run comparison.
        # It is fixed by default but remains an explicit TopKSAE constructor
        # option for deliberate ablations.
        if getattr(sae, "use_aux_loss", False):
            sae_config["dead_rate_threshold"] = float(sae.dead_rate_threshold)

    return {
        "metadata_version": 2,
        "dataset": {
            "name": getattr(cfg, "dataset_name", None),
            "activation_parent": str(activation_parent_dir(cfg)),
        },
        "model": {
            "base_model": cfg.base_model,
            "layer": cfg.layer,
            "pooling_method": cfg.pooling_method,
            "prompt_prefix": cfg.prompt_prefix,
            "d_model": cfg.d_model,
            "sae": sae_config,
        },
        "training": {
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "adam_eps": getattr(cfg, "adam_eps", DEFAULT_ADAM_EPS),
            "gradient_clip_val": getattr(cfg, "gradient_clip_val", None),
            "max_epochs": cfg.max_epochs,
            "max_steps": cfg.max_steps,
            "lr_scheduler": cfg.lr_scheduler,
            "warmup_fraction": cfg.warmup_fraction,
            "log_every_n_steps": cfg.log_every_n_steps,
            "early_stopping_patience": cfg.early_stopping_patience,
            "early_stopping_min_delta": cfg.early_stopping_min_delta,
        },
        "infrastructure": {
            "accelerator": cfg.accelerator,
            "devices": cfg.devices,
            "precision": cfg.precision,
            "seed": cfg.seed,
        },
        "train_config": getattr(cfg, "_raw_config", None),
    }


def write_model_metadata(cfg: SimpleNamespace, sae: torch.nn.Module | None = None) -> Path:
    metadata_path = cfg.save_dir / "model_metadata.yaml"
    metadata = build_model_metadata(cfg, sae)
    with metadata_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)
    return metadata_path


def wandb_run_name(cfg: SimpleNamespace) -> str:
    return (
        f"exp{exp_tag(cfg)}-layer{cfg.layer}-{cfg.pooling_method}-"
        f"{cfg.sae_type}-expansion{cfg.expansion}"
    )


def wandb_config(cfg: SimpleNamespace) -> dict:
    return {
        "exp_tag": exp_tag(cfg),
        "dataset_name": getattr(cfg, "dataset_name", None),
        "base_model": cfg.base_model,
        "layer": cfg.layer,
        "pooling_method": cfg.pooling_method,
        "prompt_prefix": cfg.prompt_prefix,
        "sae": selected_sae_config(cfg),
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "adam_eps": getattr(cfg, "adam_eps", DEFAULT_ADAM_EPS),
        "gradient_clip_val": getattr(cfg, "gradient_clip_val", None),
        "max_epochs": cfg.max_epochs,
        "max_steps": cfg.max_steps,
        "lr_scheduler": cfg.lr_scheduler,
        "warmup_fraction": cfg.warmup_fraction,
        "early_stopping_patience": cfg.early_stopping_patience,
        "early_stopping_min_delta": cfg.early_stopping_min_delta,
        "precision": cfg.precision,
        "seed": cfg.seed,
    }


def resolve_d_model(cfg: SimpleNamespace) -> int:
    """Return d_model: use explicit config value if set, otherwise infer from MODEL_DIMS."""
    if getattr(cfg, "d_model", None):
        return int(cfg.d_model)
    if cfg.base_model not in MODEL_DIMS:
        raise ValueError(
            f"Unknown base_model '{cfg.base_model}'. "
            "Add it to MODEL_DIMS or set d_model explicitly in the config."
        )
    return MODEL_DIMS[cfg.base_model]


def validate_training_config(cfg: SimpleNamespace) -> None:
    if cfg.lr_scheduler not in (None, "cosine"):
        raise ValueError(f"Unknown lr_scheduler: {cfg.lr_scheduler}")
    if not 0 <= cfg.warmup_fraction <= 1:
        raise ValueError("training.warmup_fraction must be between 0 and 1.")
    if cfg.warmup_fraction > 0 and cfg.max_steps is None:
        raise ValueError("training.warmup_fraction requires training.max_steps.")
    if cfg.lr_scheduler == "cosine" and cfg.max_steps is None:
        raise ValueError("training.lr_scheduler='cosine' requires training.max_steps.")
    if getattr(cfg, "adam_eps", DEFAULT_ADAM_EPS) <= 0:
        raise ValueError("training.adam_eps must be positive.")
    clip = getattr(cfg, "gradient_clip_val", None)
    if clip is not None and clip <= 0:
        raise ValueError("training.gradient_clip_val must be null or positive.")


def initial_data_sample(train_loader: DataLoader) -> torch.Tensor:
    """
    First training batch, used as the stats sample for mse_scale and
    (optionally) b_dec initialization.
    """
    try:
        return next(iter(train_loader))
    except StopIteration as exc:
        raise RuntimeError("No training activations available for initialization.") from exc


def build_sae(
    cfg: SimpleNamespace,
    init_data: torch.Tensor | None = None,
) -> torch.nn.Module:
    if cfg.sae_type == "vanilla":
        return VanillaSAE(
            d_model=cfg.d_model,
            expansion=cfg.expansion,
            weight_tying=cfg.weight_tying,
            l1_coeff=cfg.l1_coeff,
            init_data=init_data,
        )
    if cfg.sae_type == "topk":
        return TopKSAE(
            d_model=cfg.d_model,
            expansion=cfg.expansion,
            k=cfg.k,
            weight_tying=cfg.weight_tying,
            use_aux_loss=cfg.use_aux_loss,
            k_aux=cfg.k_aux,
            aux_alpha=cfg.aux_alpha,
            dead_rate_threshold=getattr(cfg, "dead_rate_threshold", 1e-4),
            init_data=init_data,
        )
    if cfg.sae_type == "jumprelu":
        return JumpReluSAE(
            d_model=cfg.d_model,
            expansion=cfg.expansion,
            threshold_init=cfg.threshold_init,
            bandwidth=cfg.bandwidth,
            sparsity_lambda=cfg.sparsity_lambda,
            weight_tying=cfg.weight_tying,
            init_data=init_data,
        )

    raise ValueError(f"Unknown SAE type: {cfg.sae_type}")


def build_lightning_module(cfg: SimpleNamespace, sae: torch.nn.Module) -> SAELitModule:
    return SAELitModule(
        sae=sae,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        adam_eps=getattr(cfg, "adam_eps", DEFAULT_ADAM_EPS),
        lr_scheduler=cfg.lr_scheduler,
        warmup_fraction=cfg.warmup_fraction,
        max_steps=cfg.max_steps,
    )


def uses_lr_monitor(cfg: SimpleNamespace) -> bool:
    return cfg.lr_scheduler is not None or cfg.warmup_fraction > 0


def selection_metric(cfg: SimpleNamespace) -> str:
    """
    Metric that drives best-checkpoint selection and early stopping.

    For TopK the only term val/loss adds on top of the reconstruction error is
    the dead-feature aux loss, a regularizer whose value jumps discretely as
    features cross the dead-rate threshold (and drops to exactly 0 once none
    are dead). Those jumps are orders of magnitude larger than
    early_stopping_min_delta, so they would drive stopping and selection
    instead of reconstruction quality; monitor val/recon_fvu directly.

    Vanilla and JumpReLU keep val/loss: their sparsity penalty is part of the
    objective being traded off, not a side regularizer, and selecting on
    reconstruction alone would favour the densest checkpoint.

    Both metrics are variance-normalized (~0-1), so early_stopping_min_delta
    means the same thing either way.
    """
    return "val/recon_fvu" if cfg.sae_type == "topk" else "val/loss"


def build_callbacks(cfg: SimpleNamespace) -> list:
    monitor = selection_metric(cfg)

    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.save_dir,
            filename=(
                f"{cfg.base_model}-layer{cfg.layer}-{cfg.pooling_method}-"
                f"{cfg.sae_type}-sae-{{step}}"
            ),
            monitor=monitor,
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        # Save exactly once after epoch 0; later epochs have a larger monitored value.
        ModelCheckpoint(
            dirpath=cfg.save_dir,
            filename=(
                f"{cfg.base_model}-layer{cfg.layer}-{cfg.pooling_method}-"
                f"{cfg.sae_type}-sae-first-epoch"
            ),
            monitor="epoch",
            mode="min",
            every_n_epochs=1,
            save_top_k=1,
            save_on_train_epoch_end=True,
            auto_insert_metric_name=False,
        ),
        ModelCheckpoint(
            dirpath=cfg.save_dir,
            filename=(
                f"{cfg.base_model}-layer{cfg.layer}-{cfg.pooling_method}-"
                f"{cfg.sae_type}-sae-epoch{{epoch:03d}}-step{{step}}"
            ),
            every_n_epochs=10,
            save_top_k=-1,
            save_on_train_epoch_end=True,
            auto_insert_metric_name=False,
        ),
        EarlyStopping(
            monitor=monitor,
            mode="min",
            patience=cfg.early_stopping_patience,
            min_delta=cfg.early_stopping_min_delta,
            verbose=True,
        ),
    ]

    if uses_lr_monitor(cfg):
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    return callbacks


def build_logger(cfg: SimpleNamespace):
    if cfg.wandb_enabled:
        return WandbLogger(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=wandb_run_name(cfg),
            config=wandb_config(cfg),
        )

    return CSVLogger(save_dir=cfg.save_dir, name="sae_training")


def build_trainer(cfg: SimpleNamespace) -> L.Trainer:
    return L.Trainer(
        max_epochs=cfg.max_epochs,
        max_steps=cfg.max_steps if cfg.max_steps is not None else -1,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        precision=cfg.precision,
        logger=build_logger(cfg),
        callbacks=build_callbacks(cfg),
        log_every_n_steps=cfg.log_every_n_steps,
        gradient_clip_val=getattr(cfg, "gradient_clip_val", None),
    )


def main():
    cfg = load_config()
    validate_training_config(cfg)
    L.seed_everything(cfg.seed)

    cfg.d_model = resolve_d_model(cfg)
    cfg.save_dir = resolve_save_dir(cfg)

    # ---- Data ----
    train_loader, val_loader = build_dataloaders(cfg)

    # The first batch initializes b_dec and mse_scale.
    sae = build_sae(cfg, initial_data_sample(train_loader))
    write_model_metadata(cfg, sae)

    module = build_lightning_module(cfg, sae)
    trainer = build_trainer(cfg)
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
