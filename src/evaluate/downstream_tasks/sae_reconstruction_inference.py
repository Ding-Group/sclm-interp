#!/usr/bin/env python3
"""
Evaluate downstream cell-type annotation while intervening with SAE reconstructions.

The script installs a forward hook on one or more configured transformer layers.
At each hook point it replaces the layer output hidden state with:

    sae.decode(sae.encode(hidden_state))

Generation then continues through the remaining model layers as usual. This lets
you measure how much a trained SAE reconstruction preserves downstream behavior.

Usage:
    python src/evaluate/downstream_tasks/sae_reconstruction_inference.py
    python src/evaluate/downstream_tasks/sae_reconstruction_inference.py --config configs/eval_tasks.yaml
    python src/evaluate/downstream_tasks/sae_reconstruction_inference.py --checkpoint checkpoints/gemma-2b/layer15_topk_exp8_last_no_prefix_1/last.ckpt --layer 15 --max-examples 50
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any

import torch
import yaml
from datasets import load_from_disk
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from src.data.inference import (
    InferenceConfig,
    _get_gene_token_ranges,
    generate,
    get_transformer_layers,
    load_model,
)
from src.evaluate.downstream_tasks.cell_type_annotation import (
    cell_type_annotation as cell_eval,
)
from src.evaluate.model_loading import (
    load_checkpoint_metadata,
    load_sae,
    project_path,
    resolve_checkpoint_path,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "eval_tasks.yaml"
CONFIG_SECTION = "model.reconstruct"
_OUTPUT_SAFE_RE = re.compile(r"[^A-Za-z0-9_.=-]+")


@dataclass
class InterventionConfig:
    layer_idx: int
    checkpoint_path: Path
    sae_type: str | None
    sae_device: str
    name: str | None = None


@dataclass
class SAEEvalConfig:
    cell_type: cell_eval.EvalConfig
    interventions: list[InterventionConfig]
    output_dir: Path
    save_predictions: bool
    include_prompt: bool
    reconstruct_model_name: str
    output_tag: str | None


class SAEReconstructionHook:
    """Forward hook that replaces one layer's output with its SAE reconstruction."""

    def __init__(
        self,
        model,
        layer_idx: int,
        sae,
        *,
        name: str,
        expected_d_model: int | None = None,
        pooling_method: str | None = None,
    ):
        self.model = model
        self.layer_idx = layer_idx
        self.sae = sae
        self.name = name
        self.expected_d_model = expected_d_model
        self.pooling_method = pooling_method
        self.handle = None
        self.num_calls = 0
        self.num_tokens = 0
        self._active_token_indices: list[int] = []
        self._active_prompt_len: int | None = None
        self._prefill_seen = True

    @property
    def sae_device(self) -> torch.device:
        return next(self.sae.parameters()).device

    @property
    def sae_dtype(self) -> torch.dtype:
        return next(self.sae.parameters()).dtype

    def set_active_token_indices(
        self,
        token_indices: list[int],
        *,
        prompt_len: int,
    ) -> None:
        self._active_token_indices = sorted(set(int(idx) for idx in token_indices))
        self._active_prompt_len = int(prompt_len)
        self._prefill_seen = False

    def clear_active_token_indices(self) -> None:
        self._active_token_indices = []
        self._active_prompt_len = None
        self._prefill_seen = True

    def _reconstruct_flat(self, flat: torch.Tensor) -> torch.Tensor:
        d_model = int(flat.shape[-1])
        if self.expected_d_model is not None and d_model != self.expected_d_model:
            raise RuntimeError(
                f"{self.name} expected hidden dim {self.expected_d_model}, got {d_model}."
            )

        flat = flat.to(device=self.sae_device, dtype=self.sae_dtype)
        with torch.inference_mode():
            encoded = self.sae.encode(flat)
            return self.sae.decode(encoded)

    def _reconstruct_selected(self, hidden: torch.Tensor) -> torch.Tensor | None:
        if not self._active_token_indices:
            return None
        if hidden.ndim != 3:
            raise RuntimeError(
                f"{self.name} expected a 3D hidden state for token-selective "
                f"reconstruction, got shape {tuple(hidden.shape)}."
            )

        prompt_len = self._active_prompt_len
        seq_len = int(hidden.shape[1])
        if prompt_len is not None and seq_len != prompt_len:
            return None

        token_indices = [
            idx for idx in self._active_token_indices if 0 <= idx < seq_len
        ]
        if not token_indices:
            return None

        d_model = int(hidden.shape[-1])
        if self.expected_d_model is not None and d_model != self.expected_d_model:
            raise RuntimeError(
                f"{self.name} expected hidden dim {self.expected_d_model}, got {d_model}."
            )

        selected = hidden[:, token_indices, :].reshape(-1, d_model)
        reconstructed = self._reconstruct_flat(selected)
        self.num_calls += 1
        self.num_tokens += int(selected.shape[0])

        updated = hidden.clone()
        updated[:, token_indices, :] = reconstructed.to(
            device=hidden.device,
            dtype=hidden.dtype,
        ).reshape(hidden.shape[0], len(token_indices), d_model)
        return updated

    def _make_hook(self):
        def hook(_module, _inputs, output):
            if self._prefill_seen:
                return output
            self._prefill_seen = True

            hidden = output[0] if isinstance(output, (tuple, list)) else output
            reconstructed = self._reconstruct_selected(hidden)
            if reconstructed is None:
                return output
            if isinstance(output, tuple):
                return (reconstructed, *output[1:])
            if isinstance(output, list):
                return [reconstructed, *output[1:]]
            return reconstructed

        return hook

    def register(self) -> None:
        layers = get_transformer_layers(self.model)
        if self.layer_idx < 0 or self.layer_idx >= len(layers):
            raise IndexError(
                f"layer_idx={self.layer_idx} out of range; "
                f"model has {len(layers)} layers (0-{len(layers) - 1})."
            )
        self.handle = layers[self.layer_idx].register_forward_hook(self._make_hook())

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class SAEReconstructionPatcher:
    def __init__(self, hooks: list[SAEReconstructionHook]):
        self.hooks = hooks

    def __enter__(self):
        for hook in self.hooks:
            hook.register()
        return self

    def __exit__(self, *_args):
        for hook in reversed(self.hooks):
            hook.remove()


def _project_path(value: str | Path) -> Path:
    path = project_path(value)
    if path is None:
        raise ValueError("Path must not be None.")
    return path


def _load_yaml(path: str | Path) -> dict[str, Any]:
    path = _project_path(path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return raw


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected {context} to be a mapping.")
    return value


def _checkpoint_cfg_from_item(item: dict[str, Any]) -> dict[str, Any]:
    checkpoint_cfg = _mapping(item.get("checkpoint"), "intervention.checkpoint").copy()
    if item.get("checkpoint_path"):
        checkpoint_cfg["path"] = item["checkpoint_path"]
    if item.get("checkpoint_dir"):
        checkpoint_cfg["dir"] = item["checkpoint_dir"]
    return checkpoint_cfg


def _intervention_from_item(
    item: dict[str, Any],
    *,
    inherited_sae_type: str | None,
    inherited_sae_device: str,
) -> InterventionConfig:
    checkpoint_cfg = _checkpoint_cfg_from_item(item)
    checkpoint_path = resolve_checkpoint_path(checkpoint_cfg)

    layer_idx = item.get("layer_idx", item.get("layer", checkpoint_cfg.get("layer")))
    if layer_idx is None:
        raise ValueError(
            "Each SAE reconstruction intervention must set layer_idx or layer."
        )

    sae_type = item.get(
        "sae_type",
        checkpoint_cfg.get("sae_type", inherited_sae_type),
    )
    sae_device = item.get(
        "sae_device",
        item.get("device", inherited_sae_device),
    )
    layer_idx = int(layer_idx)
    if layer_idx < 0:
        raise ValueError("SAE reconstruction layer index must be non-negative.")

    return InterventionConfig(
        layer_idx=layer_idx,
        checkpoint_path=checkpoint_path,
        sae_type=str(sae_type) if sae_type is not None else None,
        sae_device=str(sae_device),
        name=item.get("name"),
    )


def _load_interventions(module: dict[str, Any]) -> list[InterventionConfig]:
    model_cfg = _mapping(module.get("model"), f"{CONFIG_SECTION}.model")
    inherited_sae_type = module.get("sae_type", model_cfg.get("sae_type"))
    inherited_sae_device = str(module.get("sae_device", "model"))

    entries = module.get("interventions")
    if entries is None:
        intervention_cfg = _mapping(
            module.get("intervention"),
            f"{CONFIG_SECTION}.intervention",
        ).copy()
        if "checkpoint" in module:
            intervention_cfg["checkpoint"] = module["checkpoint"]
        for key in (
            "checkpoint_path",
            "checkpoint_dir",
            "layer",
            "layer_idx",
            "sae_type",
            "sae_device",
            "device",
            "name",
        ):
            if key in module:
                intervention_cfg[key] = module[key]
        entries = [intervention_cfg]

    if not isinstance(entries, list) or not entries:
        raise ValueError(
            f"{CONFIG_SECTION}.interventions must be a non-empty list, or set "
            f"{CONFIG_SECTION}.checkpoint plus {CONFIG_SECTION}.intervention.layer."
        )

    interventions: list[InterventionConfig] = []
    for idx, entry in enumerate(entries):
        item = _mapping(entry, f"{CONFIG_SECTION}.interventions[{idx}]")
        interventions.append(
            _intervention_from_item(
                item,
                inherited_sae_type=inherited_sae_type,
                inherited_sae_device=inherited_sae_device,
            )
        )
    return interventions


def load_eval_config(path: str | Path | None = None) -> SAEEvalConfig:
    path = _project_path(path or DEFAULT_CONFIG)
    raw = _load_yaml(path)
    reconstruct_model_name = cell_eval._configured_reconstruct_model_name(raw)
    base_cfg = cell_eval.load_eval_config(
        path,
        mode_override="reconstruct",
        output_name_override=reconstruct_model_name,
    )
    module = cell_eval._model_reconstruct_config(raw)
    if not module:
        raise ValueError(f"Config section '{CONFIG_SECTION}' is required.")

    output = _mapping(module.get("output"), f"{CONFIG_SECTION}.output")
    return SAEEvalConfig(
        cell_type=base_cfg,
        interventions=_load_interventions(module),
        output_dir=base_cfg.output_dir,
        save_predictions=bool(output.get("save_predictions", base_cfg.save_predictions)),
        include_prompt=bool(output.get("include_prompt", base_cfg.include_prompt)),
        reconstruct_model_name=reconstruct_model_name,
        output_tag=output.get("tag"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate cell-type annotation with SAE-reconstructed hidden states "
            "inserted during generation."
        )
    )
    parser.add_argument("--config", default=None, help=f"YAML config path (default: {DEFAULT_CONFIG}).")
    parser.add_argument(
        "--model",
        default=None,
        help="Override model.base_model: gemma, gemma-2b, gemma-27b, pythia, or a HF model id.",
    )
    parser.add_argument("--dataset", default=None, help="Override cell_type_annotation.data.c2s_dataset_dir.")
    parser.add_argument("--split", default=None, help="Override cell_type_annotation.data.split.")
    parser.add_argument("--max-examples", type=int, default=None, help="Override cell_type_annotation.data.max_examples.")
    parser.add_argument(
        "--prompt-template",
        default=None,
        help="Override cell_type_annotation.inference.prompt_template_path.",
    )
    parser.add_argument(
        "--acceptable-answers",
        default=None,
        help="Override cell_type_annotation.scoring.acceptable_answers_path. Use an empty string to disable.",
    )
    parser.add_argument("--checkpoint", default=None, help="Override the SAE checkpoint path.")
    parser.add_argument("--layer", type=int, default=None, help="Override the intervention layer index.")
    parser.add_argument("--sae-type", default=None, help="Override SAE type: topk, vanilla, or jumprelu.")
    parser.add_argument(
        "--sae-device",
        default=None,
        help="Override SAE device. Use 'model' to match the base-model device.",
    )
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--output-tag", default=None, help="Override output subdirectory tag.")
    parser.add_argument("--no-save-predictions", action="store_true")
    return parser.parse_args()


def apply_cli_overrides(cfg: SAEEvalConfig, args: argparse.Namespace) -> None:
    cell_eval.apply_cli_overrides(cfg.cell_type, args)

    if args.output_dir is not None:
        cfg.output_dir = _project_path(args.output_dir)
    if args.output_tag is not None:
        cfg.output_tag = args.output_tag
    if args.no_save_predictions:
        cfg.save_predictions = False

    if any(
        value is not None
        for value in (args.checkpoint, args.layer, args.sae_type, args.sae_device)
    ):
        base = cfg.interventions[0]
        cfg.interventions = [
            InterventionConfig(
                layer_idx=args.layer if args.layer is not None else base.layer_idx,
                checkpoint_path=_project_path(args.checkpoint)
                if args.checkpoint is not None
                else base.checkpoint_path,
                sae_type=args.sae_type if args.sae_type is not None else base.sae_type,
                sae_device=args.sae_device
                if args.sae_device is not None
                else base.sae_device,
                name=base.name,
            )
        ]


def make_inference_config(cfg: SAEEvalConfig) -> InferenceConfig:
    return cell_eval.make_inference_config(cfg.cell_type)


def _checkpoint_tag(path: Path) -> str:
    tag = path.parent.name if path.name == "last.ckpt" else path.stem
    return _OUTPUT_SAFE_RE.sub("_", tag).strip("_") or "checkpoint"


def _intervention_tag(intervention: InterventionConfig) -> str:
    if intervention.name:
        name = _OUTPUT_SAFE_RE.sub("_", intervention.name).strip("_")
        if name:
            return name
    return f"layer{intervention.layer_idx}_{_checkpoint_tag(intervention.checkpoint_path)}"


def _output_paths(cfg: SAEEvalConfig, inf_cfg: InferenceConfig) -> tuple[Path, Path, Path]:
    out_dir = (
        cfg.output_dir
        / cfg.cell_type.c2s_dataset_dir.name
        / inf_cfg.model_short_name
        / f"data_{cfg.cell_type.split}"
    )
    tag = cfg.output_tag
    if tag:
        out_dir = out_dir / _OUTPUT_SAFE_RE.sub("_", tag).strip("_")
    out_dir.mkdir(parents=True, exist_ok=True)
    return (
        out_dir / "predictions.jsonl",
        out_dir / "incorrect_predictions.jsonl",
        out_dir / "summary.json",
    )


def _reconstruction_model_metadata_card(
    cfg: SAEEvalConfig,
    inf_cfg: InferenceConfig,
) -> dict[str, Any]:
    return {
        "task": "cell_type_annotation",
        "mode": "reconstruct",
        "base_model_id": cfg.cell_type.model_id,
        "base_model_short_name": inf_cfg.model_short_name,
        "base_model_family": inf_cfg.family,
        "reconstruct_model_name": cfg.reconstruct_model_name,
        "checkpoints": [
            {
                "name": _intervention_tag(intervention),
                "checkpoint_path": str(intervention.checkpoint_path),
                "checkpoint_metadata_path": str(
                    intervention.checkpoint_path.parent
                    / cell_eval.MODEL_METADATA_FILENAME
                ),
                "layer_idx": intervention.layer_idx,
                "sae_type": intervention.sae_type,
            }
            for intervention in cfg.interventions
        ],
    }


def _resolve_sae_device(value: str, inf_cfg: InferenceConfig) -> str:
    if value in {"model", "auto"}:
        return inf_cfg.device
    return value


def _checkpoint_model_metadata(path: Path) -> dict[str, Any]:
    metadata = load_checkpoint_metadata(path)
    model = metadata.get("model")
    return model if isinstance(model, dict) else {}


def _gene_token_indices_from_ranges(
    gene_token_ranges: list[tuple[str, int, int]],
    pooling_method: str | None,
) -> list[int]:
    pooling = str(pooling_method or "all").lower()
    if pooling == "last":
        return [end - 1 for _gene, _start, end in gene_token_ranges if end > 0]

    indices: set[int] = set()
    for _gene, start, end in gene_token_ranges:
        indices.update(range(start, end))
    return sorted(indices)


def _prepare_hooks_for_prompt(
    hooks: list[SAEReconstructionHook],
    tokenizer,
    prompt: str,
    cell_sentence: str,
) -> None:
    prompt_len = len(tokenizer(prompt)["input_ids"])
    gene_token_ranges = _get_gene_token_ranges(tokenizer, prompt, cell_sentence)
    for hook in hooks:
        hook.set_active_token_indices(
            _gene_token_indices_from_ranges(
                gene_token_ranges,
                hook.pooling_method,
            ),
            prompt_len=prompt_len,
        )


def _clear_hook_prompt_state(hooks: list[SAEReconstructionHook]) -> None:
    for hook in hooks:
        hook.clear_active_token_indices()


def load_reconstruction_hooks(
    model,
    cfg: SAEEvalConfig,
    inf_cfg: InferenceConfig,
) -> tuple[list[SAEReconstructionHook], list[dict[str, Any]]]:
    hooks: list[SAEReconstructionHook] = []
    metadata: list[dict[str, Any]] = []
    expected_d_model = inf_cfg.arch.get("d_model") if inf_cfg.arch else None

    for idx, intervention in enumerate(cfg.interventions):
        sae_device = _resolve_sae_device(intervention.sae_device, inf_cfg)
        checkpoint_model = _checkpoint_model_metadata(intervention.checkpoint_path)
        checkpoint_layer = checkpoint_model.get("layer")
        checkpoint_pooling_method = checkpoint_model.get("pooling_method")
        if checkpoint_layer is not None and int(checkpoint_layer) != intervention.layer_idx:
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

        name = _intervention_tag(intervention)
        hooks.append(
            SAEReconstructionHook(
                model,
                intervention.layer_idx,
                sae,
                name=name,
                expected_d_model=sae_meta["d_model"],
                pooling_method=str(checkpoint_pooling_method)
                if checkpoint_pooling_method
                else None,
            )
        )
        meta = dict(sae_meta)
        meta.update(
            {
                "name": name,
                "layer_idx": intervention.layer_idx,
                "configured_sae_type": intervention.sae_type,
                "sae_device": str(sae_device),
                "order": idx,
                "checkpoint_layer": checkpoint_layer,
                "checkpoint_pooling_method": checkpoint_pooling_method,
                "checkpoint_prompt_prefix": checkpoint_prompt_prefix,
            }
        )
        metadata.append(meta)

    return hooks, metadata


def _safe_accuracy(correct: int, total: int) -> float:
    return float(correct / total) if total else 0.0


def run_eval(cfg: SAEEvalConfig) -> dict[str, Any]:
    cell_eval.validate_inputs(cfg.cell_type)
    split_dir = cfg.cell_type.c2s_dataset_dir / f"data_{cfg.cell_type.split}"
    dataset = load_from_disk(str(split_dir))
    required = {"cell_sentence", cfg.cell_type.cell_type_column}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"{split_dir} is missing required column(s): {sorted(missing)}")

    total_available = len(dataset)
    n_examples = (
        min(cfg.cell_type.max_examples, total_available)
        if cfg.cell_type.max_examples
        else total_available
    )
    inf_cfg = make_inference_config(cfg)
    predictions_path, incorrect_predictions_path, summary_path = _output_paths(
        cfg, inf_cfg
    )
    model_metadata_path = cell_eval.write_model_metadata_card(
        summary_path.parent,
        _reconstruction_model_metadata_card(cfg, inf_cfg),
    )
    acceptable_answers = cell_eval.load_acceptable_answers(
        cfg.cell_type.acceptable_answers_path
    )

    print("[Config]")
    print(f"  reconstruct_model: {cfg.reconstruct_model_name}")
    print(f"  model     : {cfg.cell_type.model_id}")
    print(f"  device    : {cfg.cell_type.device}")
    print(f"  dtype     : {cfg.cell_type.dtype}")
    print(f"  dataset   : {cfg.cell_type.c2s_dataset_dir}")
    print(f"  split     : data_{cfg.cell_type.split} ({total_available:,} available)")
    print(f"  examples  : {n_examples:,}")
    print(f"  prompt    : {'template' if cfg.cell_type.prompt_prefix else 'bare cell_sentence'}")
    if cfg.cell_type.prompt_prefix:
        print(f"  prompt tpl: {cfg.cell_type.prompt_template_path}")
    print("  interventions:")
    for intervention in cfg.interventions:
        print(
            f"    - layer {intervention.layer_idx}: "
            f"{intervention.checkpoint_path} "
            f"(sae_type={intervention.sae_type or 'auto'}, device={intervention.sae_device})"
        )
    if cfg.cell_type.acceptable_answers_path:
        print(
            f"  answers   : {cfg.cell_type.acceptable_answers_path} "
            f"({len(acceptable_answers):,} label entries)"
        )
    else:
        print("  answers   : disabled")

    tokenizer, model = load_model(inf_cfg)
    hooks, intervention_meta = load_reconstruction_hooks(model, cfg, inf_cfg)

    correct = 0
    per_cell_type: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "correct": 0}
    )
    prediction_writer_context = (
        open(predictions_path, "w", encoding="utf-8")
        if cfg.save_predictions
        else nullcontext(None)
    )
    incorrect_writer_context = (
        open(incorrect_predictions_path, "w", encoding="utf-8")
        if cfg.save_predictions
        else nullcontext(None)
    )

    with SAEReconstructionPatcher(hooks):
        with (
            prediction_writer_context as writer,
            incorrect_writer_context as incorrect_writer,
        ):
            for idx in tqdm(range(n_examples), desc="Evaluating"):
                sample = dataset[idx]
                gold = str(sample[cfg.cell_type.cell_type_column])
                prompt = cell_eval.build_prompt(sample, cfg.cell_type)
                _prepare_hooks_for_prompt(
                    hooks,
                    tokenizer,
                    prompt,
                    str(sample["cell_sentence"]),
                )
                try:
                    generated_text = generate(prompt, tokenizer, model, inf_cfg)
                finally:
                    _clear_hook_prompt_state(hooks)
                matched_answer = cell_eval.find_encoded_answer(
                    gold,
                    generated_text,
                    acceptable_answers,
                )
                is_correct = matched_answer is not None

                correct += int(is_correct)
                per_cell_type[gold]["total"] += 1
                per_cell_type[gold]["correct"] += int(is_correct)

                if writer is not None:
                    row = {
                        "index": idx,
                        "cell_name": sample.get("cell_name"),
                        "cell_type": gold,
                        "generated_text": generated_text,
                        "correct": is_correct,
                        "matched_answer": matched_answer,
                    }
                    if cfg.include_prompt:
                        row["prompt"] = prompt
                    writer.write(json.dumps(row) + "\n")
                    if not is_correct and incorrect_writer is not None:
                        incorrect_writer.write(json.dumps(row) + "\n")

    by_cell_type = {
        cell_type: {
            "total": stats["total"],
            "correct": stats["correct"],
            "accuracy": _safe_accuracy(stats["correct"], stats["total"]),
        }
        for cell_type, stats in sorted(per_cell_type.items())
    }

    for meta, hook in zip(intervention_meta, hooks):
        meta["hook_calls"] = hook.num_calls
        meta["hook_tokens"] = hook.num_tokens

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "cell_type_annotation",
        "mode": "reconstruct",
        "reconstruct_model_name": cfg.reconstruct_model_name,
        "model_id": cfg.cell_type.model_id,
        "dataset_dir": str(cfg.cell_type.c2s_dataset_dir),
        "split": f"data_{cfg.cell_type.split}",
        "cell_type_column": cfg.cell_type.cell_type_column,
        "num_available": total_available,
        "num_examples": n_examples,
        "num_correct": correct,
        "num_incorrect": n_examples - correct,
        "accuracy": _safe_accuracy(correct, n_examples),
        "prompt_prefix": cfg.cell_type.prompt_prefix,
        "prompt_template_path": str(cfg.cell_type.prompt_template_path)
        if cfg.cell_type.prompt_prefix
        else None,
        "max_new_tokens": cfg.cell_type.max_new_tokens,
        "temperature": cfg.cell_type.temperature,
        "interventions": intervention_meta,
        "acceptable_answers_path": str(cfg.cell_type.acceptable_answers_path)
        if cfg.cell_type.acceptable_answers_path
        else None,
        "num_acceptable_answer_entries": len(acceptable_answers),
        "label_matcher": (
            "gold label plus configured acceptable answers; normalized token sequence "
            "or all normalized label tokens with simple singular/plural variants"
        ),
        "model_metadata_path": str(model_metadata_path),
        "predictions_path": str(predictions_path) if cfg.save_predictions else None,
        "incorrect_predictions_path": str(incorrect_predictions_path)
        if cfg.save_predictions
        else None,
        "by_cell_type": by_cell_type,
    }
    summary["figures"] = cell_eval.plot_accuracy_figures(summary, summary_path.parent)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print("\n[Results]")
    print(f"  accuracy  : {summary['accuracy']:.3f} ({correct:,}/{n_examples:,})")
    if cfg.save_predictions:
        print(f"  generated : {predictions_path}")
        print(f"  incorrect : {incorrect_predictions_path}")
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
