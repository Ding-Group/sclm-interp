#!/usr/bin/env python3
"""Evaluate generated cell-type multiple-choice QA datasets."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
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

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_config import path_has_arrow_dataset
from src.data.inference import InferenceConfig, generate, load_model
from src.evaluate.downstream_tasks.cell_type_annotation import (
    cell_type_annotation as cell_eval,
)
from src.evaluate.downstream_tasks.qa.scoring import (
    clear_hook_prompt_state,
    gene_names_for_sample,
    prepare_hooks_for_prompt,
    score_generated_prediction,
    score_likelihood_prediction,
    trim_control_token,
)
from src.evaluate.model_loading import load_checkpoint_metadata, load_sae

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "eval_tasks.yaml"
DEFAULT_LIKELIHOOD_PROMPT_TEMPLATE = (
    PROJECT_ROOT / "src" / "prompts" / "cell_type_annotation_template.txt"
)
DEFAULT_ACCEPTABLE_ANSWERS = (
    PROJECT_ROOT
    / "src"
    / "evaluate"
    / "downstream_tasks"
    / "cell_type_annotation"
    / "cell_type_acceptable_answers.json"
)
MODEL_METADATA_FILENAME = "model_metadata.yaml"
TASK_NAME = "cell_type_multiple_choice"
MODEL_SECTION = "model"
BASE_OUTPUT_NAME = "base"

BASE_MODEL_IDS = {
    "gemma": "vandijklab/C2S-Scale-Gemma-2-2B",
    "gemma-2b": "vandijklab/C2S-Scale-Gemma-2-2B",
    "gemma-27b": "vandijklab/C2S-Scale-Gemma-2-27B",
    "pythia": "vandijklab/C2S-Scale-Pythia-1b-pt",
    "pythia-1b": "vandijklab/C2S-Scale-Pythia-1b-pt",
}

VALID_MODES = {"base", "reconstruct"}
VALID_SCORING_METHODS = {"likelihood", "generate"}
_OUTPUT_SAFE_RE = re.compile(r"[^A-Za-z0-9_.=-]+")
score_prediction = score_generated_prediction


@dataclass
class EvalConfig:
    mode: str
    model_id: str
    device: str
    dtype: str
    attn_implementation: str | None
    max_new_tokens: int
    temperature: float
    scoring_method: str
    stop_at_control_token: bool
    likelihood_prompt_template_path: Path
    acceptable_answers_path: Path | None
    qa_dataset_dir: Path
    split: str
    max_examples: int | None
    prompt_column: str
    answer_choice_column: str
    gold_cell_type_column: str
    choices_column: str
    choice_labels_column: str
    gene_names_column: str
    gene_sentence_column: str
    organism_column: str | None
    default_organism: str
    output_dir: Path
    save_predictions: bool
    include_prompt: bool
    reconstruct_model_name: str | None = None
    output_tag: str | None = None
    interventions: list[Any] = field(default_factory=list)


def _normalize_mode(mode: str | None) -> str:
    value = "base" if mode is None else str(mode).strip()
    if value not in VALID_MODES:
        valid = ", ".join(sorted(VALID_MODES))
        raise ValueError(f"Unknown {TASK_NAME} mode {mode!r}. Valid: {valid}.")
    return value


def _normalize_scoring_method(method: str | None) -> str:
    value = "likelihood" if method is None else str(method).strip().lower()
    if value not in VALID_SCORING_METHODS:
        valid = ", ".join(sorted(VALID_SCORING_METHODS))
        raise ValueError(f"Unknown {TASK_NAME} scoring method {method!r}. Valid: {valid}.")
    return value


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected {context} to be a mapping.")
    return value


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _optional_project_path(value: str | Path | None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return _project_path(value)


def _resolve_likelihood_prompt_template_path(value: str | Path | None) -> Path:
    if value is None or str(value).strip() == "":
        value = DEFAULT_LIKELIHOOD_PROMPT_TEMPLATE
    return _project_path(value)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    path = _project_path(path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return raw


def _safe_output_name(value: Any, fallback: str) -> str:
    candidate = fallback if value is None or str(value).strip() == "" else str(value)
    safe = _OUTPUT_SAFE_RE.sub("_", str(candidate)).strip("_")
    return safe or fallback


def _model_config(raw: dict[str, Any]) -> dict[str, Any]:
    return _mapping(raw.get(MODEL_SECTION), MODEL_SECTION)


def _model_reconstruct_config(raw: dict[str, Any]) -> dict[str, Any]:
    model = _model_config(raw)
    return _mapping(model.get("reconstruct"), f"{MODEL_SECTION}.reconstruct")


def _configured_reconstruct_model_name(raw: dict[str, Any]) -> str:
    reconstruct = _model_reconstruct_config(raw)
    for key in ("name", "output_name", "model_name"):
        if key in reconstruct:
            return _safe_output_name(reconstruct.get(key), "reconstruct")

    output = _mapping(reconstruct.get("output"), "reconstruct.output")
    if "tag" in output:
        return _safe_output_name(output.get("tag"), "reconstruct")

    interventions = reconstruct.get("interventions")
    if isinstance(interventions, list) and interventions:
        first = _mapping(interventions[0], "reconstruct.interventions[0]")
        if "name" in first:
            return _safe_output_name(first.get("name"), "reconstruct")

    return "reconstruct"


def _selected_output_name(
    raw: dict[str, Any],
    mode: str,
    output_name_override: str | None = None,
) -> str:
    if output_name_override:
        return _safe_output_name(output_name_override, "reconstruct")

    model = _model_config(raw)
    if mode == "base":
        return _safe_output_name(
            model.get("base_output_name", BASE_OUTPUT_NAME),
            BASE_OUTPUT_NAME,
        )
    return _configured_reconstruct_model_name(raw)


def _task_output_dir(
    raw: dict[str, Any],
    task_name: str,
    mode: str,
    *,
    output_name_override: str | None = None,
) -> Path:
    model = _model_config(raw)
    output_root = _project_path(
        model.get(
            "output_dir",
            model.get("results_dir", "results"),
        )
    )
    output_name = _selected_output_name(raw, mode, output_name_override)
    return output_root / task_name / output_name


def _resolve_model_id(model_value: str) -> str:
    key = model_value.strip()
    return BASE_MODEL_IDS.get(key.lower(), key)


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _default_attn_implementation(model_id: str) -> str | None:
    if "pythia" in model_id.lower():
        return None
    return "sdpa"


def _load_sae_module():
    from src.evaluate.downstream_tasks import sae_reconstruction_inference

    return sae_reconstruction_inference


def load_eval_config(
    path: str | Path | None = None,
    *,
    mode_override: str | None = None,
    output_name_override: str | None = None,
) -> EvalConfig:
    raw = _load_yaml(path or DEFAULT_CONFIG)

    model = _model_config(raw)
    module = _mapping(raw.get(TASK_NAME, raw), TASK_NAME)
    data = _mapping(module.get("data"), f"{TASK_NAME}.data")
    inference = _mapping(module.get("inference"), f"{TASK_NAME}.inference")
    scoring = _mapping(module.get("scoring"), f"{TASK_NAME}.scoring")
    output = _mapping(module.get("output"), f"{TASK_NAME}.output")

    model_value = model.get("base_model", model.get("model_id", "gemma"))
    model_id = _resolve_model_id(model_value)
    attn = model.get("attn_implementation", _default_attn_implementation(model_id))
    mode = _normalize_mode(
        mode_override or model.get("mode", module.get("mode", "base"))
    )

    dataset_dir_value = data.get("qa_dataset_dir", data.get("dataset_dir"))
    if not dataset_dir_value:
        raise ValueError(f"Config {TASK_NAME}.data.qa_dataset_dir is required.")

    max_examples = data.get("max_examples")
    if max_examples is not None:
        max_examples = int(max_examples)
        if max_examples <= 0:
            raise ValueError(f"{TASK_NAME}.data.max_examples must be positive or null.")

    scoring_method = _normalize_scoring_method(
        scoring.get("method", inference.get("scoring_method", "likelihood"))
    )
    acceptable_answers_value = scoring.get(
        "acceptable_answers_path",
        DEFAULT_ACCEPTABLE_ANSWERS,
    )

    organism_column = data.get("organism_column", "organism")
    if organism_column == "":
        organism_column = None

    reconstruct_model_name = None
    interventions: list[Any] = []
    output_tag = None
    if mode == "reconstruct":
        reconstruct_model_name = _configured_reconstruct_model_name(raw)
        reconstruct = _model_reconstruct_config(raw)
        if not reconstruct:
            raise ValueError("Config section 'model.reconstruct' is required for reconstruct mode.")
        interventions = _load_sae_module()._load_interventions(reconstruct)
        reconstruct_output = _mapping(
            reconstruct.get("output"),
            "model.reconstruct.output",
        )
        output_tag = reconstruct_output.get("tag")

    return EvalConfig(
        mode=mode,
        model_id=model_id,
        device=_resolve_device(model.get("device", "auto")),
        dtype=model.get("dtype", "bfloat16"),
        attn_implementation=attn,
        max_new_tokens=int(inference.get("max_new_tokens", 32)),
        temperature=float(inference.get("temperature", 0.0)),
        scoring_method=scoring_method,
        stop_at_control_token=bool(inference.get("stop_at_control_token", True)),
        likelihood_prompt_template_path=_resolve_likelihood_prompt_template_path(
            scoring.get("prompt_template_path")
        ),
        acceptable_answers_path=_optional_project_path(acceptable_answers_value),
        qa_dataset_dir=_project_path(dataset_dir_value),
        split=str(data.get("split", "test")).removeprefix("data_"),
        max_examples=max_examples,
        prompt_column=data.get("prompt_column", "prompt"),
        answer_choice_column=data.get("answer_choice_column", "answer_choice"),
        gold_cell_type_column=data.get("gold_cell_type_column", "gold_cell_type"),
        choices_column=data.get("choices_column", "choices"),
        choice_labels_column=data.get("choice_labels_column", "choice_labels"),
        gene_names_column=data.get("gene_names_column", "gene_names"),
        gene_sentence_column=data.get("gene_sentence_column", "gene_sentence"),
        organism_column=organism_column,
        default_organism=data.get("default_organism", "Homo sapiens"),
        output_dir=_task_output_dir(
            raw,
            TASK_NAME,
            mode,
            output_name_override=output_name_override or reconstruct_model_name,
        ),
        save_predictions=bool(output.get("save_predictions", True)),
        include_prompt=bool(output.get("include_prompt", False)),
        reconstruct_model_name=reconstruct_model_name,
        output_tag=output_tag,
        interventions=interventions,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate generated cell-type multiple-choice QA prompts, either "
            "directly or with SAE-reconstructed gene-token hidden states."
        )
    )
    parser.add_argument("--config", default=None, help=f"YAML config path (default: {DEFAULT_CONFIG}).")
    parser.add_argument(
        "--mode",
        choices=tuple(sorted(VALID_MODES)),
        default=None,
        help="Inference mode: base or reconstruct.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model.base_model: gemma, gemma-2b, gemma-27b, pythia, or a HF model id.",
    )
    parser.add_argument("--dataset", default=None, help=f"Override {TASK_NAME}.data.qa_dataset_dir.")
    parser.add_argument("--split", default=None, help=f"Override {TASK_NAME}.data.split.")
    parser.add_argument("--max-examples", type=int, default=None, help=f"Override {TASK_NAME}.data.max_examples.")
    parser.add_argument(
        "--scoring-method",
        choices=tuple(sorted(VALID_SCORING_METHODS)),
        default=None,
        help="Use likelihood scoring over choices, or free generation for debugging.",
    )
    parser.add_argument(
        "--likelihood-prompt-template",
        default=None,
        help=f"Override {TASK_NAME}.scoring.prompt_template_path.",
    )
    parser.add_argument(
        "--acceptable-answers",
        default=None,
        help=f"Override {TASK_NAME}.scoring.acceptable_answers_path. Use an empty string to disable aliases.",
    )
    parser.add_argument("--output-dir", default=None, help="Override the resolved task output directory.")
    parser.add_argument("--checkpoint", default=None, help="SAE mode: override the SAE checkpoint path.")
    parser.add_argument("--layer", type=int, default=None, help="SAE mode: override the intervention layer index.")
    parser.add_argument("--sae-type", default=None, help="SAE mode: override SAE type: topk, vanilla, or jumprelu.")
    parser.add_argument(
        "--sae-device",
        default=None,
        help="SAE mode: override SAE device. Use 'model' to match the base-model device.",
    )
    parser.add_argument("--output-tag", default=None, help="SAE mode: override output subdirectory tag.")
    parser.add_argument("--no-save-predictions", action="store_true")
    return parser.parse_args()


def apply_cli_overrides(cfg: EvalConfig, args: argparse.Namespace) -> None:
    if args.model is not None:
        cfg.model_id = _resolve_model_id(args.model)
        cfg.attn_implementation = _default_attn_implementation(cfg.model_id)
    if args.dataset is not None:
        cfg.qa_dataset_dir = _project_path(args.dataset)
    if args.split is not None:
        cfg.split = args.split.removeprefix("data_")
    if args.max_examples is not None:
        if args.max_examples <= 0:
            raise ValueError("--max-examples must be positive.")
        cfg.max_examples = args.max_examples
    if args.scoring_method is not None:
        cfg.scoring_method = _normalize_scoring_method(args.scoring_method)
    if args.likelihood_prompt_template is not None:
        cfg.likelihood_prompt_template_path = _resolve_likelihood_prompt_template_path(
            args.likelihood_prompt_template
        )
    if args.acceptable_answers is not None:
        cfg.acceptable_answers_path = _optional_project_path(args.acceptable_answers)
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
        if cfg.mode != "reconstruct":
            raise ValueError("SAE checkpoint/layer overrides require --mode reconstruct.")
        sae_eval = _load_sae_module()
        if cfg.interventions:
            base = cfg.interventions[0]
            layer_idx = args.layer if args.layer is not None else base.layer_idx
            checkpoint_path = (
                _project_path(args.checkpoint)
                if args.checkpoint is not None
                else base.checkpoint_path
            )
            sae_type = args.sae_type if args.sae_type is not None else base.sae_type
            sae_device = args.sae_device if args.sae_device is not None else base.sae_device
            name = base.name
        else:
            if args.checkpoint is None or args.layer is None:
                raise ValueError(
                    "When model.reconstruct.interventions is absent, set both "
                    "--checkpoint and --layer."
                )
            layer_idx = args.layer
            checkpoint_path = _project_path(args.checkpoint)
            sae_type = args.sae_type
            sae_device = args.sae_device or "model"
            name = None

        cfg.interventions = [
            sae_eval.InterventionConfig(
                layer_idx=int(layer_idx),
                checkpoint_path=checkpoint_path,
                sae_type=sae_type,
                sae_device=str(sae_device),
                name=name,
            )
        ]


def make_inference_config(cfg: EvalConfig) -> InferenceConfig:
    return InferenceConfig(
        model_id=cfg.model_id,
        device=cfg.device,
        dtype=cfg.dtype,
        attn_implementation=cfg.attn_implementation,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        prompt_prefix=False,
    )


def _split_dir(cfg: EvalConfig) -> Path:
    return cfg.qa_dataset_dir / f"data_{cfg.split}"


def validate_inputs(cfg: EvalConfig) -> None:
    if not cfg.qa_dataset_dir.exists():
        raise FileNotFoundError(f"QA dataset folder not found: {cfg.qa_dataset_dir}")
    split_dir = _split_dir(cfg)
    if not path_has_arrow_dataset(split_dir):
        raise FileNotFoundError(
            f"Split folder does not look like a HuggingFace dataset: {split_dir}"
        )
    if cfg.scoring_method == "likelihood" and not cfg.likelihood_prompt_template_path.exists():
        raise FileNotFoundError(
            f"Likelihood prompt template not found: {cfg.likelihood_prompt_template_path}"
        )
    if (
        cfg.scoring_method == "likelihood"
        and cfg.acceptable_answers_path is not None
        and not cfg.acceptable_answers_path.exists()
    ):
        raise FileNotFoundError(
            f"Acceptable answers JSON not found: {cfg.acceptable_answers_path}"
        )


def _require_columns(dataset, split_dir: Path, cfg: EvalConfig) -> None:
    required = {
        cfg.prompt_column,
        cfg.answer_choice_column,
        cfg.gold_cell_type_column,
        cfg.choices_column,
        cfg.choice_labels_column,
    }
    if cfg.mode == "reconstruct":
        required.add(cfg.gene_names_column)
    if cfg.scoring_method == "likelihood":
        required.add(cfg.gene_sentence_column)
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"{split_dir} is missing required column(s): {sorted(missing)}")


def _output_paths(cfg: EvalConfig, inf_cfg: InferenceConfig) -> tuple[Path, Path, Path]:
    out_dir = (
        cfg.output_dir
        / cfg.qa_dataset_dir.name
        / inf_cfg.model_short_name
        / f"data_{cfg.split}"
    )
    if cfg.mode == "reconstruct" and cfg.output_tag:
        out_dir = out_dir / _safe_output_name(cfg.output_tag, "run")
    out_dir.mkdir(parents=True, exist_ok=True)
    return (
        out_dir / "predictions.jsonl",
        out_dir / "incorrect_predictions.jsonl",
        out_dir / "summary.json",
    )


def base_model_metadata_card(
    cfg: EvalConfig,
    inf_cfg: InferenceConfig,
) -> dict[str, Any]:
    return {
        "task": TASK_NAME,
        "mode": "base",
        "model_id": cfg.model_id,
        "model_short_name": inf_cfg.model_short_name,
        "model_family": inf_cfg.family,
        "dtype": cfg.dtype,
        "attn_implementation": cfg.attn_implementation,
        "scoring_method": cfg.scoring_method,
        "likelihood_prompt_template_path": str(cfg.likelihood_prompt_template_path)
        if cfg.scoring_method == "likelihood"
        else None,
    }


def _reconstruction_model_metadata_card(
    cfg: EvalConfig,
    inf_cfg: InferenceConfig,
) -> dict[str, Any]:
    sae_eval = _load_sae_module()
    return {
        "task": TASK_NAME,
        "mode": "reconstruct",
        "base_model_id": cfg.model_id,
        "base_model_short_name": inf_cfg.model_short_name,
        "base_model_family": inf_cfg.family,
        "reconstruct_model_name": cfg.reconstruct_model_name,
        "scoring_method": cfg.scoring_method,
        "likelihood_prompt_template_path": str(cfg.likelihood_prompt_template_path)
        if cfg.scoring_method == "likelihood"
        else None,
        "checkpoints": [
            {
                "name": sae_eval._intervention_tag(intervention),
                "checkpoint_path": str(intervention.checkpoint_path),
                "checkpoint_metadata_path": str(
                    intervention.checkpoint_path.parent / MODEL_METADATA_FILENAME
                ),
                "layer_idx": intervention.layer_idx,
                "sae_type": intervention.sae_type,
            }
            for intervention in cfg.interventions
        ],
    }


def write_model_metadata_card(output_dir: Path, metadata: dict[str, Any]) -> Path:
    metadata_path = output_dir / MODEL_METADATA_FILENAME
    with open(metadata_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)
    return metadata_path


def _resolve_sae_device(value: str, inf_cfg: InferenceConfig) -> str:
    if value in {"model", "auto"}:
        return inf_cfg.device
    return value


def _checkpoint_model_metadata(path: Path) -> dict[str, Any]:
    metadata = load_checkpoint_metadata(path)
    model = metadata.get("model")
    return model if isinstance(model, dict) else {}


def load_reconstruction_hooks(
    model,
    cfg: EvalConfig,
    inf_cfg: InferenceConfig,
) -> tuple[list[Any], list[dict[str, Any]]]:
    sae_eval = _load_sae_module()
    hooks: list[Any] = []
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

        name = sae_eval._intervention_tag(intervention)
        hooks.append(
            sae_eval.SAEReconstructionHook(
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
                "checkpoint_prompt_prefix": checkpoint_model.get("prompt_prefix"),
            }
        )
        metadata.append(meta)

    return hooks, metadata


def _safe_accuracy(correct: int, total: int) -> float:
    return float(correct / total) if total else 0.0


def run_eval(cfg: EvalConfig) -> dict[str, Any]:
    validate_inputs(cfg)
    split_dir = _split_dir(cfg)
    dataset = load_from_disk(str(split_dir))
    _require_columns(dataset, split_dir, cfg)

    total_available = len(dataset)
    n_examples = (
        min(cfg.max_examples, total_available) if cfg.max_examples else total_available
    )
    inf_cfg = make_inference_config(cfg)
    predictions_path, incorrect_predictions_path, summary_path = _output_paths(
        cfg,
        inf_cfg,
    )
    metadata = (
        _reconstruction_model_metadata_card(cfg, inf_cfg)
        if cfg.mode == "reconstruct"
        else base_model_metadata_card(cfg, inf_cfg)
    )
    model_metadata_path = write_model_metadata_card(summary_path.parent, metadata)

    print("[Config]")
    print(f"  mode      : {cfg.mode}")
    if cfg.reconstruct_model_name:
        print(f"  reconstruct_model: {cfg.reconstruct_model_name}")
    print(f"  model     : {cfg.model_id}")
    print(f"  device    : {cfg.device}")
    print(f"  dtype     : {cfg.dtype}")
    print(f"  dataset   : {cfg.qa_dataset_dir}")
    print(f"  split     : data_{cfg.split} ({total_available:,} available)")
    print(f"  examples  : {n_examples:,}")
    print(f"  scoring   : {cfg.scoring_method}")
    if cfg.scoring_method == "likelihood":
        print(f"  score tpl : {cfg.likelihood_prompt_template_path}")
    if cfg.mode == "reconstruct":
        print("  interventions:")
        for intervention in cfg.interventions:
            print(
                f"    - layer {intervention.layer_idx}: "
                f"{intervention.checkpoint_path} "
                f"(sae_type={intervention.sae_type or 'auto'}, device={intervention.sae_device})"
            )

    tokenizer, model = load_model(inf_cfg)
    acceptable_answers = (
        cell_eval.load_acceptable_answers(cfg.acceptable_answers_path)
        if cfg.acceptable_answers_path is not None
        else {}
    )
    hooks: list[Any] = []
    intervention_meta: list[dict[str, Any]] = []
    patcher_context = nullcontext(None)
    if cfg.mode == "reconstruct":
        hooks, intervention_meta = load_reconstruction_hooks(model, cfg, inf_cfg)
        patcher_context = _load_sae_module().SAEReconstructionPatcher(hooks)

    correct = 0
    ambiguous = 0
    by_cell_type: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "correct": 0}
    )
    by_choice: dict[str, dict[str, int]] = defaultdict(
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

    with patcher_context:
        with (
            prediction_writer_context as writer,
            incorrect_writer_context as incorrect_writer,
        ):
            for idx in tqdm(range(n_examples), desc="Evaluating"):
                sample = dataset[idx]
                generated_text: str | None = None
                raw_generated_text: str | None = None

                if cfg.scoring_method == "likelihood":
                    score = score_likelihood_prediction(
                        sample,
                        tokenizer,
                        model,
                        inf_cfg,
                        cfg,
                        acceptable_answers,
                        hooks=hooks,
                    )
                else:
                    prompt = str(sample[cfg.prompt_column])
                    if hooks:
                        prepare_hooks_for_prompt(
                            hooks,
                            tokenizer,
                            prompt,
                            gene_names_for_sample(sample, cfg),
                        )
                    try:
                        raw_generated_text = generate(prompt, tokenizer, model, inf_cfg)
                    finally:
                        if hooks:
                            clear_hook_prompt_state(hooks)
                    generated_text = (
                        trim_control_token(raw_generated_text)
                        if cfg.stop_at_control_token
                        else raw_generated_text
                    )
                    score = score_generated_prediction(sample, generated_text, cfg)

                is_correct = bool(score["correct"])
                correct += int(is_correct)
                ambiguous += int(score["ambiguous"])

                gold_cell_type = score["gold_cell_type"]
                gold_choice = score["gold_choice_label"]
                by_cell_type[gold_cell_type]["total"] += 1
                by_cell_type[gold_cell_type]["correct"] += int(is_correct)
                by_choice[gold_choice]["total"] += 1
                by_choice[gold_choice]["correct"] += int(is_correct)

                if writer is not None:
                    row = {
                        "index": idx,
                        "id": sample.get("id"),
                        "scoring_method": cfg.scoring_method,
                        **score,
                    }
                    if generated_text is not None:
                        row["generated_text"] = generated_text
                    if raw_generated_text is not None:
                        row["raw_generated_text"] = raw_generated_text
                    if cfg.include_prompt:
                        row["prompt"] = (
                            score.get("scoring_prompt")
                            if cfg.scoring_method == "likelihood"
                            else str(sample[cfg.prompt_column])
                        )
                    writer.write(json.dumps(row) + "\n")
                    if not is_correct and incorrect_writer is not None:
                        incorrect_writer.write(json.dumps(row) + "\n")

    for meta, hook in zip(intervention_meta, hooks):
        meta["hook_calls"] = hook.num_calls
        meta["hook_tokens"] = hook.num_tokens

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": TASK_NAME,
        "mode": cfg.mode,
        "reconstruct_model_name": cfg.reconstruct_model_name,
        "model_id": cfg.model_id,
        "dataset_dir": str(cfg.qa_dataset_dir),
        "split": f"data_{cfg.split}",
        "num_available": total_available,
        "num_examples": n_examples,
        "num_correct": correct,
        "num_incorrect": n_examples - correct,
        "num_ambiguous": ambiguous,
        "accuracy": _safe_accuracy(correct, n_examples),
        "scoring_method": cfg.scoring_method,
        "max_new_tokens": cfg.max_new_tokens,
        "temperature": cfg.temperature,
        "stop_at_control_token": cfg.stop_at_control_token,
        "likelihood_prompt_template_path": str(cfg.likelihood_prompt_template_path)
        if cfg.scoring_method == "likelihood"
        else None,
        "acceptable_answers_path": str(cfg.acceptable_answers_path)
        if cfg.acceptable_answers_path
        else None,
        "num_acceptable_answer_entries": len(acceptable_answers),
        "prompt_column": cfg.prompt_column,
        "answer_choice_column": cfg.answer_choice_column,
        "gold_cell_type_column": cfg.gold_cell_type_column,
        "choices_column": cfg.choices_column,
        "choice_labels_column": cfg.choice_labels_column,
        "gene_names_column": cfg.gene_names_column,
        "gene_sentence_column": cfg.gene_sentence_column,
        "organism_column": cfg.organism_column,
        "default_organism": cfg.default_organism,
        "model_metadata_path": str(model_metadata_path),
        "predictions_path": str(predictions_path) if cfg.save_predictions else None,
        "incorrect_predictions_path": str(incorrect_predictions_path)
        if cfg.save_predictions
        else None,
        "interventions": intervention_meta,
        "by_cell_type": {
            key: {
                "total": stats["total"],
                "correct": stats["correct"],
                "accuracy": _safe_accuracy(stats["correct"], stats["total"]),
            }
            for key, stats in sorted(by_cell_type.items())
        },
        "by_choice": {
            key: {
                "total": stats["total"],
                "correct": stats["correct"],
                "accuracy": _safe_accuracy(stats["correct"], stats["total"]),
            }
            for key, stats in sorted(by_choice.items())
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print("\n[Results]")
    print(f"  accuracy  : {summary['accuracy']:.3f} ({correct:,}/{n_examples:,})")
    print(f"  ambiguous : {ambiguous:,}")
    if cfg.save_predictions:
        print(f"  generated : {predictions_path}")
        print(f"  incorrect : {incorrect_predictions_path}")
    print(f"  summary   : {summary_path}")
    print(f"  metadata  : {model_metadata_path}")
    return summary


def _requested_sae_mode(args: argparse.Namespace) -> bool:
    return any(
        getattr(args, name) is not None
        for name in ("checkpoint", "layer", "sae_type", "sae_device", "output_tag")
    )


def selected_mode(args: argparse.Namespace) -> str:
    if args.mode is not None:
        return _normalize_mode(args.mode)
    if _requested_sae_mode(args):
        return "reconstruct"

    raw = _load_yaml(args.config or DEFAULT_CONFIG)
    model = _model_config(raw)
    module = _mapping(raw.get(TASK_NAME, raw), TASK_NAME)
    return _normalize_mode(model.get("mode", module.get("mode", "base")))


def main() -> None:
    args = parse_args()
    mode = selected_mode(args)
    cfg = load_eval_config(args.config, mode_override=mode)
    apply_cli_overrides(cfg, args)
    run_eval(cfg)


if __name__ == "__main__":
    main()
