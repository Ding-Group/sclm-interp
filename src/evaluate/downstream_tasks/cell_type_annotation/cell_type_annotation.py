#!/usr/bin/env python3
"""
Evaluate C2S models on the downstream cell-type annotation task.

The evaluator loads a prepared C2S dataset split, prompts the selected base
model for each cell, saves the generated answer, and scores whether the gold
cell type or a configured acceptable answer is encoded in the generated text.
It can run the base model directly or dispatch to SAE reconstruction inference,
which replaces configured layer hidden states with SAE reconstructions before
generation continues.

Usage:
    python src/evaluate/downstream_tasks/cell_type_annotation/cell_type_annotation.py
    python src/evaluate/downstream_tasks/cell_type_annotation/cell_type_annotation.py --mode base
    python src/evaluate/downstream_tasks/cell_type_annotation/cell_type_annotation.py --mode reconstruct --max-examples 50
    python src/evaluate/downstream_tasks/cell_type_annotation/cell_type_annotation.py --config configs/eval_tasks.yaml
    python src/evaluate/downstream_tasks/cell_type_annotation/cell_type_annotation.py --model pythia --max-examples 50
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import torch
import yaml
from datasets import load_from_disk
from tqdm.auto import tqdm

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_config import path_has_arrow_dataset
from src.data.inference import (
    DEFAULT_PROMPT_TEMPLATE_PATH,
    InferenceConfig,
    build_cell_type_prompt,
    generate,
    load_model,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "eval_tasks.yaml"
DEFAULT_ACCEPTABLE_ANSWERS = EVAL_DIR / "cell_type_acceptable_answers.json"
MODEL_METADATA_FILENAME = "model_metadata.yaml"
TASK_NAME = "cell_type_annotation"
MODEL_SECTION = "model"
BASE_OUTPUT_NAME = "base"

BASE_MODEL_IDS = {
    "gemma": "vandijklab/C2S-Scale-Gemma-2-2B",
    "gemma-2b": "vandijklab/C2S-Scale-Gemma-2-2B",
    "gemma-27b": "vandijklab/C2S-Scale-Gemma-2-27B",
    "pythia": "vandijklab/C2S-Scale-Pythia-1b-pt",
    "pythia-1b": "vandijklab/C2S-Scale-Pythia-1b-pt",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_OUTPUT_SAFE_RE = re.compile(r"[^A-Za-z0-9_.=-]+")
VALID_MODES = {"base", "reconstruct"}


def _normalize_mode(mode: str | None) -> str:
    value = "base" if mode is None else str(mode).strip()
    if value not in VALID_MODES:
        valid = ", ".join(sorted(VALID_MODES))
        raise ValueError(f"Unknown cell-type annotation mode {mode!r}. Valid: {valid}.")
    return value


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected {context} to be a mapping.")
    return value


@dataclass
class EvalConfig:
    mode: str
    model_id: str
    device: str
    dtype: str
    attn_implementation: str | None
    max_new_tokens: int
    temperature: float
    prompt_prefix: bool
    prompt_template_path: Path
    c2s_dataset_dir: Path
    split: str
    max_examples: int | None
    cell_type_column: str
    organism_column: str | None
    default_organism: str
    acceptable_answers_path: Path | None
    output_dir: Path
    save_predictions: bool
    include_prompt: bool


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _optional_project_path(value: str | Path | None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return _project_path(value)


def _resolve_prompt_template_path(value: str | Path | None) -> Path:
    if value is None or str(value).strip() == "":
        value = DEFAULT_PROMPT_TEMPLATE_PATH
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
    safe = _OUTPUT_SAFE_RE.sub("_", candidate).strip("_")
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

    dataset_dir_value = data.get("c2s_dataset_dir")
    if not dataset_dir_value:
        raise ValueError("Config cell_type_annotation.data.c2s_dataset_dir is required.")

    max_examples = data.get("max_examples")
    if max_examples is not None:
        max_examples = int(max_examples)
        if max_examples <= 0:
            raise ValueError("cell_type_annotation.data.max_examples must be positive or null.")

    organism_column = data.get("organism_column")
    if organism_column == "":
        organism_column = None

    acceptable_answers_value = scoring.get(
        "acceptable_answers_path",
        module.get("acceptable_answers_path", DEFAULT_ACCEPTABLE_ANSWERS),
    )

    mode = _normalize_mode(
        mode_override or model.get("mode", module.get("mode", "base"))
    )

    return EvalConfig(
        mode=mode,
        model_id=model_id,
        device=_resolve_device(model.get("device", "auto")),
        dtype=model.get("dtype", "bfloat16"),
        attn_implementation=attn,
        max_new_tokens=int(inference.get("max_new_tokens", 32)),
        temperature=float(inference.get("temperature", 0.0)),
        prompt_prefix=bool(inference.get("prompt_prefix", True)),
        prompt_template_path=_resolve_prompt_template_path(
            inference.get("prompt_template_path")
        ),
        c2s_dataset_dir=_project_path(dataset_dir_value),
        split=str(data.get("split", "train")).removeprefix("data_"),
        max_examples=max_examples,
        cell_type_column=data.get("cell_type_column", "cell_type"),
        organism_column=organism_column,
        default_organism=data.get("default_organism", "Homo sapiens"),
        acceptable_answers_path=_optional_project_path(acceptable_answers_value),
        output_dir=_task_output_dir(
            raw,
            TASK_NAME,
            mode,
            output_name_override=output_name_override,
        ),
        save_predictions=bool(output.get("save_predictions", True)),
        include_prompt=bool(output.get("include_prompt", False)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate C2S model generations for cell-type annotation, either "
            "directly or with SAE-reconstructed hidden states."
        )
    )
    parser.add_argument("--config", default=None, help=f"YAML config path (default: {DEFAULT_CONFIG}).")
    parser.add_argument(
        "--mode",
        choices=tuple(sorted(VALID_MODES)),
        default=None,
        help=(
            "Inference mode. Use 'base' for the unmodified model or "
            "'reconstruct' to use configured SAE reconstruction hooks."
        ),
    )
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
    mode = getattr(args, "mode", None)
    if mode is not None:
        cfg.mode = _normalize_mode(mode)
    if args.model is not None:
        cfg.model_id = _resolve_model_id(args.model)
        cfg.attn_implementation = _default_attn_implementation(cfg.model_id)
    if args.dataset is not None:
        cfg.c2s_dataset_dir = _project_path(args.dataset)
    if args.split is not None:
        cfg.split = args.split.removeprefix("data_")
    if args.max_examples is not None:
        if args.max_examples <= 0:
            raise ValueError("--max-examples must be positive.")
        cfg.max_examples = args.max_examples
    if getattr(args, "prompt_template", None) is not None:
        cfg.prompt_template_path = _resolve_prompt_template_path(args.prompt_template)
    if args.acceptable_answers is not None:
        cfg.acceptable_answers_path = _optional_project_path(args.acceptable_answers)
    if args.output_dir is not None:
        cfg.output_dir = _project_path(args.output_dir)
    if args.no_save_predictions:
        cfg.save_predictions = False


def make_inference_config(cfg: EvalConfig) -> InferenceConfig:
    return InferenceConfig(
        model_id=cfg.model_id,
        device=cfg.device,
        dtype=cfg.dtype,
        attn_implementation=cfg.attn_implementation,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        prompt_prefix=cfg.prompt_prefix,
        prompt_template_path=cfg.prompt_template_path,
    )


def _split_dir(cfg: EvalConfig) -> Path:
    return cfg.c2s_dataset_dir / f"data_{cfg.split}"


def validate_inputs(cfg: EvalConfig) -> None:
    if not cfg.c2s_dataset_dir.exists():
        raise FileNotFoundError(f"C2S dataset folder not found: {cfg.c2s_dataset_dir}")
    split_dir = _split_dir(cfg)
    if not path_has_arrow_dataset(split_dir):
        raise FileNotFoundError(
            f"Split folder does not look like a HuggingFace dataset: {split_dir}"
        )


def _label_tokens(text: Any) -> list[str]:
    return _TOKEN_RE.findall(str(text).lower())


def _token_variants(token: str) -> set[str]:
    variants = {token}
    if token.endswith("ies") and len(token) > 3:
        variants.add(token[:-3] + "y")
    if token.endswith("s") and len(token) > 1:
        variants.add(token[:-1])
    else:
        variants.add(token + "s")
    return variants


def _label_key(text: Any) -> str:
    return " ".join(_label_tokens(text))


def _coerce_answer_list(label: str, value: Any) -> list[str]:
    if isinstance(value, str):
        answers = [value]
    elif isinstance(value, dict):
        answers = value.get("answers", value.get("acceptable_answers"))
        if answers is None:
            raise ValueError(
                f"Acceptable answer entry for {label!r} must contain "
                "'answers' or 'acceptable_answers'."
            )
    else:
        answers = value

    if not isinstance(answers, list):
        raise ValueError(
            f"Acceptable answers for {label!r} must be a string or list of strings."
        )

    cleaned: list[str] = []
    for answer in answers:
        if not isinstance(answer, str):
            raise ValueError(
                f"Acceptable answer for {label!r} is not a string: {answer!r}"
            )
        answer = answer.strip()
        if answer:
            cleaned.append(answer)
    return cleaned


def load_acceptable_answers(path: str | Path | None) -> dict[str, list[str]]:
    """Load cell-type answer aliases keyed by normalized gold label."""
    if path is None:
        return {}

    path = _project_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Acceptable answers JSON not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        raw_answers = raw.get("answers", raw)
        if not isinstance(raw_answers, dict):
            raise ValueError(
                f"{path} field 'answers' must be an object mapping labels to answers."
            )
        items = raw_answers.items()
    elif isinstance(raw, list):
        records = []
        for record in raw:
            if not isinstance(record, dict):
                raise ValueError(f"{path} list entries must be objects.")
            label = record.get("cell_type", record.get("label"))
            if not isinstance(label, str) or not label.strip():
                raise ValueError(
                    f"{path} list entries must include a non-empty 'cell_type'."
                )
            records.append((label, record))
        items = records
    else:
        raise ValueError(
            f"{path} must contain a JSON object or a list of cell-type records."
        )

    answer_bank: dict[str, list[str]] = {}
    for label, value in items:
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"{path} contains a non-string cell-type label: {label!r}")
        key = _label_key(label)
        if not key:
            raise ValueError(
                f"{path} cell-type label has no searchable tokens: {label!r}"
            )
        answer_bank.setdefault(key, []).extend(_coerce_answer_list(label, value))

    return {
        key: sorted(set(answers), key=str.lower)
        for key, answers in answer_bank.items()
    }


def _accepted_labels(
    gold_cell_type: str,
    acceptable_answers: dict[str, list[str]] | None = None,
) -> list[str]:
    labels = [gold_cell_type]
    if acceptable_answers:
        labels.extend(acceptable_answers.get(_label_key(gold_cell_type), []))

    deduped: list[str] = []
    seen: set[str] = set()
    for label in labels:
        key = _label_key(label)
        if key and key not in seen:
            deduped.append(label)
            seen.add(key)
    return deduped


def _contains_token_sequence(haystack: list[str], needle: list[str]) -> bool:
    if len(needle) > len(haystack):
        return False
    return any(
        haystack[i : i + len(needle)] == needle
        for i in range(len(haystack) - len(needle) + 1)
    )


def _candidate_is_encoded(candidate_label: str, generated_text: str) -> bool:
    """Return true if generated_text contains the normalized candidate label."""
    gold_tokens = _label_tokens(candidate_label)
    generated_tokens = _label_tokens(generated_text)
    if not gold_tokens or not generated_tokens:
        return False

    if _contains_token_sequence(generated_tokens, gold_tokens):
        return True

    generated_set = set(generated_tokens)
    return all(_token_variants(token) & generated_set for token in gold_tokens)


def find_encoded_answer(
    gold_cell_type: str,
    generated_text: str,
    acceptable_answers: dict[str, list[str]] | None = None,
) -> str | None:
    for candidate_label in _accepted_labels(gold_cell_type, acceptable_answers):
        if _candidate_is_encoded(candidate_label, generated_text):
            return candidate_label
    return None


def label_is_encoded(
    gold_cell_type: str,
    generated_text: str,
    acceptable_answers: dict[str, list[str]] | None = None,
) -> bool:
    return find_encoded_answer(gold_cell_type, generated_text, acceptable_answers) is not None


def build_prompt(sample: dict[str, Any], cfg: EvalConfig) -> str:
    cell_sentence = sample["cell_sentence"]
    if not cfg.prompt_prefix:
        return cell_sentence

    organism = cfg.default_organism
    if cfg.organism_column and cfg.organism_column in sample:
        organism = sample[cfg.organism_column] or organism
    return build_cell_type_prompt(
        cell_sentence,
        organism=organism,
        prompt_template_path=cfg.prompt_template_path,
    )


def _output_paths(cfg: EvalConfig, inf_cfg: InferenceConfig) -> tuple[Path, Path, Path]:
    out_dir = (
        cfg.output_dir
        / cfg.c2s_dataset_dir.name
        / inf_cfg.model_short_name
        / f"data_{cfg.split}"
    )
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
        "prompt_template_path": str(cfg.prompt_template_path)
        if cfg.prompt_prefix
        else None,
    }


def write_model_metadata_card(output_dir: Path, metadata: dict[str, Any]) -> Path:
    metadata_path = output_dir / MODEL_METADATA_FILENAME
    with open(metadata_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)
    return metadata_path


def _safe_accuracy(correct: int, total: int) -> float:
    return float(correct / total) if total else 0.0


def plot_accuracy_figures(summary: dict[str, Any], output_dir: Path) -> dict[str, str]:
    """Save accuracy diagnostic figures for a completed cell-type eval summary."""
    by_cell_type = summary.get("by_cell_type") or {}
    if not by_cell_type:
        return {}

    mpl_config_dir = Path("/tmp/matplotlib")
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "cell_type": cell_type,
            "total": int(stats["total"]),
            "correct": int(stats["correct"]),
            "accuracy": float(stats["accuracy"]),
        }
        for cell_type, stats in by_cell_type.items()
    ]
    rows_by_accuracy = sorted(rows, key=lambda row: (row["accuracy"], row["total"]))

    figure_paths: dict[str, str] = {}
    accuracy = float(summary.get("accuracy", 0.0))
    palette_green = "#4C956C"
    palette_red = "#C65D5D"
    text_dark = "#263238"
    grid = "#E6E8EB"

    stale_overall_path = figures_dir / "overall_accuracy.png"
    if stale_overall_path.exists():
        stale_overall_path.unlink()

    per_type_path = figures_dir / "per_cell_type_accuracy.png"
    height = max(3.4, 0.48 * len(rows_by_accuracy) + 1.5)
    fig, ax = plt.subplots(figsize=(8.5, height))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    labels = [row["cell_type"] for row in rows_by_accuracy]
    values = [row["accuracy"] for row in rows_by_accuracy]
    colors = [palette_red if value < accuracy else palette_green for value in values]
    ax.barh(labels, values, color=colors, height=0.58, zorder=3)
    ax.axvline(
        accuracy,
        color="#7B8790",
        linestyle=(0, (3, 3)),
        linewidth=1.2,
        zorder=2,
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Accuracy", color=text_dark)
    ax.set_title("Accuracy by Cell Type", color=text_dark, loc="left", pad=16)
    ax.xaxis.grid(True, color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.text(
        min(accuracy + 0.015, 0.86),
        1.02,
        f"Overall {accuracy:.1%}",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=9,
        color="#6F7B84",
    )
    for i, row in enumerate(rows_by_accuracy):
        x = min(row["accuracy"] + 0.015, 0.93)
        ax.text(
            x,
            i,
            f"{row['accuracy']:.1%} ({row['correct']:,}/{row['total']:,})",
            va="center",
            fontsize=9,
            color=text_dark,
        )
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(per_type_path, dpi=160)
    plt.close(fig)
    figure_paths["per_cell_type_accuracy"] = str(per_type_path)

    return figure_paths


def run_eval(cfg: EvalConfig) -> dict[str, Any]:
    validate_inputs(cfg)
    split_dir = _split_dir(cfg)
    dataset = load_from_disk(str(split_dir))
    required = {"cell_sentence", cfg.cell_type_column}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"{split_dir} is missing required column(s): {sorted(missing)}")

    total_available = len(dataset)
    n_examples = (
        min(cfg.max_examples, total_available) if cfg.max_examples else total_available
    )
    inf_cfg = make_inference_config(cfg)
    predictions_path, incorrect_predictions_path, summary_path = _output_paths(
        cfg, inf_cfg
    )
    model_metadata_path = write_model_metadata_card(
        summary_path.parent,
        base_model_metadata_card(cfg, inf_cfg),
    )
    acceptable_answers = load_acceptable_answers(cfg.acceptable_answers_path)

    print("[Config]")
    print(f"  mode      : {cfg.mode}")
    print(f"  model     : {cfg.model_id}")
    print(f"  device    : {cfg.device}")
    print(f"  dtype     : {cfg.dtype}")
    print(f"  dataset   : {cfg.c2s_dataset_dir}")
    print(f"  split     : data_{cfg.split} ({total_available:,} available)")
    print(f"  examples  : {n_examples:,}")
    print(f"  prompt    : {'template' if cfg.prompt_prefix else 'bare cell_sentence'}")
    if cfg.prompt_prefix:
        print(f"  prompt tpl: {cfg.prompt_template_path}")
    if cfg.acceptable_answers_path:
        print(
            f"  answers   : {cfg.acceptable_answers_path} "
            f"({len(acceptable_answers):,} label entries)"
        )
    else:
        print("  answers   : disabled")

    tokenizer, model = load_model(inf_cfg)

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

    with (
        prediction_writer_context as writer,
        incorrect_writer_context as incorrect_writer,
    ):
        for idx in tqdm(range(n_examples), desc="Evaluating"):
            sample = dataset[idx]
            gold = str(sample[cfg.cell_type_column])
            prompt = build_prompt(sample, cfg)
            generated_text = generate(prompt, tokenizer, model, inf_cfg)
            matched_answer = find_encoded_answer(gold, generated_text, acceptable_answers)
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

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": cfg.mode,
        "model_id": cfg.model_id,
        "dataset_dir": str(cfg.c2s_dataset_dir),
        "split": f"data_{cfg.split}",
        "cell_type_column": cfg.cell_type_column,
        "num_available": total_available,
        "num_examples": n_examples,
        "num_correct": correct,
        "num_incorrect": n_examples - correct,
        "accuracy": _safe_accuracy(correct, n_examples),
        "prompt_prefix": cfg.prompt_prefix,
        "prompt_template_path": str(cfg.prompt_template_path)
        if cfg.prompt_prefix
        else None,
        "max_new_tokens": cfg.max_new_tokens,
        "temperature": cfg.temperature,
        "acceptable_answers_path": str(cfg.acceptable_answers_path)
        if cfg.acceptable_answers_path
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
    summary["figures"] = plot_accuracy_figures(summary, summary_path.parent)

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
    return load_eval_config(args.config).mode


def run_reconstruct_eval(args: argparse.Namespace) -> dict[str, Any]:
    from src.evaluate.downstream_tasks import sae_reconstruction_inference

    cfg = sae_reconstruction_inference.load_eval_config(args.config)
    sae_reconstruction_inference.apply_cli_overrides(cfg, args)
    return sae_reconstruction_inference.run_eval(cfg)


def main() -> None:
    args = parse_args()
    mode = selected_mode(args)
    if mode == "reconstruct":
        run_reconstruct_eval(args)
        return

    cfg = load_eval_config(args.config, mode_override=mode)
    apply_cli_overrides(cfg, args)
    run_eval(cfg)


if __name__ == "__main__":
    main()
