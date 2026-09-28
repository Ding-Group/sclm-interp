#!/usr/bin/env python3
"""
Generate natural-language interpretations for groups of C2S cell sentences.

The evaluator loads a prepared C2S dataset split, groups one or more cells into
the tutorial-style natural-language interpretation prompt, saves generated
summaries, and writes a small run summary. It supports the base model directly
and the same SAE reconstruction hooks used by the cell-type annotation task.

Each invocation gets a new numbered run directory so an earlier experiment is
never overwritten::

    results/natural_language_interpretation/<experiment>/<dataset>/<model>/
        runs.json
        1/
            predictions.jsonl
            summary.json
            model_metadata.yaml
        2/
            ...

Usage:
    python src/evaluate/downstream_tasks/natural_language_interpretation/natural_language_interpretation.py
    python src/evaluate/downstream_tasks/natural_language_interpretation/natural_language_interpretation.py --max-examples 10 --cells-per-prompt 1 --num-genes 300
    python src/evaluate/downstream_tasks/natural_language_interpretation/natural_language_interpretation.py --mode reconstruct --max-examples 10
    python src/evaluate/downstream_tasks/natural_language_interpretation/natural_language_interpretation.py --model pythia --dataset datasets/c2s_datasets/pbmc-subset-test
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import random
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
from src.data.inference import InferenceConfig, load_model

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "eval_tasks.yaml"
DEFAULT_PROMPT_TEMPLATE = (
    PROJECT_ROOT / "src" / "prompts" / "natural_language_interpretation.txt"
)
MODEL_METADATA_FILENAME = "model_metadata.yaml"
RUN_INDEX_FILENAME = "runs.json"
TASK_NAME = "natural_language_interpretation"
MODEL_SECTION = "model"
BASE_OUTPUT_NAME = "base"

BASE_MODEL_IDS = {
    "gemma": "vandijklab/C2S-Scale-Gemma-2-2B",
    "gemma-2b": "vandijklab/C2S-Scale-Gemma-2-2B",
    "gemma-27b": "vandijklab/C2S-Scale-Gemma-2-27B",
    "pythia": "vandijklab/C2S-Scale-Pythia-1b-pt",
    "pythia-1b": "vandijklab/C2S-Scale-Pythia-1b-pt",
}

_OUTPUT_SAFE_RE = re.compile(r"[^A-Za-z0-9_.=-]+")
_WORD_RE = re.compile(r"\S+")
VALID_MODES = {"base", "reconstruct"}


def _normalize_mode(mode: str | None) -> str:
    value = "base" if mode is None else str(mode).strip()
    if value not in VALID_MODES:
        valid = ", ".join(sorted(VALID_MODES))
        raise ValueError(f"Unknown natural-language interpretation mode {mode!r}. Valid: {valid}.")
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
    do_sample: bool
    top_k: int | None
    top_p: float | None
    stop_at_control_token: bool
    prompt_prefix: bool
    prompt_template_path: Path
    c2s_dataset_dir: Path
    split: str
    max_examples: int | None
    cells_per_prompt: int
    num_genes: int
    drop_last: bool
    shuffle_cells: bool
    cell_sentence_column: str
    cell_name_column: str | None
    cell_type_column: str | None
    organism_column: str | None
    default_organism: str
    output_dir: Path
    save_predictions: bool
    include_prompt: bool
    seed: int | None = None
    reconstruct_model_name: str | None = None
    output_tag: str | None = None
    interventions: list[Any] = field(default_factory=list)


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _resolve_prompt_template_path(value: str | Path | None) -> Path:
    if value is None or str(value).strip() == "":
        value = DEFAULT_PROMPT_TEMPLATE
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


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch RNGs so sampled generations are reproducible."""
    from transformers import set_seed

    set_seed(int(seed))


def _example_seed(seed: int | None, group_index: int) -> int | None:
    """Derive a per-group seed.

    Seeding each prompt group independently keeps a group's generation identical
    regardless of how many groups ran before it, so a base run and a reconstruct
    run share the same sampling noise and any difference is attributable to the
    intervention rather than to RNG drift.
    """
    if seed is None:
        return None
    return (int(seed) + int(group_index)) % (2**31 - 1)


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
    output = _mapping(module.get("output"), f"{TASK_NAME}.output")

    model_value = model.get("base_model", model.get("model_id", "gemma"))
    model_id = _resolve_model_id(model_value)
    attn = model.get("attn_implementation", _default_attn_implementation(model_id))
    mode = _normalize_mode(
        mode_override or model.get("mode", module.get("mode", "base"))
    )

    dataset_dir_value = data.get("c2s_dataset_dir")
    if not dataset_dir_value:
        raise ValueError(f"Config {TASK_NAME}.data.c2s_dataset_dir is required.")

    max_examples = data.get("max_examples")
    if max_examples is not None:
        max_examples = int(max_examples)
        if max_examples <= 0:
            raise ValueError(f"{TASK_NAME}.data.max_examples must be positive or null.")

    cells_per_prompt = int(data.get("cells_per_prompt", data.get("num_cells", 2)))
    if cells_per_prompt <= 0:
        raise ValueError(f"{TASK_NAME}.data.cells_per_prompt must be positive.")

    num_genes = int(data.get("num_genes", 300))
    if num_genes <= 0:
        raise ValueError(f"{TASK_NAME}.data.num_genes must be positive.")

    organism_column = data.get("organism_column")
    if organism_column == "":
        organism_column = None

    cell_name_column = data.get("cell_name_column", "cell_name")
    if cell_name_column == "":
        cell_name_column = None

    cell_type_column = data.get("cell_type_column", "cell_type")
    if cell_type_column == "":
        cell_type_column = None

    reconstruct_model_name = None
    interventions: list[Any] = []
    output_tag = None
    if mode == "reconstruct":
        reconstruct_model_name = _configured_reconstruct_model_name(raw)
        reconstruct = _model_reconstruct_config(raw)
        if not reconstruct:
            raise ValueError("Config section 'model.reconstruct' is required for reconstruct mode.")
        sae_eval = _load_sae_module()
        interventions = sae_eval._load_interventions(reconstruct)
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
        max_new_tokens=int(inference.get("max_new_tokens", 192)),
        temperature=float(inference.get("temperature", 0.0)),
        do_sample=bool(
            inference.get(
                "do_sample",
                float(inference.get("temperature", 0.0)) > 0.0,
            )
        ),
        top_k=int(inference["top_k"]) if inference.get("top_k") is not None else None,
        top_p=float(inference["top_p"]) if inference.get("top_p") is not None else None,
        stop_at_control_token=bool(inference.get("stop_at_control_token", True)),
        seed=int(inference["seed"]) if inference.get("seed") is not None else None,
        prompt_prefix=bool(inference.get("prompt_prefix", True)),
        prompt_template_path=_resolve_prompt_template_path(
            inference.get("prompt_template_path")
        ),
        c2s_dataset_dir=_project_path(dataset_dir_value),
        split=str(data.get("split", "train")).removeprefix("data_"),
        max_examples=max_examples,
        cells_per_prompt=cells_per_prompt,
        num_genes=num_genes,
        drop_last=bool(data.get("drop_last", False)),
        shuffle_cells=bool(data.get("shuffle_cells", True)),
        cell_sentence_column=data.get("cell_sentence_column", "cell_sentence"),
        cell_name_column=cell_name_column,
        cell_type_column=cell_type_column,
        organism_column=organism_column,
        default_organism=data.get("default_organism", "Homo sapiens"),
        output_dir=_task_output_dir(
            raw,
            TASK_NAME,
            mode,
            output_name_override=output_name_override or reconstruct_model_name,
        ),
        save_predictions=bool(output.get("save_predictions", True)),
        include_prompt=bool(output.get("include_prompt", True)),
        reconstruct_model_name=reconstruct_model_name,
        output_tag=output_tag,
        interventions=interventions,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate C2S natural-language interpretation summaries, either "
            "directly or with SAE-reconstructed hidden states."
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
    parser.add_argument("--dataset", default=None, help=f"Override {TASK_NAME}.data.c2s_dataset_dir.")
    parser.add_argument("--split", default=None, help=f"Override {TASK_NAME}.data.split.")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Override the number of prompt groups to generate.",
    )
    parser.add_argument(
        "--cells-per-prompt",
        type=int,
        default=None,
        help="Override the number of cells included in each prompt.",
    )
    parser.add_argument(
        "--num-genes",
        type=int,
        default=None,
        help="Override the number of top ordered genes included for each cell.",
    )
    parser.add_argument(
        "--prompt-template",
        default=None,
        help=f"Override {TASK_NAME}.inference.prompt_template_path.",
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
    parser.add_argument(
        "--output-tag",
        default=None,
        help="SAE mode: optional label recorded in the numbered run metadata.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            f"Override {TASK_NAME}.inference.seed. Each prompt group is seeded "
            "with seed+group_index. Use -1 to disable seeding."
        ),
    )
    parser.add_argument("--no-save-predictions", action="store_true")
    return parser.parse_args()


def apply_cli_overrides(cfg: EvalConfig, args: argparse.Namespace) -> None:
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
    if args.cells_per_prompt is not None:
        if args.cells_per_prompt <= 0:
            raise ValueError("--cells-per-prompt must be positive.")
        cfg.cells_per_prompt = args.cells_per_prompt
    if args.num_genes is not None:
        if args.num_genes <= 0:
            raise ValueError("--num-genes must be positive.")
        cfg.num_genes = args.num_genes
    if args.prompt_template is not None:
        cfg.prompt_template_path = _resolve_prompt_template_path(args.prompt_template)
    if args.output_dir is not None:
        cfg.output_dir = _project_path(args.output_dir)
    if args.output_tag is not None:
        cfg.output_tag = args.output_tag
    if args.seed is not None:
        cfg.seed = None if args.seed < 0 else args.seed
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
        prompt_prefix=cfg.prompt_prefix,
        prompt_template_path=cfg.prompt_template_path,
    )


@torch.inference_mode()
def generate_interpretation(prompt: str, tokenizer, model, inf_cfg: InferenceConfig, cfg: EvalConfig) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(inf_cfg.device)
    input_len = inputs["input_ids"].shape[1]

    kwargs: dict[str, Any] = {
        **inputs,
        "max_new_tokens": inf_cfg.max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": cfg.do_sample,
    }
    if cfg.do_sample:
        kwargs["temperature"] = cfg.temperature
        if cfg.top_k is not None:
            kwargs["top_k"] = cfg.top_k
        if cfg.top_p is not None:
            kwargs["top_p"] = cfg.top_p

    output_ids = model.generate(**kwargs)
    text = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
    if cfg.stop_at_control_token:
        text = text.split("<ctrl", 1)[0].rstrip()
    return text


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


def _load_prompt_template(path: str | Path) -> str:
    return Path(path).expanduser().read_text(encoding="utf-8").strip()


def _sample_value(sample: dict[str, Any], column: str | None) -> Any:
    if column and column in sample:
        return sample[column]
    return None


def _top_gene_sentence(sample: dict[str, Any], cfg: EvalConfig) -> str:
    genes = str(sample[cfg.cell_sentence_column]).strip().split()
    return " ".join(genes[: cfg.num_genes])


def _cell_block(samples: list[dict[str, Any]], cfg: EvalConfig) -> str:
    blocks: list[str] = []
    for offset, sample in enumerate(samples, start=1):
        cell_sentence = _top_gene_sentence(sample, cfg)
        blocks.append(f"Cell {offset}:\n{cell_sentence}.")
    return "\n".join(blocks)


def _group_organism(samples: list[dict[str, Any]], cfg: EvalConfig) -> str:
    if not cfg.organism_column:
        return cfg.default_organism

    organisms = {
        str(sample[cfg.organism_column]).strip()
        for sample in samples
        if sample.get(cfg.organism_column)
    }
    if len(organisms) == 1:
        return next(iter(organisms))
    if organisms:
        return " / ".join(sorted(organisms))
    return cfg.default_organism


def build_prompt(samples: list[dict[str, Any]], cfg: EvalConfig) -> str:
    cells = _cell_block(samples, cfg)
    if not cfg.prompt_prefix:
        return cells

    template = _load_prompt_template(cfg.prompt_template_path)
    num_cells = len(samples)
    cell_noun = "cell" if num_cells == 1 else "cells"
    values = {
        "organism": _group_organism(samples, cfg),
        "num_cells": num_cells,
        "num_genes": cfg.num_genes,
        "cell_word": cell_noun,
        "cell_noun": cell_noun,
        "cell_reference": "this cell" if num_cells == 1 else "these cells",
        "cell": cells,
        "cells": cells,
    }
    return template.format(**values)


def _available_groups(total_cells: int, cfg: EvalConfig) -> int:
    if cfg.drop_last:
        groups = total_cells // cfg.cells_per_prompt
    else:
        groups = (total_cells + cfg.cells_per_prompt - 1) // cfg.cells_per_prompt
    return min(cfg.max_examples, groups) if cfg.max_examples else groups


def cell_order(total_cells: int, cfg: EvalConfig) -> list[int]:
    """Return the dataset row order used to fill prompt groups.

    With ``shuffle_cells`` the split is permuted by a dedicated ``random.Random``
    seeded from ``cfg.seed``. A dedicated instance keeps cell selection
    independent of the generation RNG, so changing sampling parameters does not
    change which cells are drawn. The permutation covers the whole split before
    ``max_examples`` truncates it, so a shorter run is a strict prefix of a
    longer one with the same seed.
    """
    order = list(range(total_cells))
    if not cfg.shuffle_cells:
        return order
    if cfg.seed is None:
        print(
            "[Warning] shuffle_cells is enabled but no seed is set; the cells "
            "selected for this run cannot be reproduced."
        )
    random.Random(cfg.seed).shuffle(order)
    return order


def _group_indices(group_index: int, order: list[int], cfg: EvalConfig) -> list[int]:
    start = group_index * cfg.cells_per_prompt
    end = min(start + cfg.cells_per_prompt, len(order))
    return order[start:end]


def _numbered_run_dirs(results_dir: Path) -> list[tuple[int, Path]]:
    """Return the numbered experiment directories under ``results_dir``."""

    if not results_dir.is_dir():
        return []
    runs = [
        (int(child.name), child)
        for child in results_dir.iterdir()
        if child.is_dir() and child.name.isdigit()
    ]
    return sorted(runs)


def _allocate_run_dir(results_dir: Path) -> Path:
    """Atomically reserve the next numbered directory for a new experiment."""

    results_dir.mkdir(parents=True, exist_ok=True)
    existing = _numbered_run_dirs(results_dir)
    run_id = max((run_id for run_id, _ in existing), default=0) + 1
    while True:
        run_dir = results_dir / str(run_id)
        try:
            run_dir.mkdir()
        except FileExistsError:
            # Another process may have allocated this number after the scan.
            run_id += 1
            continue
        return run_dir


def _output_paths(cfg: EvalConfig, inf_cfg: InferenceConfig) -> tuple[Path, Path]:
    results_dir = (
        cfg.output_dir
        / cfg.c2s_dataset_dir.name
        / inf_cfg.model_short_name
    )
    run_dir = _allocate_run_dir(results_dir)
    return run_dir / "predictions.jsonl", run_dir / "summary.json"


def _load_run_summary(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        return {}
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    return summary if isinstance(summary, dict) else {}


def update_run_index(results_dir: Path) -> Path:
    """Rebuild ``runs.json`` from all completed numbered experiments."""

    entries: list[dict[str, Any]] = []
    for run_id, run_dir in _numbered_run_dirs(results_dir):
        summary = _load_run_summary(run_dir)
        if not summary:
            continue
        entries.append(
            {
                "run": run_id,
                "dir": run_dir.name,
                "created_at_utc": summary.get("created_at_utc"),
                "mode": summary.get("mode"),
                "split": summary.get("split"),
                "output_tag": summary.get("output_tag"),
                "num_prompt_examples": summary.get("num_prompt_examples"),
                "num_cells_evaluated": summary.get("num_cells_evaluated"),
                "cells_per_prompt": summary.get("cells_per_prompt"),
                "num_genes": summary.get("num_genes"),
            }
        )

    index_path = results_dir / RUN_INDEX_FILENAME
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({"results_dir": str(results_dir), "runs": entries}, f, indent=2)
        f.write("\n")
    return index_path


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
    from src.evaluate.model_loading import load_checkpoint_metadata

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


def _cell_sentence_char_spans(prompt: str, cell_sentences: list[str]) -> list[int]:
    spans: list[int] = []
    cursor = 0
    for cell_sentence in cell_sentences:
        char_start = prompt.find(cell_sentence, cursor)
        if char_start == -1:
            char_start = prompt.find(cell_sentence)
        if char_start == -1:
            raise ValueError("cell_sentence not found in prompt.")
        spans.append(char_start)
        cursor = char_start + len(cell_sentence)
    return spans


def _gene_token_ranges_for_cell(
    offsets: list[tuple[int, int]],
    cell_sentence: str,
    sentence_char_start: int,
) -> list[tuple[str, int, int]]:
    gene_ranges: list[tuple[str, int, int]] = []
    cursor = 0
    for gene in cell_sentence.split():
        local_start = cell_sentence.find(gene, cursor)
        if local_start == -1:
            raise ValueError(f"Could not locate gene '{gene}' while parsing cell_sentence.")
        local_end = local_start + len(gene)
        cursor = local_end

        abs_start = sentence_char_start + local_start
        abs_end = sentence_char_start + local_end
        idxs = [
            idx
            for idx, (token_start, token_end) in enumerate(offsets)
            if token_start < abs_end and token_end > abs_start
        ]
        if not idxs:
            raise ValueError(f"No tokenizer tokens found for gene '{gene}'.")
        gene_ranges.append((gene, idxs[0], idxs[-1] + 1))
    return gene_ranges


def _prepare_hooks_for_prompt(
    hooks: list[Any],
    tokenizer,
    prompt: str,
    cell_sentences: list[str],
) -> None:
    tokenized = tokenizer(prompt, return_offsets_mapping=True)
    prompt_len = len(tokenized["input_ids"])
    offsets = tokenized["offset_mapping"]
    spans = _cell_sentence_char_spans(prompt, cell_sentences)

    indices_by_hook: dict[int, set[int]] = {idx: set() for idx in range(len(hooks))}
    for cell_sentence, char_start in zip(cell_sentences, spans):
        gene_ranges = _gene_token_ranges_for_cell(offsets, cell_sentence, char_start)
        for idx, hook in enumerate(hooks):
            indices_by_hook[idx].update(
                _gene_token_indices_from_ranges(gene_ranges, hook.pooling_method)
            )

    for idx, hook in enumerate(hooks):
        hook.set_active_token_indices(
            sorted(indices_by_hook[idx]),
            prompt_len=prompt_len,
        )


def _clear_hook_prompt_state(hooks: list[Any]) -> None:
    for hook in hooks:
        hook.clear_active_token_indices()


def load_reconstruction_hooks(
    model,
    cfg: EvalConfig,
    inf_cfg: InferenceConfig,
) -> tuple[list[Any], list[dict[str, Any]]]:
    from src.evaluate.model_loading import load_sae

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
        checkpoint_prompt_prefix = checkpoint_model.get("prompt_prefix")
        if (
            isinstance(checkpoint_prompt_prefix, bool)
            and checkpoint_prompt_prefix != cfg.prompt_prefix
        ):
            print(
                "[Warning] SAE checkpoint metadata says prompt_prefix="
                f"{checkpoint_prompt_prefix}, but evaluation prompt_prefix="
                f"{cfg.prompt_prefix}."
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
                "checkpoint_prompt_prefix": checkpoint_prompt_prefix,
            }
        )
        metadata.append(meta)

    return hooks, metadata


def _sample_metadata(
    sample: dict[str, Any],
    index: int,
    cfg: EvalConfig,
) -> dict[str, Any]:
    row = {"index": index}
    available_gene_count = len(str(sample[cfg.cell_sentence_column]).strip().split())
    cell_name = _sample_value(sample, cfg.cell_name_column)
    cell_type = _sample_value(sample, cfg.cell_type_column)
    organism = _sample_value(sample, cfg.organism_column)
    row["num_genes_available"] = available_gene_count
    row["num_genes_in_prompt"] = min(cfg.num_genes, available_gene_count)
    if cell_name is not None:
        row["cell_name"] = cell_name
    if cell_type is not None:
        row["cell_type"] = cell_type
    if organism is not None:
        row["organism"] = organism
    return row


def _generated_word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def run_eval(cfg: EvalConfig) -> dict[str, Any]:
    validate_inputs(cfg)
    split_dir = _split_dir(cfg)
    dataset = load_from_disk(str(split_dir))
    required = {cfg.cell_sentence_column}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"{split_dir} is missing required column(s): {sorted(missing)}")

    total_available = len(dataset)
    n_examples = _available_groups(total_available, cfg)
    order = cell_order(total_available, cfg)
    inf_cfg = make_inference_config(cfg)
    predictions_path, summary_path = _output_paths(cfg, inf_cfg)
    run_dir = summary_path.parent
    results_dir = run_dir.parent
    run_id = int(run_dir.name)
    metadata = (
        _reconstruction_model_metadata_card(cfg, inf_cfg)
        if cfg.mode == "reconstruct"
        else base_model_metadata_card(cfg, inf_cfg)
    )
    metadata.update({"run": run_id, "split": f"data_{cfg.split}"})
    model_metadata_path = write_model_metadata_card(run_dir, metadata)

    print("[Config]")
    print(f"  mode      : {cfg.mode}")
    if cfg.reconstruct_model_name:
        print(f"  reconstruct_model: {cfg.reconstruct_model_name}")
    print(f"  model     : {cfg.model_id}")
    print(f"  device    : {cfg.device}")
    print(f"  dtype     : {cfg.dtype}")
    print(f"  dataset   : {cfg.c2s_dataset_dir}")
    print(f"  split     : data_{cfg.split} ({total_available:,} cells available)")
    print(f"  examples  : {n_examples:,} prompt groups")
    print(f"  cells/prompt: {cfg.cells_per_prompt}")
    print(f"  genes/cell: {cfg.num_genes}")
    print(f"  sampling  : {cfg.do_sample}")
    if cfg.do_sample:
        print(f"  top_k/top_p: {cfg.top_k}/{cfg.top_p}")
    print(f"  seed      : {cfg.seed if cfg.seed is not None else 'disabled (not reproducible)'}")
    print(f"  cell order: {'seeded shuffle' if cfg.shuffle_cells else 'dataset order'}")
    print(f"  prompt    : {'template' if cfg.prompt_prefix else 'bare cell block'}")
    if cfg.prompt_prefix:
        print(f"  prompt tpl: {cfg.prompt_template_path}")
    print(f"  run       : {run_id}")
    print(f"  output    : {run_dir}")
    if cfg.mode == "reconstruct":
        print("  interventions:")
        for intervention in cfg.interventions:
            print(
                f"    - layer {intervention.layer_idx}: "
                f"{intervention.checkpoint_path} "
                f"(sae_type={intervention.sae_type or 'auto'}, device={intervention.sae_device})"
            )

    tokenizer, model = load_model(inf_cfg)
    hooks: list[Any] = []
    intervention_meta: list[dict[str, Any]] = []
    patcher_context = nullcontext(None)
    if cfg.mode == "reconstruct":
        hooks, intervention_meta = load_reconstruction_hooks(model, cfg, inf_cfg)
        patcher_context = _load_sae_module().SAEReconstructionPatcher(hooks)

    prediction_writer_context = (
        open(predictions_path, "w", encoding="utf-8")
        if cfg.save_predictions
        else nullcontext(None)
    )

    generated_word_counts: list[int] = []
    generated_char_counts: list[int] = []
    total_cells_evaluated = 0

    with patcher_context:
        with prediction_writer_context as writer:
            for group_idx in tqdm(range(n_examples), desc="Generating"):
                indices = _group_indices(group_idx, order, cfg)
                samples = [dataset[idx] for idx in indices]
                total_cells_evaluated += len(samples)
                prompt = build_prompt(samples, cfg)
                cell_sentences = [
                    _top_gene_sentence(sample, cfg)
                    for sample in samples
                ]

                if hooks:
                    _prepare_hooks_for_prompt(hooks, tokenizer, prompt, cell_sentences)
                group_seed = _example_seed(cfg.seed, group_idx)
                if group_seed is not None:
                    seed_everything(group_seed)
                try:
                    generated_text = generate_interpretation(
                        prompt,
                        tokenizer,
                        model,
                        inf_cfg,
                        cfg,
                    )
                finally:
                    if hooks:
                        _clear_hook_prompt_state(hooks)

                generated_word_counts.append(_generated_word_count(generated_text))
                generated_char_counts.append(len(generated_text))

                if writer is not None:
                    row = {
                        "example_index": group_idx,
                        "cell_indices": indices,
                        "cell_types": [
                            _sample_value(sample, cfg.cell_type_column)
                            for sample in samples
                        ],
                        "num_cells": len(samples),
                        "cells": [
                            _sample_metadata(sample, idx, cfg)
                            for idx, sample in zip(indices, samples)
                        ],
                        "generated_text": generated_text,
                        "seed": group_seed,
                    }
                    if cfg.include_prompt:
                        row["prompt"] = prompt
                    writer.write(json.dumps(row) + "\n")

    for meta, hook in zip(intervention_meta, hooks):
        meta["hook_calls"] = hook.num_calls
        meta["hook_tokens"] = hook.num_tokens

    avg_words = (
        sum(generated_word_counts) / len(generated_word_counts)
        if generated_word_counts
        else 0.0
    )
    avg_chars = (
        sum(generated_char_counts) / len(generated_char_counts)
        if generated_char_counts
        else 0.0
    )
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run": run_id,
        "run_dir": str(run_dir),
        "results_dir": str(results_dir),
        "task": TASK_NAME,
        "mode": cfg.mode,
        "reconstruct_model_name": cfg.reconstruct_model_name,
        "output_tag": cfg.output_tag,
        "model_id": cfg.model_id,
        "dataset_dir": str(cfg.c2s_dataset_dir),
        "split": f"data_{cfg.split}",
        "num_available_cells": total_available,
        "num_prompt_examples": n_examples,
        "num_cells_evaluated": total_cells_evaluated,
        "cells_per_prompt": cfg.cells_per_prompt,
        "num_genes": cfg.num_genes,
        "drop_last": cfg.drop_last,
        "shuffle_cells": cfg.shuffle_cells,
        "cell_sentence_column": cfg.cell_sentence_column,
        "cell_name_column": cfg.cell_name_column,
        "cell_type_column": cfg.cell_type_column,
        "organism_column": cfg.organism_column,
        "default_organism": cfg.default_organism,
        "prompt_prefix": cfg.prompt_prefix,
        "prompt_template_path": str(cfg.prompt_template_path)
        if cfg.prompt_prefix
        else None,
        "max_new_tokens": cfg.max_new_tokens,
        "temperature": cfg.temperature,
        "do_sample": cfg.do_sample,
        "top_k": cfg.top_k,
        "top_p": cfg.top_p,
        "stop_at_control_token": cfg.stop_at_control_token,
        "seed": cfg.seed,
        "avg_generated_words": avg_words,
        "avg_generated_chars": avg_chars,
        "model_metadata_path": str(model_metadata_path),
        "predictions_path": str(predictions_path) if cfg.save_predictions else None,
        "interventions": intervention_meta,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    runs_index_path = update_run_index(results_dir)

    print("\n[Results]")
    print(f"  generated groups: {n_examples:,}")
    print(f"  cells evaluated : {total_cells_evaluated:,}")
    print(f"  avg words       : {avg_words:.1f}")
    if cfg.save_predictions:
        print(f"  generated       : {predictions_path}")
    print(f"  summary         : {summary_path}")
    print(f"  metadata        : {model_metadata_path}")
    print(f"  run index       : {runs_index_path}")
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
