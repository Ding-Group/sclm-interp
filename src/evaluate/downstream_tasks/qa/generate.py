#!/usr/bin/env python3
"""Generate cell-type multiple-choice QA datasets from prepared C2S splits."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import sys
from typing import Any

import yaml
from datasets import Dataset, load_from_disk

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_config import path_has_arrow_dataset

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "qa_generation.yaml"
DEFAULT_PROMPT_TEMPLATE = (
    PROJECT_ROOT / "src" / "prompts" / "cell_type_multiple_choice_template.txt"
)
TASK_NAME = "cell_type_multiple_choice"
_OUTPUT_SAFE_RE = re.compile(r"[^A-Za-z0-9_.=-]+")


@dataclass
class QAGenerationConfig:
    seed: int
    c2s_dataset_dir: Path
    splits: list[str]
    candidate_pool_splits: list[str]
    cell_sentence_column: str
    cell_type_column: str
    cell_name_column: str | None
    max_examples_per_split: int | None
    prompt_template_path: Path
    organism: str
    num_choices: int | None
    num_genes: int
    gene_separator: str
    shuffle_choices: bool
    choice_labels: list[str]
    output_dir: Path
    output_name: str | None
    save_jsonl: bool
    overwrite: bool

    @property
    def resolved_output_name(self) -> str:
        if self.output_name:
            return _safe_output_name(self.output_name, "qa")
        return _safe_output_name(
            f"{self.c2s_dataset_dir.name}-cell-type-mcq",
            "cell-type-mcq",
        )

    @property
    def output_path(self) -> Path:
        return self.output_dir / self.resolved_output_name


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _optional_project_path(value: str | Path | None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return _project_path(value)


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected {context} to be a mapping.")
    return value


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


def _normalize_split(value: Any) -> str:
    split = str(value).strip().removeprefix("data_")
    if not split:
        raise ValueError("Split names must be non-empty.")
    return split


def _normalize_splits(values: Any, *, fallback: list[str]) -> list[str]:
    if values is None:
        values = fallback
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list) or not values:
        raise ValueError("Expected a non-empty split list.")

    splits: list[str] = []
    seen: set[str] = set()
    for value in values:
        split = _normalize_split(value)
        if split not in seen:
            splits.append(split)
            seen.add(split)
    return splits


def _optional_positive_int(value: Any, context: str) -> int | None:
    if value is None:
        return None
    result = int(value)
    if result <= 0:
        raise ValueError(f"{context} must be positive or null.")
    return result


def load_config(path: str | Path | None = None) -> QAGenerationConfig:
    raw = _load_yaml(path or DEFAULT_CONFIG)
    data = _mapping(raw.get("data"), "data")
    question = _mapping(raw.get("question"), "question")
    output = _mapping(raw.get("output"), "output")

    dataset_dir_value = data.get("c2s_dataset_dir")
    if not dataset_dir_value:
        raise ValueError("Config data.c2s_dataset_dir is required.")

    splits = _normalize_splits(data.get("splits"), fallback=["train", "val", "test"])
    candidate_pool_splits = _normalize_splits(
        data.get("candidate_pool_splits"),
        fallback=splits,
    )

    cell_name_column = data.get("cell_name_column", "cell_name")
    if cell_name_column == "":
        cell_name_column = None

    num_choices = question.get("num_choices", 5)
    if num_choices is not None:
        num_choices = int(num_choices)
        if num_choices <= 1:
            raise ValueError("question.num_choices must be greater than 1 or null.")

    choice_labels = question.get("choice_labels") or list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    if not isinstance(choice_labels, list) or not choice_labels:
        raise ValueError("question.choice_labels must be a non-empty list.")
    choice_labels = [str(label).strip() for label in choice_labels]
    if any(not label for label in choice_labels):
        raise ValueError("question.choice_labels cannot contain empty labels.")

    return QAGenerationConfig(
        seed=int(raw.get("seed", 1234)),
        c2s_dataset_dir=_project_path(dataset_dir_value),
        splits=splits,
        candidate_pool_splits=candidate_pool_splits,
        cell_sentence_column=data.get("cell_sentence_column", "cell_sentence"),
        cell_type_column=data.get("cell_type_column", "cell_type"),
        cell_name_column=cell_name_column,
        max_examples_per_split=_optional_positive_int(
            data.get("max_examples_per_split"),
            "data.max_examples_per_split",
        ),
        prompt_template_path=_optional_project_path(
            question.get("prompt_template_path")
        )
        or DEFAULT_PROMPT_TEMPLATE,
        organism=str(question.get("organism", "Homo sapiens")),
        num_choices=num_choices,
        num_genes=_optional_positive_int(question.get("num_genes", 200), "question.num_genes")
        or 200,
        gene_separator=str(question.get("gene_separator", ", ")),
        shuffle_choices=bool(question.get("shuffle_choices", True)),
        choice_labels=choice_labels,
        output_dir=_project_path(output.get("output_dir", "datasets/qas")),
        output_name=output.get("output_name"),
        save_jsonl=bool(output.get("save_jsonl", True)),
        overwrite=bool(output.get("overwrite", False)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate cell-type multiple-choice QA datasets from C2S splits."
    )
    parser.add_argument("--config", default=None, help=f"YAML config path (default: {DEFAULT_CONFIG}).")
    parser.add_argument("--dataset", default=None, help="Override data.c2s_dataset_dir.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Override data.splits, e.g. --splits train val test.",
    )
    parser.add_argument(
        "--candidate-pool-splits",
        nargs="+",
        default=None,
        help="Override data.candidate_pool_splits.",
    )
    parser.add_argument("--max-examples", type=int, default=None, help="Override max examples per split.")
    parser.add_argument("--num-genes", type=int, default=None, help="Override question.num_genes.")
    parser.add_argument(
        "--num-choices",
        type=int,
        default=None,
        help="Override question.num_choices. Use 0 to include all candidate cell types.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override the generation seed.")
    parser.add_argument("--prompt-template", default=None, help="Override question.prompt_template_path.")
    parser.add_argument("--organism", default=None, help="Override question.organism.")
    parser.add_argument("--output-dir", default=None, help="Override output.output_dir.")
    parser.add_argument("--output-name", default=None, help="Override output.output_name.")
    parser.add_argument("--no-jsonl", action="store_true", help="Do not write JSONL mirrors.")
    parser.add_argument("--no-overwrite", action="store_true", help="Fail if the output dataset already exists.")
    return parser.parse_args()


def apply_cli_overrides(cfg: QAGenerationConfig, args: argparse.Namespace) -> None:
    if args.dataset is not None:
        cfg.c2s_dataset_dir = _project_path(args.dataset)
    if args.splits is not None:
        cfg.splits = _normalize_splits(args.splits, fallback=cfg.splits)
    if args.candidate_pool_splits is not None:
        cfg.candidate_pool_splits = _normalize_splits(
            args.candidate_pool_splits,
            fallback=cfg.candidate_pool_splits,
        )
    if args.max_examples is not None:
        cfg.max_examples_per_split = _optional_positive_int(
            args.max_examples,
            "--max-examples",
        )
    if args.num_genes is not None:
        cfg.num_genes = _optional_positive_int(args.num_genes, "--num-genes") or cfg.num_genes
    if args.num_choices is not None:
        cfg.num_choices = None if args.num_choices == 0 else args.num_choices
        if cfg.num_choices is not None and cfg.num_choices <= 1:
            raise ValueError("--num-choices must be greater than 1, or 0 for all choices.")
    if args.seed is not None:
        cfg.seed = args.seed
    if args.prompt_template is not None:
        cfg.prompt_template_path = _project_path(args.prompt_template)
    if args.organism is not None:
        cfg.organism = args.organism
    if args.output_dir is not None:
        cfg.output_dir = _project_path(args.output_dir)
    if args.output_name is not None:
        cfg.output_name = args.output_name
    if args.no_jsonl:
        cfg.save_jsonl = False
    if args.no_overwrite:
        cfg.overwrite = False


def _split_dir(cfg: QAGenerationConfig, split: str) -> Path:
    return cfg.c2s_dataset_dir / f"data_{split}"


def validate_inputs(cfg: QAGenerationConfig) -> None:
    if not cfg.c2s_dataset_dir.exists():
        raise FileNotFoundError(f"C2S dataset folder not found: {cfg.c2s_dataset_dir}")
    for split in sorted(set(cfg.splits + cfg.candidate_pool_splits)):
        split_dir = _split_dir(cfg, split)
        if not path_has_arrow_dataset(split_dir):
            raise FileNotFoundError(
                f"Split folder does not look like a HuggingFace dataset: {split_dir}"
            )
    if not cfg.prompt_template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {cfg.prompt_template_path}")
    if cfg.num_choices is not None and len(cfg.choice_labels) < cfg.num_choices:
        raise ValueError(
            "question.choice_labels must contain at least question.num_choices labels."
        )


def _load_prompt_template(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _load_source_splits(
    cfg: QAGenerationConfig,
) -> dict[str, Dataset]:
    splits: dict[str, Dataset] = {}
    for split in sorted(set(cfg.splits + cfg.candidate_pool_splits)):
        splits[split] = load_from_disk(str(_split_dir(cfg, split)))
    return splits


def _require_columns(
    dataset: Dataset,
    split: str,
    cfg: QAGenerationConfig,
) -> None:
    required = {cfg.cell_sentence_column, cfg.cell_type_column}
    if cfg.cell_name_column:
        required.add(cfg.cell_name_column)
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(
            f"{_split_dir(cfg, split)} is missing required column(s): {sorted(missing)}"
        )


def _candidate_pool(
    source_splits: dict[str, Dataset],
    cfg: QAGenerationConfig,
) -> list[str]:
    candidates: set[str] = set()
    for split in cfg.candidate_pool_splits:
        dataset = source_splits[split]
        if cfg.cell_type_column not in dataset.column_names:
            raise ValueError(
                f"{_split_dir(cfg, split)} is missing {cfg.cell_type_column!r}."
            )
        candidates.update(
            str(label).strip()
            for label in dataset[cfg.cell_type_column]
            if str(label).strip()
        )
    if not candidates:
        raise ValueError("No candidate cell types found.")
    return sorted(candidates, key=str.lower)


def _example_rng(seed: int, split: str, index: int) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{split}:{index}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _top_genes(sample: dict[str, Any], cfg: QAGenerationConfig) -> list[str]:
    raw_sentence = sample.get(cfg.cell_sentence_column)
    genes = str(raw_sentence or "").strip().split()
    if not genes:
        raise ValueError(f"Encountered empty {cfg.cell_sentence_column!r}.")
    return genes[: cfg.num_genes]


def _choices_for_example(
    gold_cell_type: str,
    candidate_pool: list[str],
    rng: random.Random,
    cfg: QAGenerationConfig,
) -> list[str]:
    distractors = [label for label in candidate_pool if label != gold_cell_type]
    if cfg.num_choices is None:
        choices = [gold_cell_type, *distractors]
    else:
        n_distractors = min(cfg.num_choices - 1, len(distractors))
        choices = [gold_cell_type, *rng.sample(distractors, n_distractors)]

    if cfg.shuffle_choices:
        rng.shuffle(choices)
    return choices


def _format_choices(choices: list[str], cfg: QAGenerationConfig) -> str:
    if len(cfg.choice_labels) < len(choices):
        raise ValueError(
            f"Need {len(choices)} choice labels, got {len(cfg.choice_labels)}."
        )
    return "\n".join(
        f"{label}. {choice}"
        for label, choice in zip(cfg.choice_labels, choices, strict=False)
    )


def _build_record(
    sample: dict[str, Any],
    *,
    split: str,
    source_index: int,
    candidate_pool: list[str],
    prompt_template: str,
    cfg: QAGenerationConfig,
) -> dict[str, Any]:
    gold_cell_type = str(sample.get(cfg.cell_type_column, "")).strip()
    if not gold_cell_type:
        raise ValueError(f"Encountered empty {cfg.cell_type_column!r}.")

    genes = _top_genes(sample, cfg)
    rng = _example_rng(cfg.seed, split, source_index)
    choices = _choices_for_example(gold_cell_type, candidate_pool, rng, cfg)
    answer_index = choices.index(gold_cell_type)
    answer_choice = cfg.choice_labels[answer_index]
    gene_names = cfg.gene_separator.join(genes)
    cell_types = _format_choices(choices, cfg)
    prompt = prompt_template.format(
        gene_names=gene_names,
        cell_sentence=" ".join(genes),
        cell_types=cell_types,
        choices=cell_types,
        organism=cfg.organism,
        num_genes=len(genes),
        num_choices=len(choices),
    )

    cell_name = None
    if cfg.cell_name_column:
        value = sample.get(cfg.cell_name_column)
        cell_name = None if value is None else str(value)

    return {
        "id": f"{cfg.c2s_dataset_dir.name}:data_{split}:{source_index}",
        "task": TASK_NAME,
        "source_dataset": cfg.c2s_dataset_dir.name,
        "source_split": f"data_{split}",
        "source_index": source_index,
        "cell_name": cell_name,
        "prompt": prompt,
        "answer": f"Final Answer: {gold_cell_type}",
        "gold_cell_type": gold_cell_type,
        "answer_choice": answer_choice,
        "choices": choices,
        "choice_labels": cfg.choice_labels[: len(choices)],
        "gene_names": genes,
        "gene_sentence": " ".join(genes),
        "num_genes": len(genes),
        "organism": cfg.organism,
    }


def _records_for_split(
    dataset: Dataset,
    *,
    split: str,
    candidate_pool: list[str],
    prompt_template: str,
    cfg: QAGenerationConfig,
) -> list[dict[str, Any]]:
    total = len(dataset)
    n_examples = (
        min(cfg.max_examples_per_split, total)
        if cfg.max_examples_per_split is not None
        else total
    )
    return [
        _build_record(
            dataset[source_index],
            split=split,
            source_index=source_index,
            candidate_pool=candidate_pool,
            prompt_template=prompt_template,
            cfg=cfg,
        )
        for source_index in range(n_examples)
    ]


def _prepare_output_dir(cfg: QAGenerationConfig) -> Path:
    output_path = cfg.output_path
    if output_path.exists():
        if not cfg.overwrite:
            raise FileExistsError(
                f"Output dataset already exists: {output_path}. "
                "Set output.overwrite=true or choose a new output.output_name."
            )
        if output_path.resolve() in {PROJECT_ROOT.resolve(), cfg.output_dir.resolve()}:
            raise ValueError(f"Refusing to overwrite unsafe output path: {output_path}")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _write_metadata(
    output_path: Path,
    *,
    counts: dict[str, int],
    candidate_pool: list[str],
    cfg: QAGenerationConfig,
) -> Path:
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": TASK_NAME,
        "source_dataset_dir": str(cfg.c2s_dataset_dir),
        "output_dir": str(output_path),
        "splits": [f"data_{split}" for split in cfg.splits],
        "counts": {f"data_{split}": count for split, count in counts.items()},
        "candidate_pool_splits": [
            f"data_{split}" for split in cfg.candidate_pool_splits
        ],
        "candidate_cell_types": candidate_pool,
        "num_candidate_cell_types": len(candidate_pool),
        "prompt_template_path": str(cfg.prompt_template_path),
        "organism": cfg.organism,
        "num_genes": cfg.num_genes,
        "num_choices": cfg.num_choices,
        "shuffle_choices": cfg.shuffle_choices,
        "seed": cfg.seed,
        "schema": {
            "prompt": "Question text shown to the model.",
            "answer": "Gold answer in the requested Final Answer format.",
            "gold_cell_type": "Unformatted gold cell-type label.",
            "choices": "Candidate cell types shown in the prompt.",
            "answer_choice": "Multiple-choice label corresponding to gold_cell_type.",
        },
    }
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    return metadata_path


def generate_qa_dataset(cfg: QAGenerationConfig) -> dict[str, Any]:
    validate_inputs(cfg)
    source_splits = _load_source_splits(cfg)
    for split in cfg.splits:
        _require_columns(source_splits[split], split, cfg)

    candidate_pool = _candidate_pool(source_splits, cfg)
    prompt_template = _load_prompt_template(cfg.prompt_template_path)
    output_path = _prepare_output_dir(cfg)

    counts: dict[str, int] = {}
    saved_paths: dict[str, str] = {}
    jsonl_paths: dict[str, str] = {}
    for split in cfg.splits:
        records = _records_for_split(
            source_splits[split],
            split=split,
            candidate_pool=candidate_pool,
            prompt_template=prompt_template,
            cfg=cfg,
        )
        split_path = output_path / f"data_{split}"
        Dataset.from_list(records).save_to_disk(str(split_path))
        counts[split] = len(records)
        saved_paths[f"data_{split}"] = str(split_path)

        if cfg.save_jsonl:
            jsonl_path = output_path / f"data_{split}.jsonl"
            _write_jsonl(jsonl_path, records)
            jsonl_paths[f"data_{split}"] = str(jsonl_path)

    metadata_path = _write_metadata(
        output_path,
        counts=counts,
        candidate_pool=candidate_pool,
        cfg=cfg,
    )
    return {
        "output_path": str(output_path),
        "metadata_path": str(metadata_path),
        "saved_paths": saved_paths,
        "jsonl_paths": jsonl_paths,
        "counts": {f"data_{split}": count for split, count in counts.items()},
        "candidate_cell_types": candidate_pool,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    apply_cli_overrides(cfg, args)

    print("[Config]")
    print(f"  source    : {cfg.c2s_dataset_dir}")
    print(f"  splits    : {', '.join('data_' + split for split in cfg.splits)}")
    print(f"  output    : {cfg.output_path}")
    print(f"  template  : {cfg.prompt_template_path}")
    print(f"  num genes : {cfg.num_genes}")
    print(f"  choices   : {'all' if cfg.num_choices is None else cfg.num_choices}")

    summary = generate_qa_dataset(cfg)

    print("\n[Output]")
    for split, count in summary["counts"].items():
        print(f"  {split:<10}: {count:,} examples")
    print(f"  dataset   : {summary['output_path']}")
    print(f"  metadata  : {summary['metadata_path']}")
    if summary["jsonl_paths"]:
        print("  jsonl     : written")


if __name__ == "__main__":
    main()
