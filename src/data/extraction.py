"""
extraction.py - unified C2S activation and vocabulary extraction

Extracts sharded cell-sentence or gene activations and vocabulary-level gene embeddings
from C2S Arrow datasets already prepared by data_prep.py.

Tasks (set via --mode or the top-level 'mode' key in extraction.yaml):
    activations — for every cell in the dataset, extract the configured activation
                representation and write sharded safetensors for SAE training.
    vocab       — load vocab.json and extract one last-token embedding per gene.
    (default)   — run both activations and vocab if mode is unset.

Activation representations (set via extraction.mode in extraction.yaml):
    gene        — extract gene activations using extraction.gene_pooling.
    cell        — extract only the final hidden state of each cell sentence.

Storage layout:
    activations:
    <activations_dir>/<model_name>/layer{L}/<dataset_name>/
        gene_activations_{pool}_{prefix}/<split_name>/
    <cells_dir>/<model_name>/layer{L}/<dataset_name>/
        cell_activations_last_{prefix}/<split_name>/
    ├── shard_00000.safetensors   # key 'activations', shape (rows, d_model)
    │                              # cell mode also stores aligned 'cell_type_ids'
    ├── cell_types.json           # cell mode: ID-to-cell-type lookup
    └── ...

    vocab:
    <vocabs_dir>/<model_name>/layer{L}/<dataset_name>/
    ├── vocab_last.safetensors    # key 'embeddings', shape (V, d_model)
    └── genes.json                # ordered gene list (gene index → gene name)

Usage:
    python -m src.data.extraction --model gemma                         # both modes
    python -m src.data.extraction --model pythia                        # both modes
    python -m src.data.extraction --mode activations --model gemma
    python -m src.data.extraction --mode activations --model pythia --n_cells 500
    python -m src.data.extraction --mode vocab       --model gemma
    python -m src.data.extraction --mode vocab       --model pythia
    python -m src.data.extraction --model gemma --dataset datasets/c2s_datasets/my_dataset --config configs/extraction.yaml
"""

import argparse
from dataclasses import dataclass
import json
import sys
from pathlib import Path

import torch
import yaml
from datasets import load_from_disk
from safetensors.torch import save_file
from tqdm.auto import tqdm

_MODULE_DIR = Path(__file__).parent
_PROJECT_ROOT = _MODULE_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.inference import (
    InferenceConfig,
    load_config,
    load_model,
    build_cell_type_prompt,
    generate,
    extract_cell_sentence_activations,
    _get_token_indices_for_substring,
    _get_gene_token_ranges,
    HiddenStateExtractor,
)
from src.data.dataset_config import path_has_arrow_dataset

DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "configs" / "extraction.yaml"
N_VALIDATE    = 3        # cells to validate token mapping on
ROWS_PER_SHARD = 100_000  # activation rows per safetensors shard
SPLIT_NAMES = ("train", "val", "test")
VOCAB_OUTPUT_FILES = ("vocab_last.safetensors", "genes.json")
ACTIVATION_EXTRACTION_MODES = {"gene", "cell"}


# ─────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class PreparedDataset:
    """A data_prep.py output directory with train/val/test Arrow splits."""

    name: str
    root_dir: Path
    split_dirs: dict[str, Path]
    vocab_path: Path


@dataclass(frozen=True)
class PreparedDatasetSplit:
    """One Arrow split inside a prepared C2S dataset directory."""

    dataset_name: str
    split_name: str
    dataset_dir: Path


@dataclass(frozen=True)
class PreparedExtractionRun:
    """Loaded splits and train-derived vocabulary ready for model extraction."""

    dataset: PreparedDataset
    split_datasets: dict[str, object]
    vocab: list[str]


def _load_raw_config(config_path: Path) -> dict:
    if not config_path.exists():
        print(f"[Error] Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else _PROJECT_ROOT / path


def _resolve_dataset_entry(entry) -> PreparedDataset:
    if isinstance(entry, dict):
        extra_keys = set(entry) - {"path"}
        if "path" not in entry or extra_keys:
            print("[Error] Dataset entries must contain a 'path'.", file=sys.stderr)
            if extra_keys:
                print(
                    f"[Error] Unsupported key(s): {', '.join(sorted(extra_keys))}",
                    file=sys.stderr,
                )
            sys.exit(1)
        root_dir = _project_path(entry["path"])
    else:
        root_dir = _project_path(str(entry))

    name = root_dir.name
    split_dirs = {
        split_name: root_dir / f"data_{split_name}"
        for split_name in SPLIT_NAMES
    }
    vocab_path = root_dir / "vocab.json"
    missing = [
        str(path.relative_to(root_dir) if path.is_relative_to(root_dir) else path)
        for path, ok in [
            *(
                (split_dir, path_has_arrow_dataset(split_dir))
                for split_dir in split_dirs.values()
            ),
            (vocab_path, vocab_path.exists()),
        ]
        if not ok
    ]
    if missing:
        print(
            f"[Error] {root_dir} does not look like a data_prep.py output folder. "
            f"Missing: {', '.join(missing)}",
            file=sys.stderr,
        )
        sys.exit(1)

    return PreparedDataset(
        name=name,
        root_dir=root_dir,
        split_dirs=split_dirs,
        vocab_path=vocab_path,
    )


def _resolve_datasets(raw: dict, cli_dataset: str | None) -> list[PreparedDataset]:
    dataset_entries = [cli_dataset] if cli_dataset else raw.get("datasets")
    if not dataset_entries:
        print(
            "[Error] No datasets specified. Add prepared dataset folder paths "
            "to extraction.yaml or pass --dataset.",
            file=sys.stderr,
        )
        sys.exit(1)
    if isinstance(dataset_entries, (str, dict)):
        dataset_entries = [dataset_entries]

    return [_resolve_dataset_entry(entry) for entry in dataset_entries]


def _load_vocab(vocab_path: Path) -> list[str]:
    with open(vocab_path) as f:
        vocab = json.load(f)
    if not isinstance(vocab, list) or not all(isinstance(gene, str) for gene in vocab):
        raise ValueError(f"Expected {vocab_path} to contain a JSON list of gene names.")

    print(f"Loaded vocab.json: {len(vocab):,} train-set genes")
    return vocab


def _validate_dataset_genes_in_vocab(
    split_datasets: dict[str, object], vocab: set[str], dataset_name: str
) -> None:
    """Assert every cell sentence only contains genes present in vocab.json."""

    total = sum(len(dataset) for dataset in split_datasets.values())
    with tqdm(total=total, desc=f"Checking {dataset_name} genes", leave=False) as bar:
        for split_name, dataset in split_datasets.items():
            for i in range(len(dataset)):
                unknown = set(dataset[i]["cell_sentence"].split()) - vocab
                if unknown:
                    examples = ", ".join(sorted(unknown)[:10])
                    raise ValueError(
                        f"{dataset_name} / data_{split_name} cell {i} contains "
                        f"gene(s) absent from vocab.json: {examples}"
                    )
                bar.update(1)

    print(f"{dataset_name}: all {total:,} cells across splits are covered by vocab.json")


# ─────────────────────────────────────────────
# activations mode
# ─────────────────────────────────────────────

def _validate_activations(dataset, n, cfg, tokenizer, model, extraction_mode: str):
    print(f"\n{'='*70}\nVALIDATION  ({n} sample(s))\n{'='*70}")
    any_fail = False

    for i in range(min(n, len(dataset))):
        sample   = dataset[i]
        cs       = sample["cell_sentence"]
        organism = sample.get("organism", "Homo sapiens")
        prompt   = (
            build_cell_type_prompt(
                cs,
                organism=organism,
                prompt_template_path=cfg.prompt_template_path,
            )
            if cfg.prompt_prefix
            else cs
        )

        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        print(f"\n── Cell {i}  [{sample.get('cell_name', '?')}]  "
            f"cell_type={sample.get('cell_type', '?')!r}")
        print(f"   Genes: {len(cs.split())}   Tokens: {len(input_ids)}")

        cs_start: int | None = None
        cs_end: int | None = None
        try:
            cs_start, cs_end = _get_token_indices_for_substring(tokenizer, prompt, cs)
            decoded_span = tokenizer.decode(input_ids[cs_start:cs_end]).strip()
            span_ok = (decoded_span == cs)
        except Exception as e:
            span_ok, decoded_span = False, f"ERROR: {e}"

        span_label = f"[{cs_start},{cs_end})" if cs_start is not None else "[?,?)"
        print(f"   Cell-sentence span {span_label}  →  {'PASS' if span_ok else 'FAIL'}")
        if not span_ok:
            any_fail = True
            print(f"     decoded : {decoded_span[:100]}")
            print(f"     expected: {cs[:100]}")

        if extraction_mode == "gene":
            try:
                gene_ranges = _get_gene_token_ranges(tokenizer, prompt, cs)
                gene_fails, shown = 0, 0
                for gene, g_start, g_end in gene_ranges:
                    decoded = tokenizer.decode(input_ids[g_start:g_end]).strip()
                    ok = (decoded == gene)
                    if not ok:
                        gene_fails += 1
                        any_fail = True
                    if shown < 5 or not ok:
                        print(f"   [{g_start:4d},{g_end:4d})  {gene:<14}  {'PASS' if ok else f'FAIL got={decoded!r}'}")
                        shown += 1
                remaining_ok = len(gene_ranges) - shown - gene_fails
                if remaining_ok > 0:
                    print(f"   ... {remaining_ok} more genes PASS")
                if gene_fails:
                    print(f"   !! {gene_fails} gene(s) FAILED token mapping")
            except Exception as e:
                print(f"   Gene range extraction ERROR: {e}")
                any_fail = True

        try:
            pred = generate(prompt, tokenizer, model, cfg)
            print(f"   Predicted cell type: {pred[:120]!r}")
        except Exception as e:
            print(f"   Generation ERROR: {e}")

    print()
    if any_fail:
        print("VALIDATION RESULT: FAIL — token mapping issues detected. Aborting.")
        print("="*70)
        sys.exit(1)
    print("VALIDATION RESULT: PASS")
    print("="*70)


def _flush_shard(
    buf: list[torch.Tensor],
    shard_idx: int,
    save_dir: Path,
    cell_type_ids: list[int] | None = None,
) -> None:
    if not buf:
        return
    tensor = torch.cat(buf, dim=0).contiguous()
    tensors = {"activations": tensor}
    if cell_type_ids is not None:
        if len(cell_type_ids) != tensor.shape[0]:
            raise ValueError(
                "cell_type_ids must contain one ID per activation row: "
                f"got {len(cell_type_ids)} IDs for {tensor.shape[0]} rows."
            )
        tensors["cell_type_ids"] = torch.tensor(cell_type_ids, dtype=torch.int64)

    out = save_dir / f"shard_{shard_idx:05d}.safetensors"
    save_file(tensors, str(out))
    print(f"  wrote {out.name}:  shape={tuple(tensor.shape)}  dtype={tensor.dtype}")


def _clear_existing_shards(save_dir: Path) -> None:
    stale_shards = sorted(save_dir.glob("shard_*.safetensors"))
    for shard_path in stale_shards:
        shard_path.unlink()
    if stale_shards:
        print(f"Removed {len(stale_shards):,} existing shard(s) from {save_dir}")


def _prepare_cell_type_labels(dataset, n: int) -> tuple[list[str], list[str]]:
    """Return per-cell labels and a stable ID-to-label vocabulary."""

    labels = []
    for i in range(n):
        cell_type = dataset[i].get("cell_type")
        if not isinstance(cell_type, str) or not cell_type.strip():
            raise ValueError(
                "Cell-mode extraction requires every row to contain a non-empty "
                f"'cell_type' string; row {i} has {cell_type!r}."
            )
        labels.append(cell_type.strip())
    return labels, sorted(set(labels))


def _write_cell_types(cell_types: list[str], save_dir: Path) -> None:
    with open(save_dir / "cell_types.json", "w", encoding="utf-8") as f:
        json.dump(cell_types, f, indent=2)


def _activation_output_dir(
    activations_dir: Path,
    cells_dir: Path,
    cfg: InferenceConfig,
    split: PreparedDatasetSplit,
    extraction_mode: str,
) -> Path:
    prefix_tag = "prefix" if cfg.prompt_prefix else "no_prefix"
    if extraction_mode == "gene":
        root_dir = activations_dir
        representation_tag = f"gene_activations_{cfg.gene_pooling}_{prefix_tag}"
    else:
        root_dir = cells_dir
        representation_tag = f"cell_activations_last_{prefix_tag}"
    return (
        root_dir
        / cfg.model_short_name
        / f"layer{cfg.layer_idx}"
        / split.dataset_name
        / representation_tag
        / split.split_name
    )


def _vocab_output_dir(vocabs_dir: Path, cfg: InferenceConfig, dataset: PreparedDataset) -> Path:
    return vocabs_dir / cfg.model_short_name / f"layer{cfg.layer_idx}" / dataset.name


def _vocab_outputs_exist(out_dir: Path) -> bool:
    return all((out_dir / filename).exists() for filename in VOCAB_OUTPUT_FILES)


@torch.inference_mode()
def _extract_cell_sentence_final_state(
    cell_sentence: str,
    prompt: str,
    tokenizer,
    model,
    cfg: InferenceConfig,
) -> torch.Tensor:
    """Return the hidden state at the final token of the cell sentence."""

    inputs = tokenizer(prompt, return_tensors="pt").to(cfg.device)
    seq_len = inputs["input_ids"].shape[1]
    if cfg.max_seq_len is not None and seq_len > cfg.max_seq_len:
        raise ValueError(
            f"Prompt has {seq_len} tokens, exceeds max_seq_len={cfg.max_seq_len}."
        )

    _, cs_end = _get_token_indices_for_substring(tokenizer, prompt, cell_sentence)
    with HiddenStateExtractor(
        model,
        cfg.layer_idx,
        token_range=(cs_end - 1, cs_end),
    ) as ext:
        _ = model(**inputs, use_cache=False)
        final_state = ext.get().squeeze(0)

    if final_state.ndim == 1:
        final_state = final_state.unsqueeze(0)
    return final_state


def _run_activations(
    split: PreparedDatasetSplit,
    split_dataset,
    tokenizer,
    model,
    cfg: InferenceConfig,
    extraction_mode: str,
    n_cells: int | None,
    skip_validation: bool,
    activations_dir: Path,
    cells_dir: Path,
):
    print(f"\n{'='*70}\nDataset: {split.dataset_name} / {split.split_name}\n{'='*70}")
    print(f"Path: {split.dataset_dir}")

    out_dir = _activation_output_dir(
        activations_dir,
        cells_dir,
        cfg,
        split,
        extraction_mode,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    _clear_existing_shards(out_dir)

    print(f"Using {len(split_dataset):,} cells")
    if len(split_dataset) == 0:
        print(f"No cells available for {split.split_name}; no shards written.")
        return

    if not skip_validation:
        _validate_activations(
            split_dataset,
            N_VALIDATE,
            cfg,
            tokenizer,
            model,
            extraction_mode,
        )
    else:
        print("\n[Validation skipped]")

    n = min(n_cells, len(split_dataset)) if n_cells is not None else len(split_dataset)
    print(f"\nExtracting {n:,} cells  →  {out_dir}")
    settings = f"  layer={cfg.layer_idx}  mode={extraction_mode}"
    if extraction_mode == "gene":
        settings += f"  pooling={cfg.gene_pooling}"
    print(f"{settings}  prefix={cfg.prompt_prefix}")

    cell_type_labels: list[str] | None = None
    cell_type_to_id: dict[str, int] | None = None
    if extraction_mode == "cell":
        cell_type_labels, cell_types = _prepare_cell_type_labels(split_dataset, n)
        cell_type_to_id = {
            cell_type: cell_type_id
            for cell_type_id, cell_type in enumerate(cell_types)
        }
        _write_cell_types(cell_types, out_dir)
        print(f"  cell types={len(cell_types):,}  lookup={out_dir / 'cell_types.json'}")

    buf: list[torch.Tensor] = []
    cell_type_id_buf: list[int] = []
    buf_rows = shard_idx = total_rows = skipped = 0

    for i in tqdm(range(n), desc="Harvesting"):
        sample   = split_dataset[i]
        cs       = sample["cell_sentence"]
        organism = sample.get("organism", "Homo sapiens")
        prompt   = (
            build_cell_type_prompt(
                cs,
                organism=organism,
                prompt_template_path=cfg.prompt_template_path,
            )
            if cfg.prompt_prefix
            else cs
        )

        try:
            if extraction_mode == "gene":
                acts = extract_cell_sentence_activations(
                    cell_sentence=cs,
                    prompt=prompt,
                    tokenizer=tokenizer,
                    model=model,
                    cfg=cfg,
                )
            else:
                acts = _extract_cell_sentence_final_state(
                    cell_sentence=cs,
                    prompt=prompt,
                    tokenizer=tokenizer,
                    model=model,
                    cfg=cfg,
                )
        except ValueError as e:
            tqdm.write(f"  [skip cell {i}] {e}")
            skipped += 1
            continue

        buf.append(acts)
        if extraction_mode == "cell":
            if acts.shape[0] != 1:
                raise ValueError(
                    "Cell-mode extraction must produce exactly one activation row "
                    f"per cell; cell {i} produced {acts.shape[0]}."
                )
            cell_type = cell_type_labels[i]
            cell_type_id_buf.append(cell_type_to_id[cell_type])
        buf_rows   += acts.shape[0]
        total_rows += acts.shape[0]

        if buf_rows >= ROWS_PER_SHARD:
            _flush_shard(
                buf,
                shard_idx,
                out_dir,
                cell_type_id_buf if extraction_mode == "cell" else None,
            )
            shard_idx += 1
            buf, cell_type_id_buf, buf_rows = [], [], 0

    _flush_shard(
        buf,
        shard_idx,
        out_dir,
        cell_type_id_buf if extraction_mode == "cell" else None,
    )
    total_shards = shard_idx + (1 if buf_rows > 0 else 0)

    if extraction_mode == "cell":
        row_label = "cell final-state activations"
    else:
        row_label = (
            "gene-token activations"
            if cfg.gene_pooling == "all"
            else "gene activations"
        )
    print(f"\nDone: {total_rows:,} {row_label} in {total_shards} shard(s), "
        f"{skipped} cell(s) skipped.")
    print(f"Saved to: {out_dir}")


# ─────────────────────────────────────────────
# vocab mode
# ─────────────────────────────────────────────

@torch.inference_mode()
def _extract_gene_vocab_embedding(
    gene: str, tokenizer, model, cfg: InferenceConfig,
) -> torch.Tensor:
    """
    Return the last token embedding for one gene as a bfloat16 CPU tensor.
    Single forward pass via HiddenStateExtractor.
    """
    prompt  = (
        build_cell_type_prompt(
            gene,
            prompt_template_path=cfg.prompt_template_path,
        )
        if cfg.prompt_prefix
        else gene
    )
    g_start, g_end = _get_token_indices_for_substring(tokenizer, prompt, gene)
    inputs  = tokenizer(prompt, return_tensors="pt").to(cfg.device)

    with HiddenStateExtractor(model, cfg.layer_idx, token_range=(g_start, g_end)) as ext:
        _ = model(**inputs, use_cache=False)
        token_acts = ext.get().squeeze(0)  # (n_tokens, d_model)

    if token_acts.ndim == 1:
        token_acts = token_acts.unsqueeze(0)

    return token_acts[-1]


def _run_vocab(dataset: PreparedDataset, vocab: list[str], tokenizer, model, cfg: InferenceConfig, vocabs_dir: Path):
    print(f"\n{'='*70}\nDataset: {dataset.name} / vocab\n{'='*70}")
    print(f"Source: {dataset.vocab_path}")

    V = len(vocab)
    print(f"Vocabulary size: {V:,} genes")
    out_dir = _vocab_output_dir(vocabs_dir, cfg, dataset)
    if _vocab_outputs_exist(out_dir):
        print(f"Skipping vocab extraction: outputs already exist for '{dataset.name}' at {out_dir}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    last_buf: dict[int, torch.Tensor] = {}
    failed: list[str] = []
    d_model: int | None = None

    for idx, gene in enumerate(tqdm(vocab, desc="Extracting vocab embeddings")):
        try:
            last_vec = _extract_gene_vocab_embedding(gene, tokenizer, model, cfg)
            last_buf[idx] = last_vec
            if d_model is None:
                d_model = last_vec.shape[0]
        except Exception as e:
            tqdm.write(f"  [skip] '{gene}': {e}")
            failed.append(gene)

    if d_model is None:
        raise RuntimeError("All genes failed — cannot determine d_model.")

    zero        = torch.zeros(d_model, dtype=torch.bfloat16)
    last_tensor = torch.stack([last_buf.get(i, zero) for i in range(V)], dim=0)

    save_file({"embeddings": last_tensor.contiguous()}, str(out_dir / "vocab_last.safetensors"))
    with open(out_dir / "genes.json", "w") as f:
        json.dump(vocab, f, indent=2)

    print(f"\nSaved to {out_dir}")
    print(f"  vocab_last.safetensors : {tuple(last_tensor.shape)}")
    print(f"  genes.json             : {V} genes")
    if failed:
        print(f"  {len(failed)} gene(s) failed (zero-padded): {failed[:10]}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="C2S activation and vocabulary extraction.")
    parser.add_argument("--mode", default=None, choices=["activations", "vocab"],
                        help="Override the top-level task mode. Omit to use the config, or run both if unset.")
    parser.add_argument("--model",  required=True, choices=["gemma", "pythia"],
                        help="Model family.")
    parser.add_argument("--config", default=None,
                        help=f"Path to extraction YAML (default: {DEFAULT_CONFIG_PATH}).")
    parser.add_argument("--dataset", default=None,
                        help="Override: path to a data_prep.py output folder with data_train/data_val/data_test.")
    parser.add_argument("--prompt-template", default=None,
                        help="Override inference.prompt_template_path.")
    # activations-only flags
    parser.add_argument("--n_cells", type=int, default=None,
                        help="[activations] Max cells to process (default: all).")
    parser.add_argument("--skip_validation", action="store_true",
                        help="[activations] Skip token-mapping validation.")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else DEFAULT_CONFIG_PATH
    raw = _load_raw_config(config_path)
    cfg = load_config(config_path, model_family=args.model)
    if args.prompt_template is not None:
        cfg.prompt_template_path = _project_path(args.prompt_template)

    # Resolve task mode: CLI > config > both
    task_mode = args.mode or raw.get("mode", None)
    if task_mode not in (None, "activations", "vocab"):
        print("[Error] mode must be 'activations', 'vocab', or unset.", file=sys.stderr)
        sys.exit(1)
    run_activations = task_mode in (None, "activations")
    run_vocab       = task_mode in (None, "vocab")

    extraction_mode = raw.get("extraction", {}).get("mode", "gene")
    if extraction_mode not in ACTIVATION_EXTRACTION_MODES:
        print(
            "[Error] extraction.mode must be 'gene' or 'cell'.",
            file=sys.stderr,
        )
        sys.exit(1)

    datasets = _resolve_datasets(raw, args.dataset)
    out      = raw.get("output", {})
    activations_dir = _project_path(out.get("activations_dir", "datasets/activations"))
    cells_dir       = _project_path(out.get("cells_dir",       "datasets/cells"))
    vocabs_dir      = _project_path(out.get("vocabs_dir",      "datasets/vocabs"))

    print(f"\n[Config]")
    print(f"  task mode : {task_mode or 'both'}")
    print(f"  model     : {cfg.model_id}")
    print(f"  layer     : {cfg.layer_idx}")
    print(f"  prefix    : {cfg.prompt_prefix}")
    if cfg.prompt_prefix:
        print(f"  prompt tpl: {cfg.prompt_template_path}")
    print(f"  dtype     : {cfg.dtype}")
    print(f"  datasets  : {len(datasets)}")
    for dataset in datasets:
        print(f"    - {dataset.name}: {dataset.root_dir}")
        for split_name in SPLIT_NAMES:
            print(f"      {split_name:<5}: {dataset.split_dirs[split_name]}")
        print(f"      vocab: {dataset.vocab_path}")
    if run_activations:
        print(f"  extraction: {extraction_mode}")
        if extraction_mode == "gene":
            print(f"  pooling   : {cfg.gene_pooling}")

    datasets_to_prepare = []
    for dataset in datasets:
        out_dir = _vocab_output_dir(vocabs_dir, cfg, dataset)
        if run_vocab and not run_activations and _vocab_outputs_exist(out_dir):
            print(f"\n[Skip] Vocab extraction for '{dataset.name}' already exists: {out_dir}")
            continue
        datasets_to_prepare.append(dataset)

    prepared_runs: list[PreparedExtractionRun] = []
    for dataset in datasets_to_prepare:
        print(f"\n{'='*70}\nPreparing dataset: {dataset.name}\n{'='*70}")
        vocab = _load_vocab(dataset.vocab_path)
        vocab_set = set(vocab)
        split_datasets = {}
        for split_name in SPLIT_NAMES:
            split_dir = dataset.split_dirs[split_name]
            split_dataset = load_from_disk(str(split_dir))
            split_datasets[split_name] = split_dataset
            print(
                f"Loaded {split_name} split: "
                f"{len(split_dataset):,} cells from {split_dir}"
            )
        _validate_dataset_genes_in_vocab(split_datasets, vocab_set, dataset.name)

        prepared_runs.append(
            PreparedExtractionRun(
                dataset=dataset,
                split_datasets=split_datasets,
                vocab=vocab,
            )
        )

    vocab_runs: list[PreparedExtractionRun] = []
    if run_vocab:
        for prepared in prepared_runs:
            out_dir = _vocab_output_dir(vocabs_dir, cfg, prepared.dataset)
            if _vocab_outputs_exist(out_dir):
                print(
                    f"\n[Skip] Vocab extraction for '{prepared.dataset.name}' already exists: "
                    f"{out_dir}"
                )
                continue
            vocab_runs.append(prepared)

    if run_activations or vocab_runs:
        print("\n[Model] Loading base model for requested extraction")
        tokenizer, model = load_model(cfg)
    else:
        tokenizer = model = None
        print("\n[Model] Skipped: no extraction work remains.")

    if run_vocab:
        for prepared in vocab_runs:
            _run_vocab(
                prepared.dataset,
                prepared.vocab,
                tokenizer,
                model,
                cfg,
                vocabs_dir=vocabs_dir,
            )

    if run_activations:
        for prepared in prepared_runs:
            dataset = prepared.dataset
            for split_name in SPLIT_NAMES:
                split_dir = dataset.split_dirs[split_name]
                _run_activations(
                    PreparedDatasetSplit(
                        dataset.name,
                        f"data_{split_name}",
                        split_dir,
                    ),
                    prepared.split_datasets[split_name],
                    tokenizer,
                    model,
                    cfg,
                    extraction_mode=extraction_mode,
                    n_cells=args.n_cells,
                    skip_validation=args.skip_validation,
                    activations_dir=activations_dir,
                    cells_dir=cells_dir,
                )

    print("\nAll datasets processed.")


if __name__ == "__main__":
    main()
