"""
inference.py — unified C2S model inference and hidden-state extraction

Merges gemma_inference.py and pythia_inference.py into a single module.
All shared logic (prompting, tokenisation, hooks, pooling, generation) lives
here once; family-specific dispatch is isolated to load_model and
HiddenStateExtractor._get_layers.

Supported models:
    vandijklab/C2S-Scale-Gemma-2-2B   (Gemma-2,   26 layers, d_model=2304)
    vandijklab/C2S-Scale-Gemma-2-27B  (Gemma-2,   46 layers, d_model=4608)
    vandijklab/C2S-Scale-Pythia-1b-pt (GPT-NeoX,  16 layers, d_model=2048)

Configuration is loaded from configs/extraction.yaml.
"""

import sys
import torch
import yaml
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTNeoXForCausalLM

_MODULE_DIR = Path(__file__).parent
_PROJECT_ROOT = _MODULE_DIR.parent.parent
DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "configs" / "extraction.yaml"
DEFAULT_PROMPT_TEMPLATE_PATH = (
    _PROJECT_ROOT / "src" / "prompts" / "cell_type_annotation_template.txt"
)
GENE_POOLING_MODES = {"all", "last"}


# ─────────────────────────────────────────────
# 0. Architecture registry
# ─────────────────────────────────────────────

MODEL_ARCHITECTURES: dict[str, dict] = {
    "vandijklab/C2S-Scale-Gemma-2-2B":   {"num_layers": 26, "d_model": 2304,  "family": "gemma"},
    "vandijklab/C2S-Scale-Gemma-2-27B":  {"num_layers": 46, "d_model": 4608,  "family": "gemma"},
    "vandijklab/C2S-Scale-Pythia-1b-pt": {"num_layers": 16, "d_model": 2048, "family": "pythia"},
}

_FAMILY_DEFAULTS = {
    "gemma":  {"model_id": "vandijklab/C2S-Scale-Gemma-2-2B",   "dtype": "bfloat16", "attn_implementation": "sdpa", "layer_idx": 15, "save_dir": "datasets/activations-gemma-2b"},
    "pythia": {"model_id": "vandijklab/C2S-Scale-Pythia-1b-pt", "dtype": "bfloat16", "attn_implementation": None,   "layer_idx": 10, "save_dir": "datasets/activations-pythia-1b"},
}


# ─────────────────────────────────────────────
# 0. Config
# ─────────────────────────────────────────────

@dataclass
class InferenceConfig:
    """Unified config for both Gemma and Pythia C2S models."""
    # model
    model_id: str = "vandijklab/C2S-Scale-Gemma-2-2B"
    device: str = "cuda"
    dtype: str = "bfloat16"
    attn_implementation: str | None = "sdpa"   # None → omit kwarg (required for Pythia)
    # inference
    max_new_tokens: int = 64
    temperature: float = 0.0
    prompt_prefix: bool = False
    prompt_template_path: str | Path = DEFAULT_PROMPT_TEMPLATE_PATH
    # extraction
    layer_idx: int = 15
    gene_pooling: str = "last"
    max_seq_len: int | None = None
    # output
    save_dir: str = "datasets/activations"
    filename_template: str = "gene_activations_{gene_pooling}_{prompt_prefix}.pt"

    @property
    def family(self) -> str:
        return MODEL_ARCHITECTURES.get(self.model_id, {}).get("family", "gemma")

    @property
    def arch(self) -> dict:
        return MODEL_ARCHITECTURES.get(self.model_id, {})

    @property
    def torch_dtype(self) -> torch.dtype:
        return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[self.dtype]

    @property
    def model_short_name(self) -> str:
        return self.model_id.split("/")[-1]


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else _PROJECT_ROOT / path


def _resolve_prompt_template_path(value: str | Path | None) -> Path:
    if value is None or str(value).strip() == "":
        value = DEFAULT_PROMPT_TEMPLATE_PATH
    return _project_path(value)


def load_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    model_family: str | None = None,
) -> InferenceConfig:
    """
    Load InferenceConfig from a YAML file.

    Supports two YAML layouts:
    - Unified (new):  top-level 'models' dict with 'gemma' / 'pythia' sub-sections.
        model_family must be provided to select which sub-section to read.
    - Per-model (legacy):  top-level 'model' section (gemma_inference.yaml style).

    Args:
        config_path:  Path to the YAML file.  Defaults to configs/extraction.yaml.
        model_family: 'gemma' or 'pythia'.  Required when using unified layout.

    Returns:
        Populated InferenceConfig.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"[Error] Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    # Resolve model section: prefer unified format, fall back to legacy
    if model_family and "models" in raw:
        model_section = raw["models"].get(model_family, {})
    else:
        model_section = raw.get("model", {})

    inf = raw.get("inference", {})
    ext = raw.get("extraction", {})
    out = raw.get("output", {})

    # Determine which family's defaults to use
    model_id = model_section.get("model_id")
    if model_id is None:
        fam = model_family or "gemma"
        model_id = _FAMILY_DEFAULTS[fam]["model_id"]

    fam_key = MODEL_ARCHITECTURES.get(model_id, {}).get("family", model_family or "gemma")
    defs = _FAMILY_DEFAULTS[fam_key]

    cfg = InferenceConfig(
        model_id            = model_id,
        device              = model_section.get("device",              "cuda"),
        dtype               = model_section.get("dtype",               defs["dtype"]),
        attn_implementation = model_section.get("attn_implementation", defs["attn_implementation"]),
        max_new_tokens      = inf.get("max_new_tokens", 64),
        temperature         = inf.get("temperature",   0.0),
        prompt_prefix       = inf.get("prompt_prefix", False),
        prompt_template_path = _resolve_prompt_template_path(
            inf.get("prompt_template_path")
        ),
        layer_idx           = model_section.get("layer_idx", ext.get("layer_idx", defs["layer_idx"])),
        gene_pooling        = ext.get("gene_pooling",  "last"),
        max_seq_len         = ext.get("max_seq_len",   None),
        save_dir            = out.get("save_dir", out.get("activations_dir", defs["save_dir"])),
        filename_template   = out.get("filename_template", "gene_activations_{gene_pooling}_{prompt_prefix}.pt"),
    )
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: InferenceConfig) -> None:
    arch = cfg.arch
    if not arch:
        print(f"[Warning] Unknown model_id '{cfg.model_id}'. Skipping layer validation.")
        return

    max_layer = arch["num_layers"] - 1
    if not (0 <= cfg.layer_idx <= max_layer):
        raise ValueError(
            f"layer_idx={cfg.layer_idx} out of range for {cfg.model_id}. Valid: 0-{max_layer}."
        )
    if cfg.gene_pooling not in GENE_POOLING_MODES:
        raise ValueError(
            f"Invalid gene_pooling='{cfg.gene_pooling}'. Expected 'all' or 'last'."
        )


# ─────────────────────────────────────────────
# 1. Model loading
# ─────────────────────────────────────────────

def load_model(cfg: InferenceConfig):
    """
    Load tokenizer and model for any supported C2S model.

    Dispatches to the correct HuggingFace class based on cfg.family.
    Gemma passes attn_implementation; Pythia sets a pad token.

    Returns:
        (tokenizer, model)
    """
    print(f"Loading tokenizer from {cfg.model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)

    print(f"Loading model ({cfg.dtype}) on {cfg.device} ...")
    if cfg.family == "gemma":
        kwargs = dict(torch_dtype=cfg.torch_dtype, device_map=cfg.device)
        if cfg.attn_implementation:
            kwargs["attn_implementation"] = cfg.attn_implementation
        model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **kwargs)
    else:  # pythia
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = GPTNeoXForCausalLM.from_pretrained(
            cfg.model_id, torch_dtype=cfg.torch_dtype, device_map=cfg.device
        )

    model.eval()

    arch = cfg.arch
    arch_str = (
        f"  Architecture: {arch['num_layers']} layers, d_model={arch['d_model']}"
        if arch else "  Architecture: unknown"
    )
    print(
        f"  dtype={next(model.parameters()).dtype}  "
        f"params={sum(p.numel() for p in model.parameters())/1e9:.2f}B")
    print(arch_str)

    return tokenizer, model


# ─────────────────────────────────────────────
# 2. Prompt building
# ─────────────────────────────────────────────

@lru_cache(maxsize=16)
def _load_prompt_template(prompt_template_path: str | Path = DEFAULT_PROMPT_TEMPLATE_PATH) -> str:
    return Path(prompt_template_path).expanduser().read_text().strip()


def build_cell_type_prompt(
    cell_sentence: str,
    organism: str = "Homo sapiens",
    prompt_template_path: str | Path | None = None,
) -> str:
    """
    Construct the C2S-Scale cell type prediction prompt.

    Args:
        cell_sentence: Space-separated gene names ordered by descending expression.
        organism:      Organism string used during training.
        prompt_template_path: Prompt template file. Defaults to the bundled
            cell-type annotation template.

    Returns:
        Formatted prompt string.
    """
    template_path = _resolve_prompt_template_path(prompt_template_path)
    return _load_prompt_template(template_path).format(
        organism=organism,
        cell_sentence=cell_sentence,
    )


# ─────────────────────────────────────────────
# 3. Tokenisation helpers
# ─────────────────────────────────────────────

def _get_token_indices_for_substring(
    tokenizer, full_text: str, substring: str
) -> tuple[int, int]:
    """
    Return (start_idx, end_idx) token positions for substring inside full_text.

    Indices are bos-aware (offset mapping from the tokenizer).
    end_idx is exclusive.
    """
    char_start = full_text.find(substring)
    if char_start == -1:
        raise ValueError("Substring not found in full text.")
    char_end = char_start + len(substring)

    offsets = tokenizer(full_text, return_offsets_mapping=True)["offset_mapping"]
    idxs = [i for i, (ts, te) in enumerate(offsets) if ts < char_end and te > char_start]
    if not idxs:
        raise ValueError("No tokeniser tokens found for substring in full text.")
    return idxs[0], idxs[-1] + 1


def _get_gene_token_ranges(
    tokenizer, prompt: str, cell_sentence: str
) -> list[tuple[str, int, int]]:
    """
    Map each gene in cell_sentence to its token index range inside prompt.

    Returns:
        List of (gene_name, start_idx, end_idx), end_idx exclusive.
    """
    sentence_char_start = prompt.find(cell_sentence)
    if sentence_char_start == -1:
        raise ValueError("cell_sentence not found in prompt.")

    offsets = tokenizer(prompt, return_offsets_mapping=True)["offset_mapping"]

    gene_ranges: list[tuple[str, int, int]] = []
    cursor = 0
    for gene in cell_sentence.split():
        local_start = cell_sentence.find(gene, cursor)
        if local_start == -1:
            raise ValueError(f"Could not locate gene '{gene}' while parsing cell_sentence.")
        local_end = local_start + len(gene)
        cursor = local_end

        abs_start = sentence_char_start + local_start
        abs_end   = sentence_char_start + local_end

        idxs = [i for i, (ts, te) in enumerate(offsets) if ts < abs_end and te > abs_start]
        if not idxs:
            raise ValueError(f"No tokeniser tokens found for gene '{gene}'.")
        gene_ranges.append((gene, idxs[0], idxs[-1] + 1))

    return gene_ranges


def _pool_gene_representations(
    token_activations: torch.Tensor,
    gene_token_ranges: list[tuple[str, int, int]],
    pooling: str,
) -> torch.Tensor:
    """
    Select token activations from each gene span.

    Args:
        token_activations: (seq_len, d_model)
        gene_token_ranges: [(gene, start, end), ...]
        pooling:           'all' or 'last'

    Returns:
        (n_gene_tokens, d_model) for 'all', or (n_genes, d_model) for 'last'.
    """
    selected = []
    for gene, start, end in gene_token_ranges:
        toks = token_activations[start:end]
        if toks.shape[0] == 0:
            raise ValueError(
                f"Empty token slice for gene '{gene}' [{start},{end}). "
                f"Activation shape: {tuple(token_activations.shape)}."
            )
        if pooling == "all":
            selected.append(toks)
        elif pooling == "last":
            selected.append(toks[-1:].contiguous())
        else:
            raise ValueError(f"Unknown pooling: '{pooling}'")
    return torch.cat(selected, dim=0)


# ─────────────────────────────────────────────
# 4. Hidden-state extraction
# ─────────────────────────────────────────────

def _get_transformer_layers(model):
    """
    Return the transformer layer list for any supported model.

    Uses attribute inspection so no family string needs to be threaded through.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers          # Gemma-2
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers       # Pythia / GPT-NeoX
    raise AttributeError(
        f"Cannot locate transformer layers on {type(model).__name__}. "
        "Add a branch to _get_transformer_layers()."
    )


def get_transformer_layers(model):
    """Public wrapper for locating transformer layers on supported C2S models."""
    return _get_transformer_layers(model)


class HiddenStateExtractor:
    """
    Captures residual-stream activations at a specified transformer layer via
    a PyTorch forward hook.

    Extraction point: output of the full block (attn + MLP + residual).
    Works with any model supported by _get_transformer_layers().

    Args:
        model:        Loaded C2S model.
        layer_idx:    Layer index (0-indexed).
        token_range:  Optional (start, end) to capture only a token slice.
    """

    def __init__(self, model, layer_idx: int, token_range: tuple | None = None):
        self.model       = model
        self.layer_idx   = layer_idx
        self.token_range = token_range
        self._hooks: list = []
        self._activation: torch.Tensor | None = None

    def _make_hook(self):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if self.token_range is not None:
                s, e = self.token_range
                captured = hidden[:, s:e, :].detach()
            else:
                captured = hidden.detach()
            self._activation = captured.to(torch.bfloat16).cpu()
        return hook

    def register(self):
        layers = _get_transformer_layers(self.model)
        if self.layer_idx >= len(layers):
            raise IndexError(
                f"layer_idx={self.layer_idx} out of range; "
                f"model has {len(layers)} layers (0–{len(layers)-1})."
            )
        self._hooks.append(layers[self.layer_idx].register_forward_hook(self._make_hook()))

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def get(self) -> torch.Tensor:
        return self._activation

    def clear(self):
        self._activation = None

    def __enter__(self):
        self.register()
        return self

    def __exit__(self, *args):
        self.remove()


# ─────────────────────────────────────────────
# 5. Generation
# ─────────────────────────────────────────────

@torch.inference_mode()
def generate(prompt: str, tokenizer, model, cfg: InferenceConfig) -> str:
    """
    Run greedy (temperature=0) or sampled generation.

    Returns:
        Generated text, excluding the input prompt.
    """
    inputs    = tokenizer(prompt, return_tensors="pt").to(cfg.device)
    input_len = inputs["input_ids"].shape[1]

    kwargs: dict = dict(**inputs, max_new_tokens=cfg.max_new_tokens,
                        pad_token_id=tokenizer.eos_token_id)
    if cfg.temperature == 0.0:
        kwargs["do_sample"] = False
    else:
        kwargs["do_sample"]   = True
        kwargs["temperature"] = cfg.temperature

    output_ids = model.generate(**kwargs)
    return tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()


# ─────────────────────────────────────────────
# 6. Activation extraction
# ─────────────────────────────────────────────

@torch.inference_mode()
def extract_hidden_states(
    prompt: str,
    tokenizer,
    model,
    cfg: InferenceConfig,
    layer_idx: int | None = None,
) -> torch.Tensor:
    """
    Extract residual-stream activations for every token in prompt.

    Returns:
        bfloat16 tensor of shape (seq_len, d_model).
    """
    if layer_idx is None:
        layer_idx = cfg.layer_idx

    inputs = tokenizer(prompt, return_tensors="pt").to(cfg.device)
    with HiddenStateExtractor(model, layer_idx) as ext:
        _ = model(**inputs, use_cache=False)
        return ext.get().squeeze(0)


@torch.inference_mode()
def extract_cell_sentence_activations(
    cell_sentence: str,
    prompt: str,
    tokenizer,
    model,
    cfg: InferenceConfig,
    layer_idx: int | None = None,
    gene_pooling: str | None = None,
) -> torch.Tensor:
    """
    Extract gene-token residual-stream representations for a cell sentence.

    Args:
        cell_sentence: Space-separated gene names.
        prompt:        Full prompt containing cell_sentence.
        layer_idx:     Override cfg.layer_idx.
        gene_pooling:  Override cfg.gene_pooling ('all' or 'last').

    Returns:
        bfloat16 tensor of shape (n_gene_tokens, d_model) for 'all',
        or (n_genes, d_model) for 'last'.
    """
    if layer_idx is None:
        layer_idx = cfg.layer_idx
    if gene_pooling is None:
        gene_pooling = cfg.gene_pooling
    if gene_pooling not in GENE_POOLING_MODES:
        raise ValueError(
            f"Invalid gene_pooling='{gene_pooling}'. Expected 'all' or 'last'."
        )

    inputs  = tokenizer(prompt, return_tensors="pt").to(cfg.device)
    seq_len = inputs["input_ids"].shape[1]
    if cfg.max_seq_len is not None and seq_len > cfg.max_seq_len:
        raise ValueError(
            f"Prompt has {seq_len} tokens, exceeds max_seq_len={cfg.max_seq_len}."
        )

    cs_start, cs_end   = _get_token_indices_for_substring(tokenizer, prompt, cell_sentence)
    gene_token_ranges  = _get_gene_token_ranges(tokenizer, prompt, cell_sentence)

    with HiddenStateExtractor(model, layer_idx, token_range=(cs_start, cs_end)) as ext:
        _ = model(**inputs, use_cache=False)
        token_level = ext.get().squeeze(0)  # (cs_len, d_model)

    local_ranges = [
        (gene, g_start - cs_start, g_end - cs_start)
        for gene, g_start, g_end in gene_token_ranges
    ]
    return _pool_gene_representations(token_level, local_ranges, gene_pooling)


@torch.inference_mode()
def harvest_cell_sentence_activations_batch(
    cell_sentences: list[str],
    prompts: list[str],
    tokenizer,
    model,
    cfg: InferenceConfig,
    layer_idx: int | None = None,
    gene_pooling: str | None = None,
) -> list[torch.Tensor]:
    """
    Harvest gene-token activations for a list of (cell_sentence, prompt) pairs.

    Returns:
        List of tensors, one per cell. Each tensor has shape
        (n_gene_tokens, d_model) for 'all', or (n_genes, d_model) for 'last'.
    """
    if layer_idx is None:
        layer_idx = cfg.layer_idx
    return [
        extract_cell_sentence_activations(
            cs, p, tokenizer, model, cfg,
            layer_idx=layer_idx, gene_pooling=gene_pooling,
        )
        for cs, p in zip(cell_sentences, prompts)
    ]
