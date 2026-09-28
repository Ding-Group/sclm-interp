"""Scoring helpers for cell-type multiple-choice QA evaluation."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import torch
import torch.nn.functional as F

from src.data.inference import InferenceConfig

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_FINAL_ANSWER_RE = re.compile(
    r"(?:final\s+answer|answer|prediction)\s*(?:is|:|-)?\s*(.+)",
    re.IGNORECASE | re.DOTALL,
)


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    return [str(value)]


def label_key(text: Any) -> str:
    return " ".join(_label_tokens(text))


def trim_control_token(text: str) -> str:
    return text.split("<ctrl", 1)[0].rstrip()


def gene_names_for_sample(sample: dict[str, Any], cfg: Any) -> list[str]:
    genes = as_list(sample.get(cfg.gene_names_column))
    if genes:
        return genes

    gene_sentence = sample.get(cfg.gene_sentence_column)
    if gene_sentence:
        return str(gene_sentence).strip().split()
    raise ValueError(
        f"Sample is missing {cfg.gene_names_column!r} and "
        f"{cfg.gene_sentence_column!r}; cannot locate genes for reconstruction."
    )


def score_generated_prediction(
    sample: dict[str, Any],
    generated_text: str,
    cfg: Any,
) -> dict[str, Any]:
    choices = as_list(sample.get(cfg.choices_column))
    choice_labels = as_list(sample.get(cfg.choice_labels_column))
    gold_label = str(sample.get(cfg.answer_choice_column, "")).strip()
    gold_cell_type = str(sample.get(cfg.gold_cell_type_column, "")).strip()

    selected_label, selected_cell_type, ambiguous = find_selected_choice(
        generated_text,
        choices,
        choice_labels,
    )
    selected_is_gold = selected_label is not None and (
        _clean_choice_label(selected_label) == _clean_choice_label(gold_label)
    )
    generated_gold_text = bool(gold_cell_type) and _candidate_is_encoded(
        gold_cell_type,
        _answer_tail(generated_text),
    )
    correct = selected_is_gold or (selected_label is None and generated_gold_text)

    return {
        "gold_choice_label": gold_label,
        "gold_cell_type": gold_cell_type,
        "selected_choice_label": selected_label,
        "selected_cell_type": selected_cell_type,
        "ambiguous": ambiguous,
        "correct": correct,
    }


def find_selected_choice(
    generated_text: str,
    choices: list[str],
    choice_labels: list[str],
) -> tuple[str | None, str | None, bool]:
    explicit_label = _extract_explicit_choice_label(generated_text, choice_labels)
    if explicit_label is not None:
        try:
            choice = choices[choice_labels.index(explicit_label)]
        except (ValueError, IndexError):
            choice = None
        return explicit_label, choice, False

    answer = _answer_tail(generated_text)
    matches = [
        (label, choice)
        for label, choice in zip(choice_labels, choices, strict=False)
        if _candidate_is_encoded(choice, answer)
    ]
    if not matches and answer != generated_text:
        matches = [
            (label, choice)
            for label, choice in zip(choice_labels, choices, strict=False)
            if _candidate_is_encoded(choice, generated_text)
        ]

    if len(matches) == 1:
        return matches[0][0], matches[0][1], False
    if len(matches) > 1:
        return None, None, True
    return None, None, False


def build_likelihood_prompt(sample: dict[str, Any], cfg: Any) -> str:
    template = _load_prompt_template(cfg.likelihood_prompt_template_path)
    cell_sentence = sample.get(cfg.gene_sentence_column)
    if not cell_sentence:
        cell_sentence = " ".join(gene_names_for_sample(sample, cfg))
    return template.format(
        organism=_sample_organism(sample, cfg),
        cell_sentence=str(cell_sentence).strip(),
    )


def score_likelihood_prediction(
    sample: dict[str, Any],
    tokenizer,
    model,
    inf_cfg: InferenceConfig,
    cfg: Any,
    acceptable_answers: dict[str, list[str]],
    *,
    hooks: list[Any] | None = None,
) -> dict[str, Any]:
    choices = as_list(sample.get(cfg.choices_column))
    choice_labels = as_list(sample.get(cfg.choice_labels_column))
    gold_label = str(sample.get(cfg.answer_choice_column, "")).strip()
    gold_cell_type = str(sample.get(cfg.gold_cell_type_column, "")).strip()
    prompt = build_likelihood_prompt(sample, cfg)
    gene_names = gene_names_for_sample(sample, cfg)

    choice_scores: list[dict[str, Any]] = []
    for label, choice in zip(choice_labels, choices, strict=False):
        variant_scores = [
            _score_answer_variant(
                prompt,
                variant,
                tokenizer,
                model,
                inf_cfg,
                hooks=hooks,
                gene_names=gene_names,
            )
            for variant in _candidate_answer_variants(choice, acceptable_answers)
        ]
        best_variant = max(
            variant_scores,
            key=lambda row: row["mean_log_likelihood"],
        )
        choice_scores.append(
            {
                "choice_label": label,
                "cell_type": choice,
                "score": best_variant["mean_log_likelihood"],
                "best_answer": best_variant["answer"],
                "answer_scores": variant_scores,
            }
        )

    selected = max(choice_scores, key=lambda row: row["score"])
    selected_label = str(selected["choice_label"])
    return {
        "gold_choice_label": gold_label,
        "gold_cell_type": gold_cell_type,
        "selected_choice_label": selected_label,
        "selected_cell_type": str(selected["cell_type"]),
        "selected_answer": selected["best_answer"],
        "ambiguous": False,
        "correct": _clean_choice_label(selected_label) == _clean_choice_label(gold_label),
        "choice_scores": choice_scores,
        "scoring_prompt": prompt,
    }


def prepare_hooks_for_prompt(
    hooks: list[Any],
    tokenizer,
    prompt: str,
    gene_names: list[str],
) -> None:
    prompt_len, gene_ranges = _gene_token_ranges_for_prompt(
        tokenizer,
        prompt,
        gene_names,
    )
    for hook in hooks:
        hook.set_active_token_indices(
            _gene_token_indices_from_ranges(gene_ranges, hook.pooling_method),
            prompt_len=prompt_len,
        )


def clear_hook_prompt_state(hooks: list[Any]) -> None:
    for hook in hooks:
        hook.clear_active_token_indices()


def _score_answer_variant(
    prompt: str,
    answer: str,
    tokenizer,
    model,
    inf_cfg: InferenceConfig,
    *,
    hooks: list[Any] | None,
    gene_names: list[str],
) -> dict[str, Any]:
    total, mean, token_count = _continuation_log_likelihood(
        prompt,
        answer,
        tokenizer,
        model,
        inf_cfg,
        hooks=hooks,
        gene_names=gene_names,
    )
    return {
        "answer": answer,
        "log_likelihood": total,
        "mean_log_likelihood": mean,
        "num_tokens": token_count,
    }


def _candidate_answer_variants(
    choice: str,
    acceptable_answers: dict[str, list[str]],
) -> list[str]:
    variants = [choice]
    variants.extend(acceptable_answers.get(label_key(choice), []))

    deduped: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        variant = str(variant).strip()
        key = label_key(variant)
        if variant and key and key not in seen:
            deduped.append(variant)
            seen.add(key)
    return deduped


@torch.inference_mode()
def _continuation_log_likelihood(
    prompt: str,
    continuation: str,
    tokenizer,
    model,
    inf_cfg: InferenceConfig,
    *,
    hooks: list[Any] | None = None,
    gene_names: list[str] | None = None,
) -> tuple[float, float, int]:
    continuation_text = " " + continuation.strip()
    full_text = prompt + continuation_text
    tokenized = tokenizer(full_text, return_tensors="pt")
    continuation_ids = tokenizer(
        continuation_text,
        add_special_tokens=False,
    )["input_ids"]
    if not continuation_ids:
        raise ValueError(f"Empty continuation after tokenization: {continuation!r}")

    candidate_start = tokenized["input_ids"].shape[1] - len(continuation_ids)
    if candidate_start <= 0:
        raise ValueError(f"Could not locate continuation tokens for {continuation!r}.")

    inputs = tokenized.to(inf_cfg.device)
    if hooks:
        prepare_hooks_for_prompt(hooks, tokenizer, full_text, gene_names or [])
    try:
        outputs = model(**inputs, use_cache=False)
    finally:
        if hooks:
            clear_hook_prompt_state(hooks)

    labels = inputs["input_ids"]
    logits = outputs.logits[:, :-1, :]
    next_token_ids = labels[:, 1:]
    positions = torch.arange(next_token_ids.shape[1], device=next_token_ids.device)
    mask = positions >= (candidate_start - 1)

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(
        -1,
        next_token_ids.unsqueeze(-1),
    ).squeeze(-1)
    selected = token_log_probs[0, mask]
    total = float(selected.sum().item())
    count = int(selected.numel())
    mean = total / count if count else float("-inf")
    return total, mean, count


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


def _contains_token_sequence(haystack: list[str], needle: list[str]) -> bool:
    if len(needle) > len(haystack):
        return False
    return any(
        haystack[i : i + len(needle)] == needle
        for i in range(len(haystack) - len(needle) + 1)
    )


def _candidate_is_encoded(candidate_label: str, generated_text: str) -> bool:
    candidate_tokens = _label_tokens(candidate_label)
    generated_tokens = _label_tokens(generated_text)
    if not candidate_tokens or not generated_tokens:
        return False

    if _contains_token_sequence(generated_tokens, candidate_tokens):
        return True

    generated_set = set(generated_tokens)
    return all(_token_variants(token) & generated_set for token in candidate_tokens)


def _answer_tail(generated_text: str) -> str:
    match = _FINAL_ANSWER_RE.search(generated_text)
    if match:
        return match.group(1).strip()
    return generated_text.strip()


def _clean_choice_label(value: Any) -> str:
    return str(value).strip().strip("()[]{}.:;-").upper()


def _extract_explicit_choice_label(
    generated_text: str,
    choice_labels: list[str],
) -> str | None:
    labels_by_key = {_clean_choice_label(label): label for label in choice_labels}
    if not labels_by_key:
        return None

    answer = _answer_tail(generated_text)
    cleaned_answer = _clean_choice_label(answer)
    if cleaned_answer in labels_by_key:
        return labels_by_key[cleaned_answer]

    for raw_label in choice_labels:
        key = _clean_choice_label(raw_label)
        escaped = re.escape(str(raw_label).strip())
        strong_prefix = re.match(
            rf"^\s*\(?\s*{escaped}\s*\)?\s*[.)\]:-]",
            answer,
            flags=re.IGNORECASE,
        )
        if strong_prefix:
            return labels_by_key[key]

    option_match = re.search(
        r"\b(?:choice|option|choose|select)\s+([A-Za-z])\b",
        generated_text,
        flags=re.IGNORECASE,
    )
    if option_match:
        key = _clean_choice_label(option_match.group(1))
        return labels_by_key.get(key)

    return None


def _sample_organism(sample: dict[str, Any], cfg: Any) -> str:
    if cfg.organism_column and sample.get(cfg.organism_column):
        return str(sample[cfg.organism_column])
    return cfg.default_organism


def _load_prompt_template(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


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


def _gene_token_ranges_for_prompt(
    tokenizer,
    prompt: str,
    gene_names: list[str],
) -> tuple[int, list[tuple[str, int, int]]]:
    tokenized = tokenizer(prompt, return_offsets_mapping=True)
    offsets = tokenized["offset_mapping"]
    prompt_len = len(tokenized["input_ids"])
    cursor = 0
    ranges: list[tuple[str, int, int]] = []

    for gene in gene_names:
        gene = str(gene).strip()
        if not gene:
            continue
        char_start = prompt.find(gene, cursor)
        if char_start == -1:
            char_start = prompt.find(gene)
        if char_start == -1:
            raise ValueError(f"Could not locate gene {gene!r} in prompt.")
        char_end = char_start + len(gene)
        cursor = char_end

        idxs = [
            idx
            for idx, (token_start, token_end) in enumerate(offsets)
            if token_start < char_end and token_end > char_start
        ]
        if not idxs:
            raise ValueError(f"No tokenizer tokens found for gene {gene!r}.")
        ranges.append((gene, idxs[0], idxs[-1] + 1))

    return prompt_len, ranges
