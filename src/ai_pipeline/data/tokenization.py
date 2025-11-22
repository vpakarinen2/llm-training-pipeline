"""Tokenizer utilities."""

from __future__ import annotations

from transformers import AutoTokenizer, PreTrainedTokenizerBase
from typing import Any, Dict, Iterable, Mapping

from ai_pipeline.config.schema import DataConfig, ModelConfig


def create_tokenizer(model_cfg: ModelConfig) -> PreTrainedTokenizerBase:
    """Create tokenizer for given model config."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.model_name,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return tokenizer


def tokenize_text(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    data_cfg: DataConfig,
) -> Dict[str, Any]:
    """Tokenize single text string."""
    return tokenizer(
        text,
        max_length=data_cfg.max_seq_length,
        truncation=True,
        padding=False,
        return_tensors=None,
    )


def tokenize_batch(
    tokenizer: PreTrainedTokenizerBase,
    texts: Iterable[str],
    data_cfg: DataConfig,
) -> Dict[str, Any]:
    """Tokenize batch of text strings."""
    return tokenizer(
        list(texts),
        max_length=data_cfg.max_seq_length,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )


def format_example(example: Mapping[str, Any], data_cfg: DataConfig) -> str:
    """Format single example into prompt text."""
    if data_cfg.instruction_field and data_cfg.output_field:
        instruction = str(example.get(data_cfg.instruction_field, ""))
        input_text = ""
        if data_cfg.input_field is not None:
            input_text = str(example.get(data_cfg.input_field, ""))
        output_text = str(example.get(data_cfg.output_field, ""))

        if input_text:
            return f"Instruction:\n{instruction}\n\nInput:\n{input_text}\n\nResponse:\n{output_text}"
        return f"Instruction:\n{instruction}\n\nResponse:\n{output_text}"

    return str(example.get(data_cfg.text_field, ""))

