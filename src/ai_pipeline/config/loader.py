"""Configuration loader utilities."""

from __future__ import annotations

import json
import yaml

from typing import Any, Dict, Union
from dataclasses import asdict
from pathlib import Path

from .schema import FullConfig, ModelConfig, DataConfig, TrainingConfig, EvalConfig, RunConfig


def _read_config_file(path: Union[str, Path]) -> Dict[str, Any]:
    """Read YAML or JSON config file into dict."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    suffix = config_path.suffix.lower()
    text = config_path.read_text(encoding="utf-8")

    if suffix in {".yaml", ".yml"}:
        return yaml.safe_load(text) or {}
    if suffix == ".json":
        return json.loads(text)

    raise ValueError(f"Unsupported config file type: {suffix}")


def _merge_dict(defaults: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow merge two dictionaries."""
    merged = defaults.copy()
    merged.update(overrides)
    return merged


def _build_data_config(raw: Dict[str, Any], base_dir: Path) -> DataConfig:
    cfg_dict = _merge_dict(asdict(DataConfig()), raw)

    train_path = Path(cfg_dict["train_path"])
    val_path = cfg_dict.get("val_path")

    if not train_path.is_absolute():
        train_path = base_dir / train_path

    if val_path is not None:
        val_path = Path(val_path)
        if not val_path.is_absolute():
            val_path = base_dir / val_path

    cfg_dict["train_path"] = train_path
    cfg_dict["val_path"] = val_path

    return DataConfig(**cfg_dict)


def _build_training_config(raw: Dict[str, Any], base_dir: Path) -> TrainingConfig:
    cfg_dict = _merge_dict(asdict(TrainingConfig()), raw)

    output_dir = Path(cfg_dict["output_dir"])
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir

    cfg_dict["output_dir"] = output_dir

    return TrainingConfig(**cfg_dict)


def _build_model_config(raw: Dict[str, Any]) -> ModelConfig:
    return ModelConfig(**_merge_dict(asdict(ModelConfig()), raw))


def _build_eval_config(raw: Dict[str, Any]) -> EvalConfig:
    return EvalConfig(**_merge_dict(asdict(EvalConfig()), raw))


def _build_run_config(raw: Dict[str, Any]) -> RunConfig:
    return RunConfig(**_merge_dict(asdict(RunConfig()), raw))


def load_config(path: Union[str, Path]) -> FullConfig:
    """Load config file into `FullConfig` instance."""
    config_path = Path(path)
    raw = _read_config_file(config_path)

    base_dir = config_path.parent

    model_raw = raw.get("model", {})
    data_raw = raw.get("data", {})
    training_raw = raw.get("training", {})
    eval_raw = raw.get("eval", {})
    run_raw = raw.get("run", {})

    model_cfg = _build_model_config(model_raw)
    data_cfg = _build_data_config(data_raw, base_dir=base_dir)
    training_cfg = _build_training_config(training_raw, base_dir=base_dir)
    eval_cfg = _build_eval_config(eval_raw)
    run_cfg = _build_run_config(run_raw)

    return FullConfig(
        model=model_cfg,
        data=data_cfg,
        training=training_cfg,
        eval=eval_cfg,
        run=run_cfg,
    )
