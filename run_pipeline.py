#!/usr/bin/env python3
"""
run_pipeline.py — ML pipeline orchestrator.

Executes three stages in order:
  1. Feature engineering  (pipeline.py)       raw OHLCV  →  feature CSVs
  2. Model training       (Interface.py)      feature CSVs  →  checkpoint.pth
  3. Model export         (exportModel.py)    checkpoint.pth →  transformer.pt + scalers

Each stage is a subprocess so failures are caught immediately and the
exit code propagates to the calling shell / container orchestrator.

Usage:
  python run_pipeline.py --data-dir data/ --feature-dir features/
  python run_pipeline.py --skip-train
  python run_pipeline.py --config models/best_config.yaml  # use Optuna best params
  python run_pipeline.py --seed 42
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Configuration schema — validated before any subprocess is spawned
# ---------------------------------------------------------------------------

class PipelineConfig(BaseModel):
    """All pipeline settings, validated at startup."""

    # Paths
    data_dir: str = "data"
    feature_dir: str = "features"
    model_dir: str = "models"
    skip_train: bool = False
    config_file: str = ""  # optional path to best_config.yaml from Optuna

    # Reproducibility
    seed: int = Field(default=42, ge=0)

    # Sequence window
    seq_len: int = Field(default=30, gt=0, description="Encoder input length")
    label_len: int = Field(default=10, gt=0, description="Decoder overlap length")
    pred_len: int = Field(default=5, gt=0, description="Forecast horizon")

    # Transformer architecture
    d_model: int = Field(default=256, gt=0)
    n_heads: int = Field(default=8, gt=0)
    e_layers: int = Field(default=3, gt=0)
    d_layers: int = Field(default=2, gt=0)
    d_ff: int = Field(default=512, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, le=1.0)

    # Training
    batch_size: int = Field(default=128, gt=0)
    train_epochs: int = Field(default=100, gt=0)
    learning_rate: float = Field(default=0.0005, gt=0.0)

    @field_validator("n_heads")
    @classmethod
    def heads_must_divide_d_model(cls, v: int, info) -> int:
        d_model = (info.data or {}).get("d_model", 256)
        if d_model % v != 0:
            raise ValueError(
                f"n_heads={v} must evenly divide d_model={d_model}"
            )
        return v

    @model_validator(mode="after")
    def data_dir_must_exist(self) -> "PipelineConfig":
        if not Path(self.data_dir).exists():
            raise ValueError(
                f"data_dir '{self.data_dir}' does not exist — "
                "create the directory or pass --data-dir <path>"
            )
        return self

    @classmethod
    def from_yaml(cls, path: str, overrides: dict | None = None) -> "PipelineConfig":
        """Load config from a YAML file (e.g. models/best_config.yaml) with
        optional CLI overrides applied on top."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        if overrides:
            data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(cmd: list, *, cwd: str | None = None) -> None:
    """Run a command, stream output, and exit on failure."""
    print(f"\n{'='*60}")
    print(f">>> {' '.join(str(c) for c in cmd)}")
    print("=" * 60, flush=True)

    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {result.returncode}",
              file=sys.stderr)
        sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="TradingTransformer ML pipeline orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", default="data",
                   help="Directory containing raw OHLCV CSVs")
    p.add_argument("--feature-dir", default="features",
                   help="Directory to write enriched feature CSVs")
    p.add_argument("--model-dir", default="models",
                   help="Directory to write exported model artefacts")
    p.add_argument("--skip-train", action="store_true",
                   help="Skip training and re-export an existing checkpoint")
    p.add_argument("--config",
                   help="Path to YAML config (e.g. models/best_config.yaml)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seq-len", type=int)
    p.add_argument("--label-len", type=int)
    p.add_argument("--pred-len", type=int)
    p.add_argument("--d-model", type=int)
    p.add_argument("--n-heads", type=int)
    p.add_argument("--e-layers", type=int)
    p.add_argument("--d-layers", type=int)
    p.add_argument("--d-ff", type=int)
    p.add_argument("--dropout", type=float)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--lr", type=float)
    return p.parse_args()


def main() -> None:
    cli = _parse_args()

    # Collect explicit CLI overrides (skip Nones so YAML values aren't clobbered)
    overrides = {
        k: v for k, v in {
            "data_dir": cli.data_dir,
            "feature_dir": cli.feature_dir,
            "model_dir": cli.model_dir,
            "skip_train": cli.skip_train,
            "seed": cli.seed,
            "seq_len": cli.seq_len,
            "label_len": cli.label_len,
            "pred_len": cli.pred_len,
            "d_model": cli.d_model,
            "n_heads": cli.n_heads,
            "e_layers": cli.e_layers,
            "d_layers": cli.d_layers,
            "d_ff": cli.d_ff,
            "dropout": cli.dropout,
            "batch_size": cli.batch_size,
            "train_epochs": cli.epochs,
            "learning_rate": cli.lr,
        }.items() if v is not None
    }

    try:
        if cli.config:
            cfg = PipelineConfig.from_yaml(cli.config, overrides)
        else:
            cfg = PipelineConfig(**overrides)
    except Exception as exc:
        print(f"[PipelineConfig] Invalid configuration: {exc}", file=sys.stderr)
        sys.exit(1)

    Path(cfg.feature_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)

    python = sys.executable

    # ------------------------------------------------------------------
    # Stage 1 — Feature engineering
    # ------------------------------------------------------------------
    print("\n[Stage 1/3] Feature engineering")
    run([
        python,
        "research/features/pipeline.py",
        cfg.data_dir,
        "-o", cfg.feature_dir,
    ])

    # ------------------------------------------------------------------
    # Stage 2 — Training
    # ------------------------------------------------------------------
    if cfg.skip_train:
        print("\n[Stage 2/3] Training skipped (--skip-train)")
    else:
        print("\n[Stage 2/3] Model training")
        run([
            python,
            "research/transformer/Interface.py",
            "--data-path", cfg.feature_dir,
            "--checkpoints", "./checkpoints/",
            "--seq-len", str(cfg.seq_len),
            "--label-len", str(cfg.label_len),
            "--pred-len", str(cfg.pred_len),
            "--d-model", str(cfg.d_model),
            "--n-heads", str(cfg.n_heads),
            "--e-layers", str(cfg.e_layers),
            "--d-layers", str(cfg.d_layers),
            "--d-ff", str(cfg.d_ff),
            "--dropout", str(cfg.dropout),
            "--batch-size", str(cfg.batch_size),
            "--epochs", str(cfg.train_epochs),
            "--lr", str(cfg.learning_rate),
            "--seed", str(cfg.seed),
        ])

    # ------------------------------------------------------------------
    # Stage 3 — Export to LibTorch + scaler CSVs
    # ------------------------------------------------------------------
    print("\n[Stage 3/3] Model export")
    run([python, "research/exportModel.py"])

    default_model_dir = Path("models")
    target_model_dir = Path(cfg.model_dir)
    if target_model_dir.resolve() != default_model_dir.resolve():
        import shutil
        for artefact in ["transformer.pt", "feature_scaler.csv", "target_scaler.csv"]:
            src = default_model_dir / artefact
            if src.exists():
                shutil.copy2(src, target_model_dir / artefact)
                print(f"Copied {src} → {target_model_dir / artefact}")

    print(f"\n[Pipeline complete] Artefacts written to {cfg.model_dir}/")
    print("  transformer.pt")
    print("  feature_scaler.csv")
    print("  target_scaler.csv")


if __name__ == "__main__":
    main()
