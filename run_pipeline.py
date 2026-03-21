#!/usr/bin/env python3
"""
run_pipeline.py — ML pipeline orchestrator.

Executes three stages in order:
  1. Feature engineering  (pipeline.py)    raw OHLCV  →  feature CSVs
  2. Model training       (Train.py)       feature CSVs  →  Model3.pth
  3. Model export         (exportModel.py) Model3.pth →  transformer.pt + scalers

Each stage is a subprocess so failures are caught immediately and the
exit code propagates to the calling shell / container orchestrator.

Usage:
  python run_pipeline.py
  python run_pipeline.py --skip-train          # re-export an existing checkpoint
  python run_pipeline.py --data-dir /data      # override default paths
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list, *, cwd: str | None = None) -> None:
    """Run a command, stream output, and exit on failure."""
    print(f"\n{'='*60}")
    print(f">>> {' '.join(str(c) for c in cmd)}")
    print("=" * 60, flush=True)

    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TradingTransformer ML pipeline orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing raw OHLCV CSVs",
    )
    parser.add_argument(
        "--feature-dir",
        default="features",
        help="Directory to write enriched feature CSVs",
    )
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Directory to write exported model artefacts",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and use the existing checkpoint in research/models/",
    )
    args = parser.parse_args()

    Path(args.feature_dir).mkdir(parents=True, exist_ok=True)
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    python = sys.executable

    # ------------------------------------------------------------------
    # Stage 1 — Feature engineering
    # ------------------------------------------------------------------
    print("\n[Stage 1/3] Feature engineering")
    run([
        python,
        "research/features/pipeline.py",
        args.data_dir,
        "-o", args.feature_dir,
    ])

    # ------------------------------------------------------------------
    # Stage 2 — Training (skippable when re-exporting an existing model)
    # ------------------------------------------------------------------
    if args.skip_train:
        print("\n[Stage 2/3] Training skipped (--skip-train)")
    else:
        print("\n[Stage 2/3] Model training")
        run([python, "research/training/Train.py"])

    # ------------------------------------------------------------------
    # Stage 3 — Export to LibTorch + scaler CSVs
    # ------------------------------------------------------------------
    print("\n[Stage 3/3] Model export")
    run([python, "research/exportModel.py"])

    # exportModel.py writes to ./models/ (relative to CWD).
    # If --model-dir differs, copy the artefacts there.
    default_model_dir = Path("models")
    target_model_dir = Path(args.model_dir)
    if target_model_dir.resolve() != default_model_dir.resolve():
        import shutil
        for artefact in ["transformer.pt", "feature_scaler.csv", "target_scaler.csv"]:
            src = default_model_dir / artefact
            if src.exists():
                shutil.copy2(src, target_model_dir / artefact)
                print(f"Copied {src} → {target_model_dir / artefact}")

    print(f"\n[Pipeline complete] Artefacts written to {args.model_dir}/")
    print("  transformer.pt")
    print("  feature_scaler.csv")
    print("  target_scaler.csv")


if __name__ == "__main__":
    main()
