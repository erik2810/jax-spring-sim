"""Shared helpers for the example scripts (output directory, styling)."""

from __future__ import annotations

from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "outputs"


def output_path(name: str) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR / name
