"""Canonical local paths for APP checks; original hospital data stays in data/sysu."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYSU_DATASET = PROJECT_ROOT / 'data/processed/sysu/rebuilt-20261009'
SYSU_BASELINE_EVIDENCE = PROJECT_ROOT / 'outputs/experiments/sysu/raw-rebuild-20261009/completed_evidence'
