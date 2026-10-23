"""Load inference dependencies and project source for the mobile service."""
import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
WORKTREE = APP_DIR.parents[1]
sys.path.insert(0, str(APP_DIR / '.deps' / 'inference'))
sys.path.insert(0, str(WORKTREE / 'src'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
