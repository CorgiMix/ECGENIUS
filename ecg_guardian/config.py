"""Paths, constants and secrets. Secrets come from .env locally or real env vars (e.g. Railway)."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
# .env wins over machine-wide variables (a stale GEMINI_API_KEY exists in this user's Windows env).
# On Railway there is no .env file, so Railway's variables are used unchanged.
load_dotenv(PROJECT_DIR / ".env", override=True)

PTBXL_DIR = Path(os.environ.get("PTBXL_DIR", "C:/data/ptb-xl"))
CACHE_DIR = Path(os.environ.get("ECG_CACHE_DIR", "C:/data/ecg-guardian-cache"))
RESULTS_DIR = PROJECT_DIR / "results"

LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
FS_LO, FS_HI = 100, 500          # model signals / quality signals
RECORD_SECONDS = 10

# Sliding windows (seconds). Rhythm: 5 s, 50% overlap -> 3 windows. Quality: 2 s, no overlap -> 5.
RHYTHM_WIN, RHYTHM_STEP = 5.0, 2.5
QUALITY_WIN, QUALITY_STEP = 2.0, 2.0

JEV_MODEL = "jev-1.14.0"          # pinned for reproducibility
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
