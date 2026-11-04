"""Load PTB-XL waveforms. 100 Hz signals for all records are cached to one .npy (~1 GB)."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import wfdb

from .config import CACHE_DIR, PTBXL_DIR


def read_record(rel_path: str) -> np.ndarray:
    """Return (samples, 12) float32 in mV for one record path like 'records100/00000/00001_lr'."""
    sig, _ = wfdb.rdsamp(str(PTBXL_DIR / rel_path))
    return sig.astype(np.float32)


def load_lo_all(db: pd.DataFrame) -> np.ndarray:
    """All 100 Hz signals as (n_records, 1000, 12), in db.index order, cached on disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / "x100.npy"
    ids_cache = CACHE_DIR / "x100_ids.npy"
    if cache.exists() and ids_cache.exists() and np.array_equal(np.load(ids_cache), db.index.values):
        return np.load(cache, mmap_mode="r")
    with ThreadPoolExecutor(8) as ex:
        x = np.stack(list(ex.map(read_record, db["filename_lr"])))
    np.save(cache, x)
    np.save(ids_cache, db.index.values)
    return x


def load_hi(db: pd.DataFrame) -> list[np.ndarray]:
    """500 Hz signals (5000, 12) for the given rows. Not cached (5 GB for everything)."""
    with ThreadPoolExecutor(8) as ex:
        return list(ex.map(read_record, db["filename_hr"]))
