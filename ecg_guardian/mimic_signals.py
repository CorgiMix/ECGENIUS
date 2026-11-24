"""Load MIMIC-IV-ECG waveforms. Same interface as signals.py but for MIMIC data.

MIMIC-IV-ECG: 500 Hz, 10 s, 12-lead, WFDB format (.hea + .dat).
Files are under: {MIMIC_ECG_DIR}/files/pNNNN/pXXXXXXXX/sZZZZZZZZ/ZZZZZZZZ.{hea,dat}
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import decimate

from .config import FS_HI, FS_LO, LEADS, MIMIC_ECG_DIR

# MIMIC may use slightly different lead names; map them to our standard 12.
_LEAD_MAP = {
    "I": 0, "II": 1, "III": 2, "aVR": 3, "AVR": 3, "avr": 3,
    "aVL": 4, "AVL": 4, "avl": 4, "aVF": 5, "AVF": 5, "avf": 5,
    "V1": 6, "V2": 7, "V3": 8, "V4": 9, "V5": 10, "V6": 11,
    "v1": 6, "v2": 7, "v3": 8, "v4": 9, "v5": 10, "v6": 11,
}


def _reorder_leads(signal: np.ndarray, sig_names: list[str]) -> np.ndarray:
    """Reorder (samples, n_leads) to our standard 12-lead order (I, II, III, aVR, ..., V6)."""
    out = np.zeros((signal.shape[0], 12), dtype=np.float32)
    for i, name in enumerate(sig_names):
        idx = _LEAD_MAP.get(name.strip())
        if idx is not None:
            out[:, idx] = signal[:, i]
    return out


def read_mimic_record_hi(rel_path: str) -> np.ndarray:
    """Return (5000, 12) float32 in mV for one MIMIC-IV-ECG record (500 Hz).
    rel_path: e.g. 'files/p1000/p10001725/s41420867/41420867'
    """
    record = wfdb.rdrecord(str(MIMIC_ECG_DIR / rel_path))
    sig = np.nan_to_num(record.p_signal.astype(np.float32))   # some MIMIC records contain NaN samples
    ordered = _reorder_leads(sig, record.sig_name)
    # Ensure exactly 5000 samples (10 s at 500 Hz)
    if ordered.shape[0] > FS_HI * 10:
        ordered = ordered[: FS_HI * 10]
    elif ordered.shape[0] < FS_HI * 10:
        pad = np.zeros((FS_HI * 10 - ordered.shape[0], 12), dtype=np.float32)
        ordered = np.vstack([ordered, pad])
    return ordered


def read_mimic_record_lo(rel_path: str) -> np.ndarray:
    """Return (1000, 12) float32 in mV for one MIMIC-IV-ECG record (100 Hz).
    Decimates from 500 Hz to 100 Hz."""
    return to_lo(read_mimic_record_hi(rel_path))


def to_lo(hi: np.ndarray) -> np.ndarray:
    """(5000, 12) 500 Hz -> (1000, 12) 100 Hz, anti-aliased."""
    factor = FS_HI // FS_LO  # 5
    lo = decimate(hi, factor, axis=0, zero_phase=True).astype(np.float32)
    # Ensure exactly 1000 samples
    if lo.shape[0] > FS_LO * 10:
        lo = lo[: FS_LO * 10]
    elif lo.shape[0] < FS_LO * 10:
        pad = np.zeros((FS_LO * 10 - lo.shape[0], 12), dtype=np.float32)
        lo = np.vstack([lo, pad])
    return lo


def load_mimic_hi(paths: list[str], workers: int = 8) -> list[np.ndarray]:
    """500 Hz signals (5000, 12) for the given record paths."""
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(read_mimic_record_hi, paths))


def load_mimic_lo(paths: list[str], workers: int = 8) -> list[np.ndarray]:
    """100 Hz signals (1000, 12) for the given record paths."""
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(read_mimic_record_lo, paths))
