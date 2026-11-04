"""Per-window rhythm and signal-quality features. All numbers are computed here, in code.

Rhythm (100 Hz): R peaks are detected on the whole 10-s record, then RR intervals are taken only
from consecutive peaks that both fall inside a window, so beats cut at window edges never create
fake irregularity.

Quality (500 Hz, per lead, per 2-s window): baseline wander, high-frequency noise, mains (50 Hz),
flat line, clipping. Filters run on the whole record, then values are read per window, which
avoids filter edge effects inside short windows.
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, sosfiltfilt

from .config import FS_HI, FS_LO, QUALITY_STEP, QUALITY_WIN, RHYTHM_STEP, RHYTHM_WIN
from .segment import to_samples, windows

RHYTHM_WINDOWS = windows(RHYTHM_WIN, RHYTHM_STEP)     # 3 x 5 s
QUALITY_WINDOWS = windows(QUALITY_WIN, QUALITY_STEP)  # 5 x 2 s
QUALITY_KEYS = ("baseline_mv", "hf_mv", "mains_mv", "flat", "clipped")

_BP_QRS = butter(2, [5, 20], btype="band", fs=FS_LO)
_SOS_BASE = butter(2, 0.7, btype="low", fs=FS_HI, output="sos")
_SOS_HF = butter(4, 40, btype="high", fs=FS_HI, output="sos")
_SOS_MAINS = butter(2, [48, 52], btype="band", fs=FS_HI, output="sos")


# ---------------- rhythm ----------------

def r_peaks(x_lo: np.ndarray) -> np.ndarray:
    """R-peak sample indices from a (1000, 12) 100-Hz record, using leads I, II and V2-V5 combined."""
    leads = x_lo[:, [0, 1, 7, 8, 9, 10]]
    y = filtfilt(*_BP_QRS, leads, axis=0)
    energy = (y ** 2).sum(axis=1)
    energy = np.convolve(energy, np.ones(8) / 8, mode="same")       # ~80 ms smoothing
    if energy.max() <= 0:
        return np.array([], dtype=int)
    peaks, _ = find_peaks(energy, distance=int(0.28 * FS_LO), height=0.25 * np.percentile(energy, 99))
    return peaks


def rhythm_window_features(peaks: np.ndarray, start: float, end: float) -> dict:
    t = peaks / FS_LO
    inside = t[(t >= start) & (t < end)]
    rr = np.diff(inside)
    if len(rr) < 3:
        return {"n_beats": int(len(inside)), "hr_bpm": np.nan, "rr_cv": np.nan, "rr_rmssd_norm": np.nan}
    mean_rr = rr.mean()
    return {
        "n_beats": int(len(inside)),
        "hr_bpm": float(60.0 / mean_rr),
        "rr_cv": float(rr.std() / mean_rr),                                  # overall spread
        "rr_rmssd_norm": float(np.sqrt(np.mean(np.diff(rr) ** 2)) / mean_rr),  # beat-to-beat change
    }


def rr_pattern(peaks: np.ndarray, start: float = 0.0, end: float = 10.0) -> float:
    """Share of RR intervals within 15% of the median RR (0..1). NaN with fewer than 4 intervals.

    The textbook way to tell extra beats from AF: with occasional early beats (PVC/PAC) most
    intervals still sit at the underlying rate and only a few are off; in AF there is no
    underlying rate, so intervals scatter ("irregularly irregular").
    """
    t = peaks / FS_LO
    rr = np.diff(t[(t >= start) & (t < end)])
    if len(rr) < 4:
        return float("nan")
    return float(np.mean(np.abs(rr - np.median(rr)) <= 0.15 * np.median(rr)))


def early_beat_explained(peaks: np.ndarray, start: float = 0.0, end: float = 10.0) -> float:
    """Share of off-rate RR intervals explained by early beats (0..1). NaN with < 4 intervals.

    An early beat (PVC/PAC) gives a SHORT interval followed by a LONG compensatory pause whose sum is
    about two normal intervals. In AF there is no such pairing. Validated on folds 1-2 (irregular
    records only): AUROC 0.82 for extra beats vs AF; median 0.8 for extra beats, 0.0 for AF.
    Returns 1.0 when no interval is off-rate (nothing to explain).
    """
    t = peaks / FS_LO
    rr = np.diff(t[(t >= start) & (t < end)])
    if len(rr) < 4:
        return float("nan")
    med = np.median(rr)
    off = np.abs(rr - med) > 0.15 * med
    if not off.any():
        return 1.0
    used = np.zeros(len(rr), bool)
    for i in range(len(rr) - 1):
        if (not used[i] and rr[i] < 0.85 * med and rr[i + 1] > 1.1 * med
                and abs(rr[i] + rr[i + 1] - 2 * med) <= 0.2 * med):
            used[i] = used[i + 1] = True
    return float((used & off).sum() / off.sum())


def early_beat_shape(x_hi: np.ndarray, peaks_lo: np.ndarray) -> tuple[int, float]:
    """(number of early beats, lowest shape similarity of an early beat to the normal-beat template).

    Early beat = its preceding RR is < 85% of the median RR. Shape = R-60 ms .. R+100 ms in all 12
    leads (500 Hz), correlated with the median of the normal beats. PVCs (ventricular) look different
    (low similarity); PACs (atrial) keep the normal QRS shape (high similarity).
    Returns (0, nan) when there are no early beats or too few normal beats for a template.
    """
    r = (peaks_lo * (FS_HI // FS_LO)).astype(int)
    a, b = int(0.06 * FS_HI), int(0.10 * FS_HI)
    r = r[(r - a >= 0) & (r + b <= len(x_hi))]
    if len(r) < 4:
        return 0, float("nan")
    rr = np.diff(r).astype(float)
    med = np.median(rr)
    early = np.where(rr < 0.85 * med)[0] + 1               # index of the beat that came early
    normal = np.where(np.abs(rr - med) <= 0.15 * med)[0] + 1
    if len(early) == 0 or len(normal) < 2:
        return int(len(early)), float("nan")
    beats = np.stack([x_hi[p - a:p + b].ravel() for p in r])
    beats = beats - beats.mean(axis=1, keepdims=True)
    tmpl = np.median(beats[normal], axis=0)
    sim = (beats[early] @ tmpl) / (np.linalg.norm(beats[early], axis=1) * np.linalg.norm(tmpl) + 1e-9)
    return int(len(early)), float(sim.min())


def rate_gradualness(peaks: np.ndarray) -> float:
    """RMSSD / (sqrt(2) * SD) of RR intervals. ~1 when changes are random (AF), well below 1 when the
    rate drifts smoothly (sinus arrhythmia with breathing). NaN with fewer than 4 intervals."""
    rr = np.diff(peaks / FS_LO)
    if len(rr) < 4 or rr.std() == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.diff(rr) ** 2)) / (np.sqrt(2) * rr.std()))


def rhythm_features(x_lo: np.ndarray) -> list[dict]:
    peaks = r_peaks(x_lo)
    return [rhythm_window_features(peaks, s, e) for s, e in RHYTHM_WINDOWS]


# ---------------- P waves (500 Hz) ----------------

_SOS_P = butter(2, [1, 15], btype="band", fs=FS_HI, output="sos")
P_LEADS = (1, 6)                     # II and V1: where P waves (and AF f-waves) show best
P_START, P_END = 0.25, 0.08          # seconds before the R peak


def p_wave_consistency(x_hi: np.ndarray, peaks_lo: np.ndarray, start: float = 0.0, end: float = 10.0) -> float:
    """How alike the pre-beat segments are across beats (0..1). High = consistent P waves.

    For each beat whose pre-beat segment lies inside [start, end), take R-250 ms .. R-80 ms in leads
    II and V1, correlate it with the median template, and average. NaN if fewer than 3 usable beats.
    """
    y = sosfiltfilt(_SOS_P, x_hi[:, list(P_LEADS)], axis=0)
    r = (peaks_lo * (FS_HI // FS_LO)).astype(int)
    a, b = int(P_START * FS_HI), int(P_END * FS_HI)
    segs = [y[p - a:p - b].ravel() for p in r if p - a >= start * FS_HI and p <= end * FS_HI and p - a >= 0]
    if len(segs) < 3:
        return float("nan")
    s = np.stack(segs)
    s = s - s.mean(axis=1, keepdims=True)
    tmpl = np.median(s, axis=0)
    denom = np.linalg.norm(s, axis=1) * np.linalg.norm(tmpl) + 1e-9
    return float(np.clip((s @ tmpl) / denom, -1, 1).mean())


# ---------------- quality ----------------

def quality_features(x_hi: np.ndarray) -> np.ndarray:
    """(n_quality_windows, 12 leads, 5 features) array for a (5000, 12) 500-Hz record."""
    base = sosfiltfilt(_SOS_BASE, x_hi, axis=0)
    hf = sosfiltfilt(_SOS_HF, x_hi, axis=0)
    mains = sosfiltfilt(_SOS_MAINS, x_hi, axis=0)
    out = np.zeros((len(QUALITY_WINDOWS), x_hi.shape[1], len(QUALITY_KEYS)), dtype=np.float32)
    for i, (s, e) in enumerate(QUALITY_WINDOWS):
        sl = to_samples(s, e, FS_HI)
        raw = x_hi[sl]
        rng = raw.max(axis=0) - raw.min(axis=0)
        at_edge = (raw >= raw.max(axis=0) - 0.01 * rng) | (raw <= raw.min(axis=0) + 0.01 * rng)
        still = np.vstack([np.zeros((1, raw.shape[1]), bool), np.diff(raw, axis=0) == 0])
        out[i, :, 0] = base[sl].std(axis=0)
        out[i, :, 1] = hf[sl].std(axis=0)
        out[i, :, 2] = mains[sl].std(axis=0)
        out[i, :, 3] = raw.std(axis=0) < 0.01                           # flat line (mV)
        out[i, :, 4] = (at_edge & still).mean(axis=0) > 0.05            # stuck at the rails
    return out


def bad_lead_mask(q: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    """(n_quality_windows, 12) bool: lead is unusable in that 2-s window."""
    return (
        (q[..., 0] > thresholds["baseline_mv"])
        | (q[..., 1] > thresholds["hf_mv"])
        | (q[..., 2] > thresholds["mains_mv"])
        | (q[..., 3] > 0.5)
        | (q[..., 4] > 0.5)
    )


def quality_windows_overlapping(start: float, end: float) -> list[int]:
    return [i for i, (s, e) in enumerate(QUALITY_WINDOWS) if s < end and e > start]
