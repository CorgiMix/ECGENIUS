"""Beat-shape measurements on 500-Hz 12-lead ECG, from a median beat per lead.

Median beat: every normal-interval beat (RR within 20% of the median) is cut from R-300 ms to
R+450 ms after a 0.5-40 Hz band-pass, R peaks refined at 500 Hz, and the per-sample median taken.
Averaging cancels noise and extra beats, so intervals are measured on one clean beat per lead.

All thresholds that turn these numbers into words live in conduction_panel.py, fixed from
training folds before any evaluation.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .config import FS_HI, FS_LO, LEADS

PRE, POST = 0.55, 0.45                          # seconds around R (long pre-window: P wave + baseline)
_SOS = butter(2, [0.5, 40], btype="band", fs=FS_HI, output="sos")
_SOS_ST = butter(2, [0.05, 40], btype="band", fs=FS_HI, output="sos")   # ST/T: 0.5 Hz would distort ST levels
_I, _AVF, _V1, _V6 = LEADS.index("I"), LEADS.index("aVF"), LEADS.index("V1"), LEADS.index("V6")
_II = LEADS.index("II")


def median_beat(x_hi: np.ndarray, peaks_lo: np.ndarray, sos=_SOS) -> np.ndarray | None:
    """(12, n_samples) median beat with R at index PRE*FS_HI, or None with < 3 usable beats."""
    y = sosfiltfilt(sos, x_hi, axis=0)
    energy = np.abs(np.diff(y, axis=0, prepend=y[:1])).sum(axis=1)
    a, b = int(PRE * FS_HI), int(POST * FS_HI)
    r = []
    for p in (peaks_lo * (FS_HI // FS_LO)).astype(int):          # refine R to 500-Hz precision
        lo, hi = max(p - 15, 0), min(p + 15, len(y))
        r.append(lo + int(np.argmax(energy[lo:hi])))
    r = np.array(r)
    if len(r) >= 3:
        rr = np.diff(r)
        ok = np.r_[True, np.abs(rr - np.median(rr)) <= 0.2 * np.median(rr)]   # drop early beats
        r = r[ok]
    r = r[(r - a >= 0) & (r + b <= len(y))]
    if len(r) < 3:
        return None
    beats = np.stack([y[p - a:p + b] for p in r])                # (beats, samples, leads)
    med = np.median(beats, axis=0).T                             # (leads, samples)
    return med - np.median(med[:, : int(0.05 * FS_HI)], axis=1, keepdims=True)  # zero at start


def qrs_bounds(med: np.ndarray) -> tuple[int, int]:
    """QRS onset/offset from the spatial velocity of the median beat (10-ms smoothed, 4% of peak).

    Tuned on folds 1-2: normal QRS median 92 ms, RBBB 154 ms, LBBB 180 ms.
    """
    vel = np.abs(np.diff(med, axis=1)).sum(axis=0)
    vel = np.convolve(vel, np.ones(5) / 5, mode="same")
    rpos = int(PRE * FS_HI)
    peak = vel[rpos - 60: rpos + 60].max()
    thr = 0.04 * peak
    on = rpos
    while on > rpos - int(0.12 * FS_HI) and vel[on - 1] > thr:
        on -= 1
    off = rpos
    while off < rpos + int(0.16 * FS_HI) and vel[off] > thr:
        off += 1
    return on, off


def pr_interval_ms(med: np.ndarray, qrs_on: int) -> float:
    """P onset -> QRS onset in lead II. NaN if no clear P wave in the 60-300 ms before the QRS."""
    lead = med[_II]
    lo, hi = qrs_on - int(0.30 * FS_HI), qrs_on - int(0.04 * FS_HI)
    if lo < 0:
        return float("nan")
    base = np.median(lead[max(lo - 25, 0):lo]) if lo >= 25 else lead[qrs_on]
    seg = lead[lo:hi] - base                                     # relative to the baseline before P
    p_idx = int(np.argmax(np.abs(seg)))
    p_amp = abs(seg[p_idx])
    if p_amp < 0.04:                                             # mV: no visible P wave
        return float("nan")
    k = p_idx
    while k > 0 and abs(seg[k]) > 0.3 * p_amp:                   # walk back to P onset (tuned: 0.3)
        k -= 1
    return float((qrs_on - (lo + k)) / FS_HI * 1000)


def morphology(x_hi: np.ndarray, peaks_lo: np.ndarray) -> dict:
    """Numbers the conduction panel needs. Values are NaN when no median beat can be built."""
    nan = float("nan")
    med = median_beat(x_hi, peaks_lo)
    if med is None:
        return {k: nan for k in ("qrs_ms", "pr_ms", "axis_deg", "v1_net", "v1_terminal", "v1_late_r",
                                 "i_terminal", "v6_terminal", "i_initial", "v6_initial")}
    on, off = qrs_bounds(med)
    qrs = med[:, on:off]
    n_term = max(1, int(0.04 * FS_HI))                           # last / first 40 ms of the QRS
    area = qrs.sum(axis=1) / FS_HI                               # mV*s, net QRS area per lead
    return {
        "qrs_ms": float((off - on) / FS_HI * 1000),
        "pr_ms": pr_interval_ms(med, on),
        "axis_deg": float(np.degrees(np.arctan2(area[_AVF], area[_I]))),
        "v1_net": float(area[_V1] / (np.abs(qrs[_V1]).sum() / FS_HI + 1e-9)),   # -1 all negative .. +1
        "v1_terminal": float(qrs[_V1, -n_term:].mean()),
        "v1_late_r": float(qrs[_V1, qrs.shape[1] // 2:].max()),       # R' (late upward peak) in V1
        "i_terminal": float(qrs[_I, -n_term:].mean()),
        "v6_terminal": float(qrs[_V6, -n_term:].mean()),
        "i_initial": float(qrs[_I, : int(0.03 * FS_HI)].min()),   # septal q wave (negative dip)
        "v6_initial": float(qrs[_V6, : int(0.03 * FS_HI)].min()),
    }


# ---------------- Tier 3: per-lead Q waves, R/S amplitudes, ST level, T waves ----------------

def lead_measures(x_hi: np.ndarray, peaks_lo: np.ndarray) -> dict | None:
    """Per-lead arrays (12 values each, in LEADS order), measured on a 0.05-40 Hz median beat with the
    PR segment as zero. None when no median beat can be built.

    r_amp / s_depth: largest positive / negative QRS deflection (mV)
    q_ms / q_depth: duration (ms) / depth (mV) of an initial negative deflection before any R wave
    st_mv:         level at J + 60 ms (J = QRS offset)
    t_mv:          largest deflection between J + 80 ms and J + 360 ms (sign kept)
    """
    med = median_beat(x_hi, peaks_lo, sos=_SOS_ST)
    if med is None:
        return None
    on, off = qrs_bounds(med)
    base = np.median(med[:, max(on - 20, 0):max(on - 4, 1)], axis=1, keepdims=True)  # PR segment
    m = med - base
    qrs = m[:, on:off]
    q_ms, q_depth = np.zeros(12), np.zeros(12)
    for j in range(12):
        lead = qrs[j]
        k = 0
        while k < len(lead) and lead[k] <= 0.02:          # initial part that never rises above baseline
            k += 1
        q_ms[j] = k / FS_HI * 1000
        q_depth[j] = -lead[:max(k, 1)].min() if k else 0.0
    j60 = min(off + int(0.06 * FS_HI), m.shape[1] - 1)
    t0, t1 = off + int(0.08 * FS_HI), min(off + int(0.36 * FS_HI), m.shape[1])
    tseg = m[:, t0:t1]
    t_idx = np.argmax(np.abs(tseg), axis=1)
    return {
        "r_amp": np.clip(qrs.max(axis=1), 0, None), "s_depth": np.clip(-qrs.min(axis=1), 0, None),
        "q_ms": q_ms, "q_depth": q_depth, "st_mv": m[:, j60],
        "t_mv": tseg[np.arange(12), t_idx], "qrs_ms": (off - on) / FS_HI * 1000,
    }
