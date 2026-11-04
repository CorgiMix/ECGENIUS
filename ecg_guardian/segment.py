"""Sliding-window segmentation."""

from .config import RECORD_SECONDS


def windows(win: float, step: float, total: float = RECORD_SECONDS) -> list[tuple[float, float]]:
    """Window (start, end) times in seconds. windows(5, 2.5) -> [(0,5), (2.5,7.5), (5,10)]."""
    out, start = [], 0.0
    while start + win <= total + 1e-9:
        out.append((start, start + win))
        start += step
    return out


def to_samples(start: float, end: float, fs: int) -> slice:
    return slice(int(round(start * fs)), int(round(end * fs)))


def label(start: float, end: float) -> str:
    fmt = lambda t: f"{t:g}"
    return f"{fmt(start)}-{fmt(end)} s"
