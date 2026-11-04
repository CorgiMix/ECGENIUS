"""Jev as the AF predictor: bucket features into words, ask Jev, cache every answer.

Jev's documented weak spots: arithmetic, long irrelevant state, literal reading, first-option bias.
So: no raw floats in the state, short state, explicit criteria, option order can be shuffled.

Two request styles:
- "record":   one state for the whole 10 s  -> record-level AF question.
- "segments": 3 overlapping 5-s windows     -> per-window usable + rhythm, plus record-level AF.
"""

import asyncio
import hashlib
import json
import math
from pathlib import Path

from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul

from .config import JEV_MODEL

QUESTIONS_VERSION = "v2-jev-predicts"
EVIDENCE_VERSION = "v3-pwave-pattern-whole-record"

# ---------------- buckets (fixed in code, before looking at fold 10) ----------------

def bucket_regularity(f: dict) -> str:
    if f["n_beats"] < 4 or math.isnan(f["rr_cv"]):
        return "too few beats to judge"
    if f["rr_cv"] >= 0.12 or f["rr_rmssd_norm"] >= 0.15:
        return "irregular"
    if f["rr_cv"] >= 0.06 or f["rr_rmssd_norm"] >= 0.08:
        return "slightly irregular"
    return "regular"


def bucket_rate(f: dict) -> str:
    hr = f["hr_bpm"]
    if hr is None or math.isnan(hr):
        return "unknown"
    return "slow (under 50/min)" if hr < 50 else "fast (over 100/min)" if hr > 100 else "normal (50-100/min)"


def bucket_noise(n_bad: int) -> str:
    return "low" if n_bad == 0 else "moderate" if n_bad <= 2 else "high"


def bucket_p_waves(c: float) -> str:
    """Cut-offs from fold-1 training data: sinus median 0.95, AF median 0.71."""
    if c is None or math.isnan(c):
        return "cannot assess"
    return "consistent P waves" if c >= 0.90 else "mostly consistent P waves" if c >= 0.75 else "inconsistent or absent P waves"


def bucket_pattern(explained: float) -> str:
    """Cut-offs from folds 1-2 irregular records: AF median 0.0, extra beats median 0.8."""
    if explained is None or math.isnan(explained):
        return "cannot assess"
    if explained >= 0.6:
        return "irregularity explained by early beats with pauses"
    if explained >= 0.25:
        return "partly explained by early beats"
    return "irregularly irregular (no pattern)"


def window_state(window: str, rhythm: dict, bad_leads: list[str], p_consistency: float | None = None,
                 pattern: float | None = None) -> dict:
    """p_consistency / pattern are whole-record values (a 5-s window has too few beats to judge
    them); they are attached to every window in the 'extra evidence' (+pwave) variant only."""
    state = {
        "window": window,
        "rhythm_regularity": bucket_regularity(rhythm),
        "heart_rate": bucket_rate(rhythm),
    }
    if p_consistency is not None:
        state["p_waves"] = bucket_p_waves(p_consistency)
    if pattern is not None and state["rhythm_regularity"] in ("irregular", "slightly irregular"):
        state["irregularity_pattern"] = bucket_pattern(pattern)
    state["noise"] = bucket_noise(len(bad_leads))
    state["bad_leads"] = ", ".join(bad_leads) if bad_leads else "none"
    return state


def record_state(whole: dict) -> dict:
    """State for the whole-record arm: one 10-s window."""
    return {"ecg": whole}


def segments_state(segments: list[dict]) -> dict:
    return {"segments": segments}

# ---------------- questions ----------------

RHYTHM_OPTIONS = {
    "af": "Irregular rhythm with enough beats, in a signal clean enough to trust it. An 'irregular' "
          "rhythm in a window with high noise does NOT count as AF.",
    "not_af": "Regular or only slightly irregular rhythm with enough beats, in a usable signal.",
    "cant_tell": "Too few beats to judge, or noise is high.",
}
AF_CRITERIA = {
    "true": "the rhythm is irregular with enough beats, and noise is not high where it is irregular",
    "false": "the rhythm is regular or slightly irregular, or irregularity appears only where noise is high",
}


RHYTHM_OPTIONS_P = {
    "af": "Irregular rhythm that is 'irregularly irregular (no pattern)', or irregular with inconsistent or "
          "absent P waves, in a signal clean enough to trust.",
    "not_af": "Regular rhythm, or irregularity explained by early beats with pauses, or irregular rhythm with "
              "consistent P waves, in a usable signal.",
    "cant_tell": "Too few beats to judge, or noise is high.",
}
AF_CRITERIA_P = {
    "true": "irregular rhythm that is irregularly irregular (no pattern) or lacks consistent P waves, and noise "
            "is not high there",
    "false": "regular rhythm, or irregularity explained by early beats with pauses, or consistent P waves before "
             "the beats, or irregularity only where noise is high",
}


def _ordered(options: dict, order: list[str] | None) -> dict:
    return {k: options[k] for k in (order or list(options))}


def record_questions(rhythm_order=None, pwave=False) -> dict:
    rhythm, af = (RHYTHM_OPTIONS_P, AF_CRITERIA_P) if pwave else (RHYTHM_OPTIONS, AF_CRITERIA)
    return {
        "record_af": Noul(instructions="Does `ecg` show atrial fibrillation?", criteria=af),
        "record_rhythm": Choice(instructions="What rhythm does `ecg` show?",
                                criteria=_ordered(rhythm, rhythm_order)),
        "record_usable": Noul(instructions="Is `ecg` clean enough to judge the heart rhythm?",
                              criteria={"true": "noise is low or moderate and there are enough beats",
                                        "false": "noise is high or there are too few beats"}),
    }


def segment_questions(n_segments: int, rhythm_order=None, pwave=False) -> dict:
    rhythm, af = (RHYTHM_OPTIONS_P, AF_CRITERIA_P) if pwave else (RHYTHM_OPTIONS, AF_CRITERIA)
    q = {}
    for i in range(n_segments):
        seg = f"`segments[{i}]`"
        q[f"seg{i}_usable"] = Noul(instructions=f"Is {seg} clean enough to judge the heart rhythm?",
                                   criteria={"true": "noise is low or moderate and there are enough beats",
                                             "false": "noise is high or there are too few beats"})
        q[f"seg{i}_rhythm"] = Choice(instructions=f"What rhythm does {seg} show?",
                                     criteria=_ordered(rhythm, rhythm_order))
    q["record_af"] = Noul(instructions="Taking all `segments` together, does this ECG show atrial fibrillation?",
                          criteria=af)
    return q

# ---------------- calling Jev with an on-disk cache ----------------

def _key(state: dict, tag: str) -> str:
    blob = json.dumps({"s": state, "q": tag, "m": JEV_MODEL}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _answers_to_dict(resp) -> dict:
    out = {"model": resp.model, "input_tokens": resp.usage.input_tokens}
    for name, a in resp.nouls.items():
        out[name] = round(float(a.noul), 4)
    for name, a in resp.choices.items():
        out[name] = a.choice
        out[f"{name}_probs"] = {k: round(float(v), 4) for k, v in a.probabilities.items()}
    return out


async def ask_many(items: list[tuple[int, dict]], style: str, cache_file: Path, concurrency: int = 32,
                   rhythm_order=None, pwave=False, log=print, progress=None,
                   questions_fn=None, tag: str | None = None) -> dict[int, dict]:
    """items = [(ecg_id, state)], style = 'record' or 'segments'. Returns {ecg_id: answers}.

    progress: optional callable(n_done) for live progress display.
    questions_fn(state) -> questions, with `tag` naming that question set in the cache key: lets
    other panels (e.g. rhythm_panel) reuse the caching and concurrency here.
    """
    if questions_fn is None:
        tag = f"{QUESTIONS_VERSION}|{style}|{rhythm_order}" + (f"|pwave|{EVIDENCE_VERSION}" if pwave else "")
        questions_fn = (lambda st: record_questions(rhythm_order, pwave)) if style == "record" else \
                       (lambda st: segment_questions(len(st["segments"]), rhythm_order, pwave))
    assert tag, "a custom questions_fn needs a cache tag"
    cache = {}
    if cache_file.exists():
        for line in cache_file.read_text(encoding="utf-8").splitlines():
            row = json.loads(line); cache[row["key"]] = row["answers"]
    results, todo = {}, []
    for ecg_id, state in items:
        k = _key(state, tag)
        if k in cache:
            results[ecg_id] = cache[k]
        else:
            todo.append((ecg_id, state, k))
    log(f"jev[{style}]: {len(results)} cached, {len(todo)} to request")
    if not todo:
        return results
    sem = asyncio.Semaphore(concurrency)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncTypeSafeClient(model=JEV_MODEL) as client:
        async def one(ecg_id, state, k):
            async with sem:
                resp = await client.system_one(state=state, questions=questions_fn(state))
                return ecg_id, k, _answers_to_dict(resp)
        with cache_file.open("a", encoding="utf-8") as f:
            for coro in asyncio.as_completed([one(*t) for t in todo]):
                ecg_id, k, ans = await coro
                results[ecg_id] = ans
                f.write(json.dumps({"key": k, "ecg_id": int(ecg_id), "answers": ans}) + "\n")
                if progress:
                    progress(1)
    return results


def segments_af_prob(answers: dict, n_segments: int) -> float:
    """P(AF) combined in code from Jev's per-window answers (usable-weighted mean of P(af)).

    Overlapping windows are not independent votes, so a weighted mean is used rather than a count.
    If no window is usable, fall back to 0.5 (no information).
    """
    num = den = 0.0
    for i in range(n_segments):
        u = answers[f"seg{i}_usable"]
        num += u * answers[f"seg{i}_rhythm_probs"].get("af", 0.0)
        den += u * (1 - answers[f"seg{i}_rhythm_probs"].get("cant_tell", 0.0))
    return float(num / den) if den > 1e-6 else 0.5
