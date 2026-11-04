"""Tier 1 rhythm panel: six rhythm findings judged by Jev from 3 overlapping 5-s windows.

One Jev request per ECG: the state holds the 3 windows plus whole-record rhythm evidence, and
there is one yes/no (noul) question per finding. Jev answers each question independently, so six
findings cost about the same as one.

Labels: a finding is present when its SCP code is present (rhythm codes carry likelihood 0.0,
which in PTB-XL means "not rated", not "absent").
"""

import math

import numpy as np
from typesafe_sdk import Noul

FINDINGS = {                      # key -> (PTB-XL code, plain name)
    "af": ("AFIB", "atrial fibrillation"),
    "pvc": ("PVC", "ventricular extra beats (PVCs)"),
    "pac": ("PAC", "atrial extra beats (PACs)"),
    "brady": ("SBRAD", "sinus bradycardia"),
    "tachy": ("STACH", "sinus tachycardia"),
    "sinus_arrhythmia": ("SARRH", "sinus arrhythmia"),
}
PANEL_VERSION = "tier1-v1"


def labels(scp_codes: dict) -> dict[str, int]:
    return {k: int(code in scp_codes) for k, (code, _) in FINDINGS.items()}

# ---------------- buckets (cut-offs from training folds 1-2) ----------------

def bucket_rate60(hr: float) -> str:
    if hr is None or math.isnan(hr):
        return "unknown"
    return "slow (under 60/min)" if hr < 60 else "fast (over 100/min)" if hr > 100 else "normal (60-100/min)"


def bucket_early_beats(n_early: int, similarity: float) -> str:
    """PAC similarity median 0.89, PVC median 0.15 (AUROC 0.94 on folds 1-2)."""
    if n_early == 0:
        return "no early beats"
    count = "1 early beat" if n_early == 1 else f"{n_early} early beats"
    if similarity is None or math.isnan(similarity):
        return f"{count}, shape cannot be compared"
    shape = ("same shape as normal beats" if similarity >= 0.75
             else "somewhat different shape" if similarity >= 0.5 else "clearly different shape from normal beats")
    return f"{count}, {shape}"


def panel_state(windows: list[dict], whole: dict) -> dict:
    """windows: from jev.window_state (with 60/100 rate buckets); whole: whole-record evidence."""
    return {"segments": windows, "whole_record": whole}

# ---------------- questions ----------------

def _usable(i):
    return Noul(instructions=f"Is `segments[{i}]` clean enough to judge the heart rhythm?",
                criteria={"true": "noise is low or moderate and there are enough beats",
                          "false": "noise is high or there are too few beats"})


CRITERIA = {
    "af": ("Does this ECG show atrial fibrillation?",
           {"true": "rhythm is irregular in usable segments and `whole_record` says 'irregularly irregular (no pattern)' "
                    "or 'inconsistent or absent P waves'",
            "false": "rhythm is regular, or the irregularity is explained by early beats with pauses, or P waves are "
                     "consistent, or irregularity appears only where noise is high"}),
    "pvc": ("Does this ECG show ventricular extra beats (PVCs)?",
            {"true": "`whole_record.early_beats` reports early beats with a clearly or somewhat different shape from "
                     "normal beats",
             "false": "no early beats, or early beats with the same shape as normal beats"}),
    "pac": ("Does this ECG show atrial extra beats (PACs)?",
            {"true": "`whole_record.early_beats` reports early beats with the same shape as normal beats, in an "
                     "otherwise regular rhythm with consistent P waves",
             "false": "no early beats, or early beats with a different shape, or the rhythm is irregularly irregular"}),
    "brady": ("Does this ECG show sinus bradycardia (a regular sinus rhythm under 60 beats per minute)?",
              {"true": "`whole_record.heart_rate` is slow (under 60/min) and P waves are consistent",
               "false": "heart rate is normal or fast, or the rhythm is not sinus (e.g. irregularly irregular)"}),
    "tachy": ("Does this ECG show sinus tachycardia (a regular sinus rhythm over 100 beats per minute)?",
              {"true": "`whole_record.heart_rate` is fast (over 100/min), the rhythm is regular and P waves are consistent",
               "false": "heart rate is normal or slow, or the rhythm is irregularly irregular or lacks consistent P waves"}),
    "sinus_arrhythmia": ("Does this ECG show sinus arrhythmia (rate varying gently, with normal P waves before every beat)?",
                         {"true": "rhythm is slightly irregular or irregular, P waves are consistent, and the irregularity "
                                  "is not explained by early beats",
                          "false": "rhythm is regular, or irregularity comes from early beats, or P waves are "
                                   "inconsistent or absent"}),
}


def panel_questions(n_segments: int = 3) -> dict:
    q = {f"seg{i}_usable": _usable(i) for i in range(n_segments)}
    for k, (instr, crit) in CRITERIA.items():
        q[f"has_{k}"] = Noul(instructions=instr, criteria=crit)
    return q

# ---------------- code-rule baselines (continuous scores for AUROC) ----------------

def rule_scores(f: dict) -> dict[str, float]:
    """Simple deterministic rules on the same numbers, scaled to 0..1 (0.5 = the rule's cut-off)."""
    hr = f["hr"] if not math.isnan(f["hr"]) else 75.0
    sim = f["early_sim"] if not math.isnan(f["early_sim"]) else 1.0
    has_early = float(f["n_early"] > 0)
    pw = f["p_cons"] if not math.isnan(f["p_cons"]) else 0.5
    expl = f["explained"] if not math.isnan(f["explained"]) else 1.0
    irr = f["irregularity"]
    sig = lambda z: float(1 / (1 + np.exp(-z)))
    return {
        "af": float(np.clip(irr / 0.3, 0, 1)),
        "pvc": has_early * float(np.clip((1 - sim) / 1.0, 0, 1)) * (0.5 / 0.5),
        "pac": has_early * float(np.clip(sim, 0, 1)),
        "brady": sig((60 - hr) / 3),
        "tachy": sig((hr - 100) / 3),
        "sinus_arrhythmia": float(np.clip(irr / 0.24, 0, 1)) * pw * (1 - expl if f["n_early"] > 0 else 1.0),
    }
