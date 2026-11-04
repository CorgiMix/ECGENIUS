"""Tier 2 conduction panel: five conduction findings judged by Jev from measured beat shape.

Same request design as the rhythm panel: 3 overlapping windows (noise per window) plus a
`beat_shape` block measured on the median beat, and one yes/no question per finding.

Labels: diagnostic SCP codes with likelihood >= 50 (almost all are 100 in PTB-XL).
Bucket cut-offs below were set on training folds 1-2 (scratch validation, 2026-11-03):
  QRS width: normal median 92 ms, RBBB 154 ms, LBBB 180 ms (LBBB AUROC 0.98)
  V1 net direction: RBBB +0.88, LBBB -0.97 (AUROC 0.97)
  V1 late R': incomplete RBBB 0.19 mV, normal 0.005 mV (AUROC 0.96)
  PR: normal median 144 ms, 1st-degree AV block 208 ms (AUROC 0.78; measure reads ~10 ms short)
  Axis: normal +26 deg, LAFB -54 deg (AUROC 0.82)
"""

import math

import numpy as np
from typesafe_sdk import Noul

FINDINGS = {
    "clbbb": ("CLBBB", "complete left bundle branch block"),
    "crbbb": ("CRBBB", "complete right bundle branch block"),
    "irbbb": ("IRBBB", "incomplete right bundle branch block"),
    "avb1": ("1AVB", "first degree AV block"),
    "lafb": ("LAFB", "left anterior fascicular block"),
}
PANEL_VERSION = "tier2-v1"


def labels(scp_codes: dict) -> dict[str, int]:
    return {k: int(scp_codes.get(code, 0) >= 50) for k, (code, _) in FINDINGS.items()}


def _nan(v):
    return v is None or (isinstance(v, float) and math.isnan(v))

# ---------------- buckets ----------------

def bucket_qrs(ms):
    if _nan(ms):
        return "cannot measure"
    return "normal (under 110 ms)" if ms < 110 else "borderline (110-119 ms)" if ms < 120 else "wide (120 ms or more)"


def bucket_pr(ms):
    if _nan(ms):
        return "cannot measure (no clear P wave)"
    return "short (under 120 ms)" if ms < 120 else "normal (120-190 ms)" if ms <= 190 else "long (over 190 ms)"


def bucket_axis(deg):
    if _nan(deg):
        return "cannot measure"
    if deg < -45:
        return "marked left axis (beyond -45 degrees)"
    if deg < -30:
        return "mild left axis (-30 to -45 degrees)"
    return "right axis (beyond +90 degrees)" if deg > 90 else "normal axis (-30 to +90 degrees)"


def bucket_v1_direction(net):
    if _nan(net):
        return "cannot measure"
    return "mostly upward" if net > 0.3 else "mostly downward" if net < -0.3 else "mixed"


def bucket_v1_late_r(mv):
    if _nan(mv):
        return "cannot measure"
    return "late upward peak (R-prime) present" if mv >= 0.10 else "no late upward peak"


def bucket_terminal_s(i_term, v6_term):
    if _nan(i_term) or _nan(v6_term):
        return "cannot measure"
    return "broad terminal S wave in I and V6" if (i_term + v6_term) / 2 < -0.05 else "no terminal S wave"


def bucket_septal_q(i_init, v6_init):
    if _nan(i_init) or _nan(v6_init):
        return "cannot measure"
    return "small q waves present in I/V6" if min(i_init, v6_init) < -0.04 else "no q waves in I/V6"


def beat_shape(m: dict) -> dict:
    return {
        "qrs_width": bucket_qrs(m["qrs_ms"]),
        "pr_interval": bucket_pr(m["pr_ms"]),
        "heart_axis": bucket_axis(m["axis_deg"]),
        "v1_main_direction": bucket_v1_direction(m["v1_net"]),
        "v1_late_peak": bucket_v1_late_r(m["v1_late_r"]),
        "terminal_s_in_I_V6": bucket_terminal_s(m["i_terminal"], m["v6_terminal"]),
        "septal_q_waves": bucket_septal_q(m["i_initial"], m["v6_initial"]),
    }


def panel_state(windows: list[dict], m: dict) -> dict:
    """windows: noise per 5-s window (only noise and bad leads matter for conduction)."""
    return {"segments": [{"window": w["window"], "noise": w["noise"], "bad_leads": w["bad_leads"]} for w in windows],
            "beat_shape": beat_shape(m)}

# ---------------- questions ----------------

CRITERIA = {
    "clbbb": ("Does this ECG show complete left bundle branch block?",
              {"true": "`beat_shape.qrs_width` is wide, `v1_main_direction` is mostly downward, and there are no septal q waves in I/V6",
               "false": "QRS width is normal or borderline, or V1 is mostly upward, or septal q waves are present"}),
    "crbbb": ("Does this ECG show complete right bundle branch block?",
              {"true": "`beat_shape.qrs_width` is wide, and V1 shows a late upward peak (R-prime) or is mostly upward, "
                       "usually with a broad terminal S wave in I and V6",
               "false": "QRS width is normal or borderline, or V1 is mostly downward without a late upward peak"}),
    "irbbb": ("Does this ECG show incomplete right bundle branch block?",
              {"true": "V1 shows a late upward peak (R-prime) while `beat_shape.qrs_width` is normal or borderline (not wide)",
               "false": "no late upward peak in V1, or the QRS is wide (that would be complete block)"}),
    "avb1": ("Does this ECG show first degree AV block?",
             {"true": "`beat_shape.pr_interval` is long (over 190 ms)",
              "false": "PR interval is normal or short, or cannot be measured"}),
    "lafb": ("Does this ECG show left anterior fascicular block?",
             {"true": "`beat_shape.heart_axis` is marked left axis (beyond -45 degrees) and the QRS is not wide from left bundle branch block",
              "false": "axis is normal, right, or only mildly left"}),
}


def panel_questions(n_segments: int = 3) -> dict:
    q = {f"seg{i}_usable": Noul(instructions=f"Is `segments[{i}]` clean enough to measure beat shape?",
                                criteria={"true": "noise is low or moderate", "false": "noise is high"})
         for i in range(n_segments)}
    for k, (instr, crit) in CRITERIA.items():
        q[f"has_{k}"] = Noul(instructions=instr, criteria=crit)
    return q

# ---------------- code-rule baselines ----------------

def rule_scores(m: dict) -> dict[str, float]:
    sig = lambda z: float(1 / (1 + np.exp(-z)))
    g = lambda k, d: d if _nan(m[k]) else m[k]
    qrs, pr, ax = g("qrs_ms", 90.0), g("pr_ms", 150.0), g("axis_deg", 30.0)
    v1n, v1r = g("v1_net", 0.0), g("v1_late_r", 0.0)
    q_present = min(g("i_initial", 0.0), g("v6_initial", 0.0)) < -0.04
    wide = sig((qrs - 120) / 6)
    return {
        "clbbb": wide * sig(-v1n / 0.15) * (0.3 if q_present else 1.0),
        "crbbb": wide * max(sig((v1r - 0.10) / 0.04), sig(v1n / 0.15)),
        "irbbb": sig((v1r - 0.10) / 0.04) * (1 - wide),
        "avb1": sig((pr - 190) / 8),
        "lafb": sig((-ax - 45) / 5) * (1 - wide * sig(-v1n / 0.15)),
    }
