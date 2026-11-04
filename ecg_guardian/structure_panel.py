"""Tier 3 structure/ischemia panel: old heart attacks, enlarged heart, ischemia, ST depression.

Same request design as the other panels: 3 windows (noise per window) plus a `lead_findings`
block. Code pre-counts leads and states them in words (Jev is weak at counting and numbers).

Labels: MI / LVH / ischemia codes count as present at likelihood >= 50; a code present with a
LOWER likelihood (15/35 = cardiologist unsure) makes the record UNCERTAIN for that finding, and it
is left out of that finding's scoring. ST depression (STD_) is a form statement: present = yes.

Cut-offs set on training folds 1-2 (scratch validation, 2026-11-03):
  inferior MI: pathological Q (>= 40 ms, or deep relative to R, or QS) in II/III/aVF, AUROC 0.90
  anteroseptal MI: small R waves in V2+V3 (pos median 0.29 mV, neg 1.20 mV), AUROC 0.88
  anterior MI: Q in V3/V4 + R loss, AUROC ~0.70 (weak)
  LVH: Sokolow-Lyon SV1 + max(RV5, RV6) (pos median 3.46 mV, neg 1.98 mV), AUROC 0.92
  ischemia: ST depression + T inversion lead counts, AUROC 0.87
  ST depression: lowest ST level (not aVR), AUROC 0.83
"""

import math

import numpy as np
from typesafe_sdk import Noul

from .config import LEADS

FINDINGS = {
    "imi": (("IMI", "ILMI"), "inferior myocardial infarction (old)"),
    "asmi": (("ASMI",), "anteroseptal myocardial infarction (old)"),
    "ami": (("AMI", "ALMI"), "anterior myocardial infarction (old)"),
    "lvh": (("LVH",), "left ventricular hypertrophy"),
    "isch": (("ISC_", "ISCAL", "ISCAS", "ISCIN", "ISCIL", "ISCLA", "ISCAN"), "ischemic ST-T changes"),
    "std": (("STD_",), "ST depression"),
}
FORM = {"std"}
PANEL_VERSION = "tier3-v1"
_L = {n: i for i, n in enumerate(LEADS)}


def labels(scp_codes: dict) -> dict[str, float]:
    out = {}
    for k, (codes, _) in FINDINGS.items():
        vals = [scp_codes[c] for c in codes if c in scp_codes]
        out[k] = 0.0 if not vals else 1.0 if (k in FORM or max(vals) >= 50) else float("nan")
    return out

# ---------------- per-lead flags ----------------

def path_q(m: dict, lead: str) -> bool:
    j = _L[lead]
    return bool(m["q_ms"][j] >= 40 or (m["q_depth"][j] >= 0.1 and m["q_depth"][j] >= 0.25 * m["r_amp"][j])
                or (m["r_amp"][j] < 0.1 and m["s_depth"][j] >= 0.3))


def _in(leads: list[str], of: list[str], group: str) -> str:
    if not leads:
        return f"none in the {group} leads"
    return f"in {len(leads)} of {len(of)} {group} leads ({', '.join(leads)})"


def numbers(m: dict) -> dict:
    """Pre-computed numbers for the code rules (and nothing Jev sees directly)."""
    j = _L
    st = np.array([m["st_mv"][j[n]] for n in LEADS if n != "aVR"])
    t_leads = ["I", "II", "aVL", "V3", "V4", "V5", "V6"]
    return {
        "inf_q": sum(path_q(m, n) for n in ("II", "III", "aVF")),
        "as_q": sum(path_q(m, n) for n in ("V1", "V2", "V3")),
        "ant_q": sum(path_q(m, n) for n in ("V3", "V4")),
        "r_v2v3": float(m["r_amp"][j["V2"]] + m["r_amp"][j["V3"]]),
        "sokolow": float(m["s_depth"][j["V1"]] + max(m["r_amp"][j["V5"]], m["r_amp"][j["V6"]])),
        "st_min": float(st.min()),
        "st_dep_n": int((st <= -0.05).sum()),
        "t_inv_n": int(sum(m["t_mv"][j[n]] < -0.1 for n in t_leads)),
        "qrs_ms": float(m["qrs_ms"]),
    }


def lead_findings(m: dict | None) -> dict:
    if m is None:
        return {"status": "beat shape cannot be measured"}
    n = numbers(m)
    inf = [x for x in ("II", "III", "aVF") if path_q(m, x)]
    asq = [x for x in ("V1", "V2", "V3") if path_q(m, x)]
    antq = [x for x in ("V3", "V4") if path_q(m, x)]
    st_dep = [x for x in LEADS if x != "aVR" and m["st_mv"][_L[x]] <= -0.05]
    st_el = [x for x in LEADS if x != "aVR" and m["st_mv"][_L[x]] >= 0.1]
    t_inv = [x for x in ("I", "II", "aVL", "V3", "V4", "V5", "V6") if m["t_mv"][_L[x]] < -0.1]
    r = n["r_v2v3"]; sk = n["sokolow"]
    return {
        "qrs_width": "wide (120 ms or more)" if n["qrs_ms"] >= 120 else "not wide",
        "q_waves_inferior": _in(inf, ["II", "III", "aVF"], "inferior"),
        "q_waves_anteroseptal": _in(asq, ["V1", "V2", "V3"], "anteroseptal"),
        "q_waves_anterior": _in(antq, ["V3", "V4"], "anterior"),
        "r_waves_V2_V3": ("almost absent (QS pattern)" if r < 0.3 else "small (poor R-wave progression)" if r < 0.7
                          else "normal"),
        "qrs_voltage": ("high (meets Sokolow-Lyon criterion)" if sk >= 3.5 else "borderline high" if sk >= 3.0
                        else "normal"),
        "st_depression": "none" if not st_dep else f"in {len(st_dep)} leads ({', '.join(st_dep)})",
        "st_elevation": "none" if not st_el else f"in {len(st_el)} leads ({', '.join(st_el)})",
        "t_wave_inversion": "none" if not t_inv else f"in {len(t_inv)} leads ({', '.join(t_inv)})",
    }


def panel_state(windows: list[dict], m: dict | None) -> dict:
    return {"segments": [{"window": w["window"], "noise": w["noise"], "bad_leads": w["bad_leads"]} for w in windows],
            "lead_findings": lead_findings(m)}

# ---------------- questions ----------------

CRITERIA = {
    "imi": ("Does this ECG show an old inferior myocardial infarction?",
            {"true": "`lead_findings.q_waves_inferior` reports Q waves in at least 2 of 3 inferior leads",
             "false": "Q waves in none or only 1 of the inferior leads"}),
    "asmi": ("Does this ECG show an old anteroseptal myocardial infarction?",
             {"true": "R waves in V2-V3 are almost absent (QS pattern) or small, or Q waves in at least 2 of 3 anteroseptal "
                      "leads, and the QRS is not wide (a wide QRS from bundle branch block can mimic this)",
              "false": "R waves in V2-V3 are normal and there are no anteroseptal Q waves"}),
    "ami": ("Does this ECG show an old anterior myocardial infarction?",
            {"true": "Q waves in the anterior leads (V3, V4) together with small or absent R waves in V2-V3",
             "false": "no anterior Q waves, or normal R waves in V2-V3"}),
    "lvh": ("Does this ECG show left ventricular hypertrophy?",
            {"true": "`lead_findings.qrs_voltage` is high, especially with ST depression or T wave inversion in lateral leads",
             "false": "QRS voltage is normal"}),
    "isch": ("Does this ECG show ischemic ST-T changes?",
             {"true": "ST depression or T wave inversion in several leads, and the QRS is not wide",
              "false": "no ST depression and no T wave inversion, or changes only in one lead, or a wide QRS explains them"}),
    "std": ("Does this ECG show ST depression?",
            {"true": "`lead_findings.st_depression` lists two or more leads",
             "false": "ST depression is none or in only one lead"}),
}


def panel_questions(n_segments: int = 3) -> dict:
    q = {f"seg{i}_usable": Noul(instructions=f"Is `segments[{i}]` clean enough to measure beat shape?",
                                criteria={"true": "noise is low or moderate", "false": "noise is high"})
         for i in range(n_segments)}
    for k, (instr, crit) in CRITERIA.items():
        q[f"has_{k}"] = Noul(instructions=instr, criteria=crit)
    return q

# ---------------- code-rule baselines ----------------

def rule_scores(m: dict | None) -> dict[str, float]:
    if m is None:
        return {k: 0.5 for k in FINDINGS}
    n = numbers(m)
    sig = lambda z: float(1 / (1 + np.exp(-z)))
    return {
        "imi": sig((n["inf_q"] - 1.5) * 2),
        "asmi": sig((0.7 - n["r_v2v3"]) / 0.15) * 0.7 + 0.3 * sig((n["as_q"] - 1.5) * 2),
        "ami": sig((n["ant_q"] - 0.5) * 2) * sig((0.7 - n["r_v2v3"]) / 0.2),
        "lvh": sig((n["sokolow"] - 3.5) / 0.3),
        "isch": sig((n["st_dep_n"] + n["t_inv_n"] - 2.5) / 1.0),
        "std": sig((-n["st_min"] - 0.05) / 0.02),
    }
