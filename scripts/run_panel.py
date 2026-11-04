"""Run a finding panel: Jev (3 windows + measured evidence) vs simple code rules. Nothing trained.

Panels:  rhythm (Tier 1)      AF, PVC, PAC, sinus brady/tachycardia, sinus arrhythmia
         conduction (Tier 2)  complete LBBB/RBBB, incomplete RBBB, 1st-degree AV block, LAFB

Usage:  uv run python scripts/run_panel.py --panel conduction [--folds 9] [--no-browser]
Features are computed once per fold set and cached; Jev answers are cached per question version.
Fold 10 is refused unless --final.
"""

import argparse
import asyncio
import json
import pickle
import sys
import time
import webbrowser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from tqdm import tqdm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ecg_guardian import conduction_panel as cp, jev, rhythm_panel as rp, structure_panel as sp  # noqa: E402
from ecg_guardian.config import CACHE_DIR, LEADS, RESULTS_DIR  # noqa: E402
from ecg_guardian.features import (RHYTHM_WINDOWS, bad_lead_mask, early_beat_explained,  # noqa: E402
                                   early_beat_shape, p_wave_consistency, quality_features,
                                   quality_windows_overlapping, r_peaks, rhythm_window_features)
from ecg_guardian.labels import load_metadata  # noqa: E402
from ecg_guardian.morphology import lead_measures, morphology  # noqa: E402
from ecg_guardian.progress import RunProgress  # noqa: E402
from ecg_guardian.report import BLUE, INK, INK2, ORANGE, SURFACE  # noqa: E402
from ecg_guardian.segment import label  # noqa: E402
from ecg_guardian.signals import load_hi, load_lo_all  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_feasibility import BAR, bootstrap_ci, quality_thresholds  # noqa: E402

JEV_PRICE_PER_M = 0.042
FEATURES_VERSION = "f3"

PANELS = {
    "rhythm": {"mod": rp, "dir": "tier1", "title": "Tier 1 rhythm panel",
               "state": lambda F: rp.panel_state(F["windows"], F["whole"]), "rules": lambda F: rp.rule_scores(F["rhythm"])},
    "conduction": {"mod": cp, "dir": "tier2", "title": "Tier 2 conduction panel",
                   "state": lambda F: cp.panel_state(F["windows"], F["morph"]), "rules": lambda F: cp.rule_scores(F["morph"])},
    "structure": {"mod": sp, "dir": "tier3", "title": "Tier 3 structure & ischemia panel",
                  "state": lambda F: sp.panel_state(F["windows"], F["leads"]), "rules": lambda F: sp.rule_scores(F["leads"])},
}


def compute_features(rows, db, x_lo, thr, prog) -> dict:
    """All measurements every panel needs, per ECG."""
    out = {}
    pos = db.index.get_indexer(rows.index)
    with tqdm(total=len(rows), desc="features (500 Hz)", unit="ecg", bar_format=BAR) as bar:
        for start in range(0, len(rows), 200):
            part = rows.iloc[start:start + 200]
            for (eid, _), xi, x_hi in zip(part.iterrows(), pos[start:start + 200], load_hi(part)):
                peaks = r_peaks(np.asarray(x_lo[xi]))
                bad = bad_lead_mask(quality_features(x_hi), thr)
                bl = lambda idx: [LEADS[j] for j in range(12) if bad[idx, j].any()]
                whole = rhythm_window_features(peaks, 0, 10)
                p_cons, explained = p_wave_consistency(x_hi, peaks), early_beat_explained(peaks)
                n_early, sim = early_beat_shape(x_hi, peaks)
                windows = []
                for s, e in RHYTHM_WINDOWS:
                    f = rhythm_window_features(peaks, s, e)
                    w = jev.window_state(label(s, e), f, bl(quality_windows_overlapping(s, e)))
                    w["heart_rate"] = rp.bucket_rate60(f["hr_bpm"])
                    windows.append(w)
                reg = jev.bucket_regularity(whole)
                whole_state = {"rhythm_regularity": reg, "heart_rate": rp.bucket_rate60(whole["hr_bpm"]),
                               "p_waves": jev.bucket_p_waves(p_cons), "early_beats": rp.bucket_early_beats(n_early, sim)}
                if reg in ("irregular", "slightly irregular"):
                    whole_state["irregularity_pattern"] = jev.bucket_pattern(explained)
                out[eid] = {
                    "windows": windows, "whole": whole_state, "morph": morphology(x_hi, peaks),
                    "leads": lead_measures(x_hi, peaks),
                    "rhythm": {"hr": whole["hr_bpm"], "p_cons": p_cons, "explained": explained, "n_early": n_early,
                               "early_sim": sim, "n_bad": int(bad.sum()),
                               "irregularity": float(np.nan_to_num(max(whole["rr_cv"], whole["rr_rmssd_norm"]), nan=0.0))},
                }
                bar.update(1); prog.advance("features")
    return out


def chart(table, panel, path, title):
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(table) + 1.2))
    names = list(table.index)[::-1]
    for i, k in enumerate(names):
        r = table.loc[k]
        for pre, color, dy, lab in [("jev", BLUE, 0.14, "Jev, 3 windows"), ("rule", ORANGE, -0.14, "code rule")]:
            ax.plot([r[f"{pre}_lo"], r[f"{pre}_hi"]], [i + dy] * 2, color=color, lw=2, solid_capstyle="round")
            ax.plot(r[f"{pre}_auroc"], i + dy, "o", ms=8, color=color, mec=SURFACE, mew=2, label=lab if i == 0 else None)
            ax.text(r[f"{pre}_hi"] + 0.004, i + dy, f"{r[f'{pre}_auroc']:.3f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(names)), [f"{panel.FINDINGS[k][1]} (n={int(table.loc[k, 'n_pos'])})" for k in names])
    ax.set_xlabel("AUROC (95% bootstrap CI) - higher is better"); ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower left")
    ax.set_title(f"{title}: Jev (3 windows) vs a simple code rule", loc="left", fontsize=12, color=INK)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", choices=list(PANELS), required=True)
    ap.add_argument("--folds", default="9")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    a, _, b = args.folds.partition("-"); folds = list(range(int(a), int(b or a) + 1))
    if 10 in folds and not args.final:
        sys.exit("fold 10 is the sealed final test set: pass --final only for the one-time final evaluation")
    P = PANELS[args.panel]; mod = P["mod"]; tag = args.folds.replace("-", "_")
    t0 = time.time()

    db = load_metadata(); x_lo = load_lo_all(db)
    rows = db[db.strat_fold.isin(folds)].sort_index(); ids = list(rows.index)
    Y = pd.DataFrame([mod.labels(c) for c in rows.scp_codes], index=rows.index)
    out_dir = RESULTS_DIR / f"{P['dir']}_fold{tag}"; out_dir.mkdir(parents=True, exist_ok=True)
    feat_cache = CACHE_DIR / f"features_{FEATURES_VERSION}_fold{tag}.pkl"
    prog = RunProgress(f"{P['title']}, folds {args.folds}", len(ids), int(Y.iloc[:, 0].sum()),
                       [("features", len(ids)), ("Jev panel", len(ids)), ("scoring + CIs", len(mod.FINDINGS))])
    try:
        prog.start("features")
        if feat_cache.exists():
            # Safe: this cache is written only by compute_features() below, on this machine, in our own
            # cache folder; it never comes from outside. Delete it to force a recompute.
            F = pickle.loads(feat_cache.read_bytes())
        else:
            F = compute_features(rows, db, x_lo, quality_thresholds(db), prog)
            feat_cache.write_bytes(pickle.dumps(F))
        prog.finish("features")
        states = {i: P["state"](F[i]) for i in ids}

        prog.start("Jev panel")
        with tqdm(total=len(ids), desc=f"Jev {args.panel} panel", unit="ecg", bar_format=BAR) as bar:
            tick = lambda k: (bar.update(k), prog.advance("Jev panel", k))
            ans = asyncio.run(jev.ask_many([(i, states[i]) for i in ids], "panel", out_dir / "jev_panel.jsonl",
                                           questions_fn=lambda st: mod.panel_questions(len(st["segments"])),
                                           tag=mod.PANEL_VERSION, log=lambda *a: None, progress=tick))
            bar.update(len(ids) - bar.n)
        prog.finish("Jev panel")

        prog.start("scoring + CIs"); rows_out = []; per = {"ecg_id": ids}
        rules = {i: P["rules"](F[i]) for i in ids}
        for k in tqdm(mod.FINDINGS, desc="scoring + bootstrap CIs", unit="finding", bar_format=BAR):
            y_all = Y[k].values.astype(float)
            pj_all = np.array([ans[i][f"has_{k}"] for i in ids]); pr_all = np.array([rules[i][k] for i in ids])
            per[f"y_{k}"], per[f"jev_{k}"], per[f"rule_{k}"] = y_all, pj_all, pr_all
            keep = ~np.isnan(y_all)                     # uncertain labels (Tier 3) are left out
            y, pj, pr = y_all[keep].astype(int), pj_all[keep], pr_all[keep]
            row = {"finding": k, "n_pos": int(y.sum()), "n_uncertain": int((~keep).sum())}
            for pre, p in [("jev", pj), ("rule", pr)]:
                lo, hi = bootstrap_ci(roc_auc_score, y, p)
                row.update({f"{pre}_auroc": roc_auc_score(y, p), f"{pre}_lo": lo, f"{pre}_hi": hi,
                            f"{pre}_sens": (p[y == 1] >= 0.5).mean(), f"{pre}_spec": (p[y == 0] < 0.5).mean()})
            rows_out.append(row); prog.advance("scoring + CIs")
        prog.finish("scoring + CIs")
        table = pd.DataFrame(rows_out).set_index("finding")
        pd.set_option("display.width", 250); pd.set_option("display.precision", 3)
        print("\n" + table.to_string())

        pd.DataFrame(per).to_csv(out_dir / "per_record.csv", index=False); table.to_csv(out_dir / "summary.csv")
        toks = sum(v["input_tokens"] for v in ans.values())
        rep = {"panel": args.panel, "folds": args.folds, "n_records": len(ids), "jev_input_tokens": toks,
               "jev_cost_usd": toks / 1e6 * JEV_PRICE_PER_M, "jev_models": sorted({v["model"] for v in ans.values()}),
               "total_s": time.time() - t0}
        (out_dir / "report.json").write_text(json.dumps({**rep, "example_state": states[ids[0]]}, indent=1))
        print(json.dumps(rep, indent=1))

        chart(table, mod, out_dir / "panel_auroc.png", P["title"])
        t = table.copy(); t.index = [mod.FINDINGS[k][1] for k in t.index]
        page = out_dir / "report.html"
        page.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>{P['title']}</title>
<style>body{{font-family:system-ui,"Segoe UI",sans-serif;background:#f9f9f7;color:#0b0b0b;max-width:1000px;margin:2rem auto;padding:0 1rem}}
img{{max-width:100%;border:1px solid rgba(11,11,11,.1);border-radius:8px}} table{{border-collapse:collapse;font-size:.8rem;font-variant-numeric:tabular-nums;background:#fcfcfb}}
th,td{{padding:.3rem .5rem;border-bottom:1px solid #e1e0d9;text-align:right}} th:first-child,td:first-child{{text-align:left}} .m{{color:#52514e}}
pre{{background:#fcfcfb;border:1px solid rgba(11,11,11,.1);border-radius:8px;padding:.7rem;font-size:.8rem}}</style></head><body>
<h1>{P['title']} - PTB-XL folds {args.folds}</h1>
<p class="m">{len(ids)} ECGs. Jev ({', '.join(rep['jev_models'])}) zero-shot, one request per ECG with one yes/no question per finding.
Code rules use the same measured numbers. Sens/spec at 0.5. Jev cost ${rep['jev_cost_usd']:.2f}.</p>
<img src="panel_auroc.png" alt="AUROC per finding">
<h2>Table</h2>{t.to_html(float_format=lambda v: f"{v:.3f}", border=0)}
<h2>What Jev sees (one ECG)</h2><pre>{json.dumps(states[ids[0]], indent=1)}</pre></body></html>""", encoding="utf-8")
        if not args.no_browser:
            webbrowser.open(page.resolve().as_uri())
        res = [{"arm": f"{k} {pre}", "auroc": float(r[f"{pre}_auroc"]), "lo": float(r[f"{pre}_lo"]), "hi": float(r[f"{pre}_hi"]),
                "sens": float(r[f"{pre}_sens"]), "spec": float(r[f"{pre}_spec"]), "caught20": None}
               for k, r in table.iterrows() for pre in ("jev", "rule")]
        prog.done(res, str(page)); print(f"report: {page}")
    except Exception as ex:
        prog.fail(f"{type(ex).__name__}: {ex}"); raise


if __name__ == "__main__":
    main()
