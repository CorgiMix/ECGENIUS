"""End-to-end feasibility run on PTB-XL fold 9 (fold 10 is never touched). Nothing is trained.

Arms (each also run "+pwave", where every window additionally carries a P-wave consistency bucket):
  rule_rr          code-only baseline: RR irregularity of the whole 10 s
  jev_record       Jev reads whole-record features            -> P(AF)
  jev_segments     Jev reads 3 overlapping 5-s windows        -> P(AF)
  jev_seg_code     Jev's per-window answers combined in code  -> P(AF)
  gemini_alone     Gemini reads whole-record features         -> P(AF)
  gemini_jev       Gemini reads features + Jev record answers -> P(AF)
  gemini_jev_seg   Gemini reads windows + Jev window answers  -> P(AF)   (the planned pipeline)

Usage:
  uv run python scripts/run_feasibility.py               # all of fold 9 (default)
  uv run python scripts/run_feasibility.py --sample 400  # quick: all AF + random non-AF
  add --no-gemini to skip Gemini, --no-browser to not open the report
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ecg_guardian import gemini, jev, report  # noqa: E402
from ecg_guardian.config import CACHE_DIR, GEMINI_MODEL, LEADS, RESULTS_DIR  # noqa: E402
from ecg_guardian.features import (QUALITY_WINDOWS, RHYTHM_WINDOWS, bad_lead_mask,  # noqa: E402
                                   early_beat_explained, p_wave_consistency, quality_features,
                                   quality_windows_overlapping,
                                   r_peaks, rhythm_window_features)
from ecg_guardian.labels import TRAIN_FOLDS, VAL_FOLD, add_af_label, load_metadata  # noqa: E402
from ecg_guardian.progress import RunProgress  # noqa: E402
from ecg_guardian.metrics import aurc, errors_caught_at, failure_auroc, risk_coverage  # noqa: E402
from ecg_guardian.segment import label  # noqa: E402
from ecg_guardian.signals import load_hi, load_lo_all  # noqa: E402

JEV_PRICE_PER_M = 0.042
BAR = "{l_bar}{bar:30}{r_bar}"


def log(*a):
    tqdm.write(time.strftime("%H:%M:%S ") + " ".join(map(str, a)))


# ---------------- features ----------------

def quality_thresholds(db: pd.DataFrame) -> dict:
    """99th percentile of each quality feature on clean TRAINING records (no noise annotations)."""
    path = CACHE_DIR / "quality_thresholds.json"
    if path.exists():
        return json.loads(path.read_text())
    noise_cols = ["baseline_drift", "static_noise", "burst_noise", "electrodes_problems"]
    clean = db[db.strat_fold.isin(TRAIN_FOLDS) & db[noise_cols].isna().all(axis=1)].sample(300, random_state=0)
    q = np.stack([quality_features(x) for x in load_hi(clean)])
    thr = {k: float(np.percentile(q[..., i], 99)) for i, k in enumerate(["baseline_mv", "hf_mv", "mains_mv"])}
    path.write_text(json.dumps(thr, indent=1))
    return thr


def build_states(rows: pd.DataFrame, x_lo: np.ndarray, pos: np.ndarray, thr: dict, prog=None) -> dict:
    """Per record: states without and with P waves, plus numeric summaries for baselines."""
    out = {}
    chunks = [slice(i, i + 200) for i in range(0, len(rows), 200)]
    with tqdm(total=len(rows), desc="features (500 Hz)", unit="ecg", bar_format=BAR) as bar:
        for sl in chunks:
            part = rows.iloc[sl]
            for (ecg_id, _), xi, x_hi in zip(part.iterrows(), pos[sl], load_hi(part)):
                x = np.asarray(x_lo[xi]); peaks = r_peaks(x)
                bad = bad_lead_mask(quality_features(x_hi), thr)
                def bad_leads(idx): return [LEADS[j] for j in range(12) if bad[idx, j].any()]
                all_q = list(range(len(QUALITY_WINDOWS)))
                rr = rhythm_window_features(peaks, 0, 10)
                pw = p_wave_consistency(x_hi, peaks, 0, 10)       # whole record: windows have too few beats
                pat = early_beat_explained(peaks, 0, 10)
                wins = [(label(s, e), rhythm_window_features(peaks, s, e), bad_leads(quality_windows_overlapping(s, e)))
                        for s, e in RHYTHM_WINDOWS]
                out[ecg_id] = {
                    "record": jev.record_state(jev.window_state("0-10 s", rr, bad_leads(all_q))),
                    "segments": jev.segments_state([jev.window_state(w, f, b) for w, f, b in wins]),
                    "record+pwave": jev.record_state(jev.window_state("0-10 s", rr, bad_leads(all_q), pw, pat)),
                    "segments+pwave": jev.segments_state([jev.window_state(w, f, b, pw, pat) for w, f, b in wins]),
                    "n_bad_lead_windows": int(bad.sum()),
                    "rr_irregularity": float(np.nan_to_num(max(rr["rr_cv"], rr["rr_rmssd_norm"]), nan=0.0)),
                }
                bar.update(1)
                if prog:
                    prog.advance("features")
    return out


# ---------------- evaluation ----------------

def cv_threshold_predictions(p: np.ndarray, y: np.ndarray, seed=0) -> tuple[np.ndarray, float]:
    """Pick the Youden-optimal cut-off on 4/5 of fold 9, apply to the held-out 1/5 (no peeking)."""
    pred = np.zeros_like(y)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(p, y):
        fpr, tpr, thr = roc_curve(y[tr], p[tr])
        t = thr[np.argmax(tpr - fpr)]
        pred[te] = (p[te] >= t).astype(int)
    fpr, tpr, thr = roc_curve(y, p)
    return pred, float(thr[np.argmax(tpr - fpr)])


def bootstrap_ci(fn, *arrays, n=1000, seed=0):
    rng = np.random.default_rng(seed); vals = []
    m = len(arrays[0])
    for _ in range(n):
        i = rng.integers(0, m, m)
        try:
            v = fn(*(a[i] for a in arrays))
            if not np.isnan(v):
                vals.append(v)
        except ValueError:
            pass
    return tuple(np.percentile(vals, [2.5, 97.5])) if vals else (np.nan, np.nan)


def evaluate(name, p, y, quality):
    pred, t = cv_threshold_predictions(p, y)          # tuned cut-off: detection metrics only
    err = ((p >= 0.5).astype(int) != y).astype(int)    # error flagging: the arm's own call at 0.5 ...
    unc = 1 - np.abs(2 * p - 1)                        # ... and its own confidence in that call
    lo, hi = bootstrap_ci(roc_auc_score, y, p)
    row = {"arm": name, "auroc": roc_auc_score(y, p), "auroc_lo": lo, "auroc_hi": hi, "threshold": t,
           "sens_tuned": pred[y == 1].mean(), "spec_tuned": 1 - pred[y == 0].mean(),
           "errors_tuned": int((pred != y).sum()), "errors@0.5": int(err.sum()), "sens@0.5": (p[y == 1] >= 0.5).mean(), "spec@0.5": (p[y == 0] < 0.5).mean()}
    if 5 <= err.sum() <= len(err) - 5:
        flo, fhi = bootstrap_ci(failure_auroc, unc, err)
        row.update({"fail_auroc": failure_auroc(unc, err), "fail_auroc_lo": flo, "fail_auroc_hi": fhi,
                    "caught@20%": errors_caught_at(unc, err, 0.2), "aurc_uncertainty": aurc(unc, err)})
        lr = LogisticRegression(max_iter=1000)
        only = cross_val_predict(lr, unc[:, None], err, cv=5, method="predict_proba")[:, 1]
        comb = cross_val_predict(lr, np.column_stack([unc, quality]), err, cv=5, method="predict_proba")[:, 1]
        row.update({"aurc_unc_LR": aurc(only, err), "aurc_unc+quality_LR": aurc(comb, err)})
    return row, unc, err


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="9", help="folds to run, e.g. '9' or '1-8' (fold 10 needs --final)")
    ap.add_argument("--final", action="store_true", help="allow fold 10: the one-time final test")
    ap.add_argument("--sample", type=int, default=0, help="AF-enriched quick sample size (0 = all records)")
    ap.add_argument("--no-gemini", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    t_all = time.time()

    a, _, b = args.folds.partition("-")
    folds = list(range(int(a), int(b or a) + 1))
    if 10 in folds and not args.final:
        sys.exit("fold 10 is the sealed final test set: pass --final only for the one-time final evaluation")
    args.tag = "fold" + args.folds.replace("-", "_")

    db = add_af_label(load_metadata())
    x_lo = load_lo_all(db)
    pool = db[db.strat_fold.isin(folds)]
    if args.sample:
        af = pool[pool.af == 1]
        rows = pd.concat([af, pool[pool.af == 0].sample(max(args.sample - len(af), 0), random_state=0)])
    else:
        rows = pool
    rows = rows.sort_index(); ids = list(rows.index); y = rows.af.values
    pos = db.index.get_indexer(rows.index)
    log(f"folds {args.folds}: {len(rows)} ECGs, {y.sum()} AF ({100 * y.mean():.1f}%) {'[AF-enriched sample]' if args.sample else '[real prevalence]'}")

    n = len(ids)
    stage_names = (["features"] + [f"Jev {k}" for k in ["record", "record+pwave", "segments", "segments+pwave"]]
                   + ([] if args.no_gemini else [f"Gemini {g}{v}" for v in ["", "+pwave"] for g in ["alone", "+jev", "+jev seg"]]))
    n_arms = 7 if args.no_gemini else 13
    prog = RunProgress(f"folds {args.folds}" + (f" sample {n}" if args.sample else ""), n, int(y.sum()),
                       [(s_, n) for s_ in stage_names] + [("scoring + CIs", n_arms)])
    try:
        _run(args, db, rows, ids, y, pos, x_lo, prog, t_all)
    except Exception as ex:
        prog.fail(f"{type(ex).__name__}: {ex}")
        raise


def _run(args, db, rows, ids, y, pos, x_lo, prog, t_all):
    thr = quality_thresholds(db)
    prog.start("features")
    states = build_states(rows, x_lo, pos, thr, prog)
    prog.finish("features")
    quality = np.array([states[i]["n_bad_lead_windows"] for i in ids], float)

    out_dir = RESULTS_DIR / (f"{args.tag}_sample" if args.sample else ("fold9_full" if args.folds == "9" else args.tag))
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Jev: 4 request sets (record/segments x without/with P waves)
    jev_res, t = {}, time.time()
    for style in ["record", "segments"]:
        for variant in ["", "+pwave"]:
            key = style + variant; stage = f"Jev {key}"
            prog.start(stage)
            with tqdm(total=len(ids), desc=f"Jev {key:16}", unit="ecg", bar_format=BAR) as bar:
                tick = lambda k, bar=bar, stage=stage: (bar.update(k), prog.advance(stage, k))
                jev_res[key] = asyncio.run(jev.ask_many([(i, states[i][key]) for i in ids], style,
                                                        out_dir / f"jev_{key}.jsonl", pwave=bool(variant),
                                                        log=lambda *a: None, progress=tick))
                bar.update(len(ids) - bar.n)
            prog.finish(stage)
    jev_s = time.time() - t

    arms = {"rule_rr": np.clip(np.array([states[i]["rr_irregularity"] for i in ids]) / 0.3, 0, 1)}
    for v in ["", "+pwave"]:
        jr, js = jev_res["record" + v], jev_res["segments" + v]
        arms["jev_record" + v] = np.array([jr[i]["record_af"] for i in ids])
        arms["jev_segments" + v] = np.array([js[i]["record_af"] for i in ids])
        arms["jev_seg_code" + v] = np.array([jev.segments_af_prob(js[i], 3) for i in ids])

    # --- Gemini: 6 request sets
    gem, gemini_s = {}, 0.0
    if not args.no_gemini:
        t = time.time()
        for v in ["", "+pwave"]:
            plan = {
                "gemini_alone" + v: [(i, states[i]["record" + v], None) for i in ids],
                "gemini_jev" + v: [(i, states[i]["record" + v], jev_res["record" + v][i]) for i in ids],
                "gemini_jev_seg" + v: [(i, states[i]["segments" + v], jev_res["segments" + v][i]) for i in ids],
            }
            for name, items in plan.items():
                stage = "Gemini " + name.replace("gemini_", "").replace("jev_seg", "+jev seg").replace("jev", "+jev").replace("_", " ")
                stage = stage.replace("++", "+")
                prog.start(stage)
                with tqdm(total=len(ids), desc=f"Gemini {name:20}", unit="ecg", bar_format=BAR) as bar:
                    tick = lambda k, bar=bar, stage=stage: (bar.update(k), prog.advance(stage, k))
                    gem[name] = asyncio.run(gemini.ask_many(items, out_dir / f"{name}.jsonl", concurrency=12,
                                                            log=log, progress=tick))
                    bar.update(len(ids) - bar.n)
                prog.finish(stage)
                missing = len(ids) - len(gem[name])
                if missing:
                    log(f"{name}: {missing} failed calls scored as 0.5")
                arms[name] = np.array([gem[name][i]["af_percent"] / 100 if i in gem[name] else 0.5 for i in ids])
        gemini_s = time.time() - t

    # --- evaluate
    rows_out, curves = [], {}
    prog.start("scoring + CIs")
    for name, p in tqdm(arms.items(), desc="scoring + bootstrap CIs", unit="arm", bar_format=BAR):
        r, unc, err = evaluate(name, p, y, quality)
        rows_out.append(r); prog.advance("scoring + CIs")
        if name in ("rule_rr", "jev_seg_code+pwave", "gemini_alone+pwave", "gemini_jev_seg+pwave") and err.sum() > 0:
            cov, _ = risk_coverage(-unc, err)                       # most suspicious first
            order = np.argsort(-unc, kind="stable")
            curves[name] = (cov, np.cumsum(err[order]) / err.sum())
    prog.finish("scoring + CIs")
    table = pd.DataFrame(rows_out).set_index("arm")
    pd.set_option("display.width", 250); pd.set_option("display.precision", 3)
    print("\n" + table[["auroc", "auroc_lo", "auroc_hi", "sens_tuned", "spec_tuned", "errors_tuned",
                        "fail_auroc", "caught@20%", "aurc_unc_LR", "aurc_unc+quality_LR"]].to_string())

    # --- save + report
    pd.DataFrame({"ecg_id": ids, "af": y, **{f"p_{k}": v for k, v in arms.items()}, "n_bad_lead_windows": quality}
                 ).to_csv(out_dir / "per_record.csv", index=False)
    table.to_csv(out_dir / "summary.csv")
    all_jev = [a for res in jev_res.values() for a in res.values()]
    rep = {"folds": args.folds, "n_records": len(ids), "n_af": int(y.sum()), "jev_s": jev_s, "gemini_s": gemini_s, "total_s": time.time() - t_all,
           "jev_input_tokens": sum(a["input_tokens"] for a in all_jev),
           "jev_cost_usd": sum(a["input_tokens"] for a in all_jev) / 1e6 * JEV_PRICE_PER_M,
           "jev_models": sorted({a["model"] for a in all_jev}), "gemini_model": GEMINI_MODEL if gem else None}
    examples = []
    if gem:
        g_all = [v for res in gem.values() for v in res.values()]
        rep.update({"gemini_input_tokens": sum(v.get("input_tokens") or 0 for v in g_all),
                    "gemini_output_tokens": sum(v.get("output_tokens") or 0 for v in g_all),
                    "gemini_summaries_naming_unflagged_leads": {
                        n: sum(bool(gemini.unsupported_leads(res[i]["summary"], states[i]["segments" if "seg" in n else "record"])) for i in ids if i in res)
                        for n, res in gem.items()}})
        pipe = gem["gemini_jev_seg+pwave"]
        pick = [i for i in ids if i in pipe and rows.af[i] == 1][:2] + [i for i in ids if i in pipe and rows.af[i] == 0][:1]
        examples = [{"ecg_id": int(i), "af": int(rows.af[i]), "gemini": pipe[i]} for i in pick]
    (out_dir / "report.json").write_text(json.dumps(rep, indent=1))
    print("\n" + json.dumps(rep, indent=1))

    report.auroc_chart(table, out_dir / "auroc.png")
    if curves:
        report.review_chart(curves, out_dir / "review.png")
    report.sens_spec_chart(table, out_dir / "sens_spec.png")
    page = report.write_html(out_dir, table, rep, examples, open_browser=not args.no_browser)
    log(f"report: {page}")
    cap = lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)
    prog.done([{"arm": a, "auroc": float(r.auroc), "lo": float(r.auroc_lo), "hi": float(r.auroc_hi),
                "sens": float(r.sens_tuned), "spec": float(r.spec_tuned), "caught20": cap(r.get("caught@20%"))}
               for a, r in table.iterrows()], str(page))


if __name__ == "__main__":
    main()
