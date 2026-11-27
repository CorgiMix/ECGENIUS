"""Trained baseline: gradient boosting on the same measurements Jev's words are built from.

One model per finding (17), trained on PTB-XL folds 1-8, scored on fold 9 against the cached Jev answers
(results/tier{1,2,3}_fold9/per_record.csv). Fold 10 is never touched. Settings are fixed in advance,
not tuned on fold 9.

Models (--model): "xgb" = XGBoost (default), "hgb" = scikit-learn HistGradientBoostingClassifier.
Both are gradient boosting: many small decision trees, each one fitted to the errors of the ones
before. Both handle missing values (NaN) natively, which matters because some measurements fail on
noisy ECGs.

Usage:  uv run python scripts/run_boosting.py [--model xgb|hgb]
First run scripts/run_panel.py for the rhythm, conduction and structure panels (fold 9): this script
reads their Jev answers.
macOS: XGBoost needs the OpenMP runtime (brew install libomp).
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ecg_guardian import conduction_panel as cp, rhythm_panel as rp, structure_panel as sp  # noqa: E402
from ecg_guardian.config import CACHE_DIR, LEADS, RESULTS_DIR  # noqa: E402
from ecg_guardian.labels import TRAIN_FOLDS, VAL_FOLD, load_metadata  # noqa: E402
from ecg_guardian.report import BLUE, INK, INK2, ORANGE, SURFACE  # noqa: E402
from ecg_guardian.signals import load_lo_all  # noqa: E402
from run_feasibility import quality_thresholds  # noqa: E402
from run_panel import FEATURES_VERSION, compute_features  # noqa: E402

PANELS = [("rhythm", rp, "tier1"), ("conduction", cp, "tier2"), ("structure", sp, "tier3")]
MORPH_KEYS = ["qrs_ms", "pr_ms", "axis_deg", "v1_net", "v1_terminal", "v1_late_r",
              "i_terminal", "v6_terminal", "i_initial", "v6_initial"]
LEAD_KEYS = ["r_amp", "s_depth", "q_ms", "q_depth", "st_mv", "t_mv"]
RHYTHM_KEYS = ["hr", "p_cons", "explained", "n_early", "early_sim", "n_bad", "irregularity"]
MODELS = {   # fixed in advance, not tuned on fold 9; same capacity for both
    "xgb": ("XGBoost", lambda: __import__("xgboost").XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4, reg_lambda=1.0, random_state=0, n_jobs=4)),
    "hgb": ("Gradient boosting (scikit-learn)", lambda: HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=4, l2_regularization=1.0, random_state=0)),
}


class _NoProgress:
    """compute_features reports progress to the live panel; not needed here."""
    def advance(self, *a): pass


def features(rows, db, x_lo, tag) -> dict:
    path = CACHE_DIR / f"features_{FEATURES_VERSION}_fold{tag}.pkl"
    if path.exists():
        # Safe: written only by compute_features on this machine, in our own cache folder.
        return pickle.loads(path.read_bytes())
    F = compute_features(rows, db, x_lo, quality_thresholds(db), _NoProgress())
    path.write_bytes(pickle.dumps(F))
    return F


def to_row(f: dict) -> dict:
    """Flatten one ECG's measurements into numbers (NaN when a measurement failed)."""
    row = {f"rhythm_{k}": f["rhythm"][k] for k in RHYTHM_KEYS}
    row.update({f"morph_{k}": f["morph"][k] for k in MORPH_KEYS})
    leads = f["leads"]
    for k in LEAD_KEYS:
        for j, lead in enumerate(LEADS):
            row[f"{k}_{lead}"] = float(leads[k][j]) if leads is not None else np.nan
    return {k: (np.nan if v is None else float(v)) for k, v in row.items()}


def paired_diff_ci(y, a, b, n=1000, seed=0):
    """95% bootstrap CI of AUROC(a) - AUROC(b) on the same ECGs."""
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        d.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return tuple(np.percentile(d, [2.5, 97.5]))


def auroc_ci(y, p, n=1000, seed=0):
    rng = np.random.default_rng(seed); v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            v.append(roc_auc_score(y[i], p[i]))
    return tuple(np.percentile(v, [2.5, 97.5]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", choices=list(MODELS), default="xgb")
    model_name, make_model = MODELS[ap.parse_args().model]; model_key = ap.parse_args().model
    db = load_metadata(); x_lo = load_lo_all(db)
    train = db[db.strat_fold.isin(TRAIN_FOLDS)].sort_index()
    test = db[db.strat_fold == VAL_FOLD].sort_index()
    print(f"train: folds 1-8, {len(train)} ECGs | test: fold 9, {len(test)} ECGs | fold 10 not used")

    F_tr = features(train, db, x_lo, "1_8")
    F_te = features(test, db, x_lo, "9")
    X_tr = pd.DataFrame([to_row(F_tr[i]) for i in train.index], index=train.index)
    X_te = pd.DataFrame([to_row(F_te[i]) for i in test.index], index=test.index)
    print(f"{X_tr.shape[1]} measurements per ECG")

    out_dir = RESULTS_DIR / f"{model_key}_fold9"; out_dir.mkdir(parents=True, exist_ok=True)
    rows, per = [], {"ecg_id": list(test.index)}
    for panel, mod, tier in PANELS:
        jev = pd.read_csv(RESULTS_DIR / f"{tier}_fold9" / "per_record.csv", index_col="ecg_id").loc[test.index]
        Y_tr = pd.DataFrame([mod.labels(c) for c in train.scp_codes], index=train.index)
        Y_te = pd.DataFrame([mod.labels(c) for c in test.scp_codes], index=test.index)
        for k in mod.FINDINGS:
            ytr = Y_tr[k].values.astype(float); keep_tr = ~np.isnan(ytr)   # uncertain labels left out
            model = make_model().fit(X_tr[keep_tr], ytr[keep_tr].astype(int))
            p_all = model.predict_proba(X_te)[:, 1]
            yte = Y_te[k].values.astype(float); keep = ~np.isnan(yte)
            y, pb, pj = yte[keep].astype(int), p_all[keep], jev[f"jev_{k}"].values[keep]
            per[f"y_{k}"], per[f"boost_{k}"] = yte, p_all
            b_lo, b_hi = auroc_ci(y, pb); j_lo, j_hi = auroc_ci(y, pj); d_lo, d_hi = paired_diff_ci(y, pb, pj)
            rows.append({"type": panel, "finding": k, "name": mod.FINDINGS[k][1],
                         "n_train_pos": int(ytr[keep_tr].sum()), "n_test_pos": int(y.sum()),
                         "boost_auroc": roc_auc_score(y, pb), "boost_lo": b_lo, "boost_hi": b_hi,
                         "jev_auroc": roc_auc_score(y, pj), "jev_lo": j_lo, "jev_hi": j_hi,
                         "diff": roc_auc_score(y, pb) - roc_auc_score(y, pj), "diff_lo": d_lo, "diff_hi": d_hi})
            print(f"  {k:18s} {model_key} {rows[-1]['boost_auroc']:.3f}  Jev {rows[-1]['jev_auroc']:.3f}")

    table = pd.DataFrame(rows).set_index("finding")
    table["winner"] = np.where(table.diff_lo > 0, "boosting", np.where(table.diff_hi < 0, "Jev", "tie"))
    table.to_csv(out_dir / "summary.csv"); pd.DataFrame(per).to_csv(out_dir / "per_record.csv", index=False)
    pd.set_option("display.width", 200); pd.set_option("display.precision", 3)
    print("\n" + table[["type", "n_test_pos", "boost_auroc", "jev_auroc", "diff", "diff_lo", "diff_hi", "winner"]].to_string())
    print("\nmean AUROC by type:\n" + table.groupby("type")[["boost_auroc", "jev_auroc"]].mean().to_string())

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(table) + 1.2))
    names = list(table.index)[::-1]
    for i, k in enumerate(names):
        r = table.loc[k]
        for pre, color, dy, lab in [("jev", BLUE, 0.14, "Jev (no ECG training)"),
                                    ("boost", ORANGE, -0.14, f"{model_name} (trained, folds 1-8)")]:
            ax.plot([r[f"{pre}_lo"], r[f"{pre}_hi"]], [i + dy] * 2, color=color, lw=2, solid_capstyle="round")
            ax.plot(r[f"{pre}_auroc"], i + dy, "o", ms=7, color=color, mec=SURFACE, mew=2, label=lab if i == 0 else None)
            ax.text(r[f"{pre}_hi"] + 0.004, i + dy, f"{r[f'{pre}_auroc']:.3f}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(range(len(names)), [f"{table.loc[k, 'name']} (n={int(table.loc[k, 'n_test_pos'])})" for k in names],
                  fontsize=8)
    ax.set_xlabel("AUROC on fold 9 (95% bootstrap CI), higher is better"); ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    ax.set_title(f"Jev vs {model_name}, PTB-XL fold 9", loc="left", fontsize=12, color=INK)
    fig.tight_layout(); fig.savefig(out_dir / f"jev_vs_{model_key}.png", dpi=160); plt.close(fig)
    print(f"\nsaved: {out_dir}")


if __name__ == "__main__":
    main()
