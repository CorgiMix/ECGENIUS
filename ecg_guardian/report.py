"""Visual report: three PNG charts + an HTML page with the full tables. Opens in the browser.

Palette: reference dataviz palette, slots 1-4 (validated: CVD-safe on adjacent pairs; aqua/yellow
are below 3:1 contrast, so every line is direct-labelled and the page carries a table view).
"""

import html
import webbrowser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})

NICE = {
    "rule_rr": "Code rule (RR irregularity)",
    "jev_record": "Jev - whole record", "jev_segments": "Jev - 3 windows",
    "jev_seg_code": "Jev - windows, combined in code", "gemini_alone": "Gemini alone",
    "gemini_jev": "Gemini + Jev (whole record)", "gemini_jev_seg": "Gemini + Jev windows (pipeline)",
}


def nice(arm: str) -> str:
    base, _, variant = arm.partition("+")
    return NICE.get(base, base)


def auroc_chart(table: pd.DataFrame, path: Path):
    """Dot + 95% CI per arm; blue = without P waves, orange = with P waves."""
    bases = [b for b in NICE if any(a.split("+")[0] == b for a in table.index)]
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(bases) + 1.2))
    for i, b in enumerate(bases[::-1]):
        for arm, color, dy, lab in [(b, BLUE, 0.13, "timing + noise only"), (f"{b}+pwave", ORANGE, -0.13, "+ P waves & beat pattern")]:
            if arm not in table.index:
                continue
            r = table.loc[arm]
            ax.plot([r.auroc_lo, r.auroc_hi], [i + dy] * 2, color=color, lw=2, solid_capstyle="round")
            ax.plot(r.auroc, i + dy, "o", ms=8, color=color, mec=SURFACE, mew=2, label=lab if i == 0 or b == "rule_rr" else None)
            ax.text(r.auroc_hi + 0.003, i + dy, f"{r.auroc:.3f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(bases)), [nice(b) for b in bases[::-1]])
    ax.set_xlabel("AF detection AUROC (95% bootstrap CI)  -  higher is better")
    ax.grid(axis="y", visible=False)
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="lower right", frameon=False)
    ax.set_title("How well each arm separates AF from non-AF", loc="left", fontsize=12, color=INK)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def review_chart(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path):
    """Errors caught vs share of ECGs sent to a clinician (most suspicious first)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot([0, 100], [0, 100], ls=(0, (4, 3)), color=MUTED, lw=1.2)
    ax.text(62, 55, "random review", color=MUTED, fontsize=8, rotation=31)
    for (arm, (x, y)), color in zip(curves.items(), [BLUE, ORANGE, AQUA, YELLOW]):
        ax.plot(100 * x, 100 * y, color=color, lw=2)
        k = np.searchsorted(x, 0.2)
        ax.plot(20, 100 * y[k], "o", ms=8, color=color, mec=SURFACE, mew=2)
        ax.annotate(f"{nice(arm)}{' + P waves & beat pattern' if '+pwave' in arm else ''}: {100 * y[k]:.0f}%",
                    (20, 100 * y[k]), xytext=(26, 100 * y[k]), fontsize=8, color=INK2, va="center")
    ax.axvline(20, color=AXIS, lw=0.8)
    ax.set_xlim(0, 100); ax.set_ylim(0, 102)
    ax.set_xlabel("% of ECGs sent to a clinician (most suspicious first)")
    ax.set_ylabel("% of the arm's errors caught")
    ax.set_title("If a doctor reviews the X% most suspicious ECGs, how many errors are caught?", loc="left", fontsize=11, color=INK)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def sens_spec_chart(table: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for arm, r in table.iterrows():
        color = ORANGE if "+pwave" in arm else BLUE
        ax.plot(r.spec_tuned, r.sens_tuned, "o", ms=9, color=color, mec=SURFACE, mew=2)
        ax.annotate(nice(arm).replace(" (pipeline)", "").replace(" (whole record)", ""), (r.spec_tuned, r.sens_tuned),
                    xytext=(5, 4), textcoords="offset points", fontsize=7, color=INK2)
    ax.plot([], [], "o", color=BLUE, label="timing + noise only"); ax.plot([], [], "o", color=ORANGE, label="+ P waves & beat pattern")
    ax.legend(frameon=False, loc="lower left")
    ax.set_xlabel("Specificity (non-AF correctly cleared)"); ax.set_ylabel("Sensitivity (AF caught)")
    ax.set_title("Sensitivity vs specificity at the tuned cut-off\n(top-right is best)", loc="left", fontsize=11, color=INK)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def write_html(out_dir: Path, table: pd.DataFrame, report: dict, examples: list[dict], open_browser=True) -> Path:
    def fmt_table(df):
        return df.to_html(float_format=lambda v: f"{v:.3f}", border=0, classes="t")
    cols = ["auroc", "auroc_lo", "auroc_hi", "threshold", "sens_tuned", "spec_tuned", "errors_tuned",
            "fail_auroc", "fail_auroc_lo", "fail_auroc_hi", "caught@20%", "aurc_unc_LR", "aurc_unc+quality_LR"]
    t = table[[c for c in cols if c in table.columns]].copy()
    t.index = [f"{nice(a)}{' + P waves & beat pattern' if '+pwave' in a else ''}" for a in t.index]
    ex_html = "".join(
        f"<div class='ex'><b>ECG {e['ecg_id']}</b> - truth: {'AF' if e['af'] else 'not AF'} - "
        f"Gemini: {e['gemini']['af_percent']}% AF, <i>{html.escape(e['gemini']['action'])}</i><br>"
        f"{html.escape(e['gemini']['summary'])}</div>" for e in examples)
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>ECG Guardian - run on folds {report.get("folds", "9")}</title>
<style>
body{{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;background:#f9f9f7;color:#0b0b0b;max-width:1000px;margin:2rem auto;padding:0 1rem}}
h1{{font-size:1.5rem}} h2{{font-size:1.15rem;margin-top:2rem}} img{{max-width:100%;background:#fcfcfb;border:1px solid rgba(11,11,11,.1);border-radius:8px}}
.t{{border-collapse:collapse;font-size:.8rem;font-variant-numeric:tabular-nums;background:#fcfcfb}}
.t th,.t td{{padding:.3rem .5rem;border-bottom:1px solid #e1e0d9;text-align:right}} .t th:first-child,.t td:first-child{{text-align:left}}
.ex{{background:#fcfcfb;border:1px solid rgba(11,11,11,.1);border-radius:8px;padding:.7rem;margin:.5rem 0;font-size:.9rem}}
.meta{{color:#52514e;font-size:.9rem}}
</style></head><body>
<h1>ECG Guardian - PTB-XL folds {report.get("folds", "9")}</h1>
<p class="meta">{report['n_records']} ECGs, {report['n_af']} AF ({100 * report['n_af'] / report['n_records']:.1f}%). Nothing trained: Jev ({', '.join(report['jev_models'])}) and Gemini ({report.get('gemini_model', 'n/a')}) are zero-shot.
Cut-offs tuned by 5-fold cross-validation within these records; error flagging uses each arm's own call at 0.5. Fold 10 untouched unless this is the final run. Total time {report['total_s'] / 60:.1f} min; Jev cost ${report['jev_cost_usd']:.2f}.</p>
<h2>1. AF detection</h2><img src="auroc.png" alt="AUROC per arm with confidence intervals">
<h2>2. Flagging its own errors for review</h2><img src="review.png" alt="Errors caught vs share of ECGs reviewed">
<h2>3. Sensitivity vs specificity</h2><img src="sens_spec.png" alt="Sensitivity vs specificity per arm">
<h2>Full table</h2>{fmt_table(t)}
<p class="meta">fail_auroc: how well the arm's uncertainty ranks its own errors (95% CI). caught@20%: share of its errors among the 20% most uncertain ECGs.
aurc_unc_LR vs aurc_unc+quality_LR: error-ranking AURC (lower is better) without vs with the signal-quality count - the research question.</p>
<h2>Example Gemini summaries (pipeline arm)</h2>{ex_html}
</body></html>"""
    path = out_dir / "report.html"
    path.write_text(page, encoding="utf-8")
    if open_browser:
        webbrowser.open(path.resolve().as_uri())
    return path
