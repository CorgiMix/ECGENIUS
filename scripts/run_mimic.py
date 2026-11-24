"""External zero-shot validation on MIMIC-IV-ECG: the exact PTB-XL pipeline, nothing re-tuned.

Every bucket cut-off, prompt and question stays as fixed on PTB-XL. The only MIMIC-specific value is
the per-lead noise thresholds (99th percentile on clean 'Normal ECG' records of OTHER patients,
calib.csv), the same recipe PTB-XL used on its clean training records.

Labels come from the ECG machine's own report text, not from cardiologists: results measure
agreement with that algorithm.

First run scripts/mimic_download.py (writes subset.csv and calib.csv), then:
  uv run python scripts/run_mimic.py                 # Jev arms only (default)
  uv run python scripts/run_mimic.py --gemini        # also the 6 Gemini arms
  uv run python scripts/run_mimic.py --limit 200     # quick: first 200 records (AF-balanced)
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_feasibility import BAR, log, progress_stages, record_states, run_arms  # noqa: E402

from ecg_guardian.config import MIMIC_ECG_DIR, RESULTS_DIR  # noqa: E402
from ecg_guardian.features import quality_features  # noqa: E402
from ecg_guardian.mimic_signals import read_mimic_record_hi, to_lo  # noqa: E402
from ecg_guardian.progress import RunProgress  # noqa: E402


def quality_thresholds(calib: pd.DataFrame) -> dict:
    path = MIMIC_ECG_DIR / "quality_thresholds.json"
    if path.exists():
        return json.loads(path.read_text())
    with ThreadPoolExecutor(8) as ex:
        q = np.stack([quality_features(x) for x in ex.map(read_mimic_record_hi, calib.record_path)])
    thr = {k: float(np.percentile(q[..., i], 99)) for i, k in enumerate(["baseline_mv", "hf_mv", "mains_mv"])}
    path.write_text(json.dumps(thr, indent=1))
    log(f"noise thresholds from {len(calib)} clean MIMIC records: {thr}")
    return thr


def build_states(rows: pd.DataFrame, thr: dict, prog) -> dict:
    out = {}
    with ThreadPoolExecutor(8) as ex, tqdm(total=len(rows), desc="features (500 Hz)", unit="ecg", bar_format=BAR) as bar:
        for study_id, x_hi in zip(rows.index, ex.map(read_mimic_record_hi, rows.record_path)):
            out[study_id] = record_states(to_lo(x_hi), x_hi, thr)
            bar.update(1); prog.advance("features")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemini", action="store_true", help="also run the Gemini arms (needs GEMINI_API_KEY)")
    ap.add_argument("--limit", type=int, default=0, help="quick run on this many records, half AF")
    ap.add_argument("--subset", default="subset", help="evaluation set name: <name>.csv from mimic_download.py")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    args.no_gemini = not args.gemini
    t_all = time.time()

    subset_csv, calib_csv = MIMIC_ECG_DIR / f"{args.subset}.csv", MIMIC_ECG_DIR / "calib.csv"
    if not (subset_csv.exists() and calib_csv.exists()):
        sys.exit(f"{subset_csv.name} / {calib_csv.name} missing: run scripts/mimic_download.py first")
    rows = pd.read_csv(subset_csv, index_col="study_id")
    calib = pd.read_csv(calib_csv, index_col="study_id")
    if args.limit:
        rows = pd.concat([rows[rows.af == 1].head(args.limit // 2),
                          rows[rows.af == 0].head(args.limit - args.limit // 2)]).sort_index()
    ids, y = list(rows.index), rows.af.values
    log(f"MIMIC-IV-ECG: {len(ids)} ECGs ({rows.subject_id.nunique()} patients), {y.sum()} AF "
        f"({100 * y.mean():.1f}%) [AF-enriched, machine-report labels]")

    prog = RunProgress(f"MIMIC-IV-ECG {len(ids)}", len(ids), int(y.sum()), progress_stages(len(ids), args.no_gemini))
    try:
        thr = quality_thresholds(calib)
        prog.start("features")
        states = build_states(rows, thr, prog)
        prog.finish("features")
        name = "mimic" if args.subset == "subset" else f"mimic_{args.subset}"
        out_dir = RESULTS_DIR / (f"{name}_{len(ids)}" if args.limit else name)
        run_arms(args, ids, y, states, out_dir, prog, t_all, {
            "dataset": f"MIMIC-IV-ECG external validation, {args.subset} ({len(ids)} ECGs)",
            "note": "Zero-shot transfer: all cut-offs fixed on PTB-XL. Labels = machine report text, not cardiologists. "
                    "One ECG per patient; AF-enriched, so sensitivity/specificity do not reflect real prevalence."})
    except Exception as ex:
        prog.fail(f"{type(ex).__name__}: {ex}")
        raise


if __name__ == "__main__":
    main()
