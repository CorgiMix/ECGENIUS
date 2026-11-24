"""Download a subset of MIMIC-IV-ECG waveform files from PhysioNet (open access).

Reads machine_measurements.csv to find AF records, selects an AF-enriched evaluation subset
(one ECG per patient) plus a separate set of clean 'Normal ECG' records used only to calibrate
the noise thresholds, then downloads the corresponding .hea and .dat files via curl.

Writes {MIMIC_ECG_DIR}/subset.csv (evaluation) and {MIMIC_ECG_DIR}/calib.csv (noise calibration).
With --name X it writes X.csv instead of subset.csv, with patients disjoint from every existing *.csv
subset in that folder (and from calib.csv): a fresh, never-inspected evaluation set.

Usage:
  uv run python scripts/mimic_download.py                # default: 2000 ECGs + 300 calibration
  uv run python scripts/mimic_download.py --n 500        # smaller sample
  uv run python scripts/mimic_download.py --n 5000 --name holdout5000   # new patients only
"""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ecg_guardian.config import MIMIC_ECG_DIR  # noqa: E402
from ecg_guardian.mimic_labels import (load_mimic_metadata, mimic_af_stats,  # noqa: E402
                                       select_mimic_subset, select_quality_calibration)

BASE_URL = "https://physionet.org/files/mimic-iv-ecg/1.0"
COLS = ["record_path", "subject_id", "af", "af_flutter", "quality_warning", "report_text"]


def download_file(url: str, dest: Path, retries: int = 3) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")      # never leave a half-written file in place
    for _ in range(retries):
        try:
            r = subprocess.run(["curl", "-sS", "-f", "-o", str(tmp), url],
                               capture_output=True, text=True, timeout=60)
            if r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
                tmp.rename(dest)
                return True
        except subprocess.TimeoutExpired:
            pass
        time.sleep(1)
    tmp.unlink(missing_ok=True)
    return False


def have(p: str) -> bool:
    return all((MIMIC_ECG_DIR / (p + ext)).exists() for ext in (".hea", ".dat"))


def download_all(paths: list[str], workers: int) -> set[str]:
    """Download missing records; return the paths that failed."""
    todo = [p for p in paths if not have(p)]
    print(f"{len(paths) - len(todo)} already downloaded, {len(todo)} to download...")

    def dl(p):
        ok = all((MIMIC_ECG_DIR / (p + ext)).exists() or download_file(f"{BASE_URL}/{p}{ext}", MIMIC_ECG_DIR / (p + ext))
                 for ext in (".hea", ".dat"))
        return p, ok

    failed = set()
    with ThreadPoolExecutor(workers) as ex:
        futures = [ex.submit(dl, p) for p in todo]
        with tqdm(total=len(todo), desc="Downloading ECGs", unit="ecg") as bar:
            for f in as_completed(futures):
                p, ok = f.result()
                if not ok:
                    failed.add(p)
                bar.update(1)
    if failed:
        print(f"{len(failed)} downloads failed (excluded)")
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="target number of evaluation ECGs")
    ap.add_argument("--n-calib", type=int, default=300, help="clean ECGs for noise-threshold calibration")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8, help="parallel downloads")
    ap.add_argument("--name", default="subset", help="evaluation set name (writes <name>.csv)")
    args = ap.parse_args()

    print(f"Loading MIMIC metadata from {MIMIC_ECG_DIR}...")
    db = load_mimic_metadata()
    stats = mimic_af_stats(db)
    print(f"  Total records: {stats['n_records']:,}")
    print(f"  AF records: {stats['n_af']:,} ({100*stats['af_prevalence']:.1f}%)")
    print(f"  Flutter records: {stats['n_flutter']:,}")

    calib_csv = MIMIC_ECG_DIR / "calib.csv"
    if args.name == "subset":
        subset = select_mimic_subset(db, n_total=args.n, seed=args.seed)
        calib = select_quality_calibration(db, subset.subject_id, n=args.n_calib)
    else:
        used = set()
        for f in MIMIC_ECG_DIR.glob("*.csv"):
            if f.name not in ("record_list.csv", "machine_measurements.csv", f"{args.name}.csv"):
                used |= set(pd.read_csv(f, usecols=["subject_id"]).subject_id)
        print(f"Excluding {len(used):,} patients already used in earlier subsets / calibration")
        subset = select_mimic_subset(db, n_total=args.n, seed=args.seed, exclude_subjects=used)
        calib = pd.read_csv(calib_csv, index_col="study_id")
    sub_stats = mimic_af_stats(subset)
    print(f"\nEvaluation subset: {sub_stats['n_records']} ECGs ({subset.subject_id.nunique()} patients), "
          f"{sub_stats['n_af']} AF ({100*sub_stats['af_prevalence']:.1f}%)")
    print(f"Calibration set: {len(calib)} clean 'Normal ECG' records, other patients\n")

    failed = download_all(subset.record_path.tolist() + calib.record_path.tolist(), args.workers)
    subset = subset[~subset.record_path.isin(failed)]
    calib = calib[~calib.record_path.isin(failed)]
    subset[COLS].to_csv(MIMIC_ECG_DIR / f"{args.name}.csv")
    calib[COLS].to_csv(calib_csv)
    print(f"\nDone! {len(subset)} evaluation ECGs -> {MIMIC_ECG_DIR / (args.name + '.csv')}, "
          f"{len(calib)} calibration -> {calib_csv}")


if __name__ == "__main__":
    main()
