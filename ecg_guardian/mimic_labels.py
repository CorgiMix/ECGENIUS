"""MIMIC-IV-ECG metadata loading and AF labels from machine reports.

MIMIC-IV-ECG has no structured labels like PTB-XL's SCP codes.
AF labels are extracted from machine-generated report text (columns report_0..report_17)
in machine_measurements.csv via keyword matching.
"""

import re
from pathlib import Path

import pandas as pd

from .config import MIMIC_ECG_DIR

# Regex for atrial fibrillation mentions in machine reports.
# Matches "atrial fibrillation", "afib", "a fib", "a.fib", "a-fib" but NOT
# "no atrial fibrillation" or "atrial flutter" alone.
_AF_PATTERN = re.compile(
    r"(?<!\bno\s)(?<!\bnot\s)(?<!\brule out\s)(?<!\bwithout\s)"
    r"(?:atrial\s*fib(?:rillation)?|a[\.\-\s]*fib\b)",
    re.IGNORECASE,
)
# Also catch "AF " or "AFIB" standalone
_AF_SHORT = re.compile(r"\bAF(?:IB)?\b")

# Atrial flutter (tracked separately)
_AFLUTTER_PATTERN = re.compile(r"atrial\s*flutter", re.IGNORECASE)

REPORT_COLS = [f"report_{i}" for i in range(18)]


def load_mimic_metadata(data_dir: Path = MIMIC_ECG_DIR) -> pd.DataFrame:
    """Load record_list.csv and machine_measurements.csv, merge, and add AF labels.

    Returns a DataFrame indexed by study_id with columns:
      subject_id, path, ecg_time, report_text, af (0/1), af_flutter (0/1)
    """
    records = pd.read_csv(data_dir / "record_list.csv")
    # record_list.csv has columns: subject_id, study_id, path
    records = records.set_index("study_id")

    measurements = pd.read_csv(
        data_dir / "machine_measurements.csv",
        low_memory=False,
    )
    measurements = measurements.set_index("study_id")

    # A truncated download silently drops most records in the inner join below: fail loudly instead.
    missing = len(records.index.difference(measurements.index))
    if missing > 0.01 * len(records):
        raise ValueError(f"machine_measurements.csv lacks {missing:,} of {len(records):,} studies in "
                         "record_list.csv: the file is probably truncated, download it again")

    # Merge
    db = records.join(measurements, how="inner", rsuffix="_mm")

    # Concatenate all report columns into one text field
    for col in REPORT_COLS:
        if col not in db.columns:
            db[col] = ""
    db[REPORT_COLS] = db[REPORT_COLS].fillna("")
    db["report_text"] = db[REPORT_COLS].apply(lambda row: " | ".join(row.astype(str)).strip(), axis=1)

    # AF labels from reports
    db["af"] = db["report_text"].apply(_detect_af).astype(int)
    db["quality_warning"] = db["report_text"].str.contains("data quality", case=False).astype(int)
    db["af_flutter"] = db["report_text"].apply(
        lambda t: 1 if _AFLUTTER_PATTERN.search(str(t)) else 0
    ).astype(int)

    # Build the WFDB record path (without extension)
    # path column contains: files/p1000/p10001725/s41420867/41420867
    db["record_path"] = db["path"].astype(str)

    return db


def _detect_af(report_text: str) -> int:
    """Return 1 if report text mentions atrial fibrillation, 0 otherwise."""
    text = str(report_text)
    if _AF_PATTERN.search(text):
        return 1
    if _AF_SHORT.search(text):
        return 1
    return 0


def one_per_patient(db: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Keep one random ECG per subject_id, so bootstrap CIs are not inflated by repeat patients."""
    return db.sample(frac=1, random_state=seed).groupby("subject_id", sort=False).head(1)


def select_mimic_subset(
    db: pd.DataFrame,
    n_total: int = 2000,
    seed: int = 42,
    exclude_subjects=(),
) -> pd.DataFrame:
    """Select an AF-enriched subset of MIMIC-IV-ECG records, one ECG per patient.

    Takes AF-positive records (up to half of n_total) and fills the rest with random non-AF
    records. Patients in exclude_subjects (earlier subsets) are never drawn, so a new subset is
    patient-disjoint from everything already looked at.
    """
    db = one_per_patient(db[~db.subject_id.isin(set(exclude_subjects))], seed)
    af_pos = db[db.af == 1]
    af_neg = db[db.af == 0]

    max_af = min(len(af_pos), n_total // 2)
    af_sample = af_pos.sample(n=max_af, random_state=seed)

    n_neg = n_total - len(af_sample)
    neg_sample = af_neg.sample(n=min(n_neg, len(af_neg)), random_state=seed)

    return pd.concat([af_sample, neg_sample]).sort_index()


def select_quality_calibration(db: pd.DataFrame, exclude_subjects, n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Clean ECGs for the per-lead noise thresholds (99th percentile), like PTB-XL's clean training
    records: report says 'Normal ECG', no data-quality warning, patients not in the evaluation set.
    Only signal quality is computed on them, never AF."""
    ok = (db.report_text.str.contains("normal ecg", case=False) & (db.quality_warning == 0)
          & ~db.subject_id.isin(set(exclude_subjects)))
    return one_per_patient(db[ok], seed).sample(n=n, random_state=seed).sort_index()


def mimic_af_stats(db: pd.DataFrame) -> dict:
    """Summary statistics for AF labels in the MIMIC dataset."""
    n = len(db)
    n_af = int(db.af.sum())
    n_flutter = int(db.af_flutter.sum())
    n_both = int(((db.af == 1) & (db.af_flutter == 1)).sum())
    return {
        "n_records": n,
        "n_af": n_af,
        "af_prevalence": n_af / n if n else 0,
        "n_flutter": n_flutter,
        "n_af_and_flutter": n_both,
    }
