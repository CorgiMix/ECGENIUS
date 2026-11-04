"""PTB-XL metadata loading and the binary AF label.

Facts established from the data (2026-11-02):
- AF-related SCP codes are 'AFIB' (1,514 records) and 'AFLT' (atrial flutter, 73 records; 17 have both).
- Rhythm codes carry likelihood 0.0 in almost all cases. In PTB-XL that means "not rated",
  NOT "absent", so the label must be based on the code being PRESENT, not on its likelihood.
- Official folds (strat_fold) are patient-disjoint: no patient appears in more than one fold.
"""

import ast
from pathlib import Path

import pandas as pd

from .config import PTBXL_DIR

TRAIN_FOLDS = tuple(range(1, 9))  # 1-8
VAL_FOLD = 9
TEST_FOLD = 10  # touch once, at the end


def load_metadata(data_dir: Path = PTBXL_DIR) -> pd.DataFrame:
    """Read ptbxl_database.csv and parse scp_codes from string to dict."""
    db = pd.read_csv(data_dir / "ptbxl_database.csv", index_col="ecg_id")
    db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)
    return db


def af_label(scp_codes: dict[str, float]) -> int:
    """Return 1 if this record counts as atrial fibrillation, else 0.

    scp_codes example: {'AFIB': 0.0, 'LVOLT': 0.0} -> AFIB is present (likelihood 0.0 = not rated).

    Design decision: what counts as "AF" for this project?
      - AFIB only: the textbook definition; 1,514 positives.
      - AFIB or AFLT: 1,570 positives. Flutter is also an atrial arrhythmia, but its rhythm is
        often REGULAR (e.g., 2:1 conduction), so RR-irregularity features will not flag it.
        Including it adds positives that your rhythm features cannot explain.
      - AFIB only, and drop AFLT-only records from the dataset: cleanest classes, but those
        56 records are then never evaluated.
    """
    # Chosen: AFIB only. AFLT-only records (56) count as non-AF; change here if that proves wrong.
    return 1 if "AFIB" in scp_codes else 0


def add_af_label(db: pd.DataFrame) -> pd.DataFrame:
    """Add an integer 'af' column using af_label()."""
    db = db.copy()
    db["af"] = db["scp_codes"].apply(af_label)
    return db


def assert_patient_disjoint(db: pd.DataFrame) -> None:
    """Fail loudly if any patient appears in more than one fold."""
    folds_per_patient = db.groupby("patient_id")["strat_fold"].nunique()
    leaked = folds_per_patient[folds_per_patient > 1]
    assert leaked.empty, f"{len(leaked)} patients appear in more than one fold"
