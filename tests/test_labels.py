"""Tests for PTB-XL loading, AF labels and folds. Run with: uv run pytest"""

import pytest

from ecg_guardian.labels import (
    PTBXL_DIR, TEST_FOLD, TRAIN_FOLDS, VAL_FOLD,
    add_af_label, af_label, assert_patient_disjoint, load_metadata,
)


# --- unit tests: no data needed ---

def test_afib_with_zero_likelihood_counts_as_af():
    # Rhythm codes have likelihood 0.0 ("not rated"); presence must still count.
    assert af_label({"AFIB": 0.0}) == 1


def test_afib_with_full_likelihood_counts_as_af():
    assert af_label({"AFIB": 100.0, "LVOLT": 0.0}) == 1


def test_flutter_only_is_not_af():
    assert af_label({"AFLT": 100.0}) == 0


def test_normal_is_not_af():
    assert af_label({"NORM": 100.0, "SR": 0.0}) == 0


def test_folds_cover_1_to_10_without_overlap():
    folds = set(TRAIN_FOLDS) | {VAL_FOLD, TEST_FOLD}
    assert folds == set(range(1, 11))
    assert VAL_FOLD not in TRAIN_FOLDS and TEST_FOLD not in TRAIN_FOLDS


# --- data tests: need PTB-XL at PTBXL_DIR ---

needs_data = pytest.mark.skipif(
    not (PTBXL_DIR / "ptbxl_database.csv").exists(), reason="PTB-XL not found"
)


@pytest.fixture(scope="module")
def db():
    return add_af_label(load_metadata())


@needs_data
def test_record_and_patient_counts(db):
    assert len(db) == 21_799
    assert db["patient_id"].nunique() == 18_869


@needs_data
def test_no_patient_in_more_than_one_fold(db):
    assert_patient_disjoint(db)


@needs_data
def test_total_af_count(db):
    assert db["af"].sum() == 1_514


@needs_data
def test_af_is_balanced_across_folds(db):
    per_fold = db.groupby("strat_fold")["af"].sum()
    assert per_fold.between(150, 153).all(), per_fold.to_dict()
