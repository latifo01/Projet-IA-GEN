"""Tests unitaires pour les fonctions deterministes de outlier_agent."""

import pytest
import pandas as pd
import numpy as np

from src.agents.outlier_agent import (
    _bimodality_coefficient,
    _choose_method,
    _choose_treatment,
    HIGH_OUTLIER_PCT,
    LOW_OUTLIER_PCT,
    BIMODALITY_THRESHOLD,
    IMBALANCE_GUARD_THRESHOLD,
)


# ---------------------------------------------------------------------------
# _bimodality_coefficient
# ---------------------------------------------------------------------------

class TestBimodalityCoefficient:
    def test_returns_zero_for_tiny_series(self):
        s = pd.Series([1, 2, 3])
        assert _bimodality_coefficient(s) == 0.0

    def test_returns_zero_for_empty_series(self):
        assert _bimodality_coefficient(pd.Series([], dtype=float)) == 0.0

    def test_unimodal_normal_distribution_below_threshold(self):
        rng = np.random.default_rng(42)
        s = pd.Series(rng.normal(0, 1, 1000))
        bc = _bimodality_coefficient(s)
        assert bc < BIMODALITY_THRESHOLD

    def test_bimodal_distribution_above_threshold(self):
        rng = np.random.default_rng(42)
        # Two well-separated modes
        s = pd.Series(np.concatenate([rng.normal(-5, 0.5, 500), rng.normal(5, 0.5, 500)]))
        bc = _bimodality_coefficient(s)
        assert bc > BIMODALITY_THRESHOLD

    def test_nan_values_ignored(self):
        rng = np.random.default_rng(0)
        values = list(rng.normal(0, 1, 100)) + [np.nan] * 20
        s = pd.Series(values)
        bc = _bimodality_coefficient(s)
        assert isinstance(bc, float)


# ---------------------------------------------------------------------------
# _choose_method
# ---------------------------------------------------------------------------

class TestChooseMethod:
    def test_tiny_sample_defaults_to_iqr(self):
        s = pd.Series([1, 2, 3, 4, 5])
        method, reason = _choose_method(s)
        assert method == "iqr"
        assert "small" in reason

    def test_normal_large_sample_returns_zscore(self):
        rng = np.random.default_rng(42)
        s = pd.Series(rng.normal(0, 1, 500))
        method, _ = _choose_method(s)
        assert method == "zscore"

    def test_skewed_large_sample_not_zscore(self):
        # Skewed distributions (exponential) must never use z-score.
        # The algorithm may return "iqr" or "bimodal" depending on the
        # bimodality coefficient — both are valid non-zscore outcomes.
        rng = np.random.default_rng(42)
        s = pd.Series(rng.exponential(scale=2, size=500))
        method, _ = _choose_method(s)
        assert method != "zscore"

    def test_bimodal_distribution_returns_bimodal(self):
        rng = np.random.default_rng(42)
        s = pd.Series(np.concatenate([rng.normal(-5, 0.5, 200), rng.normal(5, 0.5, 200)]))
        method, reason = _choose_method(s)
        assert method == "bimodal"
        assert "bimodal" in reason

    def test_returns_tuple_of_two_strings(self):
        s = pd.Series(range(50))
        result = _choose_method(s)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(v, str) for v in result)


# ---------------------------------------------------------------------------
# _choose_treatment
# ---------------------------------------------------------------------------

class TestChooseTreatment:
    def test_semantic_column_always_replace_with_nan(self):
        action, confidence, _ = _choose_treatment(
            pct_outliers=50.0,
            task_type="regression",
            col="age",
            semantic_cols={"age"},
        )
        assert action == "replace_with_nan"
        assert confidence == "high"

    def test_imbalanced_classification_low_outliers_clips(self):
        action, confidence, _ = _choose_treatment(
            pct_outliers=0.5,
            task_type="classification",
            col="feature",
            semantic_cols=set(),
            imbalance_ratio=0.95,
        )
        assert action == "clip"
        assert confidence == "high"

    def test_imbalanced_classification_high_outliers_keeps(self):
        action, _, _ = _choose_treatment(
            pct_outliers=HIGH_OUTLIER_PCT + 1,
            task_type="classification",
            col="feature",
            semantic_cols=set(),
            imbalance_ratio=0.95,
        )
        assert action == "keep"

    def test_few_outliers_classification_removes(self):
        action, confidence, _ = _choose_treatment(
            pct_outliers=LOW_OUTLIER_PCT - 0.1,
            task_type="classification",
            col="feature",
            semantic_cols=set(),
            imbalance_ratio=0.0,
        )
        assert action == "remove"
        assert confidence == "high"

    def test_few_outliers_regression_clips(self):
        action, confidence, _ = _choose_treatment(
            pct_outliers=LOW_OUTLIER_PCT - 0.1,
            task_type="regression",
            col="feature",
            semantic_cols=set(),
        )
        assert action == "clip"
        assert confidence == "high"

    def test_moderate_outliers_clips(self):
        action, confidence, _ = _choose_treatment(
            pct_outliers=(LOW_OUTLIER_PCT + HIGH_OUTLIER_PCT) / 2,
            task_type="regression",
            col="feature",
            semantic_cols=set(),
        )
        assert action == "clip"
        assert confidence == "medium"

    def test_high_outlier_percentage_keeps(self):
        action, confidence, _ = _choose_treatment(
            pct_outliers=HIGH_OUTLIER_PCT + 1,
            task_type="regression",
            col="feature",
            semantic_cols=set(),
        )
        assert action == "keep"
        assert confidence == "low"

    def test_returns_tuple_of_three_strings(self):
        result = _choose_treatment(
            pct_outliers=2.0,
            task_type="regression",
            col="x",
            semantic_cols=set(),
        )
        assert isinstance(result, tuple) and len(result) == 3
        assert all(isinstance(v, str) for v in result)

    def test_non_semantic_column_not_replace_with_nan(self):
        action, _, _ = _choose_treatment(
            pct_outliers=2.0,
            task_type="regression",
            col="feature",
            semantic_cols={"other_col"},
        )
        assert action != "replace_with_nan"
