"""Tests unitaires pour les fonctions deterministes de base_agent."""

import pytest
import pandas as pd
import numpy as np

from src.agents.base_agent import (
    _validate_target,
    _audit_dtypes,
    _detect_trivial_columns,
)


# ---------------------------------------------------------------------------
# _validate_target
# ---------------------------------------------------------------------------

def _make_df(target_values):
    return pd.DataFrame({"feature": range(len(target_values)), "target": target_values})


class TestValidateTarget:
    def test_regression_high_cardinality(self):
        df = _make_df(list(range(100)))
        meta = _validate_target(df, "target")
        assert meta["task_type"] == "regression"
        assert meta["nunique"] == 100
        assert meta["pct_nan"] == 0.0

    def test_classification_low_cardinality(self):
        df = _make_df([0, 1] * 50)
        meta = _validate_target(df, "target")
        assert meta["task_type"] == "classification"
        assert meta["n_classes"] == 2

    def test_classification_object_dtype(self):
        df = _make_df(["cat", "dog", "cat", "dog"] * 10)
        meta = _validate_target(df, "target")
        assert meta["task_type"] == "classification"

    def test_classification_at_boundary_20_unique(self):
        df = _make_df(list(range(20)) * 5)
        meta = _validate_target(df, "target")
        assert meta["task_type"] == "classification"

    def test_regression_above_boundary_21_unique(self):
        df = _make_df(list(range(21)) * 5)
        meta = _validate_target(df, "target")
        assert meta["task_type"] == "regression"

    def test_raises_if_target_missing_from_columns(self):
        df = pd.DataFrame({"feature": [1, 2, 3]})
        with pytest.raises(ValueError, match="not found"):
            _validate_target(df, "target")

    def test_raises_if_target_has_too_many_nans(self):
        values = [np.nan] * 40 + [1] * 10  # 80% NaN
        df = _make_df(values)
        with pytest.raises(ValueError, match="missing values"):
            _validate_target(df, "target")

    def test_nan_below_threshold_does_not_raise(self):
        values = [np.nan] * 5 + [1] * 95  # 5% NaN
        df = _make_df(values)
        meta = _validate_target(df, "target")
        assert meta["pct_nan"] == pytest.approx(5.0, abs=0.1)

    def test_imbalance_ratio_computed(self):
        df = _make_df([1] * 95 + [0] * 5)
        meta = _validate_target(df, "target")
        assert meta["imbalance_ratio"] == pytest.approx(0.95, abs=0.01)

    def test_regression_returns_mean_std(self):
        values = list(range(100))
        df = _make_df(values)
        meta = _validate_target(df, "target")
        assert "mean" in meta
        assert "std" in meta


# ---------------------------------------------------------------------------
# _audit_dtypes
# ---------------------------------------------------------------------------

class TestAuditDtypes:
    def test_detects_numeric_as_string(self):
        df = pd.DataFrame({"target": [0, 1], "price": ["1.5", "2.3"]})
        audit = _audit_dtypes(df, "target")
        cols = {a["column"] for a in audit}
        assert "price" in cols
        match = next(a for a in audit if a["column"] == "price")
        assert match["suggested_type"] == "numeric"

    def test_detects_date_as_string(self):
        df = pd.DataFrame({
            "target": [0, 1, 0],
            "date_col": ["2023-01-01", "2023-06-15", "2024-03-10"],
        })
        audit = _audit_dtypes(df, "target")
        cols = {a["column"] for a in audit}
        assert "date_col" in cols
        match = next(a for a in audit if a["column"] == "date_col")
        assert match["suggested_type"] == "datetime"

    def test_detects_boolean_int(self):
        df = pd.DataFrame({"target": [0, 1, 0], "flag": [0, 1, 0]})
        audit = _audit_dtypes(df, "target")
        cols = {a["column"] for a in audit}
        assert "flag" in cols
        match = next(a for a in audit if a["column"] == "flag")
        assert match["suggested_type"] == "boolean"

    def test_detects_ordinal_low_cardinality_int(self):
        df = pd.DataFrame({"target": [0, 1] * 10, "rating": [1, 2, 3, 4, 5] * 4})
        audit = _audit_dtypes(df, "target")
        cols = {a["column"] for a in audit}
        assert "rating" in cols
        match = next(a for a in audit if a["column"] == "rating")
        assert match["suggested_type"] == "ordinal"

    def test_skips_target_column(self):
        df = pd.DataFrame({"target": ["1.0", "2.0", "3.0"]})
        audit = _audit_dtypes(df, "target")
        assert audit == []

    def test_clean_numeric_column_not_flagged(self):
        df = pd.DataFrame({"target": [0, 1], "value": [10.5, 20.3]})
        audit = _audit_dtypes(df, "target")
        assert not any(a["column"] == "value" for a in audit)

    def test_mixed_string_column_below_threshold_not_flagged(self):
        # Only 50% numeric — below the 95% threshold
        df = pd.DataFrame({"target": [0, 1, 0, 1], "col": ["1", "abc", "2", "xyz"]})
        audit = _audit_dtypes(df, "target")
        assert not any(a["column"] == "col" for a in audit)


# ---------------------------------------------------------------------------
# _detect_trivial_columns
# ---------------------------------------------------------------------------

class TestDetectTrivialColumns:
    def test_detects_all_nan_column(self):
        df = pd.DataFrame({"target": [1, 2, 3], "empty": [np.nan, np.nan, np.nan]})
        drops = _detect_trivial_columns(df, "target")
        reasons = {d["column"]: d["reason"] for d in drops}
        assert "empty" in reasons
        assert "missing" in reasons["empty"]

    def test_detects_constant_column(self):
        df = pd.DataFrame({"target": [1, 2, 3], "const": [5, 5, 5]})
        drops = _detect_trivial_columns(df, "target")
        reasons = {d["column"]: d["reason"] for d in drops}
        assert "const" in reasons
        assert "constant" in reasons["const"]

    def test_detects_unique_id_column(self):
        df = pd.DataFrame({
            "target": [1, 2, 3],
            "id": ["a1", "b2", "c3"],  # all unique strings
        })
        drops = _detect_trivial_columns(df, "target")
        reasons = {d["column"]: d["reason"] for d in drops}
        assert "id" in reasons
        assert "unique ID" in reasons["id"]

    def test_detects_quasi_constant_column(self):
        df = pd.DataFrame({
            "target": list(range(200)),
            "quasi": [0] * 199 + [1],  # 99.5% same value
        })
        drops = _detect_trivial_columns(df, "target")
        reasons = {d["column"]: d["reason"] for d in drops}
        assert "quasi" in reasons
        assert "quasi-constant" in reasons["quasi"]

    def test_detects_duplicate_column(self):
        df = pd.DataFrame({
            "target": [1, 2, 3],
            "a": [10, 20, 30],
            "b": [10, 20, 30],  # duplicate of a
        })
        drops = _detect_trivial_columns(df, "target")
        reasons = {d["column"]: d["reason"] for d in drops}
        assert "b" in reasons
        assert "duplicate" in reasons["b"]

    def test_target_not_flagged(self):
        df = pd.DataFrame({"target": [1, 1, 1], "feature": [1, 2, 3]})
        drops = _detect_trivial_columns(df, "target")
        assert not any(d["column"] == "target" for d in drops)

    def test_normal_column_not_flagged(self):
        df = pd.DataFrame({"target": [1, 2, 3], "x": [10, 20, 30]})
        drops = _detect_trivial_columns(df, "target")
        assert drops == []
