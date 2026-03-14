"""Tests unitaires pour les fonctions deterministes de transformation_agent."""

import pytest
import pandas as pd
import numpy as np

from src.agents.transformation_agent import (
    _dtype_audit_transforms,
    _semantic_anomaly_transforms,
    _rule_based_decisions,
    _detect_leakage,
    _detect_high_correlations,
    _cramers_v,
    MISSING_DROP_THRESHOLD,
    HIGH_CARDINALITY_THRESHOLD,
    LOW_CARDINALITY_THRESHOLD,
    LEAKAGE_CORRELATION_THRESHOLD,
    HIGH_CORRELATION_THRESHOLD,
)


# ---------------------------------------------------------------------------
# _dtype_audit_transforms
# ---------------------------------------------------------------------------

class TestDtypeAuditTransforms:
    def test_datetime_suggestion_yields_extract_datetime(self):
        audit = [{"column": "date_col", "suggested_type": "datetime", "reason": "95% date patterns"}]
        transforms = _dtype_audit_transforms(audit)
        assert len(transforms) == 1
        assert transforms[0]["column"] == "date_col"
        assert transforms[0]["action"] == "extract_datetime"
        assert transforms[0]["confidence"] == "high"
        assert transforms[0]["source"] == "rule"

    def test_numeric_suggestion_yields_cast_numeric(self):
        audit = [{"column": "price", "suggested_type": "numeric", "reason": "98% numeric"}]
        transforms = _dtype_audit_transforms(audit)
        assert len(transforms) == 1
        assert transforms[0]["action"] == "cast_numeric"

    def test_float_suggestion_yields_cast_numeric(self):
        audit = [{"column": "val", "suggested_type": "float", "reason": "100% float"}]
        transforms = _dtype_audit_transforms(audit)
        assert transforms[0]["action"] == "cast_numeric"

    def test_unknown_suggestion_ignored(self):
        audit = [{"column": "x", "suggested_type": "boolean", "reason": "binary"}]
        transforms = _dtype_audit_transforms(audit)
        assert transforms == []

    def test_multiple_entries_all_converted(self):
        audit = [
            {"column": "a", "suggested_type": "numeric", "reason": "r1"},
            {"column": "b", "suggested_type": "datetime", "reason": "r2"},
        ]
        transforms = _dtype_audit_transforms(audit)
        assert len(transforms) == 2
        actions = {t["column"]: t["action"] for t in transforms}
        assert actions["a"] == "cast_numeric"
        assert actions["b"] == "extract_datetime"

    def test_empty_audit_returns_empty(self):
        assert _dtype_audit_transforms([]) == []


# ---------------------------------------------------------------------------
# _semantic_anomaly_transforms
# ---------------------------------------------------------------------------

class TestSemanticAnomalyTransforms:
    def _df_with_col(self, values, col="col"):
        return pd.DataFrame({"target": range(len(values)), col: values})

    def test_proxy_for_missing_with_explicit_sentinels(self):
        df = self._df_with_col([1, -1, 3, -1, 5])
        anomalies = [{"column": "col", "type": "proxy_for_missing", "sentinel_values": [-1]}]
        transforms = _semantic_anomaly_transforms(anomalies, df)
        assert len(transforms) == 1
        assert transforms[0]["action"] == "replace_sentinel"
        assert -1 in transforms[0]["sentinel_values"]

    def test_proxy_for_missing_heuristic_detects_999(self):
        df = self._df_with_col([1, 999, 2, 999, 5])
        anomalies = [{"column": "col", "type": "proxy_for_missing", "sentinel_values": []}]
        transforms = _semantic_anomaly_transforms(anomalies, df)
        assert len(transforms) == 1
        assert 999 in transforms[0]["sentinel_values"]

    def test_proxy_for_missing_no_sentinels_no_common_values_no_transform(self):
        df = self._df_with_col([1, 2, 3, 4, 5])
        anomalies = [{"column": "col", "type": "proxy_for_missing", "sentinel_values": []}]
        transforms = _semantic_anomaly_transforms(anomalies, df)
        # No common sentinel candidates found -> no transform
        assert transforms == []

    def test_impossible_value_with_sentinels(self):
        df = self._df_with_col([1, -999, 2])
        anomalies = [{"column": "col", "type": "impossible_value", "sentinel_values": [-999]}]
        transforms = _semantic_anomaly_transforms(anomalies, df)
        assert len(transforms) == 1
        assert transforms[0]["action"] == "replace_sentinel"
        assert -999 in transforms[0]["sentinel_values"]

    def test_impossible_value_without_sentinels_no_transform(self):
        df = self._df_with_col([1, 2, 3])
        anomalies = [{"column": "col", "type": "impossible_value", "sentinel_values": []}]
        transforms = _semantic_anomaly_transforms(anomalies, df)
        assert transforms == []

    def test_unknown_column_skipped(self):
        df = pd.DataFrame({"target": [1, 2, 3], "other": [4, 5, 6]})
        anomalies = [{"column": "nonexistent", "type": "proxy_for_missing", "sentinel_values": [-1]}]
        transforms = _semantic_anomaly_transforms(anomalies, df)
        assert transforms == []

    def test_other_anomaly_type_ignored(self):
        df = self._df_with_col([1, 2, 3])
        anomalies = [{"column": "col", "type": "wrong_dtype", "sentinel_values": [0]}]
        transforms = _semantic_anomaly_transforms(anomalies, df)
        assert transforms == []

    def test_transform_has_required_keys(self):
        df = self._df_with_col([1, -1, 2])
        anomalies = [{"column": "col", "type": "proxy_for_missing", "sentinel_values": [-1]}]
        t = _semantic_anomaly_transforms(anomalies, df)[0]
        for key in ("column", "action", "reason", "confidence", "source", "sentinel_values"):
            assert key in t


# ---------------------------------------------------------------------------
# _rule_based_decisions
# ---------------------------------------------------------------------------

class TestRuleBasedDecisions:
    def test_high_nan_column_gets_dropped(self):
        n = 100
        df = pd.DataFrame({
            "target": range(n),
            "sparse": [np.nan] * 60 + [1] * 40,  # 60% NaN
        })
        transforms, _ = _rule_based_decisions(df, "target", [])
        actions = {t["column"]: t["action"] for t in transforms}
        assert actions.get("sparse") == "drop_column"

    def test_binary_categorical_gets_ordinal_encoding(self):
        df = pd.DataFrame({
            "target": [0, 1] * 10,
            "gender": ["M", "F"] * 10,
        })
        transforms, _ = _rule_based_decisions(df, "target", [])
        actions = {t["column"]: t["action"] for t in transforms if t["column"] == "gender"}
        assert "ordinal_encoding" in actions.values()

    def test_low_cardinality_categorical_gets_ohe(self):
        df = pd.DataFrame({
            "target": [0, 1] * 10,
            "color": ["red", "blue", "green", "yellow"] * 5,
        })
        transforms, _ = _rule_based_decisions(df, "target", [])
        actions = {t["column"]: t["action"] for t in transforms if t["column"] == "color"}
        assert "one_hot_encoding" in actions.values()

    def test_high_cardinality_categorical_gets_frequency_encoding(self):
        # 51 unique values > HIGH_CARDINALITY_THRESHOLD (50)
        df = pd.DataFrame({
            "target": range(100),
            "city": [f"city_{i}" for i in range(51)] + [f"city_{i}" for i in range(49)],
        })
        transforms, _ = _rule_based_decisions(df, "target", [])
        actions = {t["column"]: t["action"] for t in transforms if t["column"] == "city"}
        assert "frequency_encoding" in actions.values()

    def test_numeric_low_nan_gets_impute_median(self):
        n = 100
        values = [np.nan] * 3 + list(range(97))  # 3% NaN
        df = pd.DataFrame({"target": range(n), "num": values})
        transforms, _ = _rule_based_decisions(df, "target", [])
        actions = [t["action"] for t in transforms if t["column"] == "num"]
        assert "impute_median" in actions

    def test_boolean_column_gets_ordinal_encoding(self):
        df = pd.DataFrame({
            "target": [0, 1] * 10,
            "flag": [True, False] * 10,
        })
        transforms, _ = _rule_based_decisions(df, "target", [])
        actions = {t["column"]: t["action"] for t in transforms if t["column"] == "flag"}
        assert "ordinal_encoding" in actions.values()

    def test_target_column_excluded(self):
        df = pd.DataFrame({"target": ["a", "b"] * 10, "feature": range(20)})
        transforms, ambiguous = _rule_based_decisions(df, "target", [])
        assert all(t["column"] != "target" for t in transforms)
        assert "target" not in ambiguous

    def test_auto_drop_columns_excluded(self):
        df = pd.DataFrame({
            "target": range(5),
            "id": ["a1", "b2", "c3", "d4", "e5"],
            "feature": range(5),
        })
        auto_drop = [{"column": "id", "reason": "unique ID"}]
        transforms, ambiguous = _rule_based_decisions(df, "target", auto_drop)
        assert all(t["column"] != "id" for t in transforms)

    def test_continuous_numeric_gets_scaling_action(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "target": range(200),
            "x": rng.normal(50, 10, 200).tolist(),
        })
        transforms, _ = _rule_based_decisions(df, "target", [])
        scaling_actions = {"standard_scaling", "minmax_scaling", "robust_scaling",
                           "log_transform", "sqrt_transform"}
        col_actions = {t["action"] for t in transforms if t["column"] == "x"}
        assert col_actions & scaling_actions


# ---------------------------------------------------------------------------
# _detect_leakage
# ---------------------------------------------------------------------------

class TestDetectLeakage:
    def test_highly_correlated_feature_detected(self):
        n = 100
        target = np.arange(n, dtype=float)
        df = pd.DataFrame({
            "target": target,
            "leaked": target * 1.001 + 0.1,  # correlation ~1.0
            "feature": np.random.default_rng(0).normal(0, 1, n),
        })
        alerts = _detect_leakage(df, "target")
        leaked_cols = {a["column"] for a in alerts}
        assert "leaked" in leaked_cols
        assert "feature" not in leaked_cols

    def test_low_correlation_not_flagged(self):
        rng = np.random.default_rng(42)
        n = 100
        df = pd.DataFrame({
            "target": rng.normal(0, 1, n),
            "feature": rng.normal(0, 1, n),
        })
        alerts = _detect_leakage(df, "target")
        assert alerts == []

    def test_non_numeric_target_returns_empty(self):
        df = pd.DataFrame({
            "target": ["a", "b", "a", "b"],
            "feature": [1.0, 2.0, 3.0, 4.0],
        })
        alerts = _detect_leakage(df, "target")
        assert alerts == []

    def test_alert_has_column_and_correlation_keys(self):
        n = 100
        target = np.arange(n, dtype=float)
        df = pd.DataFrame({"target": target, "x": target})
        alerts = _detect_leakage(df, "target")
        assert len(alerts) >= 1
        for a in alerts:
            assert "column" in a
            assert "correlation" in a

    def test_target_not_flagged_as_its_own_leakage(self):
        n = 50
        target = np.arange(n, dtype=float)
        df = pd.DataFrame({"target": target, "feature": target * 2})
        alerts = _detect_leakage(df, "target")
        assert not any(a["column"] == "target" for a in alerts)


# ---------------------------------------------------------------------------
# _detect_high_correlations
# ---------------------------------------------------------------------------

class TestDetectHighCorrelations:
    def test_highly_correlated_numeric_pair_detected(self):
        n = 100
        x = np.arange(n, dtype=float)
        df = pd.DataFrame({
            "target": np.random.default_rng(0).normal(0, 1, n),
            "a": x,
            "b": x * 1.001,  # near-perfect correlation
        })
        alerts = _detect_high_correlations(df, "target")
        pairs = {(a["column_a"], a["column_b"]) for a in alerts}
        assert ("a", "b") in pairs or ("b", "a") in pairs

    def test_low_correlation_not_flagged(self):
        rng = np.random.default_rng(0)
        n = 100
        df = pd.DataFrame({
            "target": rng.normal(0, 1, n),
            "a": rng.normal(0, 1, n),
            "b": rng.normal(0, 1, n),
        })
        alerts = _detect_high_correlations(df, "target")
        assert alerts == []

    def test_alert_has_required_fields(self):
        n = 100
        x = np.arange(n, dtype=float)
        df = pd.DataFrame({"target": x * 0.5, "a": x, "b": x * 1.001})
        alerts = _detect_high_correlations(df, "target")
        for a in alerts:
            assert "column_a" in a
            assert "column_b" in a
            assert "correlation" in a
            assert "method" in a
            assert "severity" in a

    def test_severity_very_high_above_90(self):
        n = 200
        x = np.arange(n, dtype=float)
        df = pd.DataFrame({"target": np.ones(n), "a": x, "b": x + 0.01})
        alerts = _detect_high_correlations(df, "target")
        very_high = [a for a in alerts if a["severity"] == "very_high"]
        assert len(very_high) >= 1

    def test_target_not_in_alerts(self):
        n = 50
        x = np.arange(n, dtype=float)
        df = pd.DataFrame({"target": x, "a": x * 1.001, "b": x * 0.999})
        alerts = _detect_high_correlations(df, "target")
        for a in alerts:
            assert a["column_a"] != "target"
            assert a["column_b"] != "target"

    def test_categorical_columns_use_cramers_v(self):
        n = 200
        # Perfect association: c2 is a deterministic function of c1
        c1 = (["A", "B"] * (n // 2))[:n]
        c2 = ["X" if v == "A" else "Y" for v in c1]
        df = pd.DataFrame({
            "target": np.random.default_rng(0).normal(0, 1, n),
            "c1": c1,
            "c2": c2,
        })
        alerts = _detect_high_correlations(df, "target")
        cramers_alerts = [a for a in alerts if a["method"] == "cramers_v"]
        assert len(cramers_alerts) >= 1


# ---------------------------------------------------------------------------
# _cramers_v
# ---------------------------------------------------------------------------

class TestCramersV:
    def test_perfect_association_returns_one(self):
        n = 100
        a = pd.Series(["X", "Y"] * (n // 2))
        b = pd.Series(["P" if v == "X" else "Q" for v in a])
        v = _cramers_v(a, b)
        # Cramér's V on a 2x2 contingency table is >= 0.95 for perfect association
        assert v >= 0.95

    def test_independent_series_near_zero(self):
        rng = np.random.default_rng(42)
        n = 500
        a = pd.Series(rng.choice(["X", "Y", "Z"], n))
        b = pd.Series(rng.choice(["P", "Q", "R"], n))
        v = _cramers_v(a, b)
        assert v < 0.15  # independent series should have low association

    def test_empty_crosstab_returns_zero(self):
        a = pd.Series([], dtype=str)
        b = pd.Series([], dtype=str)
        v = _cramers_v(a, b)
        assert v == 0.0

    def test_returns_float(self):
        a = pd.Series(["A", "B", "A"])
        b = pd.Series(["X", "X", "Y"])
        assert isinstance(_cramers_v(a, b), float)
