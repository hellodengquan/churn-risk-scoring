import pytest
from typing import List

from churn_risk.models import AccountBehavior, RiskScore, ScoringConfig, FeatureWeights
from churn_risk.scoring import score_accounts, get_risk_distribution
from churn_risk.preprocessing import (
    handle_missing_values,
    handle_outliers,
    detect_outliers_iqr,
    min_max_normalize,
    preprocess_pipeline,
    accounts_to_dataframe,
)
from churn_risk.output import (
    filter_by_risk_level,
    get_top_n,
    export_to_csv,
    export_to_json,
)


def _make_accounts(n: int = 5, **overrides) -> List[AccountBehavior]:
    accounts = []
    for i in range(n):
        base = {
            "account_id": f"TEST{i:03d}",
            "login_count_last_30d": 20 - i * 3,
            "login_count_last_7d": 6 - i,
            "last_login_days_ago": i * 5,
            "total_transactions_last_30d": 15 - i * 2,
            "total_transactions_last_90d": 40 - i * 5,
            "transaction_amount_last_30d": 15000 - i * 2000,
            "avg_session_minutes": 40 - i * 5,
            "support_tickets_last_30d": i,
            "refund_count_last_90d": i // 2,
            "payment_failures_last_30d": i // 3,
            "feature_usage_count": {"dashboard": 10, "report": 5},
            "email_open_rate": 0.85 - i * 0.1,
            "subscription_age_days": 300 - i * 20,
            "plan_level": "enterprise" if i < 2 else "business" if i < 4 else "basic",
        }
        base.update(overrides)
        accounts.append(AccountBehavior(**base))
    return accounts


class TestEmptyDataset:
    def test_score_empty_list(self):
        accounts: List[AccountBehavior] = []
        config = ScoringConfig()
        scores = score_accounts(accounts, config)
        assert scores == []

    def test_distribution_empty(self):
        scores: List[RiskScore] = []
        dist = get_risk_distribution(scores)
        assert dist == {"critical": 0, "high": 0, "medium": 0, "low": 0}

    def test_filter_empty(self):
        scores: List[RiskScore] = []
        result = filter_by_risk_level(scores, "low")
        assert result == []

    def test_top_n_empty(self):
        scores: List[RiskScore] = []
        result = get_top_n(scores, 5)
        assert result == []

    def test_preprocess_empty(self):
        accounts: List[AccountBehavior] = []
        df = accounts_to_dataframe(accounts)
        assert len(df) == 0

    def test_preprocess_pipeline_empty(self):
        accounts: List[AccountBehavior] = []
        with pytest.raises(Exception):
            preprocess_pipeline(accounts)


class TestExtremeValues:
    def test_all_zero_features(self):
        accounts = _make_accounts(
            n=3,
            login_count_last_30d=0,
            login_count_last_7d=0,
            total_transactions_last_30d=0,
            total_transactions_last_90d=0,
            transaction_amount_last_30d=0.0,
            avg_session_minutes=0.0,
            support_tickets_last_30d=0,
            refund_count_last_90d=0,
            payment_failures_last_30d=0,
            email_open_rate=0.0,
        )
        config = ScoringConfig()
        scores = score_accounts(accounts, config)
        assert len(scores) == 3
        for s in scores:
            assert 0.0 <= s.total_score <= 100.0
            assert s.risk_level in {"critical", "high", "medium", "low"}

    def test_all_max_values(self):
        accounts = _make_accounts(
            n=3,
            login_count_last_30d=10000,
            login_count_last_7d=2500,
            last_login_days_ago=0,
            total_transactions_last_30d=9999,
            total_transactions_last_90d=29997,
            transaction_amount_last_30d=9999999.99,
            avg_session_minutes=999.0,
            support_tickets_last_30d=0,
            refund_count_last_90d=0,
            payment_failures_last_30d=0,
            email_open_rate=1.0,
            subscription_age_days=99999,
        )
        config = ScoringConfig()
        scores = score_accounts(accounts, config)
        assert len(scores) == 3
        for s in scores:
            assert 0.0 <= s.total_score <= 100.0

    def test_last_login_very_large(self):
        accounts = _make_accounts(n=2, last_login_days_ago=99999)
        config = ScoringConfig()
        scores = score_accounts(accounts, config)
        assert len(scores) == 2
        for s in scores:
            assert s.total_score >= 50.0

    def test_single_account(self):
        accounts = _make_accounts(n=1)
        config = ScoringConfig()
        scores = score_accounts(accounts, config)
        assert len(scores) == 1
        assert scores[0].risk_percentile == 100.0

    def test_two_accounts_extreme_contrast(self):
        good = AccountBehavior(
            account_id="GOOD001",
            login_count_last_30d=100,
            login_count_last_7d=25,
            last_login_days_ago=0,
            total_transactions_last_30d=50,
            total_transactions_last_90d=150,
            transaction_amount_last_30d=100000.0,
            avg_session_minutes=120.0,
            support_tickets_last_30d=0,
            refund_count_last_90d=0,
            payment_failures_last_30d=0,
            email_open_rate=1.0,
            subscription_age_days=1000,
            plan_level="enterprise",
        )
        bad = AccountBehavior(
            account_id="BAD001",
            login_count_last_30d=0,
            login_count_last_7d=0,
            last_login_days_ago=365,
            total_transactions_last_30d=0,
            total_transactions_last_90d=0,
            transaction_amount_last_30d=0.0,
            avg_session_minutes=0.0,
            support_tickets_last_30d=20,
            refund_count_last_90d=10,
            payment_failures_last_30d=15,
            email_open_rate=0.0,
            subscription_age_days=10,
            plan_level="basic",
        )
        scores = score_accounts([good, bad], ScoringConfig())
        assert len(scores) == 2
        assert scores[0].account_id == "BAD001"
        assert scores[1].account_id == "GOOD001"
        assert scores[0].total_score > scores[1].total_score


class TestMissingValues:
    def test_handle_missing_median(self):
        import pandas as pd
        import numpy as np

        df = pd.DataFrame({
            "account_id": ["A", "B", "C", "D", "E"],
            "login_count_last_30d": [10, None, 20, None, 30],
            "last_login_days_ago": [5, 10, None, 20, 25],
            "transaction_amount_last_30d": [100.0, 200.0, None, None, 500.0],
            "plan_level": ["basic", None, "business", "basic", None],
        })

        result, fill_map = handle_missing_values(df, strategy="median")
        assert result["login_count_last_30d"].isna().sum() == 0
        assert result["last_login_days_ago"].isna().sum() == 0
        assert result["transaction_amount_last_30d"].isna().sum() == 0
        assert result["plan_level"].isna().sum() == 0
        assert fill_map["login_count_last_30d"] == 20.0

    def test_handle_missing_mean(self):
        import pandas as pd

        df = pd.DataFrame({
            "account_id": ["A", "B", "C"],
            "login_count_last_30d": [10, None, 20],
        })
        result, _ = handle_missing_values(df, strategy="mean")
        assert result["login_count_last_30d"].iloc[1] == 15.0

    def test_handle_missing_zero(self):
        import pandas as pd

        df = pd.DataFrame({
            "account_id": ["A", "B"],
            "login_count_last_30d": [None, None],
        })
        result, _ = handle_missing_values(df, strategy="zero")
        assert result["login_count_last_30d"].tolist() == [0.0, 0.0]


class TestOutlierDetection:
    def test_iqr_detection_basic(self):
        import pandas as pd
        import numpy as np

        values = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100])
        mask, lower, upper = detect_outliers_iqr(values)
        assert mask[-1] == True
        assert mask[:-1].sum() == 0

    def test_handle_outliers_clip(self):
        import pandas as pd

        df = pd.DataFrame({
            "account_id": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"],
            "login_count_last_30d": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1000],
        })
        result, bounds, counts = handle_outliers(df, method="clip")
        assert result["login_count_last_30d"].max() <= 1000
        assert counts["login_count_last_30d"] >= 1

    def test_handle_outliers_remove(self):
        import pandas as pd

        df = pd.DataFrame({
            "account_id": [f"X{i}" for i in range(11)],
            "login_count_last_30d": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1000],
        })
        result, _, _ = handle_outliers(df, method="remove")
        assert len(result) < len(df)

    def test_no_outliers_uniform_data(self):
        import pandas as pd

        df = pd.DataFrame({
            "account_id": [f"X{i}" for i in range(10)],
            "login_count_last_30d": [5] * 10,
        })
        _, _, counts = handle_outliers(df, method="clip")
        assert counts.get("login_count_last_30d", 0) == 0


class TestNormalization:
    def test_min_max_normalization(self):
        import pandas as pd

        df = pd.DataFrame({
            "login_count_last_30d": [0, 10, 20, 30, 40, 50],
        })
        result, ranges = min_max_normalize(df)
        assert result["login_count_last_30d"].min() == 0.0
        assert result["login_count_last_30d"].max() == 1.0
        assert ranges["login_count_last_30d"] == (0.0, 50.0)

    def test_min_max_constant_values(self):
        import pandas as pd

        df = pd.DataFrame({
            "login_count_last_30d": [5, 5, 5, 5, 5],
        })
        result, _ = min_max_normalize(df)
        assert result["login_count_last_30d"].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]


class TestScoringEdgeCases:
    def test_risk_level_boundaries(self):
        from churn_risk.scoring import _determine_risk_level

        assert _determine_risk_level(95.0) == "critical"
        assert _determine_risk_level(90.0) == "critical"
        assert _determine_risk_level(80.0) == "high"
        assert _determine_risk_level(70.0) == "high"
        assert _determine_risk_level(60.0) == "medium"
        assert _determine_risk_level(40.0) == "medium"
        assert _determine_risk_level(30.0) == "low"
        assert _determine_risk_level(0.0) == "low"

    def test_custom_weights_sum(self):
        weights = FeatureWeights(
            login_frequency_30d=0.2,
            login_trend=0.1,
            recency=0.2,
            transaction_frequency=0.15,
            transaction_amount=0.1,
            engagement_depth=0.1,
            support_tickets=0.05,
            payment_issues=0.05,
            email_engagement=0.05,
        )
        total = sum([
            weights.login_frequency_30d,
            weights.login_trend,
            weights.recency,
            weights.transaction_frequency,
            weights.transaction_amount,
            weights.engagement_depth,
            weights.support_tickets,
            weights.payment_issues,
            weights.email_engagement,
        ])
        assert abs(total - 1.0) < 0.01

    def test_score_ordering(self):
        accounts = _make_accounts(n=10)
        scores = score_accounts(accounts, ScoringConfig())
        for i in range(len(scores) - 1):
            assert scores[i].total_score >= scores[i + 1].total_score

    def test_percentile_values(self):
        accounts = _make_accounts(n=20)
        scores = score_accounts(accounts, ScoringConfig())
        assert scores[0].risk_percentile == 5.0
        assert scores[-1].risk_percentile == 100.0
        for s in scores:
            assert 0.0 < s.risk_percentile <= 100.0


class TestOutputEdgeCases:
    def test_filter_invalid_level(self):
        with pytest.raises(ValueError):
            filter_by_risk_level([], "invalid_level")

    def test_top_n_larger_than_scores(self):
        accounts = _make_accounts(n=5)
        scores = score_accounts(accounts, ScoringConfig())
        result = get_top_n(scores, 100)
        assert len(result) == 5

    def test_export_empty_csv(self, tmp_path):
        output = tmp_path / "empty.csv"
        export_to_csv([], str(output))
        assert output.exists()

    def test_export_empty_json(self, tmp_path):
        output = tmp_path / "empty.json"
        export_to_json([], str(output))
        assert output.exists()


class TestPreprocessingPipeline:
    def test_full_pipeline(self):
        accounts = _make_accounts(n=15)
        result = preprocess_pipeline(accounts)
        assert "original_df" in result
        assert "clean_df" in result
        assert "normalized_df" in result
        assert "processed_accounts" in result
        assert "fill_map" in result
        assert "outlier_bounds" in result
        assert len(result["processed_accounts"]) >= 1
        assert len(result["normalized_df"]) >= 1

    def test_pipeline_all_strategies(self):
        accounts = _make_accounts(n=10)
        for norm in ["minmax", "zscore", "robust"]:
            result = preprocess_pipeline(accounts, normalization=norm)
            assert len(result["processed_accounts"]) >= 1
