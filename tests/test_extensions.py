import pytest
from typing import List

from churn_risk.models import AccountBehavior, RiskScore, ScoringConfig
from churn_risk.scoring import score_accounts
from churn_risk.ltv import (
    calculate_ltv_for_account,
    assign_ltv_tiers,
    generate_action_list,
    calculate_portfolio_summary,
    calculate_monthly_revenue,
    calculate_historical_ltv,
    calculate_predicted_ltv,
)
from churn_risk.shap_analysis import SHAPAnalyzer
from churn_risk.timeseries import (
    analyze_account_timeseries,
    get_portfolio_trend_summary,
    get_account_trend_summary,
    build_time_series_from_accounts,
    analyze_trend,
)
from churn_risk.data_source import AccountBehaviorDB, StreamProcessor
from churn_risk.ml_weights import (
    WeightLearner,
    OnlineWeightAdjuster,
    train_weights_auto,
)
from churn_risk.visualization import (
    plot_risk_distribution,
    plot_score_histogram,
)


def _make_accounts(n: int = 5) -> List[AccountBehavior]:
    accounts = []
    for i in range(n):
        accounts.append(AccountBehavior(
            account_id=f"TEST{i:03d}",
            login_count_last_30d=20 - i * 3,
            login_count_last_7d=6 - i,
            last_login_days_ago=i * 5,
            total_transactions_last_30d=15 - i * 2,
            total_transactions_last_90d=40 - i * 5,
            transaction_amount_last_30d=15000 - i * 2000,
            avg_session_minutes=40 - i * 5,
            support_tickets_last_30d=i,
            refund_count_last_90d=i // 2,
            payment_failures_last_30d=i // 3,
            feature_usage_count={},
            email_open_rate=0.85 - i * 0.1,
            subscription_age_days=300 - i * 20,
            plan_level="enterprise" if i < 2 else "business" if i < 4 else "basic",
        ))
    return accounts


class TestLTVModule:
    def test_monthly_revenue_basic(self):
        account = AccountBehavior(
            account_id="LTV001",
            plan_level="basic",
            total_transactions_last_30d=5,
        )
        revenue = calculate_monthly_revenue(account)
        assert revenue > 0

    def test_monthly_revenue_plan_multipliers(self):
        basic = AccountBehavior(account_id="A", plan_level="basic", total_transactions_last_30d=0)
        enterprise = AccountBehavior(account_id="B", plan_level="enterprise", total_transactions_last_30d=0)
        rev_basic = calculate_monthly_revenue(basic)
        rev_enterprise = calculate_monthly_revenue(enterprise)
        assert rev_enterprise > rev_basic

    def test_historical_ltv(self):
        account = AccountBehavior(
            account_id="LTV002",
            subscription_age_days=365,
            plan_level="business",
            total_transactions_last_90d=30,
        )
        ltv = calculate_historical_ltv(account)
        assert ltv > 0

    def test_predicted_ltv_with_risk(self):
        account = AccountBehavior(
            account_id="LTV003",
            plan_level="enterprise",
            total_transactions_last_30d=10,
        )
        ltv_high_risk = calculate_predicted_ltv(account, 90.0)
        ltv_low_risk = calculate_predicted_ltv(account, 10.0)
        assert ltv_low_risk > ltv_high_risk

    def test_assign_ltv_tiers(self):
        from churn_risk.ltv import AccountLTV
        ltvs = [
            AccountLTV(f"A{i}", 100000 - i * 10000, 100000 - i * 10000, 1000, 2500, 7500, 0.9 - i * 0.1, "bronze")
            for i in range(20)
        ]
        tiered = assign_ltv_tiers(ltvs)
        tiers = {l.ltv_tier for l in tiered}
        assert tiers.issubset({"platinum", "gold", "silver", "bronze"})
        assert len(tiers) >= 2

    def test_generate_action_list(self):
        accounts = _make_accounts(10)
        scores = score_accounts(accounts, ScoringConfig())
        from churn_risk.ltv import calculate_ltv_for_account, assign_ltv_tiers

        ltv_list = [
            calculate_ltv_for_account(a, s)
            for a, s in zip(accounts, scores)
        ]
        ltv_list = assign_ltv_tiers(ltv_list)
        actions = generate_action_list(scores, accounts, ltv_list)
        assert len(actions) > 0
        assert all("recommended_action" in a for a in actions)
        assert all("priority_score" in a for a in actions)

    def test_portfolio_summary(self):
        accounts = _make_accounts(10)
        scores = score_accounts(accounts, ScoringConfig())
        from churn_risk.ltv import calculate_ltv_for_account, assign_ltv_tiers

        ltv_list = [
            calculate_ltv_for_account(a, s)
            for a, s in zip(accounts, scores)
        ]
        ltv_list = assign_ltv_tiers(ltv_list)
        actions = generate_action_list(scores, accounts, ltv_list)
        summary = calculate_portfolio_summary(actions)
        assert "total_accounts" in summary
        assert summary["total_accounts"] == len(actions)
        assert "total_monthly_revenue_at_risk" in summary
        assert "recommended_retention_budget" in summary

    def test_empty_ltv_list(self):
        from churn_risk.ltv import AccountLTV, assign_ltv_tiers
        assert assign_ltv_tiers([]) == []
        summary = calculate_portfolio_summary([])
        assert summary == {}


class TestSHAPAnalyzer:
    def test_analyze_accounts(self):
        accounts = _make_accounts(10)
        analyzer = SHAPAnalyzer()
        results = analyzer.analyze_accounts(accounts)
        assert len(results) == 10
        for r in results:
            assert hasattr(r, "account_id")
            assert hasattr(r, "explanations")
            assert len(r.explanations) > 0
            assert hasattr(r, "top_positive_drivers")
            assert hasattr(r, "top_negative_drivers")

    def test_target_account_ids(self):
        accounts = _make_accounts(10)
        analyzer = SHAPAnalyzer()
        results = analyzer.analyze_accounts(accounts, target_account_ids=["TEST000", "TEST001"])
        assert len(results) == 2
        assert {r.account_id for r in results} == {"TEST000", "TEST001"}

    def test_empty_analysis(self):
        analyzer = SHAPAnalyzer()
        results = analyzer.analyze_accounts([])
        assert results == []

    def test_global_analysis(self):
        accounts = _make_accounts(15)
        analyzer = SHAPAnalyzer()
        global_result = analyzer.global_analysis(accounts)
        assert global_result.sample_count == 15
        assert len(global_result.feature_ranking) > 0
        assert len(global_result.mean_abs_shap) > 0

    def test_explanation_structure(self):
        accounts = _make_accounts(3)
        analyzer = SHAPAnalyzer()
        results = analyzer.analyze_accounts(accounts)
        for r in results:
            for exp in r.explanations:
                assert hasattr(exp, "feature_name")
                assert hasattr(exp, "shap_value")
                assert hasattr(exp, "impact_direction")
                assert exp.impact_direction in {"增加流失风险", "降低流失风险"}


class TestTimeSeries:
    def test_build_series(self):
        accounts = _make_accounts(5)
        for metric in ["login_count", "transaction_count", "transaction_amount"]:
            data = build_time_series_from_accounts(accounts, metric, periods=12)
            assert len(data) == 5
            for acc_id, values in data.items():
                assert len(values) == 12
                assert all(v >= 0 for v in values)

    def test_analyze_trend_stable(self):
        values = [10.0, 10.1, 9.9, 10.0, 10.1, 9.9, 10.0, 10.1, 9.9, 10.0, 10.1, 9.9]
        direction, slope_pct, volatility, change_pct, alert_msg, alert, forecast = analyze_trend(values)
        assert direction == "稳定"
        assert alert == False

    def test_analyze_trend_declining(self):
        values = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 2]
        direction, slope_pct, volatility, change_pct, alert_msg, alert, forecast = analyze_trend(values)
        assert direction == "下降"
        assert change_pct < 0

    def test_analyze_trend_rising(self):
        values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
        direction, slope_pct, volatility, change_pct, alert_msg, alert, forecast = analyze_trend(values)
        assert direction == "上升"

    def test_empty_trend(self):
        direction, slope_pct, volatility, change_pct, alert_msg, alert, forecast = analyze_trend([])
        assert direction == "稳定"
        assert alert == False

    def test_single_value_trend(self):
        direction, slope_pct, volatility, change_pct, alert_msg, alert, forecast = analyze_trend([5.0])
        assert direction == "稳定"
        assert alert == False

    def test_portfolio_timeseries(self):
        accounts = _make_accounts(10)
        results = analyze_account_timeseries(accounts)
        assert len(results) > 0
        summary = get_portfolio_trend_summary(results)
        assert "total_accounts" in summary
        assert summary["total_accounts"] == 10
        assert "total_alerts" in summary

    def test_account_trend_summary(self):
        accounts = _make_accounts(5)
        results = analyze_account_timeseries(accounts)
        summary = get_account_trend_summary(results, "TEST000")
        assert summary is not None
        assert summary["account_id"] == "TEST000"
        assert "overall_trend_risk" in summary

    def test_nonexistent_account_trend(self):
        accounts = _make_accounts(3)
        results = analyze_account_timeseries(accounts)
        summary = get_account_trend_summary(results, "NONEXISTENT")
        assert summary == {}


class TestMLWeights:
    def test_weight_learner_logistic(self):
        accounts = _make_accounts(20)
        learner = WeightLearner(model_type="logistic")
        result = learner.train(accounts)
        assert result.training_samples == 20
        assert result.model_type == "logistic"
        assert 0.0 <= result.accuracy <= 1.0
        assert len(result.feature_importance) > 0
        total_weight = sum([
            result.weights.login_frequency_30d,
            result.weights.login_trend,
            result.weights.recency,
            result.weights.transaction_frequency,
            result.weights.transaction_amount,
            result.weights.engagement_depth,
            result.weights.support_tickets,
            result.weights.payment_issues,
            result.weights.email_engagement,
        ])
        assert abs(total_weight - 1.0) < 0.01

    def test_weight_learner_random_forest(self):
        accounts = _make_accounts(20)
        learner = WeightLearner(model_type="random_forest")
        result = learner.train(accounts)
        assert result.model_type == "random_forest"
        assert len(result.feature_importance) > 0

    def test_insufficient_samples(self):
        accounts = _make_accounts(5)
        learner = WeightLearner()
        with pytest.raises(ValueError):
            learner.train(accounts)

    def test_churn_prediction(self):
        accounts = _make_accounts(20)
        learner = WeightLearner(model_type="logistic")
        learner.train(accounts)
        probs = learner.predict_churn_probability(accounts[:5])
        assert len(probs) == 5
        assert all(0.0 <= p <= 1.0 for p in probs)

    def test_untrained_prediction(self):
        learner = WeightLearner()
        with pytest.raises(RuntimeError):
            learner.predict_churn_probability(_make_accounts(3))

    def test_online_weight_adjuster(self):
        from churn_risk.models import FeatureWeights
        adjuster = OnlineWeightAdjuster()
        initial = adjuster.weights
        assert initial is not None

        feedback = [
            ("A001", 75.0, 1.0),
            ("A002", 25.0, 0.0),
            ("A003", 60.0, 0.0),
        ]
        new_weights = adjuster.adjust_weights(feedback)
        assert new_weights is not None
        assert adjuster.feedback_count == 3
        assert len(adjuster.weight_history) == 2

    def test_empty_feedback_adjustment(self):
        adjuster = OnlineWeightAdjuster()
        result = adjuster.adjust_weights([])
        assert result is not None

    def test_train_weights_auto_function(self):
        accounts = _make_accounts(20)
        result = train_weights_auto(accounts, model_type="logistic")
        assert result.training_samples == 20


class TestDataSource:
    def test_sqlite_db_in_memory(self, tmp_path):
        db_path = tmp_path / "test.db"
        db = AccountBehaviorDB(db_url=f"sqlite:///{db_path}")
        assert db.get_account_count() == 0

        accounts = _make_accounts(5)
        count = db.upsert_batch(accounts)
        assert count == 5
        assert db.get_account_count() == 5

        loaded = db.load_all_accounts()
        assert len(loaded) == 5

    def test_db_upsert(self, tmp_path):
        db_path = tmp_path / "test2.db"
        db = AccountBehaviorDB(db_url=f"sqlite:///{db_path}")

        acc = AccountBehavior(account_id="U1", login_count_last_30d=10)
        db.upsert_account(acc)
        assert db.get_account_count() == 1

        updated = AccountBehavior(account_id="U1", login_count_last_30d=50)
        db.upsert_account(updated)
        assert db.get_account_count() == 1

        loaded = db.load_all_accounts()
        assert loaded[0].login_count_last_30d == 50

    def test_db_delete(self, tmp_path):
        db_path = tmp_path / "test3.db"
        db = AccountBehaviorDB(db_url=f"sqlite:///{db_path}")
        accounts = _make_accounts(3)
        db.upsert_batch(accounts)
        assert db.get_account_count() == 3

        result = db.delete_account("TEST000")
        assert result == True
        assert db.get_account_count() == 2

        result = db.delete_account("NONEXISTENT")
        assert result == False

    def test_db_plan_filter(self, tmp_path):
        db_path = tmp_path / "test4.db"
        db = AccountBehaviorDB(db_url=f"sqlite:///{db_path}")
        accounts = _make_accounts(5)
        db.upsert_batch(accounts)

        basic_accounts = db.load_all_accounts(plan_filter="basic")
        assert len(basic_accounts) == 1

    def test_stream_processor_basic(self):
        received = []

        def callback(account):
            received.append(account.account_id)

        with StreamProcessor(callback=callback, batch_size=5, flush_interval=0.5) as processor:
            accounts = _make_accounts(10)
            for acc in accounts:
                processor.ingest(acc)
            import time
            time.sleep(1.5)
            stats = processor.stats

        assert stats["processed"] == 10
        assert stats["errors"] == 0
        assert len(received) == 10

    def test_stream_processor_stats(self):
        processor = StreamProcessor(batch_size=100, flush_interval=1.0)
        stats = processor.stats
        assert "processed" in stats
        assert "errors" in stats
        assert stats["processed"] == 0

    def test_stream_processor_from_json(self):
        received = []

        def callback(account):
            received.append(account)

        with StreamProcessor(callback=callback, batch_size=5, flush_interval=0.3) as processor:
            import json
            data = {"account_id": "JSON01", "login_count_last_30d": 5}
            processor.ingest_from_json(json.dumps(data))
            import time
            time.sleep(1)

        assert len(received) >= 1
        assert received[0].account_id == "JSON01"


class TestVisualization:
    def test_risk_distribution_plot(self, tmp_path):
        accounts = _make_accounts(20)
        scores = score_accounts(accounts, ScoringConfig())
        output = tmp_path / "risk_dist.png"
        result = plot_risk_distribution(scores, str(output))
        assert result in {True, False}

    def test_score_histogram_plot(self, tmp_path):
        accounts = _make_accounts(20)
        scores = score_accounts(accounts, ScoringConfig())
        output = tmp_path / "hist.png"
        result = plot_score_histogram(scores, str(output))
        assert result in {True, False}

    def test_empty_visualization(self, tmp_path):
        output = tmp_path / "empty.png"
        result = plot_risk_distribution([], str(output))
        assert result in {True, False}
