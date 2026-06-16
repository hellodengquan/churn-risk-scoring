import unittest
import pytest
from dataclasses import asdict
from typing import List

import pandas as pd

from churn_risk.models import AccountBehavior, RiskScore, ScoringConfig
from churn_risk.scoring import score_accounts
from churn_risk.preprocessing import preprocess_pipeline
from churn_risk.ltv import (
    calculate_ltv_for_account,
    assign_ltv_tiers,
    generate_action_list,
    calculate_portfolio_summary,
    calculate_monthly_revenue,
    calculate_historical_ltv,
    calculate_predicted_ltv,
    LTVFallbackConfig,
    LTVFallbackEstimator,
    LTVFallbackResult,
    calculate_ltv_with_fallback,
)
from churn_risk.shap_analysis import (
    SHAPAnalyzer,
    SHAPSamplingConfig,
    SHAPLargeScaleSampler,
    SHAPSamplingReport,
    format_sampling_report,
    _build_feature_frame,
)
from churn_risk.timeseries import (
    analyze_account_timeseries,
    get_portfolio_trend_summary,
    get_account_trend_summary,
    build_time_series_from_accounts,
    analyze_trend,
    AutoForecaster,
    ForecastingConfig,
    ForecastingResult,
    format_forecasting_report,
)
from churn_risk.data_source import (
    AccountBehaviorDB,
    StreamProcessor,
    create_data_source,
    MongoDBSource,
    ClickHouseSource,
)
from churn_risk.ml_weights import (
    WeightLearner,
    OnlineWeightAdjuster,
    train_weights_auto,
    SGDWeightLearner,
    ConceptDriftDetector,
    DriftDetectionResult,
)
from churn_risk.visualization import (
    plot_risk_distribution,
    plot_score_histogram,
)
from churn_risk.output import (
    export_to_csv,
    export_to_json,
    get_terminal_width,
    _truncate_text,
    build_rich_risk_tree,
    build_rich_account_detail_tree,
    print_terminal_width_info,
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


class TestDataSourceAdapters(unittest.TestCase):
    def test_create_data_source_sqlite(self):
        source = create_data_source("sqlite", db_url="sqlite:///:memory:")
        assert isinstance(source, AccountBehaviorDB)

    def test_create_data_source_mongodb_available(self):
        source = create_data_source("mongodb")
        if source.available:
            assert isinstance(source, MongoDBSource)
        else:
            assert isinstance(source, MongoDBSource)
            assert source.available is False

    def test_create_data_source_clickhouse_available(self):
        source = create_data_source("clickhouse")
        if source.available:
            assert isinstance(source, ClickHouseSource)
        else:
            assert isinstance(source, ClickHouseSource)
            assert source.available is False

    def test_create_data_source_invalid(self):
        with pytest.raises(ValueError, match="Unsupported data source type"):
            create_data_source("unknown_db")

    def test_mongodb_source_upsert_not_available(self):
        source = MongoDBSource()
        if not source.available:
            with pytest.raises(ImportError, match="pymongo is required"):
                source.upsert_account(_make_accounts(1)[0])

    def test_clickhouse_source_not_available(self):
        source = ClickHouseSource()
        if not source.available:
            with pytest.raises(ImportError, match="clickhouse-connect is required"):
                source.upsert_account(_make_accounts(1)[0])


class TestSGDConceptDrift(unittest.TestCase):
    def test_sgd_learner_initialization(self):
        learner = SGDWeightLearner()
        assert learner.total_updates == 0
        assert learner._model is None

    def test_sgd_partial_fit_basic(self):
        accounts = _make_accounts(15)
        learner = SGDWeightLearner()
        result = learner.partial_fit(accounts)
        assert learner.total_updates == 1
        assert result is None

    def test_sgd_partial_fit_multiple_batches(self):
        accounts = _make_accounts(15)
        learner = SGDWeightLearner()
        for i in range(5):
            result = learner.partial_fit(accounts)
        assert learner.total_updates == 5
        assert isinstance(result, DriftDetectionResult)

    def test_sgd_get_weights_untrained(self):
        learner = SGDWeightLearner()
        weights = learner.get_weights()
        assert weights is not None
        assert weights.recency > 0

    def test_drift_detector_no_baseline(self):
        detector = ConceptDriftDetector()
        result = detector.detect()
        assert result.is_drift is False
        assert result.drift_type == "none"

    def test_drift_detector_insufficient_data(self):
        detector = ConceptDriftDetector()
        features = pd.DataFrame({"f1": [1.0, 2.0]})
        detector.set_baseline(features, 0.9)
        for i in range(5):
            detector.record({"f1": float(i)}, 0.5, float(i % 2))
        result = detector.detect()
        assert result.is_drift is False
        assert "窗口数据不足" in result.message

    def test_drift_detector_record(self):
        detector = ConceptDriftDetector(window_size=10)
        features = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0, 5.0]})
        detector.set_baseline(features, 0.9)
        for i in range(15):
            detector.record({"f1": float(i)}, 0.5, float(i % 2))
        assert len(detector._recent_predictions) <= 30
        assert len(detector._recent_features) <= 30


class TestLTVFallback(unittest.TestCase):
    def test_fallback_config_default(self):
        config = LTVFallbackConfig()
        assert config.strategy == "plan_group_median"
        assert config.knn_neighbors == 5

    def test_fallback_estimator_no_reference(self):
        estimator = LTVFallbackEstimator()
        account = _make_accounts(1)[0]
        result = estimator.estimate(account)
        assert result.fallback_strategy_used == "conservative_default"
        assert result.confidence == 0.15

    def test_fallback_estimator_with_reference(self):
        ref_accounts = _make_accounts(10)
        ref_scores = score_accounts(ref_accounts, ScoringConfig())
        ref_ltvs = [calculate_ltv_for_account(a, s) for a, s in zip(ref_accounts, ref_scores)]

        estimator = LTVFallbackEstimator(ref_accounts, ref_ltvs)
        missing_account = AccountBehavior(
            account_id="NEW001",
            subscription_age_days=1,
            login_count_last_30d=0,
            total_transactions_last_30d=0,
            plan_level="basic",
        )
        result = estimator.estimate(missing_account)
        assert result is not None
        assert result.ltv_estimated > 0
        assert result.monthly_revenue_estimated > 0

    def test_calculate_ltv_with_fallback_mixed(self):
        valid_accounts = _make_accounts(8)
        missing_accounts = [
            AccountBehavior(
                account_id=f"MISS{i:03d}",
                subscription_age_days=1,
                login_count_last_30d=0,
                total_transactions_last_30d=0,
                plan_level="basic",
            )
            for i in range(3)
        ]
        all_accounts = valid_accounts + missing_accounts
        scores = score_accounts(all_accounts, ScoringConfig())

        ltvs, fallback = calculate_ltv_with_fallback(all_accounts, scores)
        assert len(ltvs) == 11
        assert len(fallback) == 3

    def test_fallback_report_generation(self):
        ref_accounts = _make_accounts(10)
        ref_scores = score_accounts(ref_accounts, ScoringConfig())
        ref_ltvs = [calculate_ltv_for_account(a, s) for a, s in zip(ref_accounts, ref_scores)]
        estimator = LTVFallbackEstimator(ref_accounts, ref_ltvs)

        results = estimator.batch_estimate(_make_accounts(5))
        report = estimator.generate_fallback_report(results)
        assert "兜底估算账号数: 5" in report
        assert "平均置信度" in report


class TestSHAPSampling(unittest.TestCase):
    def test_sampler_config_default(self):
        config = SHAPSamplingConfig()
        assert config.max_samples == 500
        assert config.strategy == "auto"

    def test_sampler_small_dataset_full(self):
        accounts = _make_accounts(50)
        sampler = SHAPLargeScaleSampler()
        sampled, report = sampler.sample(accounts)
        assert len(sampled) == 50
        assert report.strategy_used == "full"

    def test_sampler_large_dataset_stratified(self):
        accounts = _make_accounts(1000)
        sampler = SHAPLargeScaleSampler()
        sampled, report = sampler.sample(accounts)
        assert len(sampled) == 500
        assert report.strategy_used == "stratified"
        assert report.sampling_ratio < 1.0

    def test_sampler_very_large_importance_weighted(self):
        accounts = _make_accounts(2000)
        sampler = SHAPLargeScaleSampler()
        sampled, report = sampler.sample(accounts)
        assert len(sampled) == 500
        assert report.strategy_used == "importance_weighted"
        assert report.sampling_ratio < 1.0

    def test_sampler_empty_dataset(self):
        sampler = SHAPLargeScaleSampler()
        sampled, report = sampler.sample([])
        assert len(sampled) == 0
        assert report.strategy_used == "empty"

    def test_sampler_auto_select_strategy(self):
        sampler = SHAPLargeScaleSampler()
        assert sampler._auto_select_strategy(100) == "full"
        assert sampler._auto_select_strategy(1000) == "stratified"
        assert sampler._auto_select_strategy(2000) == "importance_weighted"
        assert sampler._auto_select_strategy(500) == "full"
        assert sampler._auto_select_strategy(501) == "stratified"

    def test_adaptive_background_samples(self):
        accounts = _make_accounts(200)
        X = _build_feature_frame(accounts)
        sampled = SHAPLargeScaleSampler.adaptive_background_samples(X, target=50)
        assert len(sampled) <= 50

    def test_sampling_report_format(self):
        accounts = _make_accounts(200)
        sampler = SHAPLargeScaleSampler()
        _, report = sampler.sample(accounts)
        formatted = format_sampling_report(report)
        assert "总账号数: 200" in formatted
        assert "采样策略" in formatted


class TestAutoForecaster(unittest.TestCase):
    def test_forecaster_available_models(self):
        forecaster = AutoForecaster()
        assert "linear" in forecaster.available_models

    def test_forecaster_linear_short_series(self):
        forecaster = AutoForecaster()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = forecaster.forecast(values, periods=3)
        assert result.model_used == "linear"
        assert len(result.forecast_values) == 3
        assert "样本数不足" in result.selection_reason

    def test_forecaster_analyze_characteristics(self):
        forecaster = AutoForecaster()
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        chars = forecaster._analyze_series_characteristics(values)
        assert chars["length"] == 10
        assert chars["trend_strength"] > 0

    def test_forecast_report_format(self):
        forecaster = AutoForecaster()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = forecaster.forecast(values, periods=2)
        report = format_forecasting_report(result, "登录频次")
        assert "时间序列预测报告" in report
        assert "样本数不足" in report


class TestRichTreeAdaptive(unittest.TestCase):
    def test_get_terminal_width_default(self):
        width = get_terminal_width()
        assert isinstance(width, int)
        assert width >= 80

    def test_truncate_text(self):
        assert _truncate_text("hello world", 8) == "hello..."
        assert _truncate_text("hi", 8) == "hi"
        assert _truncate_text("test", 3) == "tes"

    def test_build_rich_tree_narrow(self):
        accounts = _make_accounts(10)
        scores = score_accounts(accounts, ScoringConfig())
        tree = build_rich_risk_tree(scores, terminal_width=60)
        assert tree is not None

    def test_build_rich_tree_wide(self):
        accounts = _make_accounts(10)
        scores = score_accounts(accounts, ScoringConfig())
        tree = build_rich_risk_tree(scores, terminal_width=150)
        assert tree is not None

    def test_build_account_detail_tree(self):
        accounts = _make_accounts(1)
        scores = score_accounts(accounts, ScoringConfig())
        tree = build_rich_account_detail_tree(
            accounts[0].account_id,
            scores[0],
            account=accounts[0],
            terminal_width=120,
        )
        assert tree is not None

    def test_terminal_width_info(self):
        info = print_terminal_width_info()
        assert "终端宽度" in info
        assert "显示模式" in info


class TestUnicodeFieldBoundaries(unittest.TestCase):
    UNICODE_TEST_CASES = [
        ("ascii_simple", "USER123", False),
        ("chinese_chars", "用户_测试_账号", True),
        ("japanese_chars", "ユーザー123", True),
        ("korean_chars", "사용자테스트", True),
        ("emoji_account", "user_🔥_test", True),
        ("mixed_unicode", "用户_test_123", True),
        ("empty_string", "", False),
        ("special_ascii", "user@#$%", False),
        ("unicode_control", "user\u0000test", True),
        ("unicode_surrogate", "user𓀀test", True),
        ("cyrillic", "пользователь", True),
        ("arabic", "المستخدم", True),
        ("devanagari", "उपयोगकर्ता", True),
        ("combined_accents", "user_nm_aöe_ß", True),
        ("right_to_left", "מִשׁתַמֵשׁ", True),
        ("unicode_max_bmp", "user_" + "\uFFFF", True),
        ("long_unicode", "测试账号_" * 10, True),
        ("unicode_spaces", "user\u2003test", True),
        ("unicode_zero_width", "user\u200Btest", True),
        ("thai_chars", "ผู้ใช้", True),
    ]

    def test_account_behavior_unicode_account_id(self):
        for test_name, account_id, has_unicode in self.UNICODE_TEST_CASES:
            with self.subTest(test=test_name):
                account = AccountBehavior(
                    account_id=account_id,
                    login_count_last_30d=10,
                    last_login_days_ago=5,
                )
                assert account.account_id == account_id

                if has_unicode:
                    encoded = account_id.encode("utf-8")
                    decoded = encoded.decode("utf-8")
                    assert decoded == account_id

    def test_plan_level_unicode_boundaries(self):
        unicode_plans = [
            ("基础版", True),
            ("企业版_プレミアム", True),
            ("enterprise", False),
            ("프리미엄", True),
            ("Базовый", True),
        ]
        for plan_name, has_unicode in unicode_plans:
            with self.subTest(plan=plan_name):
                account = AccountBehavior(
                    account_id="TEST001",
                    plan_level=plan_name,
                    login_count_last_30d=5,
                )
                assert account.plan_level == plan_name

    def test_json_serialization_unicode(self):
        unicode_accounts = [
            AccountBehavior(
                account_id="账号_中文_001",
                plan_level="基础版",
                login_count_last_30d=10,
                feature_usage_count={"feature_🔥": 5, "中文功能": 10},
            ),
            AccountBehavior(
                account_id="ユーザー_002",
                plan_level="プレミアム",
                login_count_last_30d=5,
                feature_usage_count={"機能_A": 3},
            ),
        ]
        for account in unicode_accounts:
            with self.subTest(account=account.account_id):
                import json
                data = asdict(account)
                json_str = json.dumps(data, ensure_ascii=False)
                loaded = json.loads(json_str)
                assert loaded["account_id"] == account.account_id
                assert loaded["plan_level"] == account.plan_level

    def test_unicode_truncation_boundary(self):
        long_unicode = "测试" * 50
        truncated = _truncate_text(long_unicode, 20)
        assert len(truncated) <= 20 + 3
        assert truncated.endswith("...")

    def test_feature_usage_count_unicode_keys(self):
        unicode_features = {
            "登录功能": 100,
            "交易模块_💰": 50,
            "サポート機能": 20,
            "normal_feature": 80,
        }
        account = AccountBehavior(
            account_id="TEST_UNICODE",
            feature_usage_count=unicode_features,
        )
        for key, value in unicode_features.items():
            assert account.feature_usage_count[key] == value

    def test_risk_score_with_unicode(self):
        unicode_ids = ["用户_001", "ユーザー_002", "사용자_003"]
        accounts = [
            AccountBehavior(account_id=uid, login_count_last_30d=5 + i, last_login_days_ago=2 + i)
            for i, uid in enumerate(unicode_ids)
        ]
        scores = score_accounts(accounts, ScoringConfig())
        assert len(scores) == 3
        for uid in unicode_ids:
            assert any(s.account_id == uid for s in scores)

    def test_preprocessing_pipeline_unicode(self):
        unicode_accounts = [
            AccountBehavior(
                account_id=f"账号{i:03d}_{'测试' * i}",
                login_count_last_30d=10 + i,
                last_login_days_ago=i + 1,
                plan_level="基础版" if i % 2 == 0 else "premium",
            )
            for i in range(5)
        ]
        result = preprocess_pipeline(unicode_accounts)
        assert len(result["processed_accounts"]) == 5

    def test_database_roundtrip_unicode(self):
        unicode_accounts = [
            AccountBehavior(
                account_id="中文_测试_" + str(i),
                login_count_last_30d=10,
                last_login_days_ago=5,
                plan_level="企业版",
            )
            for i in range(3)
        ]

        db = AccountBehaviorDB("sqlite:///:memory:")
        for acc in unicode_accounts:
            db.upsert_account(acc)

        loaded = db.load_all_accounts()
        assert len(loaded) == 3
        for orig in unicode_accounts:
            found = [a for a in loaded if a.account_id == orig.account_id]
            assert len(found) == 1
            assert found[0].plan_level == orig.plan_level

    def test_unicode_display_width_rich(self):
        test_cases = [
            ("short_ascii", 10, "USER001"),
            ("chinese_6chars", 12, "测试账号测试"),
            ("mixed_1", 11, "test_测试"),
            ("emoji", 11, "test_🔥"),
        ]
        for name, expected_width, text in test_cases:
            with self.subTest(test=name):
                truncated = _truncate_text(text, 10)
                assert len(truncated) <= 13

    def test_utf8_encoding_boundary_values(self):
        boundary_chars = [
            "\u007F",
            "\u0080",
            "\u07FF",
            "\u0800",
            "\uFFFF",
        ]
        for char in boundary_chars:
            with self.subTest(char_code=f"U+{ord(char):04X}"):
                account_id = f"test_{char}_end"
                account = AccountBehavior(account_id=account_id)
                assert account.account_id == account_id

    def test_unicode_sorting_stability(self):
        unicode_ids = [
            "用户_B",
            "用户_A",
            "USER_C",
            "ユーザー_A",
            "사용자_B",
        ]
        accounts = [
            AccountBehavior(account_id=uid, login_count_last_30d=10, last_login_days_ago=5)
            for uid in unicode_ids
        ]
        scores = score_accounts(accounts, ScoringConfig())
        sorted_ids = sorted([s.account_id for s in scores])
        assert len(sorted_ids) == 5
        assert sorted_ids == sorted(unicode_ids)


def test_export_csv_unicode(tmp_path):
    unicode_account = AccountBehavior(
        account_id="中文账号_001",
        plan_level="企业版",
        login_count_last_30d=15,
        last_login_days_ago=3,
    )
    scores = score_accounts([unicode_account], ScoringConfig())

    output_file = tmp_path / "unicode_output.csv"
    export_to_csv(scores, str(output_file))

    content = output_file.read_text(encoding="utf-8")
    assert "中文账号_001" in content
    assert "企业版" in content


def test_export_json_unicode(tmp_path):
    unicode_account = AccountBehavior(
        account_id="테스트_계정_001",
        plan_level="프리미엄",
        login_count_last_30d=20,
        last_login_days_ago=1,
    )
    scores = score_accounts([unicode_account], ScoringConfig())

    output_file = tmp_path / "unicode_output.json"
    export_to_json(scores, str(output_file))

    content = output_file.read_text(encoding="utf-8")
    assert "테스트_계정_001" in content
    assert "프리미엄" in content


class TestRTLCharacters(unittest.TestCase):
    RTL_TEST_CASES = [
        ("hebrew_basic", "מִשׁתַמֵשׁ", "希伯来语"),
        ("hebrew_simple", "משתמש", "希伯来语简单"),
        ("arabic_basic", "المستخدم", "阿拉伯语"),
        ("arabic_with_diacritics", "المُسْتَخْدِم", "阿拉伯语变音符号"),
        ("persian_farsi", "کاربر", "波斯语"),
        ("urdu", "صارف", "乌尔都语"),
        ("syriac", "ܡܫܬܡܫܐ", "叙利亚语"),
        ("arabic_indic_numerals", "用户١٢٣", "阿拉伯-印度数字"),
        ("hebrew_numerals", "אחדשניםשלשה", "希伯来语数字"),
        ("mixed_ltr_rtl", "user_מִשׁתַמֵשׁ_test", "混合 LTR+RTL"),
        ("mixed_rtl_ltr", "المستخدم_test_العربي", "混合 RTL+LTR"),
        ("rtl_with_emoji", "מִשׁתַמֵשׁ🔥עברית", "RTL+表情"),
        ("rtl_with_punctuation", "מִשׁתַמֵשׁ, עברית!", "RTL+标点"),
        ("bidi_control_chars", "\u202Bמשתמש\u202Ctest", "双向控制字符"),
        ("rtl_lro", "\u202Dמשתמשtest\u202C", "LRO 强制左到右"),
        ("rtl_rlo", "\u202Etestמשתמש\u202C", "RLO 强制右到左"),
        ("long_rtl_string", "المستخدم" * 5, "长 RTL 字符串"),
        ("rtl_with_zwj", "👨‍👩‍👧‍👦_משתמש", "RTL+零宽连字符"),
        ("rtl_with_newline", "משתמש\nעברית", "RTL+换行"),
        ("rtl_account_id_complex", "מִשׁתַמֵשׁ_العربي_کاربر", "多语言 RTL"),
    ]

    BIDI_TEST_STRINGS = [
        ("HelloשלוםWorld", "HelloשלוםWorld"),
        ("العربيEnglishعربي", "العربيEnglishعربي"),
        ("123עברית456", "123עברית456"),
    ]

    def test_rtl_account_behavior_creation(self):
        """测试包含 RTL 字符的 AccountBehavior 创建"""
        for test_name, account_id, description in self.RTL_TEST_CASES:
            with self.subTest(test=test_name, description=description):
                account = AccountBehavior(
                    account_id=account_id,
                    plan_level="פְרִימְיָם",
                    login_count_last_30d=10,
                    last_login_days_ago=5,
                )
                assert account.account_id == account_id
                assert account.plan_level == "פְרִימְיָם"

                utf8_encoded = account_id.encode("utf-8")
                utf8_decoded = utf8_encoded.decode("utf-8")
                assert utf8_decoded == account_id

    def test_rtl_risk_score(self):
        """测试包含 RTL 字符的 RiskScore 处理"""
        for test_name, account_id, description in self.RTL_TEST_CASES[:10]:
            with self.subTest(test=test_name):
                score = RiskScore(
                    account_id=account_id,
                    total_score=75.5,
                    feature_scores={},
                    risk_level="high",
                    risk_percentile=85.0,
                    plan_level="عالي",
                )
                assert score.account_id == account_id
                assert score.plan_level == "عالي"
                assert score.risk_level == "high"

                score_dict = asdict(score)
                assert score_dict["account_id"] == account_id
                assert score_dict["plan_level"] == "عالي"

    def test_rtl_json_serialization(self):
        """测试 RTL 字符的 JSON 序列化"""
        rtl_accounts = [
            AccountBehavior(
                account_id="מִשׁתַמֵשׁ_001",
                plan_level="פְרִימְיָם",
                login_count_last_30d=10,
                feature_usage_count={"תכונה_אחת": 5, "משתמש": 10},
            ),
            AccountBehavior(
                account_id="المستخدم_002",
                plan_level="بريميوم",
                login_count_last_30d=5,
                feature_usage_count={"الميزة_الاولى": 3},
            ),
            AccountBehavior(
                account_id="کاربر_003",
                plan_level="پریمیوم",
                login_count_last_30d=8,
                feature_usage_count={"ویژگی_اول": 7},
            ),
        ]

        for account in rtl_accounts:
            with self.subTest(account=account.account_id):
                import json
                data = asdict(account)
                json_str = json.dumps(data, ensure_ascii=False)
                loaded = json.loads(json_str)
                assert loaded["account_id"] == account.account_id
                assert loaded["plan_level"] == account.plan_level

                json_str_ascii = json.dumps(data, ensure_ascii=True)
                loaded_ascii = json.loads(json_str_ascii)
                assert loaded_ascii["account_id"] == account.account_id

    def test_rtl_csv_export(self, tmp_path=None):
        """测试 RTL 字符的 CSV 导出"""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = os.path.join(tmp_dir, "rtl_test_output.csv")

            rtl_accounts = [
                AccountBehavior(account_id="מִשׁתַמֵשׁ_001", plan_level="פְרִימְיָם",
                              login_count_last_30d=10, last_login_days_ago=3),
                AccountBehavior(account_id="المستخدم_002", plan_level="بريميوم",
                              login_count_last_30d=5, last_login_days_ago=7),
                AccountBehavior(account_id="کاربر_003", plan_level="پریمیوم",
                              login_count_last_30d=8, last_login_days_ago=2),
            ]
            scores = score_accounts(rtl_accounts, ScoringConfig())

            export_to_csv(scores, output_file)

            content = open(output_file, encoding="utf-8").read()

            for acc in rtl_accounts:
                assert acc.account_id in content
                assert acc.plan_level in content

            assert "account_id" in content
            assert "total_score" in content

    def test_rtl_bidi_text_preservation(self):
        """测试双向文本（Bidi）的正确保留"""
        for original, expected in self.BIDI_TEST_STRINGS:
            with self.subTest(text=original):
                account = AccountBehavior(
                    account_id=original,
                    login_count_last_30d=5,
                )
                assert account.account_id == expected

                import json
                data = asdict(account)
                json_str = json.dumps(data, ensure_ascii=False)
                loaded = json.loads(json_str)
                assert loaded["account_id"] == expected

    def test_rtl_string_operations(self):
        """测试 RTL 字符串的各种操作"""
        rtl_accounts = [
            AccountBehavior(account_id="מִשׁתַמֵשׁ_A", login_count_last_30d=15, last_login_days_ago=2),
            AccountBehavior(account_id="מִשׁתַמֵשׁ_B", login_count_last_30d=5, last_login_days_ago=10),
            AccountBehavior(account_id="المستخدم_A", login_count_last_30d=20, last_login_days_ago=1),
            AccountBehavior(account_id="المستخدم_B", login_count_last_30d=10, last_login_days_ago=5),
        ]

        scores = score_accounts(rtl_accounts, ScoringConfig())
        assert len(scores) == 4

        sorted_scores = sorted(scores, key=lambda s: s.account_id)
        assert sorted_scores[0].account_id < sorted_scores[1].account_id
        assert sorted_scores[2].account_id < sorted_scores[3].account_id

    def test_rtl_truncation(self):
        """测试 RTL 字符串截断处理"""
        from churn_risk.output import _truncate_text

        rtl_long_strings = [
            "מִשׁתַמֵשׁ" * 10,
            "المستخدم" * 10,
            "mixed_" + "עברית" * 8 + "_ltr",
        ]

        for rtl_text in rtl_long_strings:
            with self.subTest(length=len(rtl_text)):
                truncated = _truncate_text(rtl_text, 20)
                assert len(truncated) <= 20 + 3
                assert truncated.endswith("...") or len(truncated) == len(rtl_text)

                utf8_len = len(truncated.encode("utf-8"))
                assert utf8_len > 0

    def test_rtl_plan_level_boundaries(self):
        """测试 RTL 字符作为 plan_level 的边界情况"""
        rtl_plans = [
            ("פְרִימְיָם", "希伯来语高级版"),
            ("بريميوم", "阿拉伯语高级版"),
            ("پریمیوم", "波斯语高级版"),
            ("عادي", "阿拉伯语普通版"),
            ("בָּסִיס", "希伯来语基础版"),
        ]

        for plan_name, description in rtl_plans:
            with self.subTest(plan=plan_name, description=description):
                account = AccountBehavior(
                    account_id="TEST_RTL_001",
                    plan_level=plan_name,
                    login_count_last_30d=5,
                )
                assert account.plan_level == plan_name

                utf8_bytes = plan_name.encode("utf-8")
                decoded = utf8_bytes.decode("utf-8")
                assert decoded == plan_name

    def test_rtl_bidi_control_chars(self):
        """测试包含双向控制字符的 RTL 文本处理"""
        bidi_control_cases = [
            ("\u202Bמשתמש\u202C", "RLE 包裹"),
            ("\u202Dמשתמשtest\u202C", "LRO 强制 LTR"),
            ("\u202Etestמשתמש\u202C", "RLO 强制 RTL"),
            ("\u200Fמשתמש\u200Ftest", "RLM 标记"),
            ("\u200Etest\u200Em�תמש", "LRM 标记"),
        ]

        for text, description in bidi_control_cases:
            with self.subTest(desc=description):
                account = AccountBehavior(
                    account_id=text,
                    login_count_last_30d=10,
                )
                assert account.account_id == text

                import json
                json_str = json.dumps({"id": text}, ensure_ascii=False)
                loaded = json.loads(json_str)
                assert loaded["id"] == text

    def test_rtl_export_json(self, tmp_path=None):
        """测试 RTL 字符的 JSON 导出"""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = os.path.join(tmp_dir, "rtl_json_output.json")

            rtl_account = AccountBehavior(
                account_id="מִשׁתַמֵשׁ_العربي_کاربر",
                plan_level="פְרִימְיָם_بريميوم_پریمیوم",
                login_count_last_30d=15,
                last_login_days_ago=3,
            )
            scores = score_accounts([rtl_account], ScoringConfig())

            export_to_json(scores, output_file)

            content = open(output_file, encoding="utf-8").read()
            assert "מִשׁתַמֵשׁ_العربي_کاربر" in content
            assert "פְרִימְיָם_بريميوم_پریمیوم" in content

            import json
            data = json.loads(content)
            assert isinstance(data, list)
            assert data[0]["account_id"] == rtl_account.account_id
            assert data[0]["plan_level"] == rtl_account.plan_level

    def test_rtl_mixed_ltr_sorting(self):
        """测试混合 LTR 和 RTL 字符串的排序"""
        mixed_ids = [
            "abc_עברית",
            "עברית_abc",
            "xyz_العربي",
            "العربي_xyz",
            "a_משתמש_b",
        ]

        accounts = [
            AccountBehavior(account_id=uid, login_count_last_30d=10, last_login_days_ago=5)
            for uid in mixed_ids
        ]

        scores = score_accounts(accounts, ScoringConfig())
        sorted_ids = sorted([s.account_id for s in scores])
        expected_sorted = sorted(mixed_ids)
        assert sorted_ids == expected_sorted
        assert len(sorted_ids) == len(mixed_ids)

    def test_rtl_boundary_edge_cases(self):
        """测试 RTL 字符的边界情况"""
        edge_cases = [
            ("", "空字符串"),
            ("\u0590", "单个希伯来语字符"),
            ("\u0600", "单个阿拉伯语字符"),
            ("\u0590\u0600\u0750", "各 RTL 语系各一"),
            ("a" * 100 + "עברית" * 10 + "z" * 100, "超长混合字符串"),
            ("\u200E\u200F\u202A\u202B\u202C\u202D\u202E", "仅控制字符"),
        ]

        for text, description in edge_cases:
            with self.subTest(desc=description):
                account = AccountBehavior(
                    account_id=text,
                    login_count_last_30d=5,
                )
                assert account.account_id == text

                encoded = text.encode("utf-8")
                decoded = encoded.decode("utf-8")
                assert decoded == text

                import json
                json_str = json.dumps({"id": text}, ensure_ascii=False)
                loaded = json.loads(json_str)
                assert loaded["id"] == text

