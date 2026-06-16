from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .models import AccountBehavior


@dataclass
class TrendPoint:
    period: str
    value: float
    period_index: int


@dataclass
class TimeSeriesResult:
    account_id: str
    metric_name: str
    values: List[float]
    trend_slope: float
    trend_direction: str
    volatility: float
    change_pct: float
    forecast_next: float
    is_alert: bool
    alert_message: str


def build_time_series_from_accounts(
    accounts: List[AccountBehavior],
    metric: str = "login_count",
    periods: int = 12,
) -> Dict[str, List[float]]:
    series_data: Dict[str, List[float]] = {}
    np.random.seed(42)

    for account in accounts:
        if metric == "login_count":
            base = account.login_count_last_30d / 4.0
            trend = -0.1 if account.last_login_days_ago > 14 else 0.05
        elif metric == "transaction_count":
            base = account.total_transactions_last_30d / 4.0
            trend = -0.08 if account.total_transactions_last_30d < 3 else 0.03
        elif metric == "transaction_amount":
            base = account.transaction_amount_last_30d / 4.0
            trend = -0.12 if account.transaction_amount_last_30d < 1000 else 0.04
        elif metric == "session_minutes":
            base = account.avg_session_minutes
            trend = -0.06 if account.avg_session_minutes < 15 else 0.02
        elif metric == "support_tickets":
            base = account.support_tickets_last_30d / 4.0
            trend = 0.1 if account.support_tickets_last_30d > 2 else 0.0
        elif metric == "payment_failures":
            base = account.payment_failures_last_30d / 4.0
            trend = 0.15 if account.payment_failures_last_30d > 0 else 0.0
        else:
            base = 1.0
            trend = 0.0

        values = []
        current = max(0.1, base * (1 - trend * periods / 2))
        for i in range(periods):
            noise = np.random.normal(0, max(0.1, base * 0.15))
            current = max(0.0, current * (1 + trend) + noise)
            values.append(round(current, 4))

        series_data[account.account_id] = values

    return series_data


def _linear_regression_slope(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values))
    y = np.array(values, dtype=float)
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return 0.0
    x = x[mask]
    y = y[mask]
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def analyze_trend(
    values: List[float],
    alert_threshold_pct: float = -20.0,
) -> Tuple[str, float, float, float, str, bool, float]:
    if not values:
        return "稳定", 0.0, 0.0, 0.0, "", False, 0.0

    values_arr = np.array(values, dtype=float)
    valid = values_arr[~np.isnan(values_arr)]

    if len(valid) < 2:
        return "稳定", 0.0, 0.0, 0.0, "", False, float(valid[-1]) if len(valid) > 0 else 0.0

    slope = _linear_regression_slope(list(valid))

    mean_val = float(np.mean(valid))
    if mean_val != 0:
        slope_pct = slope / mean_val * 100
    else:
        slope_pct = 0.0

    if len(valid) >= 2 and valid[-1] != 0:
        change_pct = (valid[-1] - valid[0]) / valid[-1] * 100
    else:
        change_pct = 0.0

    if len(valid) >= 2:
        volatility = float(np.std(valid) / (np.abs(np.mean(valid)) + 1e-9))
    else:
        volatility = 0.0

    if slope_pct > 5:
        direction = "上升"
    elif slope_pct < -5:
        direction = "下降"
    else:
        direction = "稳定"

    forecast = float(valid[-1] + slope) if len(valid) > 0 else 0.0
    forecast = max(0.0, forecast)

    alert = change_pct < alert_threshold_pct or (slope_pct < -10 and len(valid) >= 4)

    alert_msg = ""
    if alert:
        if change_pct < alert_threshold_pct:
            alert_msg = f"指标较初期下降 {abs(change_pct):.1f}%"
        elif slope_pct < -10:
            alert_msg = f"持续下滑趋势 (月均 {abs(slope_pct):.1f}%)"

    return direction, slope_pct, volatility, change_pct, alert_msg, alert, forecast


def analyze_account_timeseries(
    accounts: List[AccountBehavior],
    metrics: Optional[List[str]] = None,
    periods: int = 12,
) -> List[Dict]:
    if metrics is None:
        metrics = [
            "login_count",
            "transaction_count",
            "transaction_amount",
            "session_minutes",
            "support_tickets",
            "payment_failures",
        ]

    all_results = []
    for metric in metrics:
        series_data = build_time_series_from_accounts(accounts, metric, periods)

        for account_id, values in series_data.items():
            direction, slope_pct, volatility, change_pct, alert_msg, alert, forecast = analyze_trend(values)

            all_results.append({
                "account_id": account_id,
                "metric": metric,
                "values": values,
                "trend_direction": direction,
                "trend_slope_pct": round(slope_pct, 4),
                "volatility": round(volatility, 4),
                "change_pct": round(change_pct, 2),
                "forecast_next": round(forecast, 4),
                "is_alert": alert,
                "alert_message": alert_msg,
            })

    return all_results


def get_account_trend_summary(
    all_results: List[Dict],
    account_id: str,
) -> Dict:
    account_results = [r for r in all_results if r["account_id"] == account_id]
    if not account_results:
        return {}

    alert_count = sum(1 for r in account_results if r["is_alert"])
    declining_count = sum(1 for r in account_results if r["trend_direction"] == "下降")
    avg_slope = float(np.mean([r["trend_slope_pct"] for r in account_results]))

    alerts = []
    for r in account_results:
        if r["is_alert"]:
            alerts.append({
                "metric": r["metric"],
                "message": r["alert_message"],
                "change_pct": r["change_pct"],
            })

    if alert_count >= 3 or declining_count >= 4:
        overall_risk = "high"
    elif alert_count >= 1 or declining_count >= 2:
        overall_risk = "medium"
    else:
        overall_risk = "low"

    return {
        "account_id": account_id,
        "total_metrics": len(account_results),
        "alert_count": alert_count,
        "declining_metrics": declining_count,
        "avg_trend_slope_pct": round(avg_slope, 4),
        "overall_trend_risk": overall_risk,
        "active_alerts": alerts,
        "metrics": account_results,
    }


def get_portfolio_trend_summary(all_results: List[Dict]) -> Dict:
    if not all_results:
        return {}

    total_accounts = len(set(r["account_id"] for r in all_results))
    total_alerts = sum(1 for r in all_results if r["is_alert"])
    total_declining = sum(1 for r in all_results if r["trend_direction"] == "下降")

    metric_alerts: Dict[str, int] = {}
    for r in all_results:
        if r["is_alert"]:
            metric_alerts[r["metric"]] = metric_alerts.get(r["metric"], 0) + 1

    alert_accounts = set()
    for r in all_results:
        if r["is_alert"]:
            alert_accounts.add(r["account_id"])

    return {
        "total_accounts": total_accounts,
        "total_metric_points": len(all_results),
        "total_alerts": total_alerts,
        "accounts_with_alerts": len(alert_accounts),
        "accounts_with_alerts_pct": round(len(alert_accounts) / total_accounts * 100, 2) if total_accounts > 0 else 0,
        "total_declining_metrics": total_declining,
        "alerts_by_metric": metric_alerts,
        "alert_account_ids": sorted(list(alert_accounts)),
    }


def format_timeseries_report(summary: Dict) -> str:
    if not summary:
        return "无时间序列分析数据"

    lines = []
    lines.append("=" * 60)
    lines.append("时间序列趋势分析汇总")
    lines.append("=" * 60)
    lines.append(f"分析账号数: {summary.get('total_accounts', 0)}")
    lines.append(f"监控指标点: {summary.get('total_metric_points', 0)}")
    lines.append(f"告警总数: {summary.get('total_alerts', 0)}")
    lines.append(f"存在告警的账号: {summary.get('accounts_with_alerts', 0)} "
                 f"({summary.get('accounts_with_alerts_pct', 0):.1f}%)")
    lines.append(f"下降趋势指标数: {summary.get('total_declining_metrics', 0)}")
    lines.append("")

    lines.append("各指标告警分布:")
    for metric, count in summary.get("alerts_by_metric", {}).items():
        bar = "█" * count
        lines.append(f"  {metric:<25s}: {count:3d} {bar}")
    lines.append("")

    alert_accounts = summary.get("alert_account_ids", [])
    if alert_accounts:
        lines.append(f"告警账号列表 ({len(alert_accounts)} 个):")
        for acc_id in alert_accounts[:10]:
            lines.append(f"  - {acc_id}")
        if len(alert_accounts) > 10:
            lines.append(f"  ... 还有 {len(alert_accounts) - 10} 个")

    lines.append("=" * 60)
    return "\n".join(lines)


def format_account_trend_report(account_summary: Dict) -> str:
    if not account_summary:
        return "无该账号的趋势数据"

    lines = []
    lines.append("=" * 60)
    lines.append(f"账号趋势详情 - {account_summary.get('account_id', '')}")
    lines.append("=" * 60)
    lines.append(f"整体趋势风险: {account_summary.get('overall_trend_risk', 'unknown')}")
    lines.append(f"监控指标数: {account_summary.get('total_metrics', 0)}")
    lines.append(f"告警数量: {account_summary.get('alert_count', 0)}")
    lines.append(f"下降指标数: {account_summary.get('declining_metrics', 0)}")
    lines.append(f"平均趋势斜率: {account_summary.get('avg_trend_slope_pct', 0):.2f}%")
    lines.append("")

    active_alerts = account_summary.get("active_alerts", [])
    if active_alerts:
        lines.append("活跃告警:")
        for alert in active_alerts:
            lines.append(f"  ⚠ {alert['metric']}: {alert['message']} (变化 {alert['change_pct']:+.2f}%)")
        lines.append("")

    lines.append(f"{'指标':<25s}  {'方向':<6s}  {'斜率%':>10s}  {'波动率':>8s}  {'变化%':>8s}  {'预测':>8s}")
    lines.append("-" * 70)

    for metric_result in account_summary.get("metrics", []):
        alert_marker = " ⚠" if metric_result.get("is_alert") else ""
        lines.append(
            f"{metric_result['metric']:<25s}  "
            f"{metric_result['trend_direction']:<6s}  "
            f"{metric_result['trend_slope_pct']:+9.2f}%  "
            f"{metric_result['volatility']:7.4f}  "
            f"{metric_result['change_pct']:+7.2f}%  "
            f"{metric_result['forecast_next']:8.4f}{alert_marker}"
        )

    lines.append("=" * 60)
    return "\n".join(lines)


@dataclass
class ForecastingConfig:
    auto_select: bool = True
    preferred_model: str = "auto"
    arima_order: Tuple[int, int, int] = (1, 1, 1)
    prophet_seasonality_mode: str = "additive"
    prophet_yearly_seasonality: bool = False
    prophet_weekly_seasonality: bool = True
    min_samples_arima: int = 20
    min_samples_prophet: int = 30
    min_forecast_periods: int = 1
    max_forecast_periods: int = 12
    cv_folds: int = 3
    use_volatility_threshold: float = 0.3
    use_ensemble: bool = False
    ensemble_method: str = "weighted_average"
    ensemble_models: List[str] = field(default_factory=lambda: ["arima", "prophet", "linear"])
    ensemble_weights: Optional[Dict[str, float]] = None
    dynamic_weighting: bool = True
    dynamic_weight_window: int = 10
    bma_prior_strength: float = 1.0
    ensemble_confidence_method: str = "conservative"
    min_ensemble_models: int = 2


@dataclass
class EnsembleForecastDetail:
    model_name: str
    forecast_values: List[float]
    weight: float
    model_metrics: Dict[str, float]


@dataclass
class ForecastingResult:
    model_used: str
    selection_reason: str
    forecast_values: List[float]
    confidence_lower: List[float]
    confidence_upper: List[float]
    model_metrics: Dict[str, float]
    trend_direction: str
    forecast_periods: int
    ensemble_details: Optional[List[EnsembleForecastDetail]] = None


class AutoForecaster:
    def __init__(self, config: Optional[ForecastingConfig] = None):
        self.config = config or ForecastingConfig()
        self._arima_available = False
        self._prophet_available = False
        try:
            import statsmodels
            self._arima_available = True
        except ImportError:
            pass
        try:
            import prophet
            self._prophet_available = True
        except ImportError:
            pass

    @property
    def available_models(self) -> List[str]:
        models = ["linear"]
        if self._arima_available:
            models.append("arima")
        if self._prophet_available:
            models.append("prophet")
        return models

    def _analyze_series_characteristics(
        self,
        values: List[float],
    ) -> Dict[str, Any]:
        arr = np.array(values, dtype=float)
        arr = arr[~np.isnan(arr)]

        if len(arr) < 2:
            return {
                "length": len(arr),
                "volatility": 0.0,
                "trend_strength": 0.0,
                "seasonality_strength": 0.0,
                "is_stationary": True,
                "non_zero_ratio": 0.0,
            }

        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr))
        volatility = std_val / (abs(mean_val) + 1e-9)

        slope = _linear_regression_slope(list(arr))
        trend_strength = abs(slope) / (mean_val + 1e-9)

        seasonality_strength = 0.0
        if len(arr) >= 14:
            try:
                weekly = arr.reshape(-1, 7) if len(arr) % 7 == 0 else None
                if weekly is not None and weekly.shape[0] >= 2:
                    week_std = np.std(weekly, axis=0).mean()
                    total_std = std_val + 1e-9
                    seasonality_strength = min(1.0, week_std / total_std)
            except Exception:
                pass

        is_stationary = volatility < self.config.use_volatility_threshold

        non_zero_ratio = float(np.sum(arr > 0) / len(arr)) if len(arr) > 0 else 0.0

        return {
            "length": len(arr),
            "volatility": round(volatility, 4),
            "trend_strength": round(trend_strength, 4),
            "seasonality_strength": round(seasonality_strength, 4),
            "is_stationary": is_stationary,
            "non_zero_ratio": round(non_zero_ratio, 4),
        }

    def _select_model(self, characteristics: Dict[str, Any]) -> Tuple[str, str]:
        n = characteristics["length"]
        reasons = []

        if not self.config.auto_select:
            preferred = self.config.preferred_model
            if preferred == "arima" and not self._arima_available:
                return "linear", "ARIMA 不可用，回退到线性回归"
            if preferred == "prophet" and not self._prophet_available:
                return "linear", "Prophet 不可用，回退到线性回归"
            if preferred in self.available_models:
                return preferred, f"用户指定模型: {preferred}"
            return "linear", f"未知模型 {preferred}，回退到线性回归"

        if n < self.config.min_samples_arima:
            return "linear", f"样本数不足 ({n} < {self.config.min_samples_arima})，使用线性回归"
        reasons.append(f"样本数足够: {n}")

        if n < self.config.min_samples_prophet or not self._prophet_available:
            if self._arima_available:
                if characteristics["is_stationary"]:
                    reasons.append("序列平稳")
                    return "arima", "; ".join(reasons) + ": 使用 ARIMA"
                reasons.append("序列非平稳但样本不足 Prophet，使用 ARIMA")
                return "arima", "; ".join(reasons)
            return "linear", "ARIMA/Prophet 均不可用，使用线性回归"

        if characteristics["seasonality_strength"] > 0.3:
            reasons.append(f"季节性强 ({characteristics['seasonality_strength']:.2f})")
            return "prophet", "; ".join(reasons) + ": 使用 Prophet"

        if characteristics["trend_strength"] > 0.5 and not characteristics["is_stationary"]:
            reasons.append(f"趋势强且非平稳 (trend={characteristics['trend_strength']:.2f})")
            return "prophet", "; ".join(reasons) + ": 使用 Prophet"

        reasons.append(f"波动率 {characteristics['volatility']:.2f}")
        return "arima", "; ".join(reasons) + ": 使用 ARIMA"

    def _forecast_linear(
        self,
        values: List[float],
        periods: int,
    ) -> ForecastingResult:
        arr = np.array(values, dtype=float)
        valid = arr[~np.isnan(arr)]

        if len(valid) < 2:
            flat = [float(valid[-1]) if len(valid) > 0 else 0.0] * periods
            return ForecastingResult(
                model_used="linear",
                selection_reason="样本不足，使用最后值外推",
                forecast_values=flat,
                confidence_lower=flat,
                confidence_upper=flat,
                model_metrics={"r2": 0.0, "mae": 0.0, "rmse": 0.0},
                trend_direction="稳定",
                forecast_periods=periods,
            )

        x = np.arange(len(valid))
        slope, intercept = np.polyfit(x, valid, 1)
        residuals = valid - (slope * x + intercept)
        std_resid = float(np.std(residuals)) if len(residuals) > 1 else 0.0

        future_x = np.arange(len(valid), len(valid) + periods)
        forecast = slope * future_x + intercept
        forecast = np.maximum(0.0, forecast)

        if slope > 0.01:
            direction = "上升"
        elif slope < -0.01:
            direction = "下降"
        else:
            direction = "稳定"

        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((valid - np.mean(valid)) ** 2))
        r2 = 1.0 - ss_res / (ss_tot + 1e-9)
        mae = float(np.mean(np.abs(residuals)))
        rmse = float(np.sqrt(np.mean(residuals ** 2)))

        return ForecastingResult(
            model_used="linear",
            selection_reason="线性回归外推",
            forecast_values=[round(float(v), 4) for v in forecast],
            confidence_lower=[round(float(max(0.0, v - 1.96 * std_resid)), 4) for v in forecast],
            confidence_upper=[round(float(v + 1.96 * std_resid), 4) for v in forecast],
            model_metrics={"r2": round(r2, 4), "mae": round(mae, 4), "rmse": round(rmse, 4)},
            trend_direction=direction,
            forecast_periods=periods,
        )

    def _forecast_arima(
        self,
        values: List[float],
        periods: int,
    ) -> ForecastingResult:
        if not self._arima_available:
            return self._forecast_linear(values, periods)

        try:
            from statsmodels.tsa.arima.model import ARIMA

            arr = np.array(values, dtype=float)
            valid = arr[~np.isnan(arr)]
            if len(valid) < self.config.min_samples_arima:
                return self._forecast_linear(values, periods)

            valid_log = np.log1p(valid)
            model = ARIMA(valid_log, order=self.config.arima_order)
            fitted = model.fit()

            forecast_res = fitted.get_forecast(steps=periods)
            forecast_log = forecast_res.predicted_mean
            conf_int = forecast_res.conf_int(alpha=0.05)

            forecast = np.expm1(forecast_log)
            lower = np.expm1(conf_int[:, 0])
            upper = np.expm1(conf_int[:, 1])
            forecast = np.maximum(0.0, forecast)
            lower = np.maximum(0.0, lower)
            upper = np.maximum(0.0, upper)

            slope = _linear_regression_slope(list(valid))
            if slope > 0.01:
                direction = "上升"
            elif slope < -0.01:
                direction = "下降"
            else:
                direction = "稳定"

            return ForecastingResult(
                model_used="arima",
                selection_reason=f"ARIMA{self.config.arima_order} 拟合",
                forecast_values=[round(float(v), 4) for v in forecast],
                confidence_lower=[round(float(v), 4) for v in lower],
                confidence_upper=[round(float(v), 4) for v in upper],
                model_metrics={
                    "aic": round(float(fitted.aic), 4),
                    "bic": round(float(fitted.bic), 4),
                    "mae": round(float(np.mean(np.abs(fitted.resid))), 4),
                },
                trend_direction=direction,
                forecast_periods=periods,
            )
        except Exception:
            return self._forecast_linear(values, periods)

    def _forecast_prophet(
        self,
        values: List[float],
        periods: int,
    ) -> ForecastingResult:
        if not self._prophet_available:
            return self._forecast_arima(values, periods)

        try:
            from prophet import Prophet

            arr = np.array(values, dtype=float)
            valid = arr[~np.isnan(arr)]
            if len(valid) < self.config.min_samples_prophet:
                return self._forecast_arima(values, periods)

            dates = pd.date_range(end=pd.Timestamp.now(), periods=len(valid), freq="D")
            df = pd.DataFrame({"ds": dates, "y": valid})

            model = Prophet(
                seasonality_mode=self.config.prophet_seasonality_mode,
                yearly_seasonality=self.config.prophet_yearly_seasonality,
                weekly_seasonality=self.config.prophet_weekly_seasonality,
                changepoint_prior_scale=0.05,
            )
            model.fit(df)

            future = model.make_future_dataframe(periods=periods)
            forecast_df = model.predict(future)

            forecast_vals = forecast_df["yhat"].tail(periods).values
            lower_vals = forecast_df["yhat_lower"].tail(periods).values
            upper_vals = forecast_df["yhat_upper"].tail(periods).values

            forecast_vals = np.maximum(0.0, forecast_vals)
            lower_vals = np.maximum(0.0, lower_vals)
            upper_vals = np.maximum(0.0, upper_vals)

            slope = _linear_regression_slope(list(valid))
            if slope > 0.01:
                direction = "上升"
            elif slope < -0.01:
                direction = "下降"
            else:
                direction = "稳定"

            return ForecastingResult(
                model_used="prophet",
                selection_reason=f"Prophet ({self.config.prophet_seasonality_mode} 季节性)",
                forecast_values=[round(float(v), 4) for v in forecast_vals],
                confidence_lower=[round(float(v), 4) for v in lower_vals],
                confidence_upper=[round(float(v), 4) for v in upper_vals],
                model_metrics={
                    "trend_strength": round(abs(slope) / (np.mean(valid) + 1e-9), 4),
                    "n_changepoints": int(len(getattr(model, 'changepoints', []))),
                },
                trend_direction=direction,
                forecast_periods=periods,
            )
        except Exception:
            return self._forecast_arima(values, periods)

    def forecast(
        self,
        values: List[float],
        periods: Optional[int] = None,
        use_ensemble: Optional[bool] = None,
    ) -> ForecastingResult:
        if periods is None:
            periods = self.config.min_forecast_periods
        periods = max(
            self.config.min_forecast_periods,
            min(self.config.max_forecast_periods, periods),
        )

        if use_ensemble is None:
            use_ensemble = self.config.use_ensemble

        characteristics = self._analyze_series_characteristics(values)

        if use_ensemble:
            return self._forecast_ensemble(values, periods, characteristics)

        model_name, reason = self._select_model(characteristics)

        if model_name == "prophet":
            result = self._forecast_prophet(values, periods)
        elif model_name == "arima":
            result = self._forecast_arima(values, periods)
        else:
            result = self._forecast_linear(values, periods)
        result.selection_reason = reason
        return result

    def _cross_validation_score(
        self,
        values: List[float],
        model_name: str,
        horizon: int = 3,
    ) -> Dict[str, float]:
        arr = np.array(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        n = len(arr)

        if n < self.config.min_samples_arima + horizon:
            return {"mae": float('inf'), "rmse": float('inf'), "mape": float('inf')}

        cv_errors = []
        fold_size = n // self.config.cv_folds if self.config.cv_folds > 0 else n // 3
        fold_size = max(fold_size, horizon + 5)

        for fold in range(max(1, self.config.cv_folds)):
            test_start = n - (fold + 1) * fold_size
            if test_start < horizon + 5:
                break
            train_end = test_start
            test_end = min(test_start + horizon, n)

            train_data = list(arr[:train_end])
            test_data = arr[test_start:test_end]

            try:
                if model_name == "arima":
                    forecast = self._forecast_arima(train_data, len(test_data))
                elif model_name == "prophet":
                    forecast = self._forecast_prophet(train_data, len(test_data))
                else:
                    forecast = self._forecast_linear(train_data, len(test_data))

                pred = np.array(forecast.forecast_values)
                errors = np.abs(pred - test_data)
                cv_errors.extend(errors.tolist())
            except Exception:
                continue

        if not cv_errors:
            return {"mae": float('inf'), "rmse": float('inf'), "mape": float('inf')}

        cv_arr = np.array(cv_errors)
        return {
            "mae": float(np.mean(cv_arr)),
            "rmse": float(np.sqrt(np.mean(cv_arr ** 2))),
            "mape": float(np.mean(cv_arr / (np.abs(arr[-len(cv_arr):]) + 1e-9))) if len(arr) >= len(cv_arr) else 0.0,
        }

    def _calculate_dynamic_weights(
        self,
        values: List[float],
        available_models: List[str],
    ) -> Dict[str, float]:
        if self.config.ensemble_weights and not self.config.dynamic_weighting:
            return {m: self.config.ensemble_weights.get(m, 0.0) for m in available_models}

        model_scores = {}
        for model in available_models:
            try:
                scores = self._cross_validation_score(values, model)
                if scores["mae"] == float('inf'):
                    model_scores[model] = 0.0
                    continue

                weight = 1.0 / (scores["mae"] + 1e-9)
                if self.config.ensemble_method == "bayesian":
                    weight = np.exp(-self.config.bma_prior_strength * scores["rmse"])
                model_scores[model] = weight
            except Exception:
                model_scores[model] = 0.0

        total = sum(model_scores.values())
        if total <= 0:
            n = len(available_models)
            return {m: 1.0 / n for m in available_models}

        return {m: w / total for m, w in model_scores.items()}

    def _forecast_ensemble(
        self,
        values: List[float],
        periods: int,
        characteristics: Dict[str, Any],
    ) -> ForecastingResult:
        available_models = [
            m for m in self.config.ensemble_models
            if m in self.available_models
        ]

        if len(available_models) < self.config.min_ensemble_models:
            model_name, reason = self._select_model(characteristics)
            if model_name == "prophet":
                result = self._forecast_prophet(values, periods)
            elif model_name == "arima":
                result = self._forecast_arima(values, periods)
            else:
                result = self._forecast_linear(values, periods)
            result.selection_reason = f"ensemble_not_enough_models_{len(available_models)}_" + reason
            return result

        weights = self._calculate_dynamic_weights(values, available_models)

        individual_results = {}
        for model in available_models:
            try:
                if model == "prophet":
                    res = self._forecast_prophet(values, periods)
                elif model == "arima":
                    res = self._forecast_arima(values, periods)
                else:
                    res = self._forecast_linear(values, periods)
                individual_results[model] = res
            except Exception:
                individual_results[model] = None

        valid_results = {m: r for m, r in individual_results.items() if r is not None}
        if not valid_results:
            return self._forecast_linear(values, periods)

        if len(valid_results) < len(available_models):
            available_models = list(valid_results.keys())
            weights = self._calculate_dynamic_weights(values, available_models)

        n_periods = periods
        ensemble_forecast = np.zeros(n_periods)
        ensemble_lower = np.zeros(n_periods)
        ensemble_upper = np.zeros(n_periods)
        ensemble_details = []

        for model in available_models:
            res = valid_results[model]
            w = weights.get(model, 0.0)
            if w <= 0:
                continue

            forecasts = np.array(res.forecast_values)
            lower = np.array(res.confidence_lower)
            upper = np.array(res.confidence_upper)

            if self.config.ensemble_method in ["simple_average", "weighted_average", "bayesian"]:
                ensemble_forecast += forecasts * w
                if self.config.ensemble_confidence_method == "conservative":
                    ensemble_lower = np.minimum(ensemble_lower if np.any(ensemble_lower) else lower, lower)
                    ensemble_upper = np.maximum(ensemble_upper if np.any(ensemble_upper) else upper, upper)
                elif self.config.ensemble_confidence_method == "weighted":
                    ensemble_lower += lower * w
                    ensemble_upper += upper * w

            ensemble_details.append(EnsembleForecastDetail(
                model_name=model,
                forecast_values=res.forecast_values,
                weight=round(w, 4),
                model_metrics=res.model_metrics,
            ))

        trend = characteristics.get("trend_direction", "稳定")
        if isinstance(trend, (int, float)):
            trend = "上升" if trend > 0.05 else "下降" if trend < -0.05 else "稳定"
        else:
            trend = str(trend)

        model_metrics = {
            "n_models": len(valid_results),
            "ensemble_method": self.config.ensemble_method,
            "dynamic_weighting": self.config.dynamic_weighting,
        }
        for m, w in weights.items():
            model_metrics[f"weight_{m}"] = round(w, 4)

        reason_parts = [
            f"ensemble_{self.config.ensemble_method}",
            f"models={','.join(available_models)}",
        ]
        if self.config.dynamic_weighting:
            reason_parts.append("dynamic_weights")

        return ForecastingResult(
            model_used="ensemble",
            selection_reason="; ".join(reason_parts),
            forecast_values=[round(float(v), 4) for v in ensemble_forecast],
            confidence_lower=[round(float(v), 4) for v in ensemble_lower],
            confidence_upper=[round(float(v), 4) for v in ensemble_upper],
            model_metrics=model_metrics,
            trend_direction=trend,
            forecast_periods=periods,
            ensemble_details=ensemble_details,
        )

    def batch_forecast(
        self,
        series_data: Dict[str, List[float]],
        periods: Optional[int] = None,
    ) -> Dict[str, ForecastingResult]:
        results = {}
        for account_id, values in series_data.items():
            results[account_id] = self.forecast(values, periods)
        return results


def format_forecasting_report(
    result: ForecastingResult,
    metric_name: str = "指标",
) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append(f"时间序列预测报告 - {metric_name}")
    lines.append("=" * 60)
    lines.append(f"选用模型: {result.model_used}")
    lines.append(f"选择依据: {result.selection_reason}")
    lines.append(f"趋势方向: {result.trend_direction}")
    lines.append(f"预测期数: {result.forecast_periods}")
    lines.append("")

    if result.model_metrics:
        lines.append("模型评估指标:")
        for k, v in result.model_metrics.items():
            lines.append(f"  {k}: {v}")
        lines.append("")

    lines.append(f"{'期数':>6s}  {'预测值':>12s}  {'下界(95%)':>12s}  {'上界(95%)':>12s}")
    lines.append("-" * 50)
    for i in range(result.forecast_periods):
        f = result.forecast_values[i] if i < len(result.forecast_values) else 0.0
        lo = result.confidence_lower[i] if i < len(result.confidence_lower) else 0.0
        hi = result.confidence_upper[i] if i < len(result.confidence_upper) else 0.0
        lines.append(f"  t+{i+1:<3d}  {f:12.4f}  {lo:12.4f}  {hi:12.4f}")

    lines.append("=" * 60)
    return "\n".join(lines)

