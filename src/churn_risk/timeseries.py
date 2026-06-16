from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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
