import math
from typing import List, Dict, Tuple

from .models import AccountBehavior, RiskScore, ScoringConfig, FeatureWeights


def _normalize_inverse(value: float, min_val: float, max_val: float) -> float:
    if max_val == min_val:
        return 50.0
    normalized = (1 - (value - min_val) / (max_val - min_val)) * 100
    return max(0.0, min(100.0, normalized))


def _normalize_direct(value: float, min_val: float, max_val: float) -> float:
    if max_val == min_val:
        return 50.0
    normalized = ((value - min_val) / (max_val - min_val)) * 100
    return max(0.0, min(100.0, normalized))


def _calculate_feature_ranges(accounts: List[AccountBehavior]) -> Dict[str, Dict[str, float]]:
    login_counts_30d = [a.login_count_last_30d for a in accounts]
    login_counts_7d = [a.login_count_last_7d for a in accounts]
    last_logins = [a.last_login_days_ago for a in accounts]
    transactions_30d = [a.total_transactions_last_30d for a in accounts]
    tx_amounts = [a.transaction_amount_last_30d for a in accounts]
    session_times = [a.avg_session_minutes for a in accounts]
    support_tickets = [a.support_tickets_last_30d for a in accounts]
    payment_failures = [a.payment_failures_last_30d for a in accounts]
    refund_counts = [a.refund_count_last_90d for a in accounts]
    email_rates = [a.email_open_rate for a in accounts]

    ranges = {
        "login_frequency_30d": {"min": min(login_counts_30d), "max": max(login_counts_30d)},
        "login_trend": {"min": 0.0, "max": 1.0},
        "recency": {"min": min(last_logins), "max": max(last_logins)},
        "transaction_frequency": {"min": min(transactions_30d), "max": max(transactions_30d)},
        "transaction_amount": {"min": min(tx_amounts), "max": max(tx_amounts)},
        "engagement_depth": {"min": min(session_times), "max": max(session_times)},
        "support_tickets": {"min": min(support_tickets), "max": max(support_tickets) + max(refund_counts)},
        "payment_issues": {"min": min(payment_failures), "max": max(payment_failures)},
        "email_engagement": {"min": min(email_rates), "max": max(email_rates)},
    }
    return ranges


def _calculate_login_trend(account: AccountBehavior) -> float:
    weekly_avg_30d = account.login_count_last_30d / 4.0 if account.login_count_last_30d > 0 else 0
    if weekly_avg_30d == 0:
        return 1.0 if account.login_count_last_7d == 0 else 0.0
    ratio = account.login_count_last_7d / weekly_avg_30d
    return max(0.0, min(1.0, ratio))


def _score_account(
    account: AccountBehavior,
    weights: FeatureWeights,
    ranges: Dict[str, Dict[str, float]],
) -> RiskScore:
    feature_scores = {}

    login_freq_score = _normalize_inverse(
        account.login_count_last_30d,
        ranges["login_frequency_30d"]["min"],
        ranges["login_frequency_30d"]["max"],
    )
    feature_scores["login_frequency_30d"] = login_freq_score

    login_trend_ratio = _calculate_login_trend(account)
    login_trend_score = _normalize_inverse(
        login_trend_ratio,
        ranges["login_trend"]["min"],
        ranges["login_trend"]["max"],
    )
    feature_scores["login_trend"] = login_trend_score

    recency_score = _normalize_direct(
        account.last_login_days_ago,
        ranges["recency"]["min"],
        ranges["recency"]["max"],
    )
    feature_scores["recency"] = recency_score

    tx_freq_score = _normalize_inverse(
        account.total_transactions_last_30d,
        ranges["transaction_frequency"]["min"],
        ranges["transaction_frequency"]["max"],
    )
    feature_scores["transaction_frequency"] = tx_freq_score

    tx_amount_score = _normalize_inverse(
        account.transaction_amount_last_30d,
        ranges["transaction_amount"]["min"],
        ranges["transaction_amount"]["max"],
    )
    feature_scores["transaction_amount"] = tx_amount_score

    engagement_score = _normalize_inverse(
        account.avg_session_minutes,
        ranges["engagement_depth"]["min"],
        ranges["engagement_depth"]["max"],
    )
    feature_scores["engagement_depth"] = engagement_score

    support_total = account.support_tickets_last_30d + account.refund_count_last_90d
    support_score = _normalize_direct(
        support_total,
        ranges["support_tickets"]["min"],
        ranges["support_tickets"]["max"],
    )
    feature_scores["support_tickets"] = support_score

    payment_score = _normalize_direct(
        account.payment_failures_last_30d,
        ranges["payment_issues"]["min"],
        ranges["payment_issues"]["max"],
    )
    feature_scores["payment_issues"] = payment_score

    email_score = _normalize_inverse(
        account.email_open_rate,
        ranges["email_engagement"]["min"],
        ranges["email_engagement"]["max"],
    )
    feature_scores["email_engagement"] = email_score

    total_score = (
        feature_scores["login_frequency_30d"] * weights.login_frequency_30d
        + feature_scores["login_trend"] * weights.login_trend
        + feature_scores["recency"] * weights.recency
        + feature_scores["transaction_frequency"] * weights.transaction_frequency
        + feature_scores["transaction_amount"] * weights.transaction_amount
        + feature_scores["engagement_depth"] * weights.engagement_depth
        + feature_scores["support_tickets"] * weights.support_tickets
        + feature_scores["payment_issues"] * weights.payment_issues
        + feature_scores["email_engagement"] * weights.email_engagement
    )

    risk_level = _determine_risk_level(total_score)

    return RiskScore(
        account_id=account.account_id,
        total_score=round(total_score, 2),
        feature_scores={k: round(v, 2) for k, v in feature_scores.items()},
        risk_level=risk_level,
    )


def _determine_risk_level(score: float) -> str:
    if score >= 90:
        return "critical"
    elif score >= 70:
        return "high"
    elif score >= 40:
        return "medium"
    else:
        return "low"


def score_accounts(
    accounts: List[AccountBehavior],
    config: ScoringConfig,
) -> List[RiskScore]:
    if not accounts:
        return []

    ranges = config.feature_ranges if config.feature_ranges else _calculate_feature_ranges(accounts)

    scores = []
    for account in accounts:
        score = _score_account(account, config.weights, ranges)
        scores.append(score)

    scores.sort(key=lambda s: s.total_score, reverse=True)

    total = len(scores)
    for i, score in enumerate(scores):
        score.risk_percentile = round((i + 1) / total * 100, 2)

    return scores


def get_risk_distribution(scores: List[RiskScore]) -> Dict[str, int]:
    distribution = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for score in scores:
        distribution[score.risk_level] = distribution.get(score.risk_level, 0) + 1
    return distribution
