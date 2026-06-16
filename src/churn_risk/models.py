from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class AccountBehavior:
    account_id: str
    login_count_last_30d: int = 0
    login_count_last_7d: int = 0
    last_login_days_ago: int = 999
    total_transactions_last_30d: int = 0
    total_transactions_last_90d: int = 0
    transaction_amount_last_30d: float = 0.0
    avg_session_minutes: float = 0.0
    support_tickets_last_30d: int = 0
    refund_count_last_90d: int = 0
    payment_failures_last_30d: int = 0
    feature_usage_count: Dict[str, int] = field(default_factory=dict)
    email_open_rate: float = 0.0
    subscription_age_days: int = 0
    plan_level: str = "basic"


@dataclass
class FeatureWeights:
    login_frequency_30d: float = 0.15
    login_trend: float = 0.10
    recency: float = 0.20
    transaction_frequency: float = 0.15
    transaction_amount: float = 0.10
    engagement_depth: float = 0.08
    support_tickets: float = 0.07
    payment_issues: float = 0.10
    email_engagement: float = 0.05


@dataclass
class RiskScore:
    account_id: str
    total_score: float
    feature_scores: Dict[str, float]
    risk_level: str
    risk_percentile: float = 0.0


@dataclass
class ScoringConfig:
    weights: FeatureWeights = field(default_factory=FeatureWeights)
    risk_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "low": 30.0,
            "medium": 60.0,
            "high": 80.0,
            "critical": 95.0,
        }
    )
    feature_ranges: Dict[str, Dict[str, float]] = field(default_factory=dict)
