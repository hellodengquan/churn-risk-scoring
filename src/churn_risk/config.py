from pathlib import Path
from typing import Optional

import yaml

from .models import ScoringConfig, FeatureWeights


DEFAULT_CONFIG = ScoringConfig()


def load_config(config_path: Optional[str] = None) -> ScoringConfig:
    if config_path is None:
        return DEFAULT_CONFIG

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    weights_data = data.get("weights", {})
    weights = FeatureWeights(
        login_frequency_30d=weights_data.get("login_frequency_30d", 0.15),
        login_trend=weights_data.get("login_trend", 0.10),
        recency=weights_data.get("recency", 0.20),
        transaction_frequency=weights_data.get("transaction_frequency", 0.15),
        transaction_amount=weights_data.get("transaction_amount", 0.10),
        engagement_depth=weights_data.get("engagement_depth", 0.08),
        support_tickets=weights_data.get("support_tickets", 0.07),
        payment_issues=weights_data.get("payment_issues", 0.10),
        email_engagement=weights_data.get("email_engagement", 0.05),
    )

    thresholds = data.get("risk_thresholds", {
        "low": 30.0,
        "medium": 60.0,
        "high": 80.0,
        "critical": 95.0,
    })

    feature_ranges = data.get("feature_ranges", {})

    config = ScoringConfig(
        weights=weights,
        risk_thresholds=thresholds,
        feature_ranges=feature_ranges,
    )

    return config


def save_default_config(output_path: str) -> None:
    config = {
        "weights": {
            "login_frequency_30d": 0.15,
            "login_trend": 0.10,
            "recency": 0.20,
            "transaction_frequency": 0.15,
            "transaction_amount": 0.10,
            "engagement_depth": 0.08,
            "support_tickets": 0.07,
            "payment_issues": 0.10,
            "email_engagement": 0.05,
        },
        "risk_thresholds": {
            "low": 30.0,
            "medium": 60.0,
            "high": 80.0,
            "critical": 95.0,
        },
        "feature_ranges": {},
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
