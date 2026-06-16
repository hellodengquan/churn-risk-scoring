from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .models import AccountBehavior, RiskScore


PLAN_LTV_MULTIPLIERS = {
    "basic": 1.0,
    "business": 3.0,
    "enterprise": 8.0,
    "premium": 15.0,
}

DEFAULT_LTV_CONFIG = {
    "plan_multipliers": PLAN_LTV_MULTIPLIERS,
    "avg_revenue_per_transaction": 150.0,
    "subscription_monthly_fee": {
        "basic": 99.0,
        "business": 499.0,
        "enterprise": 1999.0,
        "premium": 4999.0,
    },
    "churn_cost_multiplier": 2.5,
    "retention_target_months": 12,
}


@dataclass
class AccountLTV:
    account_id: str
    historical_ltv: float
    predicted_ltv: float
    monthly_revenue: float
    churn_cost: float
    retention_value: float
    priority_score: float
    ltv_tier: str


LTV_TIERS = ["platinum", "gold", "silver", "bronze"]
LTV_TIER_LABELS = {
    "platinum": "铂金客户",
    "gold": "黄金客户",
    "silver": "白银客户",
    "bronze": "青铜客户",
}


def calculate_monthly_revenue(
    account: AccountBehavior,
    config: Optional[Dict] = None,
) -> float:
    cfg = config or DEFAULT_LTV_CONFIG
    plan = account.plan_level.lower()

    subscription_fee = cfg["subscription_monthly_fee"].get(plan, 99.0)
    avg_rev = cfg["avg_revenue_per_transaction"]
    monthly_txn_revenue = (account.total_transactions_last_30d) * avg_rev

    return round(subscription_fee + monthly_txn_revenue, 2)


def calculate_historical_ltv(
    account: AccountBehavior,
    config: Optional[Dict] = None,
) -> float:
    cfg = config or DEFAULT_LTV_CONFIG
    plan = account.plan_level.lower()
    multiplier = cfg["plan_multipliers"].get(plan, 1.0)

    months_active = max(1, account.subscription_age_days / 30.0)
    tx_90d_monthly_avg = account.total_transactions_last_90d / 3.0 if account.total_transactions_last_90d > 0 else 0
    avg_rev = cfg["avg_revenue_per_transaction"]
    sub_fee = cfg["subscription_monthly_fee"].get(plan, 99.0)

    avg_monthly_revenue = sub_fee + tx_90d_monthly_avg * avg_rev
    historical = avg_monthly_revenue * months_active * multiplier

    return round(historical, 2)


def calculate_predicted_ltv(
    account: AccountBehavior,
    risk_score: float,
    config: Optional[Dict] = None,
) -> float:
    cfg = config or DEFAULT_LTV_CONFIG
    target_months = cfg["retention_target_months"]
    churn_cost_mult = cfg["churn_cost_multiplier"]

    monthly_rev = calculate_monthly_revenue(account, cfg)

    retention_prob = max(0.0, 1.0 - risk_score / 100.0)
    expected_months = retention_prob * target_months

    plan = account.plan_level.lower()
    multiplier = cfg["plan_multipliers"].get(plan, 1.0)

    predicted = monthly_rev * expected_months * multiplier
    return round(predicted, 2)


def calculate_ltv_for_account(
    account: AccountBehavior,
    risk_score: RiskScore,
    config: Optional[Dict] = None,
) -> AccountLTV:
    cfg = config or DEFAULT_LTV_CONFIG

    historical_ltv = calculate_historical_ltv(account, cfg)
    predicted_ltv = calculate_predicted_ltv(account, risk_score.total_score, cfg)
    monthly_revenue = calculate_monthly_revenue(account, cfg)
    churn_cost = round(monthly_revenue * cfg["churn_cost_multiplier"], 2)

    risk_normalized = risk_score.total_score / 100.0
    ltv_normalized = min(1.0, predicted_ltv / 100000.0) if predicted_ltv > 0 else 0.0

    priority_score = round(
        risk_normalized * 0.6 + ltv_normalized * 0.4,
        4,
    )

    return AccountLTV(
        account_id=account.account_id,
        historical_ltv=historical_ltv,
        predicted_ltv=predicted_ltv,
        monthly_revenue=monthly_revenue,
        churn_cost=churn_cost,
        retention_value=round(predicted_ltv - churn_cost, 2),
        priority_score=priority_score,
        ltv_tier="bronze",
    )


def assign_ltv_tiers(ltv_list: List[AccountLTV]) -> List[AccountLTV]:
    if not ltv_list:
        return ltv_list

    sorted_ltv = sorted(ltv_list, key=lambda x: x.predicted_ltv, reverse=True)
    total = len(sorted_ltv)

    tier_thresholds = {
        "platinum": int(total * 0.1),
        "gold": int(total * 0.3),
        "silver": int(total * 0.6),
    }

    for i, ltv in enumerate(sorted_ltv):
        if i < tier_thresholds["platinum"]:
            ltv.ltv_tier = "platinum"
        elif i < tier_thresholds["gold"]:
            ltv.ltv_tier = "gold"
        elif i < tier_thresholds["silver"]:
            ltv.ltv_tier = "silver"
        else:
            ltv.ltv_tier = "bronze"

    ltv_map = {ltv.account_id: ltv for ltv in sorted_ltv}
    return [ltv_map[ltv.account_id] for ltv in ltv_list]


def generate_action_list(
    scores: List[RiskScore],
    accounts: List[AccountBehavior],
    ltv_list: List[AccountLTV],
    config: Optional[Dict] = None,
) -> List[Dict]:
    account_map = {a.account_id: a for a in accounts}
    ltv_map = {l.account_id: l for l in ltv_list}
    score_map = {s.account_id: s for s in scores}

    actions = []
    for account_id in score_map:
        if account_id not in ltv_map or account_id not in account_map:
            continue

        ltv = ltv_map[account_id]
        score = score_map[account_id]
        account = account_map[account_id]

        action = {
            "account_id": account_id,
            "risk_level": score.risk_level,
            "risk_score": score.total_score,
            "ltv_tier": ltv.ltv_tier,
            "predicted_ltv": ltv.predicted_ltv,
            "monthly_revenue": ltv.monthly_revenue,
            "churn_cost": ltv.churn_cost,
            "retention_value": ltv.retention_value,
            "priority_score": ltv.priority_score,
            "recommended_action": _recommend_action(score.risk_level, ltv.ltv_tier),
            "retention_budget": round(ltv.churn_cost * 0.3, 2),
        }
        actions.append(action)

    actions.sort(key=lambda x: x["priority_score"], reverse=True)
    return actions


def _recommend_action(risk_level: str, ltv_tier: str) -> str:
    high_value_tiers = {"platinum", "gold"}
    high_risk = {"critical", "high"}

    if ltv_tier in high_value_tiers and risk_level in high_risk:
        return "紧急挽留 - VP/总监级人工回访，定制优惠方案"
    elif ltv_tier in high_value_tiers and risk_level == "medium":
        return "主动关怀 - 客户经理跟进，赠送增值服务"
    elif ltv_tier == "silver" and risk_level in high_risk:
        return "重点挽留 - 标准优惠包 + 专属客服"
    elif ltv_tier == "silver" and risk_level == "medium":
        return "常规关怀 - 产品推荐 + 满意度调研"
    elif risk_level in high_risk:
        return "低成本挽留 - 自动化优惠触达"
    elif risk_level == "medium":
        return "保持关注 - 常规运营触达"
    else:
        return "正常维护 - 常规内容推送"


def calculate_portfolio_summary(
    actions: List[Dict],
) -> Dict:
    if not actions:
        return {}

    total_revenue = sum(a["monthly_revenue"] for a in actions)
    total_churn_cost = sum(a["churn_cost"] for a in actions)
    total_retention_value = sum(max(0, a["retention_value"]) for a in actions)
    total_budget = sum(a["retention_budget"] for a in actions)

    risk_distribution: Dict[str, int] = {}
    ltv_distribution: Dict[str, int] = {}
    for a in actions:
        risk_distribution[a["risk_level"]] = risk_distribution.get(a["risk_level"], 0) + 1
        ltv_distribution[a["ltv_tier"]] = ltv_distribution.get(a["ltv_tier"], 0) + 1

    return {
        "total_accounts": len(actions),
        "total_monthly_revenue_at_risk": round(total_revenue, 2),
        "total_churn_cost_exposure": round(total_churn_cost, 2),
        "total_potential_retention_value": round(total_retention_value, 2),
        "recommended_retention_budget": round(total_budget, 2),
        "risk_distribution": risk_distribution,
        "ltv_distribution": ltv_distribution,
    }


def format_action_list(actions: List[Dict], max_items: int = 20) -> str:
    if not actions:
        return "无挽留操作建议"

    display_actions = actions[:max_items]

    lines = []
    lines.append("=" * 100)
    lines.append("客户挽留优先级清单 (LTV + 风险联动)")
    lines.append("=" * 100)
    header = (
        f"{'优先级':>4s}  {'账号ID':<12s}  {'风险':<6s}  {'LTV等级':<8s}  "
        f"{'月收入':>10s}  {'流失成本':>10s}  {'挽留价值':>10s}  {'建议'}"
    )
    lines.append(header)
    lines.append("-" * 100)

    for i, action in enumerate(display_actions, 1):
        line = (
            f"{i:4d}  {action['account_id']:<12s}  {action['risk_level']:<6s}  "
            f"{LTV_TIER_LABELS.get(action['ltv_tier'], action['ltv_tier']):<8s}  "
            f"{action['monthly_revenue']:10.2f}  {action['churn_cost']:10.2f}  "
            f"{action['retention_value']:10.2f}  {action['recommended_action']}"
        )
        lines.append(line)

    if len(actions) > max_items:
        lines.append(f"\n... 共 {len(actions)} 条建议，仅显示前 {max_items} 条")

    lines.append("=" * 100)
    return "\n".join(lines)


def format_portfolio_summary(summary: Dict) -> str:
    if not summary:
        return "无组合汇总数据"

    lines = []
    lines.append("=" * 60)
    lines.append("客户组合风险与价值汇总")
    lines.append("=" * 60)
    lines.append(f"账号总数: {summary['total_accounts']}")
    lines.append(f"风险敞口月收入: ¥{summary['total_monthly_revenue_at_risk']:,.2f}")
    lines.append(f"潜在流失总成本: ¥{summary['total_churn_cost_exposure']:,.2f}")
    lines.append(f"可挽留总价值: ¥{summary['total_potential_retention_value']:,.2f}")
    lines.append(f"建议挽留预算: ¥{summary['recommended_retention_budget']:,.2f}")
    lines.append("")
    lines.append("风险等级分布:")
    for level, count in summary["risk_distribution"].items():
        lines.append(f"  {level}: {count}")
    lines.append("")
    lines.append("LTV 等级分布:")
    for tier, count in summary["ltv_distribution"].items():
        lines.append(f"  {LTV_TIER_LABELS.get(tier, tier)}: {count}")
    lines.append("=" * 60)
    return "\n".join(lines)
