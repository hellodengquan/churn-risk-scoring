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


@dataclass
class LTVFallbackConfig:
    strategy: str = "plan_group_median"
    fallback_strategies: List[str] = field(default_factory=lambda: [
        "cold_start_bayesian",
        "plan_group_median",
        "knn_similar",
        "global_quantile",
        "conservative_default",
    ])
    knn_neighbors: int = 5
    quantile: float = 0.5
    min_group_size: int = 3
    use_subscription_age: bool = True
    cold_start_threshold_days: int = 30
    cold_start_bayesian_strength: float = 0.3
    cold_start_decay_rate: float = 0.05
    use_ensemble_for_cold_start: bool = True
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        "plan_group_median": 0.35,
        "bayesian_shrinkage": 0.35,
        "knn_similar": 0.20,
        "global_quantile": 0.10,
    })
    shrinkage_prior_strength: float = 10.0


@dataclass
class LTVFallbackResult:
    account_id: str
    ltv_estimated: float
    monthly_revenue_estimated: float
    fallback_strategy_used: str
    confidence: float
    reference_group_size: int
    reference_metrics: Dict[str, float]


class LTVFallbackEstimator:
    def __init__(
        self,
        reference_accounts: Optional[List[AccountBehavior]] = None,
        reference_ltvs: Optional[List[AccountLTV]] = None,
        config: Optional[LTVFallbackConfig] = None,
    ):
        self.config = config or LTVFallbackConfig()
        self._plan_stats: Dict[str, Dict[str, float]] = {}
        self._global_stats: Dict[str, float] = {}
        self._reference_accounts: List[AccountBehavior] = []
        self._reference_ltvs: Dict[str, AccountLTV] = {}

        if reference_accounts and reference_ltvs:
            self.fit(reference_accounts, reference_ltvs)

    def fit(
        self,
        reference_accounts: List[AccountBehavior],
        reference_ltvs: List[AccountLTV],
    ) -> None:
        self._reference_accounts = list(reference_accounts)
        self._reference_ltvs = {l.account_id: l for l in reference_ltvs}

        plan_groups: Dict[str, List[Tuple[AccountBehavior, AccountLTV]]] = {}
        all_monthly_rev = []
        all_pred_ltv = []

        for acc in reference_accounts:
            ltv = self._reference_ltvs.get(acc.account_id)
            if not ltv:
                continue
            plan = acc.plan_level.lower()
            plan_groups.setdefault(plan, []).append((acc, ltv))
            all_monthly_rev.append(ltv.monthly_revenue)
            all_pred_ltv.append(ltv.predicted_ltv)

        for plan, items in plan_groups.items():
            if len(items) >= self.config.min_group_size:
                ltv_values = sorted([i[1].predicted_ltv for i in items])
                rev_values = sorted([i[1].monthly_revenue for i in items])
                sub_ages = [i[0].subscription_age_days for i in items]

                q_idx = int(len(ltv_values) * self.config.quantile)
                q_idx = max(0, min(q_idx, len(ltv_values) - 1))

                self._plan_stats[plan] = {
                    "count": len(items),
                    "ltv_median": float(np.median(ltv_values)),
                    "ltv_mean": float(np.mean(ltv_values)),
                    "ltv_std": float(np.std(ltv_values)),
                    "ltv_quantile": float(ltv_values[q_idx]),
                    "rev_median": float(np.median(rev_values)),
                    "rev_mean": float(np.mean(rev_values)),
                    "rev_std": float(np.std(rev_values)),
                    "rev_quantile": float(rev_values[q_idx]),
                    "sub_age_mean": float(np.mean(sub_ages)),
                }

        if all_pred_ltv:
            ltv_values = sorted(all_pred_ltv)
            rev_values = sorted(all_monthly_rev)
            q_idx = int(len(ltv_values) * self.config.quantile)
            q_idx = max(0, min(q_idx, len(ltv_values) - 1))
            self._global_stats = {
                "count": len(all_pred_ltv),
                "ltv_median": float(np.median(ltv_values)),
                "ltv_mean": float(np.mean(ltv_values)),
                "ltv_std": float(np.std(ltv_values)),
                "ltv_quantile": float(ltv_values[q_idx]),
                "rev_median": float(np.median(rev_values)),
                "rev_mean": float(np.mean(rev_values)),
                "rev_std": float(np.std(rev_values)),
                "rev_quantile": float(rev_values[q_idx]),
            }

    def _feature_similarity(
        self,
        target: AccountBehavior,
        reference: AccountBehavior,
    ) -> float:
        score = 0.0
        weight_sum = 0.0

        if target.plan_level.lower() == reference.plan_level.lower():
            score += 0.3
        weight_sum += 0.3

        age_diff = abs(target.subscription_age_days - reference.subscription_age_days)
        age_sim = 1.0 / (1.0 + age_diff / 365.0)
        score += 0.2 * age_sim
        weight_sum += 0.2

        target_login = target.login_count_last_30d
        ref_login = reference.login_count_last_30d
        login_sim = 1.0 / (1.0 + abs(target_login - ref_login) / (max(target_login, ref_login, 1)))
        score += 0.2 * login_sim
        weight_sum += 0.2

        target_txn = target.total_transactions_last_30d
        ref_txn = reference.total_transactions_last_30d
        txn_sim = 1.0 / (1.0 + abs(target_txn - ref_txn) / (max(target_txn, ref_txn, 1)))
        score += 0.2 * txn_sim
        weight_sum += 0.2

        email_sim = 1.0 - abs(target.email_open_rate - reference.email_open_rate)
        score += 0.1 * max(0.0, email_sim)
        weight_sum += 0.1

        return score / weight_sum if weight_sum > 0 else 0.0

    def _estimate_knn(
        self,
        account: AccountBehavior,
    ) -> Optional[Tuple[float, float, int, float]]:
        if not self._reference_accounts:
            return None

        similarities = []
        for ref_acc in self._reference_accounts:
            ref_ltv = self._reference_ltvs.get(ref_acc.account_id)
            if not ref_ltv:
                continue
            sim = self._feature_similarity(account, ref_acc)
            similarities.append((sim, ref_ltv))

        if len(similarities) < self.config.min_group_size:
            return None

        similarities.sort(key=lambda x: x[0], reverse=True)
        neighbors = similarities[:self.config.knn_neighbors]

        total_sim = sum(s for s, _ in neighbors)
        if total_sim <= 0:
            return None

        weighted_ltv = sum(s * l.predicted_ltv for s, l in neighbors) / total_sim
        weighted_rev = sum(s * l.monthly_revenue for s, l in neighbors) / total_sim
        avg_sim = total_sim / len(neighbors)

        return weighted_ltv, weighted_rev, len(neighbors), avg_sim

    def _is_cold_start(self, account: AccountBehavior) -> bool:
        return account.subscription_age_days <= self.config.cold_start_threshold_days

    def _confidence_decay_factor(self, account: AccountBehavior) -> float:
        age = account.subscription_age_days
        threshold = self.config.cold_start_threshold_days
        if age >= threshold:
            return 1.0
        decay_rate = self.config.cold_start_decay_rate
        factor = 1.0 - np.exp(-decay_rate * (age + 1))
        return max(0.2, min(1.0, factor))

    def _estimate_bayesian_shrinkage(
        self,
        account: AccountBehavior,
    ) -> Optional[Tuple[float, float, float, Dict[str, float]]]:
        if not self._global_stats:
            return None

        plan = account.plan_level.lower()
        plan_group = self._plan_stats.get(plan)
        prior_strength = self.config.shrinkage_prior_strength

        global_ltv_mean = self._global_stats["ltv_mean"]
        global_ltv_std = max(self._global_stats["ltv_std"], 1e-9)
        global_rev_mean = self._global_stats["rev_mean"]
        global_rev_std = max(self._global_stats["rev_std"], 1e-9)

        if plan_group and plan_group["count"] >= self.config.min_group_size:
            group_count = plan_group["count"]
            group_ltv_mean = plan_group["ltv_mean"]
            group_ltv_std = max(plan_group["ltv_std"], 1e-9)
            group_rev_mean = plan_group["rev_mean"]
            group_rev_std = max(plan_group["rev_std"], 1e-9)

            shrinkage_ltv = prior_strength / (prior_strength + group_count)
            shrinkage_rev = prior_strength / (prior_strength + group_count)

            shrunk_ltv = shrinkage_ltv * global_ltv_mean + (1 - shrinkage_ltv) * group_ltv_mean
            shrunk_rev = shrinkage_rev * global_rev_mean + (1 - shrinkage_rev) * group_rev_mean

            shrunk_ltv_std = np.sqrt(
                (shrinkage_ltv ** 2) * (global_ltv_std ** 2) +
                ((1 - shrinkage_ltv) ** 2) * (group_ltv_std ** 2)
            )
            shrunk_rev_std = np.sqrt(
                (shrinkage_rev ** 2) * (global_rev_std ** 2) +
                ((1 - shrinkage_rev) ** 2) * (group_rev_std ** 2)
            )

            metrics = {
                "global_ltv_mean": global_ltv_mean,
                "group_ltv_mean": group_ltv_mean,
                "shrinkage_factor_ltv": round(shrinkage_ltv, 4),
                "shrunk_ltv_std": round(shrunk_ltv_std, 2),
                "group_count": group_count,
            }
            confidence = min(0.85, 0.4 + (group_count / (group_count + prior_strength)) * 0.45)

        else:
            shrinkage_ltv = 0.5
            shrinkage_rev = 0.5
            shrunk_ltv = global_ltv_mean
            shrunk_rev = global_rev_mean
            shrunk_ltv_std = global_ltv_std
            shrunk_rev_std = global_rev_std

            metrics = {
                "global_ltv_mean": global_ltv_mean,
                "shrinkage_factor_ltv": round(shrinkage_ltv, 4),
                "shrunk_ltv_std": round(shrunk_ltv_std, 2),
                "note": "no_plan_group_used_global",
            }
            confidence = 0.35

        decay_factor = self._confidence_decay_factor(account)
        if self.config.use_subscription_age:
            age_factor = account.subscription_age_days / max(1, self.config.cold_start_threshold_days)
            age_factor = max(0.3, min(1.5, age_factor))
            shrunk_ltv = shrunk_ltv * age_factor
            shrunk_rev = shrunk_rev * (0.5 + 0.5 * age_factor)

        return shrunk_ltv, shrunk_rev, confidence * decay_factor, metrics

    def _estimate_ensemble(
        self,
        account: AccountBehavior,
    ) -> Optional[Tuple[float, float, float, Dict[str, float], Dict[str, float]]]:
        estimates = {}
        weights = {}
        available_weights = dict(self.config.ensemble_weights)

        if "plan_group_median" in available_weights:
            plan = account.plan_level.lower()
            plan_group = self._plan_stats.get(plan)
            if plan_group:
                ltv_val = plan_group["ltv_median"]
                rev_val = plan_group["rev_median"]
                if self.config.use_subscription_age and plan_group["sub_age_mean"] > 0:
                    age_factor = account.subscription_age_days / plan_group["sub_age_mean"]
                    age_factor = max(0.3, min(3.0, age_factor))
                    ltv_val = ltv_val * age_factor
                    rev_val = rev_val * (0.5 + 0.5 * age_factor)
                estimates["plan_group_median"] = (ltv_val, rev_val)
                weights["plan_group_median"] = available_weights["plan_group_median"]

        if "bayesian_shrinkage" in available_weights:
            bayes_result = self._estimate_bayesian_shrinkage(account)
            if bayes_result:
                ltv_val, rev_val, conf, _ = bayes_result
                estimates["bayesian_shrinkage"] = (ltv_val, rev_val)
                weights["bayesian_shrinkage"] = available_weights["bayesian_shrinkage"] * (0.5 + 0.5 * conf)

        if "knn_similar" in available_weights:
            knn_result = self._estimate_knn(account)
            if knn_result:
                ltv_val, rev_val, n, avg_sim = knn_result
                estimates["knn_similar"] = (ltv_val, rev_val)
                knn_conf = max(0.3, min(0.8, avg_sim * 0.7 + (n / 20.0) * 0.3))
                weights["knn_similar"] = available_weights["knn_similar"] * (0.5 + 0.5 * knn_conf)

        if "global_quantile" in available_weights and self._global_stats:
            ltv_val = self._global_stats["ltv_quantile"]
            rev_val = self._global_stats["rev_quantile"]
            estimates["global_quantile"] = (ltv_val, rev_val)
            global_conf = 0.4 if self._global_stats["count"] >= 50 else 0.25
            weights["global_quantile"] = available_weights["global_quantile"] * (0.5 + 0.5 * global_conf)

        if not estimates:
            return None

        total_weight = sum(weights.values())
        if total_weight <= 0:
            return None

        normalized_weights = {k: v / total_weight for k, v in weights.items()}

        ensemble_ltv = sum(estimates[k][0] * normalized_weights[k] for k in estimates)
        ensemble_rev = sum(estimates[k][1] * normalized_weights[k] for k in estimates)

        decay_factor = self._confidence_decay_factor(account)
        base_confidence = min(0.9, 0.5 + len(estimates) * 0.1)
        final_confidence = base_confidence * decay_factor

        individual_estimates = {
            k: {"ltv": round(v[0], 2), "rev": round(v[1], 2), "weight": round(normalized_weights[k], 4)}
            for k, v in estimates.items()
        }

        ensemble_metrics = {
            "n_estimators": len(estimates),
            "total_weight": round(total_weight, 4),
            "decay_factor": round(decay_factor, 4),
            "individual_estimates": individual_estimates,
        }

        return ensemble_ltv, ensemble_rev, final_confidence, ensemble_metrics, individual_estimates

    def estimate(self, account: AccountBehavior) -> LTVFallbackResult:
        plan = account.plan_level.lower()
        plan_group = self._plan_stats.get(plan)

        is_cold_start = self._is_cold_start(account)
        decay_factor = self._confidence_decay_factor(account)

        if is_cold_start and self.config.use_ensemble_for_cold_start:
            ensemble_result = self._estimate_ensemble(account)
            if ensemble_result:
                ltv_val, rev_val, confidence, metrics, indiv = ensemble_result
                return LTVFallbackResult(
                    account_id=account.account_id,
                    ltv_estimated=round(ltv_val, 2),
                    monthly_revenue_estimated=round(rev_val, 2),
                    fallback_strategy_used="cold_start_ensemble",
                    confidence=round(confidence, 4),
                    reference_group_size=metrics.get("n_estimators", 0),
                    reference_metrics={
                        "is_cold_start": True,
                        "subscription_age_days": account.subscription_age_days,
                        "decay_factor": round(decay_factor, 4),
                        **metrics,
                    },
                )

        if is_cold_start and "cold_start_bayesian" in self.config.fallback_strategies:
            bayes_result = self._estimate_bayesian_shrinkage(account)
            if bayes_result:
                ltv_val, rev_val, confidence, metrics = bayes_result
                return LTVFallbackResult(
                    account_id=account.account_id,
                    ltv_estimated=round(ltv_val, 2),
                    monthly_revenue_estimated=round(rev_val, 2),
                    fallback_strategy_used="cold_start_bayesian",
                    confidence=round(confidence, 4),
                    reference_group_size=metrics.get("group_count", 0) or self._global_stats.get("count", 0),
                    reference_metrics={
                        "is_cold_start": True,
                        "subscription_age_days": account.subscription_age_days,
                        "decay_factor": round(decay_factor, 4),
                        **metrics,
                    },
                )

        strategies_tried = []

        if "plan_group_median" in self.config.fallback_strategies and plan_group:
            ltv_val = plan_group["ltv_median"]
            rev_val = plan_group["rev_median"]
            if self.config.use_subscription_age and plan_group["sub_age_mean"] > 0:
                age_factor = account.subscription_age_days / plan_group["sub_age_mean"]
                age_factor = max(0.3, min(3.0, age_factor))
                ltv_val = ltv_val * age_factor
                rev_val = rev_val * (0.5 + 0.5 * age_factor)

            confidence = min(0.95, 0.6 + plan_group["count"] * 0.01)
            return LTVFallbackResult(
                account_id=account.account_id,
                ltv_estimated=round(ltv_val, 2),
                monthly_revenue_estimated=round(rev_val, 2),
                fallback_strategy_used="plan_group_median",
                confidence=round(confidence, 4),
                reference_group_size=plan_group["count"],
                reference_metrics={
                    "plan_ltv_median": plan_group["ltv_median"],
                    "plan_rev_median": plan_group["rev_median"],
                    "plan_mean_sub_age": plan_group["sub_age_mean"],
                    "age_factor_applied": round(account.subscription_age_days / plan_group["sub_age_mean"] if plan_group["sub_age_mean"] > 0 else 1.0, 4),
                },
            )
        strategies_tried.append("plan_group_median")

        if "knn_similar" in self.config.fallback_strategies:
            knn_result = self._estimate_knn(account)
            if knn_result:
                ltv_val, rev_val, n, avg_sim = knn_result
                confidence = max(0.3, min(0.8, avg_sim * 0.7 + (n / 20.0) * 0.3))
                return LTVFallbackResult(
                    account_id=account.account_id,
                    ltv_estimated=round(ltv_val, 2),
                    monthly_revenue_estimated=round(rev_val, 2),
                    fallback_strategy_used="knn_similar",
                    confidence=round(confidence, 4),
                    reference_group_size=n,
                    reference_metrics={
                        "avg_similarity": round(avg_sim, 4),
                        "knn_neighbors": n,
                    },
                )
        strategies_tried.append("knn_similar")

        if "global_quantile" in self.config.fallback_strategies and self._global_stats:
            ltv_val = self._global_stats["ltv_quantile"]
            rev_val = self._global_stats["rev_quantile"]
            confidence = 0.4 if self._global_stats["count"] >= 50 else 0.25
            return LTVFallbackResult(
                account_id=account.account_id,
                ltv_estimated=round(ltv_val, 2),
                monthly_revenue_estimated=round(rev_val, 2),
                fallback_strategy_used="global_quantile",
                confidence=round(confidence, 4),
                reference_group_size=self._global_stats["count"],
                reference_metrics={
                    "quantile": self.config.quantile,
                    "global_ltv_quantile": ltv_val,
                    "global_rev_quantile": rev_val,
                },
            )
        strategies_tried.append("global_quantile")

        cfg = DEFAULT_LTV_CONFIG
        sub_fee = cfg["subscription_monthly_fee"].get(plan, 99.0)
        default_months = cfg["retention_target_months"]
        default_ltv = sub_fee * default_months * 0.5
        return LTVFallbackResult(
            account_id=account.account_id,
            ltv_estimated=round(default_ltv, 2),
            monthly_revenue_estimated=round(sub_fee, 2),
            fallback_strategy_used="conservative_default",
            confidence=0.15,
            reference_group_size=0,
            reference_metrics={
                "plan_default_monthly_fee": sub_fee,
                "conservative_discount": 0.5,
            },
        )

    def batch_estimate(
        self,
        accounts: List[AccountBehavior],
    ) -> List[LTVFallbackResult]:
        return [self.estimate(acc) for acc in accounts]

    def generate_fallback_report(
        self,
        results: List[LTVFallbackResult],
    ) -> str:
        if not results:
            return "无兜底估算数据"

        strategy_counts: Dict[str, int] = {}
        total_confidence = 0.0
        for r in results:
            strategy_counts[r.fallback_strategy_used] = strategy_counts.get(
                r.fallback_strategy_used, 0) + 1
            total_confidence += r.confidence

        lines = []
        lines.append("=" * 60)
        lines.append("LTV 缺失客户统计学兜底估算报告")
        lines.append("=" * 60)
        lines.append(f"兜底估算账号数: {len(results)}")
        lines.append(f"平均置信度: {total_confidence / len(results):.4f}")
        lines.append("")
        lines.append("各兜底策略使用分布:")
        for strategy, count in sorted(strategy_counts.items(), key=lambda x: -x[1]):
            pct = count / len(results) * 100
            bar = "█" * int(pct / 5)
            lines.append(f"  {strategy:<25s}: {count:4d} ({pct:5.1f}%) {bar}")
        lines.append("")

        low_conf = [r for r in results if r.confidence < 0.5]
        if low_conf:
            lines.append(f"低置信度账号 (<0.5): {len(low_conf)} 个")
            for r in low_conf[:5]:
                lines.append(
                    f"  - {r.account_id}: conf={r.confidence:.4f}, "
                    f"strategy={r.fallback_strategy_used}, "
                    f"est_ltv=¥{r.ltv_estimated:,.2f}"
                )
            if len(low_conf) > 5:
                lines.append(f"  ... 还有 {len(low_conf) - 5} 个")

        lines.append("=" * 60)
        return "\n".join(lines)


def calculate_ltv_with_fallback(
    accounts: List[AccountBehavior],
    scores: List[RiskScore],
    config: Optional[Dict] = None,
    fallback_config: Optional[LTVFallbackConfig] = None,
) -> Tuple[List[AccountLTV], List[LTVFallbackResult]]:
    valid_ltvs: List[AccountLTV] = []
    fallback_results: List[LTVFallbackResult] = []
    valid_accounts_for_fallback: List[AccountBehavior] = []
    missing_accounts: List[AccountBehavior] = []

    score_map = {s.account_id: s for s in scores}

    for acc in accounts:
        has_enough_data = (
            acc.subscription_age_days > 7
            and (acc.total_transactions_last_30d > 0 or acc.login_count_last_30d > 0)
        )
        if has_enough_data and acc.account_id in score_map:
            ltv = calculate_ltv_for_account(acc, score_map[acc.account_id], config)
            valid_ltvs.append(ltv)
            valid_accounts_for_fallback.append(acc)
        else:
            missing_accounts.append(acc)

    if missing_accounts and valid_ltvs:
        estimator = LTVFallbackEstimator(
            reference_accounts=valid_accounts_for_fallback,
            reference_ltvs=valid_ltvs,
            config=fallback_config,
        )
        fallback_results = estimator.batch_estimate(missing_accounts)

        for fb, acc in zip(fallback_results, missing_accounts):
            default_score = RiskScore(
                account_id=acc.account_id,
                total_score=50.0,
                feature_scores={},
                risk_level="medium",
                risk_percentile=50.0,
            )
            risk_score = score_map.get(acc.account_id, default_score)
            fallback_ltv = AccountLTV(
                account_id=acc.account_id,
                historical_ltv=round(fb.ltv_estimated * 0.5, 2),
                predicted_ltv=fb.ltv_estimated,
                monthly_revenue=fb.monthly_revenue_estimated,
                churn_cost=round(fb.monthly_revenue_estimated * (config or DEFAULT_LTV_CONFIG).get("churn_cost_multiplier", 2.5), 2),
                retention_value=round(fb.ltv_estimated - fb.monthly_revenue_estimated * 2.5, 2),
                priority_score=round(
                    (risk_score.total_score / 100.0) * 0.6 + min(1.0, fb.ltv_estimated / 100000.0) * 0.4,
                    4,
                ),
                ltv_tier="bronze",
            )
            valid_ltvs.append(fallback_ltv)

    elif missing_accounts and not valid_ltvs:
        for acc in missing_accounts:
            default_ltv = DEFAULT_LTV_CONFIG["subscription_monthly_fee"].get(
                acc.plan_level.lower(), 99.0)
            fallback_results.append(LTVFallbackResult(
                account_id=acc.account_id,
                ltv_estimated=round(default_ltv * 6, 2),
                monthly_revenue_estimated=round(default_ltv, 2),
                fallback_strategy_used="conservative_default",
                confidence=0.1,
                reference_group_size=0,
                reference_metrics={},
            ))

    return valid_ltvs, fallback_results

